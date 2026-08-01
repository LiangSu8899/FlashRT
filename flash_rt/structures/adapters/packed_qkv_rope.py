"""Route packed-QKV vision attention through the generic qkv_rope seam."""

from __future__ import annotations

import importlib
import types

import torch
import torch.nn.functional as F

from ..impls.qkv_rope import bind_packed_bias_qkv_rope
from ..guard import GuardRefused
from ..discover import discover


class PackedQkvRopeAdapter:
    """Recognize packed biased QKV plus rotate-half RoPE by capability."""

    __name__ = "packed_qkv_rope"

    def __call__(self, model, plan, caps):
        routes = []
        observed = {}
        refused = []

        capacities = {}
        for seam in discover(model, ("vision_ffn",)):
            block = seam.parent_path
            rows = int(caps.get(seam.path, {}).get("rows", 0))
            if rows > 0:
                capacities[block] = max(capacities.get(block, 0), rows)

        for path, module in model.named_modules():
            qkv = getattr(module, "qkv", None)
            proj = getattr(module, "proj", None)
            if not (
                isinstance(qkv, torch.nn.Linear)
                and isinstance(proj, torch.nn.Linear)
            ):
                continue
            site = f"{path}::packed_qkv_rope"

            def refuse(reason, where=site):
                refused.append((where, f"qkv_rope refused: {reason}"))

            required = (
                "num_heads",
                "head_dim",
                "scaling",
                "config",
                "attention_dropout",
                "is_causal",
            )
            if not all(hasattr(module, attr) for attr in required):
                refuse("host lacks the packed-attention capability slots")
                continue
            if module.training:
                refuse("training/dropout form is outside the inference seam")
                continue
            if bool(module.is_causal):
                refuse("causal attention is outside the bidirectional vision seam")
                continue
            heads, head_dim = int(module.num_heads), int(module.head_dim)
            dim = heads * head_dim
            if (
                qkv.in_features != dim
                or qkv.out_features != 3 * dim
                or proj.in_features != dim
                or proj.out_features != dim
                or qkv.bias is None
                or qkv.weight.dtype is not torch.bfloat16
                or qkv.bias.dtype is not torch.bfloat16
            ):
                refuse("projections do not form BF16 packed equal-head QKV")
                continue
            block_path = path.rsplit(".", 1)[0] if "." in path else ""
            row_capacity = capacities.get(block_path, 0)
            if row_capacity <= 0:
                refuse(
                    "no real vision-token capacity was observed for the "
                    "sibling block"
                )
                continue
            try:
                source = importlib.import_module(type(module).__module__)
                eager_attention = getattr(source, "eager_attention_forward")
                attention_functions = getattr(source, "ALL_ATTENTION_FUNCTIONS")
            except (ImportError, AttributeError, ValueError) as exc:
                refuse(f"cannot resolve the host attention dispatcher: {exc}")
                continue
            implementation = getattr(module.config, "_attn_implementation", None)
            try:
                attention = attention_functions.get_interface(
                    implementation, eager_attention
                )
            except (AttributeError, KeyError, TypeError) as exc:
                refuse(
                    f"attention implementation {implementation!r} is "
                    f"unavailable: {exc}"
                )
                continue
            try:
                bound = bind_packed_bias_qkv_rope(
                    qkv.bias,
                    row_capacity=row_capacity,
                    q_heads=heads,
                    kv_heads=heads,
                    head_dim=head_dim,
                )
            except (ValueError, RuntimeError) as exc:
                refuse(str(exc))
                continue

            original = module.forward
            had_instance_forward = "forward" in module.__dict__

            def routed(
                self,
                hidden_states,
                cu_seqlens,
                position_embeddings=None,
                *,
                rope=bound,
                qkv_proj=qkv,
                output_proj=proj,
                attention_fn=attention,
                attention_scale=float(module.scaling),
                implementation_name=implementation,
                **kwargs,
            ):
                tokens = hidden_states.shape[0]
                if (
                    hidden_states.dim() != 2
                    or hidden_states.shape[1] != qkv_proj.in_features
                    or hidden_states.dtype is not torch.bfloat16
                    or hidden_states.device != qkv_proj.weight.device
                ):
                    raise GuardRefused(
                        "qkv_rope: hidden state is outside the bound "
                        "packed-attention form"
                    )
                if not (
                    isinstance(position_embeddings, tuple)
                    and len(position_embeddings) == 2
                ):
                    raise GuardRefused(
                        "qkv_rope: host did not provide a (cos, sin) table"
                    )
                cos, sin = position_embeddings
                packed = F.linear(
                    hidden_states, qkv_proj.weight, None
                ).view(1, tokens, -1)
                query, key, value = rope(
                    packed, cos.view(1, tokens, -1), sin.view(1, tokens, -1)
                )
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

                if implementation_name == "flash_attention_2":
                    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
                    output, _ = attention_fn(
                        self,
                        query,
                        key,
                        value,
                        attention_mask=None,
                        scaling=attention_scale,
                        dropout=0.0,
                        cu_seq_lens_q=cu_seqlens,
                        cu_seq_lens_k=cu_seqlens,
                        max_length_q=max_seqlen,
                        max_length_k=max_seqlen,
                        is_causal=False,
                        **kwargs,
                    )
                else:
                    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                    splits = [
                        torch.split(tensor, lengths.tolist(), dim=2)
                        for tensor in (query, key, value)
                    ]
                    outputs = [
                        attention_fn(
                            self,
                            q,
                            k,
                            v,
                            attention_mask=None,
                            scaling=attention_scale,
                            dropout=0.0,
                            is_causal=False,
                            **kwargs,
                        )[0]
                        for q, k, v in zip(*splits)
                    ]
                    output = torch.cat(outputs, dim=1)
                output = output.reshape(tokens, -1).contiguous()
                return output_proj(output)

            routed_method = types.MethodType(routed, module)
            routes.append(
                (module, routed_method, original, had_instance_forward)
            )
            observed[site] = bound

        if not routes:
            return {"refused": refused} if refused else None

        def enable() -> None:
            for module, routed, _, _ in routes:
                module.forward = routed

        def disable() -> None:
            for module, _, original, _ in routes:
                module.forward = original

        def revert() -> None:
            for module, _, original, had_instance_forward in routes:
                if had_instance_forward:
                    module.forward = original
                elif "forward" in module.__dict__:
                    del module.forward

        enable()
        return {
            "observed": observed,
            "revert": [revert],
            "toggle": (enable, disable),
            "refused": refused,
        }

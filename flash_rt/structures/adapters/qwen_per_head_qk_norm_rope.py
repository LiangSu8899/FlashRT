"""Compose packed QKV with per-head Q/K norm + RoPE in Qwen-style hosts."""

from __future__ import annotations

import importlib
import types

from ..impls.qk_norm_rope import bind_per_head_gqa_qk_norm_rope
from ..impls.qkv_pack.fp8_static import PackedLinear, StashReader


class QwenPerHeadQkNormRopeAdapter:
    """Route capability-compatible Qwen attention modules through one seam."""

    __name__ = "qwen_per_head_qk_norm_rope"

    def __call__(self, model, plan):
        modules = dict(model.named_modules())
        routes = []
        observed = {}

        for path, module in modules.items():
            pack = plan.swaps.get(f"{path}.q_proj")
            k_reader = plan.swaps.get(f"{path}.k_proj")
            v_reader = plan.swaps.get(f"{path}.v_proj")
            if not (
                isinstance(pack, PackedLinear)
                and isinstance(k_reader, StashReader)
                and isinstance(v_reader, StashReader)
                and k_reader._packed[0] is pack
                and v_reader._packed[0] is pack
            ):
                continue
            if not all(
                hasattr(module, attr)
                for attr in (
                    "q_norm",
                    "k_norm",
                    "head_dim",
                    "num_key_value_groups",
                    "config",
                    "o_proj",
                    "scaling",
                )
            ):
                continue
            if int(module.head_dim) != 128 or len(pack.splits) < 3:
                continue
            q_heads, q_rem = divmod(int(pack.splits[0]), 128)
            kv_heads, k_rem = divmod(int(pack.splits[1]), 128)
            if (
                q_rem
                or k_rem
                or int(pack.splits[2]) != int(pack.splits[1])
                or q_heads != kv_heads * int(module.num_key_value_groups)
            ):
                continue
            q_weight = getattr(module.q_norm, "weight", None)
            k_weight = getattr(module.k_norm, "weight", None)
            if q_weight is None or k_weight is None:
                continue
            eps = getattr(
                module.q_norm,
                "variance_epsilon",
                getattr(module.q_norm, "eps", None),
            )
            if eps is None:
                continue

            source = importlib.import_module(type(module).__module__)
            eager_attention = getattr(source, "eager_attention_forward", None)
            attention_functions = getattr(source, "ALL_ATTENTION_FUNCTIONS", None)
            if eager_attention is None or attention_functions is None:
                continue

            impl = bind_per_head_gqa_qk_norm_rope(
                q_weight,
                k_weight,
                row_capacity=pack.rows,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=128,
                eps=float(eps),
            )
            original = module.forward
            had_instance_forward = "forward" in module.__dict__

            def routed(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                cache_position=None,
                *,
                bound=impl,
                packed=pack,
                eager_fn=eager_attention,
                interfaces=attention_functions,
                **kwargs,
            ):
                batch, tokens, _ = hidden_states.shape
                packed_qkv = packed.joint(hidden_states).view(
                    batch, tokens, -1
                )
                cos, sin = position_embeddings
                query, key, value = bound(packed_qkv, cos, sin)
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                if past_key_values is not None:
                    cache_kwargs = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                    }
                    key, value = past_key_values.update(
                        key,
                        value,
                        self.layer_idx,
                        cache_kwargs,
                    )

                attention = eager_fn
                implementation = self.config._attn_implementation
                if implementation != "eager":
                    attention = interfaces[implementation]
                attention_kwargs = dict(
                    dropout=(
                        0.0 if not self.training else self.attention_dropout
                    ),
                    scaling=self.scaling,
                    **kwargs,
                )
                if hasattr(self, "sliding_window"):
                    attention_kwargs["sliding_window"] = self.sliding_window
                output, weights = attention(
                    self,
                    query,
                    key,
                    value,
                    attention_mask,
                    **attention_kwargs,
                )
                output = output.reshape(batch, tokens, -1).contiguous()
                return self.o_proj(output), weights

            routed_method = types.MethodType(routed, module)
            routes.append(
                (
                    module,
                    pack,
                    routed_method,
                    original,
                    had_instance_forward,
                )
            )
            observed[f"{path}::per_head_qk_norm_rope"] = impl

        if not routes:
            return None

        def enable() -> None:
            for module, pack, routed, _, _ in routes:
                pack.enable_joint(3)
                module.forward = routed

        def disable() -> None:
            for module, pack, _, original, _ in routes:
                module.forward = original
                pack.disable_joint()

        def revert() -> None:
            for module, pack, _, original, had_instance_forward in routes:
                pack.disable_joint()
                if had_instance_forward:
                    module.forward = original
                elif "forward" in module.__dict__:
                    del module.forward

        enable()
        return {
            "observed": observed,
            "revert": [revert],
            "toggle": (enable, disable),
        }

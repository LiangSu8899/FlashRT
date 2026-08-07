"""Q/K norm + RoPE adapter for single-stream packed attention hosts.

The host capability is one sibling-QKV projection group consumed by
per-head RMSNorm and a rotate-half rotary table before a dispatched
self-attention — the plain diffusers processor form, one stream, no
cross attention, no cache. The factored two-way adapter requires two
groups; hosts with a single group fell through it entirely, leaving
their norm/rope chains eager at every layer.

Partial rotary is absorbed at assembly time. The per-head kernel
rotates all 128 channels with a half-split at 64; a host that rotates
only the leading ``R`` channels (half-split ``R/2``) is served by a
channel permutation: the two rotary halves move to kernel slots
``[0:R/2)`` and ``[64:64+R/2)`` — the kernel's pairing then *is* the
host's pairing — and the pass-through channels land in slots whose
tables read ``cos=1, sin=0``. The permutation is one row gather of the
pack's FP8 weight rows (bit-exact, no requantize), the same gather on
the norm weights, and a table remap per call. Q and K share the
permutation, so QK^T is unchanged and nothing downstream un-permutes.

The permutation needs the rotary width, which is a runtime fact — so
it applies lazily on the first routed call (eager warmup), and the
revert path restores the pack rows with the inverse gather.
"""

from __future__ import annotations

import sys
import types

import torch

from ..guard import GuardRefused
from ..impls.qk_norm_rope import bind_per_head_gqa_qk_norm_rope
from ..impls.qkv_pack.fp8_static import PackedLinear, StashReader


def _pack_at(plan, path: str):
    head = plan.swaps.get(f"{path}.to_q")
    key = plan.swaps.get(f"{path}.to_k")
    value = plan.swaps.get(f"{path}.to_v")
    if not (
        isinstance(head, PackedLinear)
        and isinstance(key, StashReader)
        and isinstance(value, StashReader)
        and key._packed[0] is head
        and value._packed[0] is head
    ):
        return None
    return head


def _epsilon(norm) -> float | None:
    value = getattr(norm, "variance_epsilon", getattr(norm, "eps", None))
    return None if value is None else float(value)


class PackedStreamQkNormRopeAdapter:
    """Compose one packed QKV group with per-head Q/K norm and RoPE."""

    __name__ = "packed_stream_qk_norm_rope"

    def __call__(self, model, plan):
        routes = []
        observed = {}
        refused = []

        for path, module in model.named_modules():
            pack = _pack_at(plan, path)
            if pack is None:
                continue
            if hasattr(module, "add_q_proj"):
                continue  # the factored two-way adapter's territory
            site = f"{path}::packed_stream_qk_norm_rope"

            def refuse(reason: str) -> None:
                refused.append((site, f"qk_norm_rope refused: {reason}"))

            if module.training:
                refuse("training/dropout form is outside the inference seam")
                continue
            heads = getattr(module, "heads",
                            getattr(module, "num_attention_heads", None))
            head_dim = getattr(module, "head_dim", None)
            processor = getattr(module, "processor", None)
            to_out = getattr(module, "to_out", None)
            if (heads is None or head_dim is None or processor is None
                    or to_out is None):
                refuse("host lacks the single-stream attention slots")
                continue
            if int(head_dim) != 128:
                refuse("current Hub entry requires head_dim=128")
                continue
            heads = int(heads)
            q_norm = getattr(module, "norm_q", None)
            k_norm = getattr(module, "norm_k", None)
            q_w = getattr(q_norm, "weight", None)
            k_w = getattr(k_norm, "weight", None)
            eps = _epsilon(q_norm)
            if q_w is None or k_w is None or eps is None:
                refuse("Q/K norm weights or epsilon are absent")
                continue
            if tuple(q_w.shape) != (128,) or tuple(k_w.shape) != (128,):
                refuse("per-head norm weights must have shape (head_dim,)")
                continue
            expected = (heads * 128, heads * 128, heads * 128)
            if tuple(pack.splits[:3]) != expected:
                refuse(f"packed widths {tuple(pack.splits[:3])} "
                       f"!= {expected}")
                continue
            dispatch = getattr(
                sys.modules.get(type(processor).__module__),
                "dispatch_attention_fn", None)
            if dispatch is None:
                refuse("host processor module lacks dispatch_attention_fn")
                continue

            try:
                bound = bind_per_head_gqa_qk_norm_rope(
                    q_w, k_w, row_capacity=pack.rows, q_heads=heads,
                    kv_heads=heads, head_dim=128, eps=eps,
                    workspace_lane="stream")
            except (ValueError, RuntimeError) as exc:
                refuse(str(exc))
                continue

            original = module.forward
            had_instance_forward = "forward" in module.__dict__
            state = {"perm": None, "inv": None, "r": None}

            def lazy_permute(rotary_dim: int, _pack=pack, _bound=bound,
                             _state=state, _heads=heads, _qw=q_w, _kw=k_w):
                if rotary_dim % 2 or rotary_dim > 128:
                    raise GuardRefused(
                        "qk_norm_rope: rotary width must be even and "
                        "<= head_dim")
                half = rotary_dim // 2
                if half > 64:
                    raise GuardRefused(
                        "qk_norm_rope: rotary half exceeds the kernel's "
                        "pairing distance")
                perm = torch.empty(128, dtype=torch.long)
                free = list(range(rotary_dim, 128))
                # kernel slot <- host channel
                slot_src = {}
                for i in range(half):
                    slot_src[i] = i
                    slot_src[64 + i] = half + i
                spare = [s for s in range(128) if s not in slot_src]
                for s, c in zip(spare, free):
                    slot_src[s] = c
                for s in range(128):
                    perm[s] = slot_src[s]
                inv = torch.empty_like(perm)
                inv[perm] = torch.arange(128)
                _state.update(perm=perm, inv=inv, r=rotary_dim)
                if rotary_dim == 128:
                    return  # identity pairing, nothing to move
                dev = _pack.w8.device
                pdev = perm.to(dev)
                with torch.no_grad():
                    for g in range(2):        # q rows, k rows
                        base = g * _heads * 128
                        for h in range(_heads):
                            rows = slice(base + h * 128,
                                         base + (h + 1) * 128)
                            _pack.w8[rows] = _pack.w8[rows][pdev].clone()
                    _bound.q_norm_weight.copy_(
                        _qw.detach().to(dev, torch.bfloat16)[pdev])
                    _bound.k_norm_weight.copy_(
                        _kw.detach().to(dev, torch.bfloat16)[pdev])

            def remap_tables(cos, sin, _state=state):
                r = cos.shape[-1]
                if _state["r"] is None:
                    lazy_permute(r)
                elif _state["r"] != r:
                    raise GuardRefused(
                        "qk_norm_rope: rotary width changed after binding")
                if r == 128:
                    return cos, sin
                half = r // 2
                c = cos.new_ones(*cos.shape[:-1], 128)
                s_ = sin.new_zeros(*sin.shape[:-1], 128)
                c[..., :half] = cos[..., :half]
                c[..., 64:64 + half] = cos[..., half:r]
                s_[..., :half] = sin[..., :half]
                s_[..., 64:64 + half] = sin[..., half:r]
                return c, s_

            def routed(self, hidden_states, rotary_emb=None,
                       attention_mask=None, *, _pack=pack, _bound=bound,
                       _remap=remap_tables, _dispatch=dispatch,
                       _proc=processor):
                if rotary_emb is None:
                    raise GuardRefused(
                        "qk_norm_rope: the bound form needs the rotary "
                        "tables")
                cos, sin = rotary_emb
                cos, sin = _remap(cos.to(torch.bfloat16),
                                  sin.to(torch.bfloat16))
                flat = _pack.joint(hidden_states)
                lead = hidden_states.shape[:-1]
                packed = flat.reshape(1, -1, flat.shape[-1])
                if cos.dim() == 2:
                    cos = cos.unsqueeze(0)
                    sin = sin.unsqueeze(0)
                q, k, v = _bound(packed.contiguous(), cos.contiguous(),
                                 sin.contiguous())
                out = _dispatch(
                    q, k, v, attn_mask=attention_mask, dropout_p=0.0,
                    is_causal=False,
                    backend=getattr(_proc, "_attention_backend", None),
                    parallel_config=getattr(_proc, "_parallel_config",
                                            None))
                out = out.flatten(2, 3).type_as(q)
                out = out.reshape(*lead, out.shape[-1])
                for layer in self.to_out:
                    out = layer(out)
                return out

            routes.append((module, pack, state,
                           types.MethodType(routed, module), original,
                           had_instance_forward))
            observed[f"{path}::per_head_qk_norm_rope"] = bound

        if not routes:
            return {"refused": refused} if refused else None

        def enable() -> None:
            for module, pack, _state, routed, _, _ in routes:
                pack.enable_joint(3)
                module.forward = routed

        def disable() -> None:
            for module, pack, _state, _, original, _ in routes:
                module.forward = original
                pack.disable_joint()

        def revert() -> None:
            for module, pack, state, _, original, had in routes:
                pack.disable_joint()
                inv = state.get("inv")
                if inv is not None and state.get("r") != 128 \
                        and state.get("r") is not None:
                    dev = pack.w8.device
                    idev = inv.to(dev)
                    heads = pack.splits[0] // 128
                    with torch.no_grad():
                        for g in range(2):
                            base = g * heads * 128
                            for h in range(heads):
                                rows = slice(base + h * 128,
                                             base + (h + 1) * 128)
                                pack.w8[rows] = pack.w8[rows][idev].clone()
                if had:
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

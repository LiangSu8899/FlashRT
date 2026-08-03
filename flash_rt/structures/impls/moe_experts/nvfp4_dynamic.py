"""NVFP4 (W4A4, dynamic activation scales) ``moe_experts`` implementation.

The expert bank of a sparse-MoE block stores every expert's projections
as stacked 3D tensors (``gate_up_proj [E, 2I, H]``, ``down_proj
[E, H, I]``). Each expert's two matrices are packed to the Hub NVFP4
layout at bind time; activations are quantized per call with dynamic
block scales, same as the ``linear_proj`` sibling. A packed expert row
is a contiguous 2D view of the 3D buffer, so the per-expert GEMM/GEMV
entries consume slices directly — no per-expert module objects, no
gather copies of weight data.

The forward contract mirrors the host bank it replaces:
``forward(hidden, top_k_index, top_k_weights)`` over a flattened token
batch. Experts are visited in ascending index order — a fixed reduction
order, never atomics — and contributions accumulate in FP32 before the
single cast back to the host dtype.

There is no host fallback: binding exists to retire the dense weights
whose footprint keeps the checkpoint off the card, so the guard refuses
out-of-form calls instead of falling back.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam

KERNEL_DEP = {
    "provider": "huggingface_kernels",
    "repo": "flashrt/fp4-gemm",
    "version": ">=1",
}

#: mirrors the kernel's own checks (K divisible by 16 for the per-block
#: scale factors) — both contraction dims of an expert bank are K once:
#: H for gate_up, I for down
SUPPORT = {
    "K": {"min": 16, "multiple_of": 16},
    "N": {"min": 1},
    "E": {"min": 1},
}

#: experts are streamed to the GPU in slabs of this many during bind so
#: the transient footprint stays at slab size, not the full bank
_BIND_SLAB = 32


@lru_cache(maxsize=1)
def _kernel():
    from flash_rt.structures.impls import hub_kernel

    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


def check_experts(weights: Mapping[str, torch.Tensor]) -> tuple[int, int, int]:
    """Validate an expert bank's shapes; returns ``(E, H, I)``."""
    gu, dn = weights["gate_up_proj"], weights["down_proj"]
    if gu.dim() != 3 or dn.dim() != 3:
        raise ValueError(
            f"expert bank must be 3D stacks, got gate_up "
            f"{tuple(gu.shape)}, down {tuple(dn.shape)}")
    e, two_i, h = gu.shape
    e2, h2, i = dn.shape
    if e != e2 or h != h2 or two_i != 2 * i:
        raise ValueError(
            f"inconsistent expert bank: gate_up {tuple(gu.shape)} vs "
            f"down {tuple(dn.shape)}")
    if e < SUPPORT["E"]["min"]:
        raise ValueError(f"E={e} outside support envelope")
    for name, dim in (("H", h), ("I", i)):
        if dim < SUPPORT["K"]["min"] or dim % SUPPORT["K"]["multiple_of"]:
            raise ValueError(
                f"{name}={dim} must be a positive multiple of "
                f"{SUPPORT['K']['multiple_of']} (it is a contraction dim)")
    return e, h, i


def _quantize_activation(kern, flat: torch.Tensor):
    if flat.dtype is torch.bfloat16:
        direct = getattr(kern, "quantize_fp4_sfa_bf16", None)
        if direct is not None:
            return direct(flat.contiguous())
    return kern.quantize_fp4_sfa_fp16(
        flat.to(torch.float16).contiguous())


class MoeExpertsNvfp4Dynamic(GuardedSeam, torch.nn.Module):
    """Packed expert bank: per-expert FP4 GEMMs behind the host contract."""

    _frt_can_fallback = False

    def __init__(self, gu_packed, gu_sfb, dn_packed, dn_sfb, act_fn,
                 num_experts, hidden, inter):
        super().__init__()
        self.register_buffer("_gu_packed", gu_packed)
        self.register_buffer("_gu_sfb", gu_sfb)
        self.register_buffer("_dn_packed", dn_packed)
        self.register_buffer("_dn_sfb", dn_sfb)
        self._act = act_fn
        self._e = num_experts
        self._h = hidden
        self._i = inter
        kern = _kernel()
        self._kern = kern
        self._gemm = kern.fp4_w4a16_linear_bf16
        gemv = getattr(kern, "fp4_w4a4_gemv_warpsplit_bf16", None)

        def _fits(n, k):
            return gemv is not None and n % 8 == 0 and k % (64 * 4) == 0

        self._gemv_gu = gemv if _fits(2 * inter, hidden) else None
        self._gemv_dn = gemv if _fits(hidden, inter) else None
        self._frt_arm(dtypes=CAST_OK, device=gu_packed.device, k=hidden)

    def _expert_mm(self, a_packed, a_sfa, w_packed, w_sfb, m, gemv):
        if m == 1 and gemv is not None:
            return gemv(a_packed, w_packed, a_sfa, w_sfb)
        return self._gemm(a_packed, w_packed, a_sfa, w_sfb, variant=2)

    def forward(self, hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(hidden_states)
        if admitted is not PROCEED:
            return admitted
        out = torch.zeros(hidden_states.shape[0], self._h,
                          device=hidden_states.device, dtype=torch.float32)
        if hidden_states.shape[0] == 1:
            # decode row: gather-then-fixed-shape. The routed experts'
            # packed rows are gathered device-side (index_select reads
            # the routing tensor, the host never does), then a fixed
            # top-k loop of GEMVs runs — no host sync, and the whole
            # step stays legal inside a compiled region or a captured
            # graph, where a host-read of the routing would freeze it.
            # The reduction order is the fixed top-k position order.
            idx = top_k_index[0]
            wts = top_k_weights[0].float()
            gu_p = self._gu_packed.index_select(0, idx)
            gu_s = self._gu_sfb.index_select(0, idx)
            dn_p = self._dn_packed.index_select(0, idx)
            dn_s = self._dn_sfb.index_select(0, idx)
            a_packed, a_sfa = _quantize_activation(self._kern, hidden_states)
            for j in range(int(idx.shape[0])):
                y = self._expert_mm(a_packed, a_sfa, gu_p[j], gu_s[j],
                                    1, self._gemv_gu)
                gate, up = y.chunk(2, dim=-1)
                inter = self._act(gate) * up
                b_packed, b_sfa = _quantize_activation(self._kern, inter)
                d = self._expert_mm(b_packed, b_sfa, dn_p[j], dn_s[j],
                                    1, self._gemv_dn)
                out += wts[j] * d.float()
        elif hidden_states.shape[0] <= 16:
            # short multi-token rows (a verify or rewrite pass): the
            # same gather-then-fixed-shape form, per token — T*k is
            # small and fixed, so the pass stays capturable where the
            # routed-loop form below would sync the host. Beyond the
            # bound the routed loop wins: many tokens share experts and
            # its per-expert GEMMs amortise the weight reads.
            for r in range(int(hidden_states.shape[0])):
                row = hidden_states[r:r + 1]
                idx = top_k_index[r]
                wts = top_k_weights[r].float()
                gu_p = self._gu_packed.index_select(0, idx)
                gu_s = self._gu_sfb.index_select(0, idx)
                dn_p = self._dn_packed.index_select(0, idx)
                dn_s = self._dn_sfb.index_select(0, idx)
                a_packed, a_sfa = _quantize_activation(self._kern, row)
                for j in range(int(idx.shape[0])):
                    y = self._expert_mm(a_packed, a_sfa, gu_p[j],
                                        gu_s[j], 1, self._gemv_gu)
                    gate, up = y.chunk(2, dim=-1)
                    inter = self._act(gate) * up
                    b_packed, b_sfa = _quantize_activation(
                        self._kern, inter)
                    d = self._expert_mm(b_packed, b_sfa, dn_p[j],
                                        dn_s[j], 1, self._gemv_dn)
                    out[r] += wts[j] * d.float().view(-1)
        else:
            for e in torch.unique(top_k_index).tolist():
                pos, tok = torch.where(top_k_index.t() == e)
                cur = hidden_states[tok]
                a_packed, a_sfa = _quantize_activation(self._kern, cur)
                m = cur.shape[0]
                y = self._expert_mm(a_packed, a_sfa, self._gu_packed[e],
                                    self._gu_sfb[e], m, self._gemv_gu)
                gate, up = y.chunk(2, dim=-1)
                inter = self._act(gate) * up
                b_packed, b_sfa = _quantize_activation(self._kern, inter)
                d = self._expert_mm(b_packed, b_sfa, self._dn_packed[e],
                                    self._dn_sfb[e], m, self._gemv_dn)
                scaled = d.float() * top_k_weights[tok, pos, None].float()
                out.index_add_(0, tok, scaled)
        return out.to(hidden_states.dtype)


@torch.no_grad()
def _pack_bank(kern, bank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Pack one 3D stack ``[E, N, K]`` to NVFP4; returns worst relL2."""
    e = bank.shape[0]
    packed, sfb = [], []
    worst = 0.0
    for lo in range(0, e, _BIND_SLAB):
        slab = bank[lo:lo + _BIND_SLAB].to("cuda", torch.float16)
        for j in range(slab.shape[0]):
            w = slab[j].contiguous()
            p, s = kern.quantize_fp4_sfa_fp16(w, is_sfb=True)
            deq = kern.dequantize_fp4_sfa_fp16(p, s)
            num = float((deq.float() - w.float()).square().sum())
            den = float(w.float().square().sum())
            worst = max(worst, (num ** 0.5) / max(den ** 0.5, 1e-12))
            packed.append(p)
            sfb.append(s)
            del deq
        del slab
    return torch.stack(packed), torch.stack(sfb), worst


@torch.no_grad()
def bind_experts_seam(
    weights: Mapping[str, torch.Tensor], act_fn,
) -> tuple[MoeExpertsNvfp4Dynamic, dict[str, float]]:
    """Bind one expert bank from its dense 3D stacks.

    Weights stream to the GPU in expert slabs and pack there; the
    returned dict carries the worst pack-and-unpack relative L2 per
    stack, for the adoption receipt. The bound module holds only the
    packed layout — retiring the dense bank is the caller's move (and
    the point).
    """
    e, h, i = check_experts(weights)
    kern = _kernel()
    gu_packed, gu_sfb, gu_rel = _pack_bank(kern, weights["gate_up_proj"])
    dn_packed, dn_sfb, dn_rel = _pack_bank(kern, weights["down_proj"])
    bound = MoeExpertsNvfp4Dynamic(gu_packed, gu_sfb, dn_packed, dn_sfb,
                                   act_fn, e, h, i)
    # bind-time smoke: one decode-shaped call through the real entries
    probe = bound(
        torch.zeros(1, h, device=gu_packed.device, dtype=torch.bfloat16),
        torch.zeros(1, 1, device=gu_packed.device, dtype=torch.long),
        torch.ones(1, 1, device=gu_packed.device, dtype=torch.bfloat16))
    if probe.shape != (1, h) or not torch.isfinite(probe).all():
        raise ValueError(
            f"refused: moe_experts nvfp4 bind smoke produced shape "
            f"{tuple(probe.shape)}, "
            f"finite={bool(torch.isfinite(probe).all())}")
    return bound, {"gate_up_proj": gu_rel, "down_proj": dn_rel}

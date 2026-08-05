"""qkv_pack in NVFP4 W4A4 with a shared channel balance.

Sibling projections share one input, so under an activation-only
balance they share one balance vector, one dynamic FP4 quantization,
and one packed ``[sum(N_i), K]`` NVFP4 GEMM per call — the structural
form of what a hand-written chain does with a per-forward quantization
memo, held by construction instead of keyed by data pointer. Later
siblings read stashed outputs exactly as the FP8 pack does (the
:class:`~.fp8_static.StashReader` tail is shared; it only needs the
head's stash buffers and guard).

Activation scales are per-block and computed per call, so nothing is
static enough to drift; the calibrated per-channel amax feeds the
balance fold only.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .. import hub_kernel
from ...guard import CAST_OK, PROCEED, GuardedSeam
from .fp8_static import StashReader

KERNEL_DEP = {
    "provider": "huggingface_kernels",
    "repo": "flashrt/fp4-gemm",
    "version": ">=1",
}

_VARIANT = 2


class PackedLinearNvfp4(GuardedSeam, torch.nn.Module):
    """Leaf-form head: one balanced FP4 GEMM, later siblings stashed."""

    _frt_host_attr = "host_linear"
    _frt_can_fallback = True

    def __init__(self, mods: Sequence[torch.nn.Module], w_packed,
                 w_sfb, inv_s, bias_cat, splits, rows: int):
        super().__init__()
        self.host_linear = mods[0]
        kern = hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])
        self._kern = kern
        self._gemm = kern.fp4_w4a16_linear_bf16
        self.splits = splits
        self.rows = rows
        self.register_buffer("wp", w_packed)
        self.register_buffer("wsfb", w_sfb)
        self.register_buffer("inv_s", inv_s)
        self.no_bias = not bool(bias_cat.any())
        self.register_buffer("bias_cat", bias_cat)
        dev = w_packed.device
        for i, n in enumerate(splits[1:], 1):
            self.register_buffer(f"stash{i}", torch.zeros(
                rows, n, device=dev, dtype=torch.bfloat16))
        self._frt_arm(dtypes=CAST_OK, device=dev,
                      k=int(mods[0].weight.shape[1]), row_capacity=rows)

    def forward(self, x):
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        flat = x.reshape(-1, x.shape[-1])
        src = (flat.to(torch.float16) * self.inv_s).contiguous()
        a_packed, a_sfa = self._kern.quantize_fp4_sfa_fp16(src)
        y = self._gemm(a_packed, self.wp, a_sfa, self.wsfb,
                       variant=_VARIANT)
        if not self.no_bias:
            y = y + self.bias_cat
        logical_rows = flat.shape[0]
        off = self.splits[0]
        for i, n in enumerate(self.splits[1:], 1):
            getattr(self, f"stash{i}")[:logical_rows].copy_(
                y[:, off:off + n])
            off += n
        out = y[:, :self.splits[0]].contiguous()
        out = out.reshape(*x.shape[:-1], self.splits[0])
        return out.to(x.dtype)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host_linear"), name)


@torch.no_grad()
def bind_qkv_pack(mods: Sequence[torch.nn.Module], *, channel_amax,
                  rows: int, alpha: float = 0.5, clamp=(0.25, 4.0)):
    """Bind a sibling group; returns replacements in sibling order.

    ``channel_amax`` is the per-input-channel amax at the shared input
    (``[K]``); the balance it fits folds into the concatenated weight
    once — the vector depends on the activation alone, so it is the
    same for every sibling by construction.
    """
    if len(mods) < 2:
        raise ValueError("qkv_pack: need at least two siblings")
    kdims = {m.weight.shape[1] for m in mods}
    if len(kdims) != 1:
        raise ValueError(f"qkv_pack: sibling K dims differ {kdims}")
    from flash_rt.core.quantization import fit_input_channel_balance

    kern = hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])
    w = torch.cat([m.weight.detach() for m in mods], 0)
    amax = torch.as_tensor(channel_amax, device=w.device,
                           dtype=torch.float32)
    w_bal, inv_s = fit_input_channel_balance(
        w.float(), amax, alpha=alpha,
        clamp=(float(clamp[0]), float(clamp[1])),
        out_dtype=torch.float32)
    w_packed, w_sfb = kern.quantize_fp4_sfa_fp16(
        w_bal.to(torch.float16).contiguous(), is_sfb=True)
    splits = [m.weight.shape[0] for m in mods]
    biases = []
    for m in mods:
        b = getattr(m, "bias", None)
        biases.append(b.detach().to(torch.bfloat16) if b is not None
                      else torch.zeros(m.weight.shape[0],
                                       device=w.device,
                                       dtype=torch.bfloat16))
    packed = PackedLinearNvfp4(
        mods, w_packed, w_sfb, inv_s.to(torch.float16),
        torch.cat(biases), splits, rows)
    out = [packed]
    for i, m in enumerate(mods[1:], 1):
        out.append(StashReader(m, packed, i))
    return out

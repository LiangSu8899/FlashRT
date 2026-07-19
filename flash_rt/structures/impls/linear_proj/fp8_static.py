"""FP8-static implementation of the ``linear_proj`` structure.

Wraps the fused BF16-entry FP8 projection from ``flashrt/flashrt-fp8-ffn``
(``bf16_fp8_linear_bias_bf16``: fused input quantization, FP8 weights,
BF16 bias/output) behind the linear_proj boundary. Activation scale is
static per-tensor, calibrated from caller-provided representative
inputs.

Qualification is work-based, derived from standalone preflight on the
target shapes: the fused entry pays a fixed input-quantization cost, so
projections whose GEMM is too small to amortize it (M*N*K below the
floor) are refused at bind time — the caller records the refusal, the
host keeps its own Linear.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

import torch

KERNEL_DEP = {
    "provider": "hf",
    "repo": "flashrt/flashrt-fp8-ffn",
    "version": ">=1",
}

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0

SUPPORT = {
    # measured on RTX 5090: wins at M=712 K=2048 (1.2-1.9x), loses at
    # M=51 K=1024 (quant cost > GEMM). Floor sits between those bands.
    "flops_min": 2.0e8,
    "K": {"min": 512, "max": 16384},
    "N": {"min": 128, "max": 16384},
}


@lru_cache(maxsize=1)
def _kernel():
    from flash_rt.structures.impls import hub_kernel

    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


def _amax_scale(t: torch.Tensor) -> torch.Tensor:
    return (t.float().abs().max() / _FP8_MAX).clamp(min=1e-8)


class FusedLinearProj(torch.nn.Module):
    """Drop-in replacement for one nn.Linear projection.

    ``original`` is retained whole and attribute lookups fall through to
    it, so host code that introspects ``weight``/``bias``/``in_features``
    keeps working.
    """

    def __init__(self, w_fp8, bias, input_scale, weight_scale,
                 original: torch.nn.Module | None = None):
        super().__init__()
        self._w_fp8 = w_fp8
        self._bias = bias
        self._input_scale = input_scale
        self._weight_scale = weight_scale
        self._bufs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if original is not None:
            self.host_linear = original

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_linear":
                raise
            return getattr(super().__getattr__("host_linear"), name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        m = flat.shape[0]
        bufs = self._bufs.get(m)
        if bufs is None:
            bufs = (torch.empty_like(flat, dtype=_FP8),
                    torch.empty(m, self._w_fp8.shape[0], device=x.device,
                                dtype=torch.bfloat16))
            self._bufs[m] = bufs
        x_fp8, out = bufs
        y = _kernel().bf16_fp8_linear_bias_bf16(
            flat.to(torch.bfloat16).contiguous(), self._w_fp8, self._bias,
            self._input_scale, self._weight_scale,
            input_fp8=x_fp8, out=out)
        return y.reshape(*shape[:-1], y.shape[-1]).to(x.dtype)


@torch.no_grad()
def bind_proj_seam(
    weights: Mapping[str, torch.Tensor],
    *,
    calibration: Sequence[torch.Tensor],
    original: torch.nn.Module | None = None,
) -> FusedLinearProj:
    """Bind one projection: ``weights['w']`` is checkpoint-layout [N, K].

    ``calibration``: real inputs of the projection. Work-based
    qualification uses the median calibration M.
    """
    if not calibration:
        raise ValueError("calibration must be non-empty")
    w = weights["w"]
    n, k = w.shape
    for name, dim in (("K", k), ("N", n)):
        lo, hi = SUPPORT[name]["min"], SUPPORT[name]["max"]
        if not lo <= dim <= hi:
            raise ValueError(f"{name}={dim} outside support envelope")
    ms = sorted(int(t.reshape(-1, t.shape[-1]).shape[0])
                for t in calibration)
    m_med = ms[len(ms) // 2]
    if m_med * n * k < SUPPORT["flops_min"]:
        raise ValueError(
            f"projection work {m_med}x{n}x{k} below amortization floor "
            f"({SUPPORT['flops_min']:.0e}) — fused quant cost would not "
            "pay for itself; host keeps its Linear")
    if not w.is_cuda:
        raise ValueError("fp8_static requires CUDA-resident weights")

    device = w.device
    w_scale = _amax_scale(w)
    w_fp8 = (w.float() / w_scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
    amax = torch.zeros((), device=device)
    for t in calibration:
        amax = torch.maximum(amax, t.float().abs().max().to(device))
    input_scale = (amax / _FP8_MAX).clamp(min=1e-8)
    bias = weights.get("b")
    if bias is None:
        bias = torch.zeros(n, device=device, dtype=torch.bfloat16)
    else:
        bias = bias.detach().to(torch.bfloat16)
    bound = FusedLinearProj(w_fp8, bias, input_scale.view(1),
                            w_scale.view(1), original=original)
    for m in set(ms):  # pre-allocate per calibrated M: keeps the hot
        bound._bufs[m] = (  # path allocation-free (graph/compile safe)
            torch.empty(m, k, device=device, dtype=_FP8),
            torch.empty(m, n, device=device, dtype=torch.bfloat16))
    return bound

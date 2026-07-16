"""FP8-static implementation of the ``decoder_ffn`` structure.

Composes the fused FP8 gate/up -> activation -> down block from the
``flashrt/flashrt-fp8-swiglu-ffn`` Hub kernel behind the structure
boundary. The norm and optional AdaLN modulation run in torch and feed
the fused block through static-scale FP8 quantization. Activation
scales are calibrated at bind time from caller-provided representative
inputs; weight scales are per-tensor amax computed while packing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

import torch

KERNEL_DEP = {
    "provider": "hf",
    "repo": "flashrt/flashrt-fp8-swiglu-ffn",
    "version": ">=1",
}

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0
_ENTRYPOINTS = {"gelu": "fp8_geglu_mlp_bf16", "silu": "fp8_swiglu_mlp_bf16"}

SUPPORT = {
    "D": {"min": 512, "max": 16384},
    "F": {"min": 1024, "max": 16384},
    "m_classes": ("micro", "small", "medium"),
}


@lru_cache(maxsize=1)
def _kernel():
    from kernels import get_kernel

    return get_kernel(KERNEL_DEP["repo"], version=KERNEL_DEP["version"])


def _amax_scale(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.float().abs().max() / _FP8_MAX).clamp(min=1e-8)


def _quantize(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (tensor.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)


def _normalize(
    x: torch.Tensor,
    w_norm: torch.Tensor,
    mode: str,
    cond_scale: torch.Tensor | None,
    cond_shift: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
    if mode == "offset":
        h = h * (1.0 + w_norm.float())
    elif mode == "direct":
        h = h * w_norm.float()
    else:
        raise ValueError(f"unknown norm_weight_mode: {mode!r}")
    if cond_scale is not None:
        h = h * (1.0 + cond_scale.float())
    if cond_shift is not None:
        h = h + cond_shift.float()
    return h.to(torch.bfloat16)


@dataclass(frozen=True)
class BoundDecoderFfnFp8:
    """Bound callable: boundary inputs in, boundary output out."""

    fused_mlp: Callable[..., torch.Tensor]
    w_norm: torch.Tensor
    gate_up_fp8: torch.Tensor
    down_fp8: torch.Tensor
    input_scale: torch.Tensor
    gate_up_scale: torch.Tensor
    hidden_scale: torch.Tensor
    down_scale: torch.Tensor
    norm_weight_mode: str
    eps: float

    def __call__(
        self,
        x: torch.Tensor,
        *,
        cond_scale: torch.Tensor | None = None,
        cond_shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = _normalize(x, self.w_norm, self.norm_weight_mode,
                       cond_scale, cond_shift, self.eps)
        out = self.fused_mlp(
            _quantize(h, self.input_scale),
            self.gate_up_fp8,
            self.down_fp8,
            self.input_scale.view(1),
            self.gate_up_scale.view(1),
            self.hidden_scale.view(1),
            self.down_scale.view(1),
        )
        return x + out.to(x.dtype)


def bind(
    weights: Mapping[str, torch.Tensor],
    *,
    variant: Mapping[str, str],
    calibration_inputs: Sequence[Mapping[str, torch.Tensor]],
    eps: float = 1e-6,
) -> BoundDecoderFfnFp8:
    """Pack weights, calibrate activation scales, return a bound callable.

    ``calibration_inputs`` must be non-empty and drawn from the real
    input distribution of the target binding; static FP8 scales are only
    as trustworthy as the data they were measured on.
    """
    activation = variant.get("activation", "gelu")
    if activation not in _ENTRYPOINTS:
        raise ValueError(f"unsupported activation: {activation!r}")
    mode = variant.get("norm_weight_mode", "offset")
    if not calibration_inputs:
        raise ValueError("calibration_inputs must be non-empty")

    w_norm = weights["w_norm"]
    w_gate, w_up, w_down = weights["w_gate"], weights["w_up"], weights["w_down"]
    dim_d, dim_f = w_gate.shape
    if w_up.shape != (dim_d, dim_f) or w_down.shape != (dim_f, dim_d):
        raise ValueError(
            f"inconsistent weight dims: gate {tuple(w_gate.shape)}, "
            f"up {tuple(w_up.shape)}, down {tuple(w_down.shape)}"
        )
    for name, dim in (("D", dim_d), ("F", dim_f)):
        bounds = SUPPORT[name]
        if not bounds["min"] <= dim <= bounds["max"]:
            raise ValueError(
                f"{name}={dim} outside support envelope "
                f"[{bounds['min']}, {bounds['max']}]"
            )
    if not (w_gate.is_cuda and w_up.is_cuda and w_down.is_cuda):
        raise ValueError("fp8_static requires CUDA-resident weights")

    gate_up = torch.cat([w_gate.t(), w_up.t()], dim=0).contiguous()
    down = w_down.t().contiguous()
    gate_up_scale = _amax_scale(gate_up)
    down_scale = _amax_scale(down)

    act: Callable[[torch.Tensor], torch.Tensor]
    if activation == "gelu":
        act = lambda t: torch.nn.functional.gelu(t, approximate="tanh")
    else:
        act = torch.nn.functional.silu
    input_amax = torch.zeros((), device=w_gate.device)
    hidden_amax = torch.zeros((), device=w_gate.device)
    with torch.no_grad():
        for sample in calibration_inputs:
            h = _normalize(sample["x"], w_norm, mode,
                           sample.get("cond_scale"), sample.get("cond_shift"),
                           eps)
            hidden = act(h.float() @ w_gate.float()) * (h.float() @ w_up.float())
            input_amax = torch.maximum(input_amax, h.float().abs().max())
            hidden_amax = torch.maximum(hidden_amax, hidden.abs().max())

    return BoundDecoderFfnFp8(
        fused_mlp=getattr(_kernel(), _ENTRYPOINTS[activation]),
        w_norm=w_norm,
        gate_up_fp8=_quantize(gate_up, gate_up_scale),
        down_fp8=_quantize(down, down_scale),
        input_scale=(input_amax / _FP8_MAX).clamp(min=1e-8),
        gate_up_scale=gate_up_scale,
        hidden_scale=(hidden_amax / _FP8_MAX).clamp(min=1e-8),
        down_scale=down_scale,
        norm_weight_mode=mode,
        eps=eps,
    )

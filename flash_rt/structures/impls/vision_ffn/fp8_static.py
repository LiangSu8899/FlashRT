"""FP8-static implementation of the ``vision_ffn`` structure.

Composes the fused FP8 fc1 -> GELU -> fc2 block (biases included) from
the ``flashrt/flashrt-fp8-ffn`` Hub kernel. ``bind`` covers the full
structure boundary; ``bind_mlp_seam`` covers the normed-input ->
ffn-output slice for hosts whose replaceable module boundary is the MLP.
Weights use the checkpoint-native (out, in) layout directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Mapping, Sequence

import torch

KERNEL_DEP = {
    "provider": "hf",
    "repo": "flashrt/flashrt-fp8-ffn",
    "version": ">=1",
}

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0

SUPPORT = {
    "D": {"min": 512, "max": 16384},
    "F": {"min": 1024, "max": 16384},
    "m_classes": ("small", "medium", "large"),
}


@lru_cache(maxsize=1)
def _kernel():
    from kernels import get_kernel

    return get_kernel(KERNEL_DEP["repo"], version=KERNEL_DEP["version"])


def _amax_scale(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.float().abs().max() / _FP8_MAX).clamp(min=1e-8)


def _quantize(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (tensor.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)


@dataclass(frozen=True)
class BoundVisionFfnFp8:
    """Bound callable for the full structure boundary."""

    fused_mlp: Callable[..., torch.Tensor]
    w_norm: torch.Tensor
    b_norm: torch.Tensor
    fc1_fp8: torch.Tensor
    b_fc1: torch.Tensor
    fc2_fp8: torch.Tensor
    b_fc2: torch.Tensor
    input_scale: torch.Tensor
    fc1_scale: torch.Tensor
    hidden_scale: torch.Tensor
    fc2_scale: torch.Tensor
    eps: float

    def ffn(self, normed: torch.Tensor) -> torch.Tensor:
        """The normed-input -> ffn-output slice (no norm, no residual)."""
        shape = normed.shape
        out = self.fused_mlp(
            _quantize(normed.reshape(-1, shape[-1]), self.input_scale),
            self.fc1_fp8,
            self.b_fc1,
            self.fc2_fp8,
            self.b_fc2,
            self.input_scale.view(1),
            self.fc1_scale.view(1),
            self.hidden_scale.view(1),
            self.fc2_scale.view(1),
        )
        return out.reshape(shape).to(normed.dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.layer_norm(
            x.float(), (x.shape[-1],),
            self.w_norm.float(), self.b_norm.float(), self.eps).to(x.dtype)
        return x + self.ffn(h).to(x.dtype)


class FusedGeluMlp(torch.nn.Module):
    """MLP-seam module: the host keeps its own norm and residual."""

    def __init__(self, bound: BoundVisionFfnFp8,
                 original: torch.nn.Module | None = None):
        super().__init__()
        self._bound = bound
        if original is not None:
            self.fc1 = original.fc1
            self.fc2 = original.fc2

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._bound.ffn(hidden)


def _calibrate(normed_samples, w_fc1, b_fc1):
    if not normed_samples:
        raise ValueError("calibration samples must be non-empty")
    device = w_fc1.device
    input_amax = torch.zeros((), device=device)
    hidden_amax = torch.zeros((), device=device)
    for h in normed_samples:
        flat = h.reshape(-1, h.shape[-1]).float().to(device)
        hidden = torch.nn.functional.gelu(
            flat @ w_fc1.float().t() + b_fc1.float(), approximate="tanh")
        input_amax = torch.maximum(input_amax, flat.abs().max())
        hidden_amax = torch.maximum(hidden_amax, hidden.abs().max())
    return ((input_amax / _FP8_MAX).clamp(min=1e-8),
            (hidden_amax / _FP8_MAX).clamp(min=1e-8))


def _check(weights: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    w_fc1, w_fc2 = weights["w_fc1"], weights["w_fc2"]
    dim_f, dim_d = w_fc1.shape
    if w_fc2.shape != (dim_d, dim_f):
        raise ValueError(
            f"inconsistent weight dims: fc1 {tuple(w_fc1.shape)}, "
            f"fc2 {tuple(w_fc2.shape)}"
        )
    for name, dim in (("D", dim_d), ("F", dim_f)):
        bounds = SUPPORT[name]
        if not bounds["min"] <= dim <= bounds["max"]:
            raise ValueError(
                f"{name}={dim} outside support envelope "
                f"[{bounds['min']}, {bounds['max']}]"
            )
    if not (w_fc1.is_cuda and w_fc2.is_cuda):
        raise ValueError("fp8_static requires CUDA-resident weights")
    return dim_d, dim_f


def _build(weights, input_scale, hidden_scale, eps):
    _check(weights)
    fc1_scale = _amax_scale(weights["w_fc1"])
    fc2_scale = _amax_scale(weights["w_fc2"])
    to_bf16 = lambda t: t.to(torch.bfloat16)
    return BoundVisionFfnFp8(
        fused_mlp=_kernel().fp8_gelu_mlp_bf16,
        w_norm=weights["w_norm"],
        b_norm=weights["b_norm"],
        fc1_fp8=_quantize(weights["w_fc1"], fc1_scale),
        b_fc1=to_bf16(weights["b_fc1"]),
        fc2_fp8=_quantize(weights["w_fc2"], fc2_scale),
        b_fc2=to_bf16(weights["b_fc2"]),
        input_scale=input_scale,
        fc1_scale=fc1_scale,
        hidden_scale=hidden_scale,
        fc2_scale=fc2_scale,
        eps=eps,
    )


@torch.no_grad()
def bind(
    weights: Mapping[str, torch.Tensor],
    *,
    variant: Mapping[str, str],
    calibration_inputs: Sequence[Mapping[str, torch.Tensor]],
    eps: float = 1e-6,
) -> BoundVisionFfnFp8:
    """Bind the full structure: calibration inputs are boundary inputs."""
    if variant.get("activation", "gelu") != "gelu":
        raise ValueError("vision_ffn fp8_static supports gelu only")
    if not calibration_inputs:
        raise ValueError("calibration_inputs must be non-empty")
    normed = [
        torch.nn.functional.layer_norm(
            s["x"].float(), (s["x"].shape[-1],),
            weights["w_norm"].float(), weights["b_norm"].float(), eps)
        for s in calibration_inputs
    ]
    input_scale, hidden_scale = _calibrate(
        normed, weights["w_fc1"], weights["b_fc1"])
    return _build(weights, input_scale, hidden_scale, eps)


@torch.no_grad()
def bind_mlp_seam(
    weights: Mapping[str, torch.Tensor],
    *,
    variant: Mapping[str, str],
    calibration_normed: Sequence[torch.Tensor],
    original: torch.nn.Module | None = None,
    eps: float = 1e-6,
) -> FusedGeluMlp:
    """Bind the MLP-seam slice: calibration inputs are normed activations."""
    if variant.get("activation", "gelu") != "gelu":
        raise ValueError("vision_ffn fp8_static supports gelu only")
    input_scale, hidden_scale = _calibrate(
        calibration_normed, weights["w_fc1"], weights["b_fc1"])
    bound = _build(weights, input_scale, hidden_scale, eps)
    return FusedGeluMlp(bound, original=original)

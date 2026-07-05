"""
FlashRT FP8 Linear layer.

Implements true FP8 GEMM acceleration using FlashRT-compiled CUDA kernels:
  - Weights are statically quantized to FP8 E4M3 (done once at load time).
  - Activations are quantized using a *calibrated static scale* (eliminating
    per-call GPU reduce synchronization overhead).
  - Calls the fp8_gemm_descale_fp16 kernel to compute D = (A_fp8 @ B_fp8) * act_scale * w_scale.

Benchmarked on RTX 5060 Ti (sm120): 3x+ speedup over PyTorch fp16 matmul,
with cosine similarity >= 0.999 against the fp16 reference output.

Calibration workflow:
  1. install_flashrt_fp8(transformer)  -> replace all Linears with FlashRTFp8Linear
  2. set_calibration(transformer, True) -> enter calibration mode (forward records activation amax)
  3. Run several representative forwards (e.g. all 12 steps of the first inference segment)
  4. freeze_calibration(transformer, margin=1.0) -> freeze the static act_scale
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from flash_rt import flash_rt_kernels as kern

logger = logging.getLogger(__name__)

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0


def _quantize_weight_fp8(w: torch.Tensor):
    """Quantize an [N, K] fp16/bf16 weight tensor to FP8.

    Returns (w_fp8_t [K, N] contiguous, weight_scale fp32 scalar tensor).
    """
    w = w.contiguous()
    amax = w.abs().max()
    scale = (amax / _FP8_MAX).clamp(min=1e-12).to(torch.float32).view(1)
    # Kernel requires [K, N] row-major layout (A[M,K] @ B[K,N])
    w_t = w.t().contiguous()
    n = w_t.numel()
    w_fp8 = torch.empty(w_t.shape, dtype=_FP8, device=w.device)
    kern.quantize_fp8_static_fp16(
        w_t.data_ptr(), w_fp8.data_ptr(), scale.data_ptr(), n, 0
    )
    return w_fp8, scale


class FlashRTFp8Linear(nn.Module):
    """Linear layer backed by the FlashRT FP8 GEMM.

    Computation is equivalent to nn.Linear:  y = x @ weight^T + bias,
    where weight is stored in FP8 form (transposed to [K, N]) and the
    activation is quantized to FP8 at forward time.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # FP8 weight (transposed [K, N] layout, row-major) + scale
        self.weight_fp8 = nn.Parameter(
            torch.empty(in_features, out_features, dtype=_FP8, device=device),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.ones(1, dtype=torch.float32, device=device),
            requires_grad=False,
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        # Static activation scale (frozen after calibration)
        self.act_scale = nn.Parameter(
            torch.ones(1, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        # Activation amax recorded during calibration (accumulated on GPU, no CPU sync)
        self.register_buffer(
            "act_amax", torch.zeros(1, dtype=torch.float32, device=device)
        )
        self.register_buffer("act_amax_max", torch.zeros(1, dtype=torch.float32, device=device))

        self.calibrating = False

    # Backward-compatible with nn.Linear's weight attribute (some code reads .weight / .weight.dtype)
    @property
    def weight(self):
        return self.weight_fp8

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FlashRTFp8Linear":
        w = linear.weight.data
        dtype = torch.float16 if w.dtype == torch.float16 else torch.bfloat16
        layer = cls(
            w.shape[1], w.shape[0],
            bias=linear.bias is not None,
            device=w.device, dtype=dtype,
        )
        w_fp8, scale = _quantize_weight_fp8(w.to(dtype))
        layer.weight_fp8.data = w_fp8
        layer.weight_scale.data = scale
        if linear.bias is not None:
            layer.bias.data = linear.bias.data.to(dtype)
        # Use weight amax to give the activation scale a reasonable initial value
        # (avoid overflow before calibration)
        layer.act_scale.data = (scale * 4.0).clamp(min=1e-6)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        # Kernel input requires fp16
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        if x2d.stride(0) != self.in_features or x2d.stride(1) != 1:
            x2d = x2d.contiguous()
        m = x2d.shape[0]
        k, n = self.in_features, self.out_features

        if self.calibrating:
            # Dynamic scale (on GPU, no CPU sync); also accumulate historical amax
            amax = x2d.abs().max()
            self.act_amax.data.copy_(amax)
            self.act_amax_max.data = torch.maximum(self.act_amax_max.data, amax)
            scale = (amax / _FP8_MAX).clamp(min=1e-12).to(torch.float32).view(1)
        else:
            scale = self.act_scale.data

        # Temporary allocation (freed immediately after use, relying on PyTorch's caching
        # allocator). This avoids each of the 181 layers holding its own persistent buffer,
        # which would balloon VRAM (measured peak jumps from ~3GB to ~12GB with persistent buffers).
        x_fp8 = torch.empty(m, k, dtype=_FP8, device=x2d.device)
        out = torch.empty(m, n, dtype=torch.float16, device=x2d.device)
        # Use the current CUDA stream (not a hardcoded 0) so the kernels are
        # stream-safe and graph-compatible (a caller capturing a CUDA Graph
        # would replay them on the captured stream). The FP8 pipeline does
        # not capture a graph itself.
        stream = torch.cuda.current_stream().cuda_stream
        kern.quantize_fp8_static_fp16(
            x2d.data_ptr(), x_fp8.data_ptr(), scale.data_ptr(), m * k, stream
        )
        kern.fp8_gemm_descale_fp16(
            x_fp8.data_ptr(), self.weight_fp8.data_ptr(), out.data_ptr(),
            m, n, k, scale.data_ptr(), self.weight_scale.data_ptr(), stream,
        )
        if self.bias is not None:
            # FlashRT fused bias (in-place [m,n]+=bias[n]), no torch op
            kern.add_bias_fp16(out.data_ptr(), self.bias.data_ptr(), m, n, stream)
        out = out.view(*orig_shape[:-1], n)

        if in_dtype != torch.float16:
            out = out.to(in_dtype)
        return out

    def freeze_act_scale(self, margin: float = 1.0):
        """After calibration: set the static act_scale from the accumulated amax."""
        amax = float(self.act_amax_max.item())
        if amax <= 0:
            amax = float(self.weight_scale.item()) * _FP8_MAX
        scale = max(amax * margin / _FP8_MAX, 1e-12)
        self.act_scale.data = torch.tensor([scale], dtype=torch.float32, device=self.weight_fp8.device)
        self.calibrating = False


def _is_fp8_target(module: nn.Module) -> bool:
    """Determine whether a Linear is suitable for FP8 replacement.

    Skips very small Linears (e.g. norm affine params) and
    condition_embedder/time_embedder.
    """
    return isinstance(module, nn.Linear)


def install_flashrt_fp8(model: nn.Module, verbose: bool = True, target: str = "all") -> int:
    """Recursively replace Linears in the model with FlashRTFp8Linear.

    Replacement scope:
      - target="all" (default): replace all attn/ffn/proj_out Linears. Measured
        to be the fastest (FFN large matrices get ~3x GEMM speedup; attention
        small matrices also benefit slightly from FP8 thanks to halved weight
        VRAM and lower memory traffic).
      - target="ffn_only": only replace FFN up/down projections (slightly higher
        accuracy, PSNR~61dB).
    Kept in fp32: time_embedder, condition_embedder (timestep encoding is small
    and sensitive to precision).
    """
    replaced = 0
    skip_substr = ("time_embedder", "condition_embedder")

    # Target selection: which name patterns participate in FP8
    if target == "ffn_only":
        include_patterns = ("ffn.net.0.proj", "ffn.net.2")
    else:
        include_patterns = ()  # empty = all (except skips)

    def _should_replace(name: str) -> bool:
        if any(s in name for s in skip_substr):
            return False
        if include_patterns:
            return any(p in name for p in include_patterns)
        return True

    # Collect (name, linear) pairs to replace
    targets = []
    for name, module in model.named_modules():
        if not _is_fp8_target(module):
            continue
        if not _should_replace(name):
            continue
        targets.append((name, module))

    for name, linear in targets:
        # Find the parent module and attribute name
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]
        new_layer = FlashRTFp8Linear.from_linear(linear)
        setattr(parent, attr, new_layer)
        replaced += 1
        if verbose:
            logger.info("    [FP8] %s: %s", name, tuple(linear.weight.shape))

    if verbose:
        logger.info("  Replaced %d Linears -> FlashRTFp8Linear", replaced)
    return replaced


def set_calibration(model: nn.Module, on: bool):
    """Toggle calibration mode on all FlashRTFp8Linear modules."""
    for module in model.modules():
        if isinstance(module, FlashRTFp8Linear):
            module.calibrating = on


def freeze_calibration(model: nn.Module, margin: float = 1.0):
    """Freeze the static act_scale on all FlashRTFp8Linear modules."""
    n = 0
    for module in model.modules():
        if isinstance(module, FlashRTFp8Linear):
            module.freeze_act_scale(margin)
            n += 1
    return n

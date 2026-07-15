"""FlashRT -- MiniMax-Remover VAE optimization: fp16-native fused kernels.

Replaces diffusers WanRMS_norm.forward (4 full-tensor fp32 passes,
~0.45 ms each at [1,384,1,240,432]) with the FlashRT
``fp16_rms_norm_ncdhw`` CUDA kernel (single-pass, fp16 in/out, fp32
internal statistics, ~0.07 ms -- a ~6x speed-up per call).

Additionally fuses RMS_norm + SiLU in every WanResidualBlock via
``fp16_rms_silu_ncdhw`` (one pass instead of norm->write->silu->write),
and eliminates the redundant fp32 cast in WanUpsample (nearest-exact
upsample is index-only, so fp16 == fp32 bit-for-bit).

Key design decision: **no dtype cast**. The VAE stays in fp16 (cuDNN
already dispatches fp16 tensorop conv kernels). Only the norm/activation
ops are replaced. This preserves fp16's 10-bit mantissa end-to-end,
keeping PSNR at ~40 dB vs the fp16 reference (vs ~15 dB for the
bf16-cast path which loses 3 bits of mantissa across 52 RMS_norm
layers).
"""
from __future__ import annotations

import logging
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The fp16 fused kernels live in a standalone pybind module
# (flash_rt_minimax_remover) that is opt-in:
#   cmake -DFLASHRT_ENABLE_MINIMAX_REMOVER=ON -DGPU_ARCH=120 ...
_fvk = None
try:
    from flash_rt import flash_rt_minimax_remover as _fvk
except ImportError:
    try:
        import flash_rt_minimax_remover as _fvk
    except ImportError:
        pass

_FP16 = torch.float16
_EPS = 1e-6


def _shape_ncdhw(x: torch.Tensor):
    """Return (B, C, T, H, W) for 4D/5D NCDHW tensors, else None."""
    if x.dim() == 4:
        B, C, H, W = x.shape
        return B, C, 1, H, W
    if x.dim() == 5:
        B, C, T, H, W = x.shape
        return B, C, T, H, W
    return None


def _prep_gamma_bias(gamma, bias):
    """Return contiguous fp16 (gamma_flat, bias_ptr) for the kernels."""
    if gamma.dtype != _FP16:
        gamma = gamma.to(_FP16)
    gamma_flat = gamma.contiguous().view(-1)
    if isinstance(bias, torch.Tensor):
        bias_flat = bias.contiguous().view(-1).to(_FP16)
        return gamma_flat, bias_flat.data_ptr()
    return gamma_flat, 0


def _ref_rms_norm(gamma, bias, x):
    """Reference fallback (fp32 stats, fp16 out) -- WanRMS_norm semantics."""
    C = x.shape[1]
    scale = C ** 0.5
    out = torch.nn.functional.normalize(x.float(), dim=1).to(x.dtype)
    return out * scale * gamma + (bias if isinstance(bias, torch.Tensor) else 0.0)


def _flashrt_fp16_rms_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """FlashRT fp16-native RMS_norm replacement for WanRMS_norm.forward.

    Computes: y = (x / rms(x)) * gamma + bias  (fp16 in/out, fp32 stats)
    which equals WanRMS_norm's F.normalize(x, dim=1) * sqrt(C) * gamma.
    """
    shp = _shape_ncdhw(x)
    if shp is None:
        return self._orig_forward(x)
    B, C, T, H, W = shp

    if x.dtype != _FP16:
        x = x.to(_FP16)
    if not x.is_contiguous():
        x = x.contiguous()

    gamma_flat, bias_ptr = _prep_gamma_bias(self.gamma, self.bias)
    out = torch.empty_like(x)
    stream = torch.cuda.current_stream().cuda_stream
    rc = _fvk.fp16_rms_norm_ncdhw(
        x.data_ptr(), gamma_flat.data_ptr(), bias_ptr,
        out.data_ptr(), B, C, T, H, W, _EPS, stream)
    if rc != 0:
        return _ref_rms_norm(self.gamma, self.bias, x)
    return out


class _FusedRmsSilu(nn.Module):
    """Drop-in for WanRMS_norm that outputs silu(rms_norm(x)) in one kernel.

    Installed as WanResidualBlock.norm1/norm2 while the block's
    ``nonlinearity`` is swapped to ``Identity`` -- so the existing
    ``forward`` (norm1 -> nonlinearity -> conv1 -> norm2 -> nonlinearity
    -> conv2) silently becomes a fused norm+silu path with no rewrite of
    the complex causal-cache logic.
    """

    def __init__(self, gamma, bias):
        super().__init__()
        self.gamma = gamma
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = _shape_ncdhw(x)
        if shp is None:
            return torch.nn.functional.silu(_ref_rms_norm(self.gamma, self.bias, x))
        B, C, T, H, W = shp

        if x.dtype != _FP16:
            x = x.to(_FP16)
        if not x.is_contiguous():
            x = x.contiguous()

        gamma_flat, bias_ptr = _prep_gamma_bias(self.gamma, self.bias)
        out = torch.empty_like(x)
        stream = torch.cuda.current_stream().cuda_stream
        rc = _fvk.fp16_rms_silu_ncdhw(
            x.data_ptr(), gamma_flat.data_ptr(), bias_ptr,
            out.data_ptr(), B, C, T, H, W, _EPS, stream)
        if rc != 0:
            return torch.nn.functional.silu(_ref_rms_norm(self.gamma, self.bias, x))
        return out


def install_flashrt_fp16_rms_norm(vae) -> int:
    """Replace WanRMS_norm.forward (attention sites) with the FlashRT fp16
    kernel, and fuse norm+silu inside every WanResidualBlock.

    Attention-block norms (WanAttentionBlock) keep the plain rms_norm
    kernel (no SiLU follows them).  Residual-block norms (norm1/norm2)
    are swapped to the fused ``fp16_rms_silu_ncdhw`` kernel and the
    block's SiLU is set to Identity, so the existing ``forward`` runs the
    fused path without touching the causal-cache logic.

    Returns the count of patched modules.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        WanRMS_norm, WanResidualBlock)

    n_fused = 0
    for blk in vae.modules():
        if isinstance(blk, WanResidualBlock):
            blk.norm1 = _FusedRmsSilu(blk.norm1.gamma, blk.norm1.bias)
            blk.norm2 = _FusedRmsSilu(blk.norm2.gamma, blk.norm2.bias)
            blk.nonlinearity = nn.Identity()
            n_fused += 1

    if not getattr(WanRMS_norm, "_flashrt_fp16_patched", False):
        WanRMS_norm._orig_forward = WanRMS_norm.forward
        WanRMS_norm.forward = _flashrt_fp16_rms_norm_forward
        WanRMS_norm._flashrt_fp16_patched = True
        logger.info("[minimax-vae] patched WanRMS_norm.forward -> FlashRT "
                    "fp16_rms_norm_ncdhw (fp16-native, no cast, ~6x faster)")

    logger.info("[minimax-vae] %d WanResidualBlock(s) now use fused "
                "fp16_rms_silu_ncdhw (norm+silu in one pass)", n_fused)
    return n_fused


def _install_wan_upsample_no_cast(vae) -> int:
    """Eliminate the redundant fp32 cast in WanUpsample.

    WanUpsample.forward does ``super().forward(x.float()).type_as(x)``.
    For ``nearest-exact`` mode the upsample is pure index selection (no
    arithmetic), so fp16 and fp32 give bit-identical results -- the cast
    is wasted bandwidth.  This swaps it to a fp16-native forward.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanUpsample

    if not getattr(WanUpsample, "_flashrt_nocast", False):
        _orig_upsample_forward = WanUpsample.forward

        def _no_cast_forward(self, x):
            if self.mode == "nearest-exact":
                return nn.Upsample.forward(self, x)
            return _orig_upsample_forward(self, x)

        WanUpsample.forward = _no_cast_forward
        WanUpsample._flashrt_nocast = True
        logger.info("[minimax-vae] patched WanUpsample.forward -> "
                    "fp16-native (nearest-exact cast eliminated)")

    return sum(1 for m in vae.modules() if isinstance(m, WanUpsample))


def install_vae_optimizations(vae, dtype=None, use_fp8_conv: bool = True) -> Dict:
    """Apply VAE optimizations: fp16-native fused RMS_norm + RMS_SiLU kernels
    + WanUpsample cast elimination + channels-last 3D pipeline + FP8
    implicit-GEMM conv3d.

    No dtype cast is applied -- the VAE stays fp16. Only norm/activation
    ops are replaced with FlashRT fp16 CUDA kernels, and applicable 3x3x3
    conv3d layers are replaced with FP8 implicit-GEMM kernels (fp16 in/out,
    FP8 e4m3 internal compute).

    Requires the standalone ``flash_rt_minimax_remover`` module, built with:
        cmake -DFLASHRT_ENABLE_MINIMAX_REMOVER=ON -DGPU_ARCH=120 ...

    Args:
        vae: loaded ``diffusers.AutoencoderKLWan`` instance.
        dtype: ignored (kept for API compat with the old bf16 interface).
        use_fp8_conv: if True, replace applicable 3x3x3 conv3d with the
            FP8 implicit-GEMM kernel (default True).

    Returns:
        stats dict.

    Raises:
        ImportError if flash_rt_minimax_remover is not built.
    """
    if _fvk is None:
        raise ImportError(
            "flash_rt_minimax_remover not found; rebuild FlashRT with: "
            "cmake -DFLASHRT_ENABLE_MINIMAX_REMOVER=ON -DGPU_ARCH=120 ...")
    n_fused_blocks = install_flashrt_fp16_rms_norm(vae)
    n_upsample = _install_wan_upsample_no_cast(vae)
    _install_channels_last_pipeline(vae)
    n_fp8_conv = _install_fp8_conv3d_pipeline(vae, enabled=use_fp8_conv)
    return {
        "n_fused_res_blocks": n_fused_blocks,
        "n_upsample_nocast": n_upsample,
        "n_fp8_conv3d": n_fp8_conv,
        "vae_dtype": str(next(vae.parameters()).dtype),
    }


def _install_channels_last_pipeline(vae) -> int:
    """Enable channels-last 3D (NDHWC) throughout the VAE pipeline.

    Converts Conv3d weights to channels-last, patches WanCausalConv3d
    to preserve the format, and replaces the FlashRT norm kernels with
    channels-last variants so norm output stays NDHWC.

    This eliminates ALL nchw↔nhwc conversion kernels that cuDNN inserts
    (~287 ms / decode) and gives a ~1.3x speedup on the dominant conv
    layers.  Zero precision loss (identical fp16 computation, only the
    memory layout changes).
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        WanCausalConv3d, WanRMS_norm)

    n = 0

    # 1. Convert Conv3d weights to channels-last 3D.
    for m in vae.modules():
        if isinstance(m, WanCausalConv3d):
            with torch.no_grad():
                m.weight.data = m.weight.data.to(
                    memory_format=torch.channels_last_3d)
            n += 1

    # 2. Patch WanCausalConv3d.forward to preserve channels-last.
    if not getattr(WanCausalConv3d, "_flashrt_cl_fwd", False):
        def _cl_conv_forward(self, x, cache_x=None):
            if x.dim() == 5 and not x.is_contiguous(
                    memory_format=torch.channels_last_3d):
                x = x.to(memory_format=torch.channels_last_3d)
            padding = list(self._padding)
            if cache_x is not None and self._padding[4] > 0:
                cache_x = cache_x.to(x.device)
                if not cache_x.is_contiguous(
                        memory_format=torch.channels_last_3d):
                    cache_x = cache_x.to(
                        memory_format=torch.channels_last_3d)
                x = torch.cat([cache_x, x], dim=2)
                padding[4] -= cache_x.shape[2]
            x = F.pad(x, padding)
            return F.conv3d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        WanCausalConv3d.forward = _cl_conv_forward
        WanCausalConv3d._flashrt_cl_fwd = True

    # 3. Replace norm kernels with channels-last variants.
    def _cl_rms_norm_forward(self, x):
        if x.dim() == 4:
            # Attention block reshapes to 4D [B*T, C, H, W] — use NCDHW
            # kernel (attention is ~2% of decode time, conversion negligible).
            return _flashrt_fp16_rms_norm_forward(self, x)
        B, C, T, H, W = x.shape
        if x.dtype != _FP16:
            x = x.to(_FP16)
        if not x.is_contiguous(memory_format=torch.channels_last_3d):
            x = x.to(memory_format=torch.channels_last_3d)
        gamma_flat, bias_ptr = _prep_gamma_bias(self.gamma, self.bias)
        out = torch.empty_like(x)
        stream = torch.cuda.current_stream().cuda_stream
        rc = _fvk.fp16_rms_norm_ndhwc(
            x.data_ptr(), gamma_flat.data_ptr(), bias_ptr,
            out.data_ptr(), B, C, T, H, W, _EPS, stream)
        if rc != 0:
            return _ref_rms_norm(self.gamma, self.bias, x)
        return out

    class _FusedRmsSiluCL(nn.Module):
        """Fused RMSNorm+SiLU with optional amax computation for FP8 conv.

        When _fp8_sister_conv is set (by the FP8 conv pipeline), the
        forward computes the output amax via the fused
        fp16_rms_silu_amax_ndhwc kernel and accumulates it into a
        shared running-amax buffer via atomicMax.

        The running-max buffer is shared with the sister conv so the
        conv can skip its own amax pass over both x AND cache_x (the
        cache was a previous output of this same norm, so its amax
        was already accumulated in a prior iteration).
        """
        def __init__(self, gamma, bias):
            super().__init__()
            self.gamma = gamma
            self.bias = bias
            self._fp8_sister_conv = None
            self._amax_buf = None
            self._running_mode = False

        def _ensure_amax_buf(self, device):
            if self._amax_buf is None or self._amax_buf.device != device:
                self._amax_buf = torch.zeros(
                    1, dtype=torch.float32, device=device)

        def forward(self, x):
            if x.dim() == 4:
                return F.silu(_flashrt_fp16_rms_norm_forward(
                    type('_DummyNorm', (), {
                        'gamma': self.gamma, 'bias': self.bias,
                        '_orig_forward': None})(), x))
            B, C, T, H, W = x.shape
            if x.dtype != _FP16:
                x = x.to(_FP16)
            if not x.is_contiguous(memory_format=torch.channels_last_3d):
                x = x.to(memory_format=torch.channels_last_3d)
            gamma_flat, bias_ptr = _prep_gamma_bias(self.gamma, self.bias)
            out = torch.empty_like(x)
            stream = torch.cuda.current_stream().cuda_stream

            if self._fp8_sister_conv is not None:
                self._ensure_amax_buf(x.device)
                if not self._running_mode:
                    self._amax_buf.zero_()
                rc = _fvk.fp16_rms_silu_amax_ndhwc(
                    x.data_ptr(), gamma_flat.data_ptr(), bias_ptr,
                    out.data_ptr(), self._amax_buf.data_ptr(),
                    B, C, T, H, W, _EPS, stream)
            else:
                rc = _fvk.fp16_rms_silu_ndhwc(
                    x.data_ptr(), gamma_flat.data_ptr(), bias_ptr,
                    out.data_ptr(), B, C, T, H, W, _EPS, stream)

            if rc != 0:
                return F.silu(_ref_rms_norm(self.gamma, self.bias, x))
            return out

    # Swap residual-block norms to CL fused variant.
    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        WanResidualBlock)
    for blk in vae.modules():
        if isinstance(blk, WanResidualBlock):
            blk.norm1 = _FusedRmsSiluCL(blk.norm1.gamma, blk.norm1.bias)
            blk.norm2 = _FusedRmsSiluCL(blk.norm2.gamma, blk.norm2.bias)

    # Swap attention-block norm to CL plain variant.
    WanRMS_norm.forward = _cl_rms_norm_forward
    WanRMS_norm._flashrt_cl_norm = True

    logger.info("[minimax-vae] channels-last 3D pipeline enabled: "
                "%d Conv3d weights converted, norm kernels swapped to "
                "NDHWC variant (eliminates ~287ms format conversion)",
                n)
    return n


# ================================================================
# FP8 implicit-GEMM conv3d pipeline
# ================================================================

_FP8_E4M3_MAX = 448.0


def _prequantize_conv3d_weight(conv):
    """Pre-quantize a WanCausalConv3d 3x3x3 weight to FP8 e4m3.

    Returns (w_fp8 [Co,3,3,3,Ci] fp8, w_scale [Co] float32) or
    (None, None) if the conv is not applicable.
    """
    kt, kh, kw = conv.kernel_size
    Ci, Co = conv.in_channels, conv.out_channels
    if ((kt, kh, kw) != (3, 3, 3) or Ci % 32 != 0 or Co < 8 or
            conv.groups != 1):
        return None, None

    w = conv.weight.data
    if not w.is_contiguous(memory_format=torch.channels_last_3d):
        w = w.to(memory_format=torch.channels_last_3d)
    w_perm = w.permute(0, 2, 3, 4, 1).contiguous().float()

    w_scale = w_perm.reshape(Co, -1).abs().amax(dim=1) / _FP8_E4M3_MAX
    w_scale = w_scale.clamp(min=1e-6)

    w_scaled = w_perm / w_scale.reshape(-1, 1, 1, 1, 1)
    w_fp8 = w_scaled.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(
        torch.float8_e4m3fn)
    return w_fp8.contiguous(), w_scale


def _fp8_conv3d_forward(self, x, cache_x=None):
    """FP8 implicit-GEMM conv3d forward for 3x3x3 causal convs.

    Uses a running-max amax strategy: the sister-norm module
    accumulates the output amax of x via atomicMax into a shared
    buffer across iterations.  Since cache_x was a previous output
    of the SAME norm module, its amax is already covered by the
    running max — so we skip the cache amax pass entirely.

    This saves one full read of the cache tensor per layer (~40 MB
    for the largest layers).
    """
    B, Ci, T_new, H, W = x.shape
    Co = self._fp8_w.shape[0]
    stream = torch.cuda.current_stream().cuda_stream

    if not x.is_contiguous(memory_format=torch.channels_last_3d):
        x = x.to(memory_format=torch.channels_last_3d)

    # ── Resolve cache ────────────────────────────────────────────
    if cache_x is not None and cache_x.shape[2] >= 2:
        cache_2 = cache_x[:, :, -2:]
        if not cache_2.is_contiguous(memory_format=torch.channels_last_3d):
            cache_2 = cache_2.to(memory_format=torch.channels_last_3d)
    elif cache_x is not None and cache_x.shape[2] >= 1:
        cache_2 = torch.empty(
            B, Ci, 2, H, W, dtype=x.dtype, device=x.device,
            memory_format=torch.channels_last_3d).zero_()
        cache_2[:, :, 1:2] = cache_x[:, :, -1:]
    else:
        cache_2 = torch.empty(
            B, Ci, 2, H, W, dtype=x.dtype, device=x.device,
            memory_format=torch.channels_last_3d).zero_()

    n_cache = cache_2.numel()
    n_new = x.numel()

    # ── Determine amax ───────────────────────────────────────────
    sister_norm = getattr(self, '_sister_norm', None)
    sister_amax = getattr(sister_norm, '_amax_buf', None) if sister_norm else None
    running_mode = getattr(sister_norm, '_running_mode', False) if sister_norm else False

    if sister_amax is not None and sister_amax.is_cuda and running_mode:
        # Running-max path: amax of x already accumulated by norm.
        # cache_x amax was accumulated in a previous iteration (same norm).
        # No cache amax call needed!
        shared_amax = sister_amax
    elif sister_amax is not None and sister_amax.is_cuda:
        # First-iteration path: norm computed x's amax, accumulate cache amax.
        shared_amax = sister_amax
        _fvk.amax_fp16(cache_2.data_ptr(), shared_amax.data_ptr(),
                       n_cache, stream)
    else:
        # Fallback: compute amax from scratch.
        shared_amax = self._fp8_amax
        shared_amax.zero_()
        _fvk.amax_fp16(cache_2.data_ptr(), shared_amax.data_ptr(),
                       n_cache, stream)
        _fvk.amax_fp16(x.data_ptr(), shared_amax.data_ptr(),
                       n_new, stream)

    # ── Quantize cache + new with shared amax (single launch) ──
    cache_fp8 = torch.empty(
        n_cache, dtype=torch.float8_e4m3fn, device=x.device)
    new_fp8 = torch.empty(
        n_new, dtype=torch.float8_e4m3fn, device=x.device)
    if hasattr(_fvk, 'quantize_fp16_fp8_with_amax_dual'):
        _fvk.quantize_fp16_fp8_with_amax_dual(
            cache_2.data_ptr(), cache_fp8.data_ptr(), n_cache,
            x.data_ptr(), new_fp8.data_ptr(), n_new,
            shared_amax.data_ptr(), self._fp8_scale.data_ptr(),
            stream)
    else:
        _fvk.quantize_fp16_fp8_with_amax(
            cache_2.data_ptr(), cache_fp8.data_ptr(),
            shared_amax.data_ptr(), self._fp8_scale.data_ptr(),
            n_cache, stream)
        _fvk.quantize_fp16_fp8_with_amax(
            x.data_ptr(), new_fp8.data_ptr(),
            shared_amax.data_ptr(), self._fp8_scale.data_ptr(),
            n_new, stream)

    alpha_vec = self._fp8_scale * self._w_scale

    out = torch.empty(
        B, Co, T_new, H, W, dtype=torch.float16, device=x.device,
        memory_format=torch.channels_last_3d)

    rc = _fvk.fp8_conv3d_mm_ndhwc_fp16out(
        cache_fp8.data_ptr(), new_fp8.data_ptr(),
        self._fp8_w.data_ptr(), out.data_ptr(),
        self.bias.data_ptr() if self.bias is not None else 0,
        alpha_vec.data_ptr(),
        B, 2, T_new, H, W, Ci, Co, stream)

    if rc != 0:
        logger.warning("[fp8_conv3d_mm] kernel rc=%d, falling back to "
                       "cuDNN for [%d,%d,%d,%d,%d]", rc, B, Ci, T_new, H, W)
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cx = cache_x.to(x.device)
            if not cx.is_contiguous(memory_format=torch.channels_last_3d):
                cx = cx.to(memory_format=torch.channels_last_3d)
            x_cat = torch.cat([cx, x], dim=2)
            padding[4] -= cx.shape[2]
        else:
            x_cat = x
        x_pad = F.pad(x_cat, padding)
        return F.conv3d(x_pad, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    return out


# ────────────────────────────────────────────────────────────────────────
# FP8 fused norm+silu+running-amax+quant → prequant FP8 conv3d
#
# Patches WanResidualBlock.forward so that, for blocks whose conv1 AND
# conv2 are FP8-eligible, the sister norm produces the FP8 activation
# directly via ``fp16_rms_silu_amax_quant_fp8_ndhwc_nozero`` (one kernel:
# RMSNorm + SiLU + accumulate-running-amax + FP8-quant). The conv then
# only has to quantize the small causal cache (2 frames) and reuse the
# prequant FP8 new-activation -- eliminating the large per-call fp16
# round-trip between norm and conv (the bulk of ``quantize_..._dual``).
#
# Cache correctness: the causal cache (``feat_cache``) stays fp16 (last
# CACHE_T frames of the norm output, computed via a tiny rms_silu on the
# 2-frame slice) so it is re-quantized each call with the current running
# scale -- exactly matching the baseline's shared-scale invariant.
# ────────────────────────────────────────────────────────────────────────
_FUSE_FP8_NORMQUANT = os.environ.get("FLASHRT_FP8_FUSED_NORMQUANT", "1") == "1"


def _make_fp8_fused_residual_forward(orig_forward):
    def _patched_forward(self, x, feat_cache=None, feat_idx=[0]):
        # Only handle blocks where BOTH convs are FP8 (and NOT promoted to
        # NVFP4 — install_vae_nvfp4 sets _nvfp4_w without clearing _fp8_w).
        def _is_fp8(c):
            return (getattr(c, '_fp8_w', None) is not None
                    and getattr(c, '_nvfp4_w', None) is None)
        c1_fp8 = _is_fp8(self.conv1)
        c2_fp8 = _is_fp8(self.conv2)
        if not (c1_fp8 and c2_fp8):
            return orig_forward(self, x, feat_cache, feat_idx)

        try:
            from diffusers.models.autoencoders.autoencoder_kl_wan import CACHE_T
        except Exception:
            CACHE_T = 2
        stream = torch.cuda.current_stream().cuda_stream
        h = self.conv_shortcut(x)  # fp16 shortcut, unchanged

        def _to_cl(t):
            if not t.is_contiguous(memory_format=torch.channels_last_3d):
                return t.to(memory_format=torch.channels_last_3d)
            return t

        def _norm_quant_fp8(norm, xin):
            """RMSNorm+SiLU+running-amax+FP8-quant -> (new_fp8 flat, amax_buf)."""
            B, C, T, Hh, Ww = xin.shape
            xin = _to_cl(xin)
            gamma_flat, bias_ptr = _prep_gamma_bias(norm.gamma, getattr(norm, 'bias', 0))
            if norm._amax_buf is None or norm._amax_buf.device != xin.device:
                norm._amax_buf = torch.zeros(1, dtype=torch.float32, device=xin.device)
            n = B * T * Hh * Ww
            new_fp8 = torch.empty(n * C, dtype=torch.float8_e4m3fn, device=xin.device)
            scale_out = torch.empty(1, dtype=torch.float32, device=xin.device)
            rc = _fvk.fp16_rms_silu_amax_quant_fp8_ndhwc_nozero(
                xin.data_ptr(), gamma_flat.data_ptr(), bias_ptr,
                new_fp8.data_ptr(), scale_out.data_ptr(),
                norm._amax_buf.data_ptr(),
                B, C, T, Hh, Ww, _EPS, stream)
            if rc != 0:
                raise RuntimeError(f"[fp8_fused] norm_quant rc={rc}")
            return new_fp8, norm._amax_buf

        def _cache_fp16_for_next(norm, xin):
            """Tiny rms_silu on last CACHE_T frames -> fp16 cache slice."""
            sl = xin[:, :, -CACHE_T:, :, :]
            sl = _to_cl(sl)
            B, C, Tc, Hh, Ww = sl.shape
            gamma_flat, bias_ptr = _prep_gamma_bias(norm.gamma, getattr(norm, 'bias', 0))
            out = torch.empty_like(sl)
            _fvk.fp16_rms_silu_ndhwc(
                sl.data_ptr(), gamma_flat.data_ptr(), bias_ptr,
                out.data_ptr(), B, C, Tc, Hh, Ww, _EPS, stream)
            return out

        def _conv_prequant(conv, new_fp8, amax_buf, shape_in, cache_x_fp16):
            B, Ci, T_new, Hh, Ww = shape_in
            Co = conv._fp8_w.shape[0]
            # resolve cache_2 [B,Ci,2,H,W] channels-last (mirror _fp8_conv3d_forward)
            if cache_x_fp16 is not None and cache_x_fp16.shape[2] >= 2:
                cache_2 = cache_x_fp16[:, :, -2:]
                if not cache_2.is_contiguous(memory_format=torch.channels_last_3d):
                    cache_2 = cache_2.to(memory_format=torch.channels_last_3d)
            elif cache_x_fp16 is not None and cache_x_fp16.shape[2] >= 1:
                cache_2 = torch.empty(B, Ci, 2, Hh, Ww, dtype=_FP16,
                                      device=conv._fp8_w.device,
                                      memory_format=torch.channels_last_3d).zero_()
                cache_2[:, :, 1:2] = cache_x_fp16[:, :, -1:]
            else:
                cache_2 = torch.empty(B, Ci, 2, Hh, Ww, dtype=_FP16,
                                      device=conv._fp8_w.device,
                                      memory_format=torch.channels_last_3d).zero_()
            n_cache = cache_2.numel()
            cache_fp8 = torch.empty(n_cache, dtype=torch.float8_e4m3fn,
                                    device=cache_2.device)
            # quant cache with the running amax -> writes conv._fp8_scale
            _fvk.quantize_fp16_fp8_with_amax(
                cache_2.data_ptr(), cache_fp8.data_ptr(),
                amax_buf.data_ptr(), conv._fp8_scale.data_ptr(),
                n_cache, stream)
            alpha_vec = conv._fp8_scale * conv._w_scale
            out = torch.empty(B, Co, T_new, Hh, Ww, dtype=_FP16,
                              device=new_fp8.device,
                              memory_format=torch.channels_last_3d)
            bias_ptr = conv.bias.data_ptr() if conv.bias is not None else 0
            rc = _fvk.fp8_conv3d_mm_ndhwc_fp16out(
                cache_fp8.data_ptr(), new_fp8.data_ptr(),
                conv._fp8_w.data_ptr(), out.data_ptr(),
                bias_ptr, alpha_vec.data_ptr(),
                B, 2, T_new, Hh, Ww, Ci, Co, stream)
            if rc != 0:
                raise RuntimeError(f"[fp8_fused] conv3d_prequant rc={rc}")
            return out

        # ── norm1 + conv1 ──────────────────────────────────────────
        x_cl = _to_cl(x)
        B0, C0, T0, H0, W0 = x_cl.shape
        new_fp8_1, amax_1 = _norm_quant_fp8(self.norm1, x_cl)
        cache_in_1 = feat_cache[feat_idx[0]] if feat_cache is not None else None
        x_out = _conv_prequant(self.conv1, new_fp8_1, amax_1,
                               (B0, C0, T0, H0, W0), cache_in_1)
        if feat_cache is not None:
            feat_cache[feat_idx[0]] = _cache_fp16_for_next(self.norm1, x_cl)
            feat_idx[0] += 1

        # ── norm2 + conv2 ──────────────────────────────────────────
        x_out_cl = _to_cl(x_out)
        B1, C1, T1, H1, W1 = x_out_cl.shape
        new_fp8_2, amax_2 = _norm_quant_fp8(self.norm2, x_out_cl)
        cache_in_2 = feat_cache[feat_idx[0]] if feat_cache is not None else None
        x_out2 = _conv_prequant(self.conv2, new_fp8_2, amax_2,
                                (B1, C1, T1, H1, W1), cache_in_2)
        if feat_cache is not None:
            feat_cache[feat_idx[0]] = _cache_fp16_for_next(self.norm2, x_out_cl)
            feat_idx[0] += 1

        return x_out2 + h
    return _patched_forward


def _install_fp8_conv3d_pipeline(vae, enabled: bool = True) -> int:
    """Pre-quantize applicable Conv3d weights to FP8 e4m3 and patch
    WanCausalConv3d.forward to dispatch to the FP8 implicit-GEMM kernel.

    For each 3x3x3 WanCausalConv3d with Ci % 32 == 0 and Co >= 8:
      - Pre-quantize weight to FP8 e4m3 with per-output-channel scale.
      - Store fp8 weight, scale, and scratch buffers as non-persistent
        buffers on the module.

    Non-applicable layers (1x1x1, 3x1x1, Ci%32!=0, Co<8) fall back to
    the channels-last cuDNN path.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        WanCausalConv3d)

    if not enabled:
        return 0

    n_fp8 = 0
    n_skip = 0
    for m in vae.modules():
        if isinstance(m, WanCausalConv3d):
            w_fp8, w_scale = _prequantize_conv3d_weight(m)
            if w_fp8 is not None:
                device = m.weight.device
                m.register_buffer('_fp8_w', w_fp8.to(device),
                                  persistent=False)
                m.register_buffer('_w_scale', w_scale.to(device),
                                  persistent=False)
                m.register_buffer('_fp8_amax',
                                  torch.empty(1, dtype=torch.float32,
                                              device=device),
                                  persistent=False)
                m.register_buffer('_fp8_scale',
                                  torch.empty(1, dtype=torch.float32,
                                              device=device),
                                  persistent=False)
                n_fp8 += 1
            else:
                m._fp8_w = None
                n_skip += 1

    if n_fp8 == 0:
        logger.info("[minimax-vae] FP8 conv3d: 0 layers applicable")
        return 0

    _saved_forward = WanCausalConv3d.forward

    def _fp8_dispatch_forward(self, x, cache_x=None):
        if getattr(self, '_fp8_w', None) is not None:
            return _fp8_conv3d_forward(self, x, cache_x)
        return _saved_forward(self, x, cache_x)

    WanCausalConv3d.forward = _fp8_dispatch_forward
    WanCausalConv3d._flashrt_fp8_patched = True

    # ── Wire up sister-norm references for fused amax ───────────
    # Each FP8-enabled conv gets a reference to its sister norm module
    # so it can reuse the amax computed by fp16_rms_silu_amax_ndhwc.
    # The norm accumulates amax into a running-max buffer (shared with
    # the conv) so the conv can skip amax over both x and cache_x.
    try:
        from diffusers.models.autoencoders.autoencoder_kl_wan import (
            WanResidualBlock)
        n_sister = 0
        for blk in vae.modules():
            if isinstance(blk, WanResidualBlock):
                if getattr(blk.conv1, '_fp8_w', None) is not None:
                    norm1 = blk.norm1
                    if hasattr(norm1, '_fp8_sister_conv'):
                        norm1._fp8_sister_conv = blk.conv1
                        blk.conv1._sister_norm = norm1
                        norm1._running_mode = True
                        n_sister += 1
                if getattr(blk.conv2, '_fp8_w', None) is not None:
                    norm2 = blk.norm2
                    if hasattr(norm2, '_fp8_sister_conv'):
                        norm2._fp8_sister_conv = blk.conv2
                        blk.conv2._sister_norm = norm2
                        norm2._running_mode = True
                        n_sister += 1
        if n_sister > 0:
            logger.info("[minimax-vae] fused norm+silu+amax + running-max: "
                        "%d norm→conv links (skips amax over cache_x entirely)",
                        n_sister)
    except Exception as e:
        logger.debug("[minimax-vae] sister-norm link skipped: %s", e)

    # ── FP8 fused norm+silu+running-amax+quant at residual-block level ──
    # For blocks where BOTH conv1+conv2 are FP8-eligible, patch the block
    # forward to produce the FP8 activation directly in the norm (one
    # kernel) and feed it pre-quantized to the conv (which only re-quants
    # the small causal cache). Eliminates the large fp16 round-trip.
    if _FUSE_FP8_NORMQUANT:
        try:
            from diffusers.models.autoencoders.autoencoder_kl_wan import (
                WanResidualBlock)
            _orig_res_fwd = WanResidualBlock.forward
            _patched = _make_fp8_fused_residual_forward(_orig_res_fwd)
            n_blk = 0
            for blk in vae.modules():
                if isinstance(blk, WanResidualBlock):
                    def _is_fp8(c):
                        return (getattr(c, '_fp8_w', None) is not None
                                and getattr(c, '_nvfp4_w', None) is None)
                    if _is_fp8(blk.conv1) and _is_fp8(blk.conv2):
                        n_blk += 1
            if n_blk > 0:
                WanResidualBlock.forward = _patched
                logger.info("[minimax-vae] FP8 fused norm+silu+quant: "
                            "%d WanResidualBlocks (prequant FP8 conv path)",
                            n_blk)
        except Exception as e:
            logger.warning("[minimax-vae] FP8 fused residual patch failed: %s", e)

    logger.info("[minimax-vae] FP8 implicit-GEMM conv3d: %d layers "
                "quantized, %d skipped (non-3x3x3 or Ci%%32!=0)",
                n_fp8, n_skip)
    return n_fp8




@torch.no_grad()
def profile_vae(pipe, images_tensor, masks_infer, height, width, num_frames,
                iterations=6, num_inference_steps=12, seed=42,
                device=torch.device("cuda:0")) -> Dict[str, float]:
    """Time VAE encode + transformer denoise + VAE decode separately."""
    from einops import rearrange
    from diffusers.utils.torch_utils import randn_tensor

    vae = pipe.vae
    transformer = pipe.transformer
    scheduler = pipe.scheduler

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    num_channels_latents = 16
    vae_scale_factor_temporal = pipe.vae_scale_factor_temporal
    vae_scale_factor_spatial = pipe.vae_scale_factor_spatial
    num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1

    shape = (1, num_channels_latents, num_latent_frames,
             height // vae_scale_factor_spatial,
             width // vae_scale_factor_spatial)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = randn_tensor(shape, generator=generator, device=device,
                           dtype=torch.float16)

    masks = pipe.expand_masks(masks_infer, iterations)
    masks = pipe.resize(masks, height, width).to(device).half()
    masks[masks > 0] = 1
    images = rearrange(images_tensor, "f h w c -> c f h w")
    images = pipe.resize(images[None, ...], height, width).to(device).half()
    masked_images = images * (1 - masks)

    latents_mean = (torch.tensor(vae.config.latents_mean)
                    .view(1, vae.config.z_dim, 1, 1, 1)
                    .to(vae.device, torch.float16))
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1).to(vae.device, torch.float16)

    vae_dtype = next(vae.parameters()).dtype

    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev_enc = torch.cuda.Event(enable_timing=True)
    ev0.record()
    masked_latents = vae.encode(masked_images.to(vae_dtype)).latent_dist.mode()
    masks_latents = vae.encode((2 * masks - 1.0).to(vae_dtype)).latent_dist.mode()
    masked_latents = (masked_latents - latents_mean) * latents_std
    masks_latents = (masks_latents - latents_mean) * latents_std
    ev_enc.record()
    torch.cuda.synchronize()
    vae_encode_ms = ev0.elapsed_time(ev_enc)

    ev_denoise_start = torch.cuda.Event(enable_timing=True)
    ev_denoise_end = torch.cuda.Event(enable_timing=True)
    ev_denoise_start.record()
    for i, t in enumerate(timesteps):
        latent_model_input = latents.to(torch.float16)
        latent_model_input = torch.cat(
            [latent_model_input, masked_latents, masks_latents], dim=1)
        timestep = t.expand(latents.shape[0])
        noise_pred = transformer(
            hidden_states=latent_model_input.half(), timestep=timestep)[0]
        latents = scheduler.step(noise_pred, t, latents,
                                 return_dict=False)[0]
    ev_denoise_end.record()
    torch.cuda.synchronize()
    denoise_ms = ev_denoise_start.elapsed_time(ev_denoise_end)

    latents = latents.half() / latents_std + latents_mean
    ev_dec_start = torch.cuda.Event(enable_timing=True)
    ev_dec_end = torch.cuda.Event(enable_timing=True)
    ev_dec_start.record()
    video = vae.decode(latents.to(vae_dtype), return_dict=False)[0]
    ev_dec_end.record()
    torch.cuda.synchronize()
    vae_decode_ms = ev_dec_start.elapsed_time(ev_dec_end)

    return {
        "vae_encode_ms": vae_encode_ms,
        "denoise_ms": denoise_ms,
        "vae_decode_ms": vae_decode_ms,
        "total_ms": vae_encode_ms + denoise_ms + vae_decode_ms,
    }

"""FlashRT — MiniMax-Remover WanVAE NVFP4 (W4A4) conv3d integration.

Replaces eligible 3×3×3 WanCausalConv3d layers in the WanVAE with an
NVFP4 (W4A4) path that uses:

  1. **FlashRT fp16_quant_nvfp4_ndhwc** (novel CUDA kernel in this repo):
     fp16 NCDHW → NVFP4 packed + UE4M3 block-scale (NDHWC layout).
     Fuses the layout conversion + quantization into one pass.

  2. **motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out** (existing SM120 kernel):
     NVFP4 W4A4 implicit-GEMM conv3d with bias + optional residual.
     Uses mma.sync.kind::mxf4nvf4 (e2m1 × e2m1, UE4M3 block scales).

Weight is pre-quantized once at install time (bf16 → NVFP4 via
``quantize_bf16_to_nvfp4``).  Activations are quantized dynamically
per call (fp16 → FP4, on-GPU, no CPU sync).

Eligibility: 3×3×3 WanCausalConv3d with Ci % 96 == 0 (WanVAE channels
96/192/384 all qualify).  The FP4 path is installed ADDITIVELY over
the existing FP8 path — eligible layers switch to FP4, ineligible
layers stay on FP8.

Env knobs:
  FLASHRT_NVFP4_VAE=1            enable (default ON)
  FLASHRT_NVFP4_VAE_DECODE_ONLY=0  only patch decoder (default 0 = enc+dec)
  FLASHRT_NVFP4_VAE_MIN_CI=192   minimum Ci for FP4 (default 192)
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_FP16 = torch.float16
_BF16 = torch.bfloat16

# Direction 3: block-level fused norm+silu+NVFP4-quant (single kernel).
# Default ON (FLASHRT_NVFP4_FUSED_NORMQUANT=1): the fused kernel emits the
# FP4 activation directly from the norm (no fp16 round-trip) and the conv
# reuses the rolling 2-frame FP4 cache. End-to-end PSNR 35.23 dB vs 35.11
# baseline. The actual gate is read in install_vae_nvfp4() below.
# Set FLASHRT_NVFP4_FUSED_NORMQUANT=0 to fall back to the separate
# norm→fp16→quant path (Direction-2 cache only).

# Lazy-loaded kernel modules
_fvk = None       # flash_rt_minimax_remover (our new kernels)
_frk = None       # flash_rt_kernels (motus FP4 conv3d + weight quant)


def _load_kernels():
    global _fvk, _frk
    if _fvk is not None:
        return True
    try:
        from flash_rt import flash_rt_minimax_remover as _fvk
    except ImportError:
        try:
            import flash_rt_minimax_remover as _fvk
        except ImportError:
            logger.error("[nvfp4_vae] flash_rt_minimax_remover not found")
            return False
    try:
        from flash_rt import flash_rt_kernels as _frk
    except ImportError:
        import flash_rt_kernels as _frk
    if not hasattr(_frk, 'quantize_bf16_to_nvfp4'):
        logger.error("[nvfp4_vae] quantize_bf16_to_nvfp4 not in flash_rt_kernels")
        return False
    if not hasattr(_fvk, 'nvfp4_conv3d_ndhwc_fp16out'):
        logger.error("[nvfp4_vae] nvfp4_conv3d_ndhwc_fp16out not in flash_rt_minimax_remover")
        return False
    return True


# ── Weight pre-quantization (one-time at install) ──

def _quantize_weight_nvfp4(conv_weight: torch.Tensor):
    """Quantize a 3×3×3 conv3d weight [Co,Ci,3,3,3] fp16 to NVFP4.

    Returns (w_fp4 [Co,3,3,3,Ci/2] uint8, w_sf [Co,3,3,3,Ci/16] uint8).
    """
    Co, Ci = conv_weight.shape[0], conv_weight.shape[1]
    # Permute to [Co, kT, kH, kW, Ci] and flatten to [Co*27, Ci]
    w_bf16 = conv_weight.to(_BF16).permute(0, 2, 3, 4, 1).contiguous()
    rows = Co * 27
    w_2d = w_bf16.view(rows, Ci)

    w_fp4 = torch.empty((rows, Ci // 2), dtype=torch.uint8, device=w_2d.device)
    w_sf = torch.empty((rows, Ci // 16), dtype=torch.uint8, device=w_2d.device)
    _frk.quantize_bf16_to_nvfp4(
        w_2d.data_ptr(), w_fp4.data_ptr(), w_sf.data_ptr(),
        rows, Ci, 0)
    torch.cuda.synchronize(w_2d.device)

    return (w_fp4.view(Co, 3, 3, 3, Ci // 2).contiguous(),
            w_sf.view(Co, 3, 3, 3, Ci // 16).contiguous())


# ── Activation quantization (per-call, dynamic) ──

def _quant_act_nvfp4(x: torch.Tensor, B: int, C: int, T: int, H: int, W: int):
    """Quantize fp16 [B,C,T,H,W] → NVFP4 NDHWC flat + SF flat.

    Automatically detects channels-last 3D layout and uses the CL kernel
    variant to avoid a contiguous() copy.
    """
    n = B * T * H * W
    fp4 = torch.empty(n * (C // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty(n * (C // 16), dtype=torch.uint8, device=x.device)
    s = torch.cuda.current_stream().cuda_stream
    if x.is_contiguous(memory_format=torch.channels_last_3d):
        # Channels-last: use CL kernel, no copy needed
        rc = _fvk.fp16_quant_nvfp4_cl_ndhwc(
            x.data_ptr(), fp4.data_ptr(), sf.data_ptr(),
            B, C, T, H, W, s)
    else:
        x_c = x.contiguous(memory_format=torch.contiguous_format)
        rc = _fvk.fp16_quant_nvfp4_ndhwc(
            x_c.data_ptr(), fp4.data_ptr(), sf.data_ptr(),
            B, C, T, H, W, s)
    if rc != 0:
        raise RuntimeError(f"[nvfp4_vae] quant_nvfp4 rc={rc} "
                           f"shape=({B},{C},{T},{H},{W})")
    return fp4, sf


# ── FP4 conv3d forward ──

def _nvfp4_conv3d_forward(self, x: torch.Tensor, cache_x=None):
    """NVFP4 W4A4 conv3d forward with FP4 cache (Direction 2).

    Caches the quantized FP4 activation for the next call's cache,
    eliminating per-call cache quantization.
    """
    B, Ci, T_new, H, W = x.shape
    Co = self._nvfp4_w.shape[0]
    s = torch.cuda.current_stream().cuda_stream

    # Quantize new activation
    new_fp4, new_sf = _quant_act_nvfp4(x, B, Ci, T_new, H, W)

    # FP4 cache: reuse stored FP4 from previous call if available (Direction 2)
    # Set FLASHRT_NVFP4_NO_CACHE=1 to disable (always quantize per call)
    use_fp4_cache = os.environ.get('FLASHRT_NVFP4_NO_CACHE', '0') != '1'
    stored_fp4 = getattr(self, '_nvfp4_stored_fp4', None) if use_fp4_cache else None
    stored_sf = getattr(self, '_nvfp4_stored_sf', None) if use_fp4_cache else None
    stored_T = getattr(self, '_nvfp4_stored_T', 0) if use_fp4_cache else 0

    if stored_fp4 is not None and stored_T >= 2:
        # Use stored 2-frame cache directly (already the right format)
        cache_fp4 = stored_fp4
        cache_sf = stored_sf
    elif stored_fp4 is not None and stored_T == 1:
        # Previous call had T_new=1: pad [zero, prev_frame]
        cache_fp4 = torch.zeros(B * _CACHE_T * H * W * (Ci // 2), dtype=torch.uint8, device=x.device)
        cache_sf = torch.zeros(B * _CACHE_T * H * W * (Ci // 16), dtype=torch.uint8, device=x.device)
        off = B * H * W * (Ci // 2)
        cache_fp4[off:off + stored_fp4.numel()] = stored_fp4
        off_sf = B * H * W * (Ci // 16)
        cache_sf[off_sf:off_sf + stored_sf.numel()] = stored_sf
    elif cache_x is not None and cache_x.shape[2] >= 1:
        # Fallback: quantize fp16 cache per call (same as non-fused path)
        cache_2 = cache_x[:, :, -2:]
        if cache_2.shape[2] < 2:
            pad = torch.zeros(B, Ci, 2 - cache_2.shape[2], H, W,
                              dtype=x.dtype, device=x.device)
            cache_2 = torch.cat([pad, cache_2], dim=2)
        if not cache_2.is_contiguous():
            cache_2 = cache_2.contiguous(
                memory_format=torch.channels_last_3d
                if cache_2.is_contiguous(memory_format=torch.channels_last_3d)
                else torch.contiguous_format)
        cache_fp4, cache_sf = _quant_act_nvfp4(cache_2, B, Ci, 2, H, W)
    else:
        cache_fp4 = torch.zeros(B * _CACHE_T * H * W * (Ci // 2), dtype=torch.uint8, device=x.device)
        cache_sf = torch.zeros(B * _CACHE_T * H * W * (Ci // 16), dtype=torch.uint8, device=x.device)

    # Purpose-built kernel: fp16 NDHWC output
    M_total = B * T_new * H * W
    out_ndhwc = torch.empty(M_total * Co, dtype=_FP16, device=x.device)
    bias_ptr = self.bias.data_ptr() if self.bias is not None else 0

    rc = _fvk.nvfp4_conv3d_ndhwc_fp16out(
        cache_fp4.data_ptr(), new_fp4.data_ptr(),
        self._nvfp4_w.data_ptr(),
        cache_sf.data_ptr(), new_sf.data_ptr(),
        self._nvfp4_sf.data_ptr(),
        out_ndhwc.data_ptr(), bias_ptr,
        B, _CACHE_T, T_new, H, W, Ci, Co, 1.0, s)
    if rc != 0:
        logger.warning("[nvfp4_vae] conv3d kernel rc=%d, falling back to cuDNN", rc)
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cx = cache_x.to(x.device)
            x_cat = torch.cat([cx, x], dim=2)
            padding[4] -= cx.shape[2]
        else:
            x_cat = x
        x_pad = torch.nn.functional.pad(x_cat, padding)
        return torch.nn.functional.conv3d(
            x_pad, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups)

    # Store rolling 2-frame cache for next call (Direction 2)
    if use_fp4_cache:
        if T_new >= 2:
            self._nvfp4_stored_fp4 = new_fp4.view(B, T_new, H, W, Ci // 2)[:, -_CACHE_T:].contiguous().view(-1).clone()
            self._nvfp4_stored_sf = new_sf.view(B, T_new, H, W, Ci // 16)[:, -_CACHE_T:].contiguous().view(-1).clone()
            self._nvfp4_stored_T = _CACHE_T
        elif T_new == 1 and stored_fp4 is not None and stored_T >= 1:
            # Rolling: [prev_last_frame, current_frame]
            prev_fp4_5d = stored_fp4.view(B, stored_T, H, W, Ci // 2)[:, -1:]
            curr_fp4_5d = new_fp4.view(B, 1, H, W, Ci // 2)
            self._nvfp4_stored_fp4 = torch.cat([prev_fp4_5d, curr_fp4_5d], dim=1).contiguous().view(-1).clone()
            prev_sf_5d = stored_sf.view(B, stored_T, H, W, Ci // 16)[:, -1:]
            curr_sf_5d = new_sf.view(B, 1, H, W, Ci // 16)
            self._nvfp4_stored_sf = torch.cat([prev_sf_5d, curr_sf_5d], dim=1).contiguous().view(-1).clone()
            self._nvfp4_stored_T = _CACHE_T
        else:
            self._nvfp4_stored_fp4 = new_fp4.clone()
            self._nvfp4_stored_sf = new_sf.clone()
            self._nvfp4_stored_T = T_new

    out = out_ndhwc.view(B, T_new, H, W, Co).permute(0, 4, 1, 2, 3)
    return out.contiguous(memory_format=torch.channels_last_3d)


# ════════════════════════════════════════════════════════════════════
# Direction 2+3: Fused norm+silu+quant + FP4 cache
# Eliminates both activation quant and cache quant per call.
# ════════════════════════════════════════════════════════════════════

_CACHE_T = 2

def _fused_norm_silu_quant_cl(x, gamma, bias, B, C, T, H, W):
    """Call fused RMS+SiLU+NVFP4quant kernel (channels-last input).
    Returns (fp4_flat, sf_flat).
    """
    n = B * T * H * W
    fp4 = torch.empty(n * (C // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty(n * (C // 16), dtype=torch.uint8, device=x.device)
    s = torch.cuda.current_stream().cuda_stream
    # Ensure channels-last contiguous
    if not x.is_contiguous(memory_format=torch.channels_last_3d):
        x = x.to(memory_format=torch.channels_last_3d)
    bias_ptr = bias.data_ptr() if isinstance(bias, torch.Tensor) else 0
    rc = _fvk.fp16_rms_silu_quant_nvfp4_cl_ndhwc(
        x.data_ptr(), gamma.data_ptr(), bias_ptr,
        fp4.data_ptr(), sf.data_ptr(),
        B, C, T, H, W, 1e-6, s)
    if rc != 0:
        raise RuntimeError(f"[nvfp4_vae] fused_norm_silu_quant rc={rc}")
    return fp4, sf


def _conv3d_prequant(conv, new_fp4, new_sf, cache_fp4, cache_sf,
                     B, Ci, Co, T_new, H, W):
    """Conv3d with pre-quantized FP4 data (no activation/cache quant needed)."""
    s = torch.cuda.current_stream().cuda_stream
    M_total = B * T_new * H * W
    out_ndhwc = torch.empty(M_total * Co, dtype=_FP16, device=conv._nvfp4_w.device)
    bias_ptr = conv.bias.data_ptr() if conv.bias is not None else 0
    rc = _fvk.nvfp4_conv3d_ndhwc_fp16out(
        cache_fp4.data_ptr(), new_fp4.data_ptr(),
        conv._nvfp4_w.data_ptr(),
        cache_sf.data_ptr(), new_sf.data_ptr(),
        conv._nvfp4_sf.data_ptr(),
        out_ndhwc.data_ptr(), bias_ptr,
        B, _CACHE_T, T_new, H, W, Ci, Co, 1.0, s)
    if rc != 0:
        raise RuntimeError(f"[nvfp4_vae] conv3d_prequant rc={rc}")
    out = out_ndhwc.view(B, T_new, H, W, Co).permute(0, 4, 1, 2, 3)
    return out.contiguous(memory_format=torch.channels_last_3d)


def _slice_fp4_cache(fp4_flat, sf_flat, B, C, T, H, W):
    """Extract last 2 frames from flat FP4+SF as cache."""
    fp4_5d = fp4_flat.view(B, T, H, W, C // 2)
    sf_5d = sf_flat.view(B, T, H, W, C // 16)
    return (fp4_5d[:, -_CACHE_T:].contiguous().view(-1),
            sf_5d[:, -_CACHE_T:].contiguous().view(-1))


def _build_fp4_cache(conv, B, C, H, W, device):
    """Construct 2-frame FP4 cache from previous call's stored output.
    Handles T_prev < 2 by zero-padding (matches FP8 path behavior)."""
    prev_fp4 = getattr(conv, '_nvfp4_cache_fp4', None)
    prev_sf = getattr(conv, '_nvfp4_cache_sf', None)
    prev_T = getattr(conv, '_nvfp4_cache_T', 0)

    if prev_fp4 is not None and prev_T >= 2:
        fp4_5d = prev_fp4.view(B, prev_T, H, W, C // 2)
        sf_5d = prev_sf.view(B, prev_T, H, W, C // 16)
        return (fp4_5d[:, -_CACHE_T:].contiguous().view(-1),
                sf_5d[:, -_CACHE_T:].contiguous().view(-1))
    elif prev_fp4 is not None and prev_T == 1:
        # Pad: [zero_frame, prev_frame] (prev at index 1)
        cache_fp4 = torch.zeros(B * _CACHE_T * H * W * (C // 2),
                                 dtype=torch.uint8, device=device)
        cache_sf = torch.zeros(B * _CACHE_T * H * W * (C // 16),
                                dtype=torch.uint8, device=device)
        off = B * H * W * (C // 2)  # offset to second frame
        cache_fp4[off:off + prev_fp4.numel()] = prev_fp4
        off_sf = B * H * W * (C // 16)
        cache_sf[off_sf:off_sf + prev_sf.numel()] = prev_sf
        return cache_fp4, cache_sf
    else:
        return (torch.zeros(B * _CACHE_T * H * W * (C // 2),
                             dtype=torch.uint8, device=device),
                torch.zeros(B * _CACHE_T * H * W * (C // 16),
                             dtype=torch.uint8, device=device))


def _make_patched_residual_forward(orig_forward):
    """Create a patched WanResidualBlock.forward that fuses norm+quant
    and uses FP4 cache for blocks where both conv1 and conv2 are NVFP4-eligible.
    """

    def _patched_forward(self, x, feat_cache=None, feat_idx=[0]):
        # Only handle blocks where BOTH convs are NVFP4-eligible.
        conv1_nvfp4 = getattr(self.conv1, '_nvfp4_w', None) is not None
        conv2_nvfp4 = getattr(self.conv2, '_nvfp4_w', None) is not None
        if not (conv1_nvfp4 and conv2_nvfp4):
            return orig_forward(self, x, feat_cache, feat_idx)

        try:
            from diffusers.models.autoencoders.autoencoder_kl_wan import CACHE_T
        except Exception:
            CACHE_T = 2
        stream = torch.cuda.current_stream().cuda_stream

        def _to_cl(t):
            if not t.is_contiguous(memory_format=torch.channels_last_3d):
                return t.to(memory_format=torch.channels_last_3d)
            return t

        # Shortcut (fp16, unchanged) -- conv_shortcut may itself be NVFP4/FP8.
        h = self.conv_shortcut(x)

        def _norm_quant(norm, xin):
            """Fused RMS+SiLU+NVFP4-quant -> (new_fp4 flat, new_sf flat)."""
            B, C, T, Hh, Ww = xin.shape
            gamma = norm.gamma
            bias = getattr(norm, 'bias', 0)
            bias_ptr = bias.data_ptr() if isinstance(bias, torch.Tensor) else 0
            gamma_flat = gamma.contiguous().view(-1).to(_FP16) if gamma.dtype != _FP16 else gamma.contiguous().view(-1)
            n = B * T * Hh * Ww
            fp4 = torch.empty(n * (C // 2), dtype=torch.uint8, device=xin.device)
            sf = torch.empty(n * (C // 16), dtype=torch.uint8, device=xin.device)
            rc = _fvk.fp16_rms_silu_quant_nvfp4_cl_ndhwc(
                xin.data_ptr(), gamma_flat.data_ptr(), bias_ptr,
                fp4.data_ptr(), sf.data_ptr(),
                B, C, T, Hh, Ww, 1e-6, stream)
            if rc != 0:
                raise RuntimeError(f"[nvfp4_fused] norm_quant rc={rc}")
            return fp4, sf

        def _conv_prequant(conv, new_fp4, new_sf, B, Ci, T_new, Hh, Ww, device):
            """NVFP4 conv with pre-quantized new activation + rolling FP4 cache
            reuse (mirrors Direction-2's _nvfp4_conv3d_forward cache logic, but
            skips the per-call new-x quant since new_fp4 comes fused from the
            norm). This keeps the cache-reuse win AND the norm+quant fusion."""
            Co = conv._nvfp4_w.shape[0]
            use_fp4_cache = os.environ.get('FLASHRT_NVFP4_NO_CACHE', '0') != '1'
            stored_fp4 = getattr(conv, '_nvfp4_stored_fp4', None) if use_fp4_cache else None
            stored_sf = getattr(conv, '_nvfp4_stored_sf', None) if use_fp4_cache else None
            stored_T = getattr(conv, '_nvfp4_stored_T', 0) if use_fp4_cache else 0

            if stored_fp4 is not None and stored_T >= 2:
                cache_fp4 = stored_fp4; cache_sf = stored_sf
            elif stored_fp4 is not None and stored_T == 1:
                # Previous call had T_new=1: pad [zero, prev_frame] (Direction-2)
                cache_fp4 = torch.zeros(B * _CACHE_T * Hh * Ww * (Ci // 2),
                                        dtype=torch.uint8, device=device)
                cache_sf = torch.zeros(B * _CACHE_T * Hh * Ww * (Ci // 16),
                                       dtype=torch.uint8, device=device)
                off = B * Hh * Ww * (Ci // 2)
                cache_fp4[off:off + stored_fp4.numel()] = stored_fp4
                off_sf = B * Hh * Ww * (Ci // 16)
                cache_sf[off_sf:off_sf + stored_sf.numel()] = stored_sf
            else:
                cache_fp4 = torch.zeros(B * _CACHE_T * Hh * Ww * (Ci // 2),
                                        dtype=torch.uint8, device=device)
                cache_sf = torch.zeros(B * _CACHE_T * Hh * Ww * (Ci // 16),
                                       dtype=torch.uint8, device=device)

            M_total = B * T_new * Hh * Ww
            out_ndhwc = torch.empty(M_total * Co, dtype=_FP16, device=device)
            bias_ptr = conv.bias.data_ptr() if conv.bias is not None else 0
            rc = _fvk.nvfp4_conv3d_ndhwc_fp16out(
                cache_fp4.data_ptr(), new_fp4.data_ptr(),
                conv._nvfp4_w.data_ptr(),
                cache_sf.data_ptr(), new_sf.data_ptr(),
                conv._nvfp4_sf.data_ptr(),
                out_ndhwc.data_ptr(), bias_ptr,
                B, _CACHE_T, T_new, Hh, Ww, Ci, Co, 1.0, stream)
            if rc != 0:
                raise RuntimeError(f"[nvfp4_fused] conv3d_prequant rc={rc}")

            # Store rolling 2-frame FP4 cache for next call (Direction-2 logic).
            if use_fp4_cache:
                if T_new >= 2:
                    conv._nvfp4_stored_fp4 = new_fp4.view(B, T_new, Hh, Ww, Ci // 2)[:, -_CACHE_T:].contiguous().view(-1).clone()
                    conv._nvfp4_stored_sf = new_sf.view(B, T_new, Hh, Ww, Ci // 16)[:, -_CACHE_T:].contiguous().view(-1).clone()
                    conv._nvfp4_stored_T = _CACHE_T
                elif T_new == 1 and stored_fp4 is not None and stored_T >= 1:
                    prev_fp4 = stored_fp4.view(B, stored_T, Hh, Ww, Ci // 2)[:, -1:]
                    curr_fp4 = new_fp4.view(B, 1, Hh, Ww, Ci // 2)
                    conv._nvfp4_stored_fp4 = torch.cat([prev_fp4, curr_fp4], dim=1).contiguous().view(-1).clone()
                    prev_sf = stored_sf.view(B, stored_T, Hh, Ww, Ci // 16)[:, -1:]
                    curr_sf = new_sf.view(B, 1, Hh, Ww, Ci // 16)
                    conv._nvfp4_stored_sf = torch.cat([prev_sf, curr_sf], dim=1).contiguous().view(-1).clone()
                    conv._nvfp4_stored_T = _CACHE_T
                else:
                    conv._nvfp4_stored_fp4 = new_fp4.clone()
                    conv._nvfp4_stored_sf = new_sf.clone()
                    conv._nvfp4_stored_T = T_new

            out = out_ndhwc.view(B, T_new, Hh, Ww, Co).permute(0, 4, 1, 2, 3)
            return out.contiguous(memory_format=torch.channels_last_3d)

        # ── norm1 + conv1 ──────────────────────────────────────────
        x_cl = _to_cl(x)
        B0, C0, T0, H0, W0 = x_cl.shape
        new_fp4_1, new_sf_1 = _norm_quant(self.norm1, x_cl)
        x_out = _conv_prequant(self.conv1, new_fp4_1, new_sf_1,
                               B0, C0, T0, H0, W0, x_cl.device)
        if feat_cache is not None:
            feat_idx[0] += 1

        # ── norm2 + conv2 ──────────────────────────────────────────
        x_out_cl = _to_cl(x_out)
        B1, C1, T1, H1, W1 = x_out_cl.shape
        new_fp4_2, new_sf_2 = _norm_quant(self.norm2, x_out_cl)
        x_out2 = _conv_prequant(self.conv2, new_fp4_2, new_sf_2,
                                B1, C1, T1, H1, W1, x_out_cl.device)
        if feat_cache is not None:
            feat_idx[0] += 1

        return x_out2 + h

    return _patched_forward


# ── Install ──

def install_vae_nvfp4(vae, enabled: bool = True) -> Dict:
    """Install NVFP4 W4A4 conv3d for eligible WanVAE layers.

    Must be called AFTER ``install_vae_optimizations`` (the FP8 path
    must already be installed — FP4 overrides eligible layers).

    Returns summary dict.
    """
    if not enabled:
        return {'enabled': False, 'reason': 'disabled'}

    if os.environ.get('FLASHRT_NVFP4_VAE', '1') != '1':
        return {'enabled': False, 'reason': 'env disabled'}

    if not _load_kernels():
        return {'enabled': False, 'reason': 'kernels not found'}

    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        WanCausalConv3d)

    decode_only = os.environ.get('FLASHRT_NVFP4_VAE_DECODE_ONLY', '0') == '1'
    min_ci = int(os.environ.get('FLASHRT_NVFP4_VAE_MIN_CI', '192'))

    n_quantized = 0
    n_skipped = 0
    skip_reasons: Dict[str, int] = {}

    for name, m in vae.named_modules():
        if not isinstance(m, WanCausalConv3d):
            continue
        kt, kh, kw = m.kernel_size
        Ci, Co = m.in_channels, m.out_channels
        if (kt, kh, kw) != (3, 3, 3):
            continue
        if Ci % 96 != 0 or Ci < min_ci or Ci % 64 != 0:
            n_skipped += 1
            r = f'Ci={Ci} (<{min_ci} or not %64 for NVFP4 MMA)'
            skip_reasons[r] = skip_reasons.get(r, 0) + 1
            continue
        if decode_only and name.startswith('encoder.'):
            n_skipped += 1
            r = 'encoder (decode_only)'
            skip_reasons[r] = skip_reasons.get(r, 0) + 1
            continue

        # Pre-quantize weight
        w_fp4, w_sf = _quantize_weight_nvfp4(m.weight.data)
        m.register_buffer('_nvfp4_w', w_fp4, persistent=False)
        m.register_buffer('_nvfp4_sf', w_sf, persistent=False)
        n_quantized += 1

    if n_quantized == 0:
        logger.warning("[nvfp4_vae] 0 layers eligible for FP4")
        return {'enabled': False, 'n_quantized': 0, 'n_skipped': n_skipped,
                'skip_reasons': skip_reasons}

    # Patch forward — override FP8 dispatch for FP4-eligible layers
    _saved_forward = WanCausalConv3d.forward

    def _nvfp4_dispatch_forward(self, x, cache_x=None):
        if getattr(self, '_nvfp4_w', None) is not None:
            return _nvfp4_conv3d_forward(self, x, cache_x)
        return _saved_forward(self, x, cache_x)

    WanCausalConv3d.forward = _nvfp4_dispatch_forward
    WanCausalConv3d._flashrt_nvfp4_patched = True

    # ── Direction 2+3: Patch WanResidualBlock for fused norm+quant + FP4 cache ──
    n_fused_blocks = 0
    # Direction 3 ON by default: fused norm+silu+NVFP4-quant (kernel verified
    # byte-correct 99.7%) + streaming-correct cache (mirrors diffusers'
    # [<2 frames] padding for the per-frame decode loop). End-to-end PSNR
    # 35.23 dB vs 35.11 baseline. Disable with FLASHRT_NVFP4_FUSED_NORMQUANT=0.
    no_fused = os.environ.get('FLASHRT_NVFP4_FUSED_NORMQUANT', '1') == '0'
    try:
        from diffusers.models.autoencoders.autoencoder_kl_wan import (
            WanResidualBlock)
        _orig_res_forward = WanResidualBlock.forward
        _patched_fwd = _make_patched_residual_forward(_orig_res_forward)

        # Check which blocks have both conv1+conv2 NVFP4-eligible
        for blk in vae.modules():
            if isinstance(blk, WanResidualBlock):
                c1_ok = getattr(blk.conv1, '_nvfp4_w', None) is not None
                c2_ok = getattr(blk.conv2, '_nvfp4_w', None) is not None
                if c1_ok and c2_ok:
                    blk._nvfp4_fused = True
                    n_fused_blocks += 1

        if n_fused_blocks > 0 and not no_fused:
            WanResidualBlock.forward = _patched_fwd
            logger.info("[nvfp4_vae] %d WanResidualBlocks use fused "
                        "norm+quant + FP4 cache", n_fused_blocks)

            # Patch clear_cache to also clear FP4 caches between encode/decode passes
            import types
            _orig_clear = vae.clear_cache
            def _clear_with_fp4(self_vae):
                _orig_clear()
                for m in self_vae.modules():
                    m._nvfp4_stored_fp4 = None
                    m._nvfp4_stored_sf = None
                    m._nvfp4_stored_T = 0
            vae.clear_cache = types.MethodType(_clear_with_fp4, vae)
    except Exception as e:
        logger.debug("[nvfp4_vae] fused block patch skipped: %s", e)

    summary = {
        'enabled': True,
        'n_quantized': n_quantized,
        'n_skipped': n_skipped,
        'n_fused_blocks': n_fused_blocks,
        'skip_reasons': skip_reasons,
        'decode_only': decode_only,
        'min_ci': min_ci,
    }
    logger.info("[nvfp4_vae] install summary: %s", summary)
    return summary

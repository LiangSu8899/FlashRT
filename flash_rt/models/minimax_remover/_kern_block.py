"""
FlashRT pure-kernel Transformer block fusion (FP8 path).

Replaces every element-wise/norm/gate/residual/gelu op inside each
Transformer3DModel block with fused kernels; attention is installed
separately by ``install_attention`` (from ``_attention``, shared with the
NVFP4 path -- supports sage_*/triton_fp8/triton_fp16/fa2 via
``FLASHRT_ATTN_MODE``); all Linear layers still run through the FlashRT
FP8 GEMM installed by ``install_flashrt_fp8``.

Fusion points (per block, compared to the original video_subtitle_remover.py):
  Original: norm1(fp32) -> .float -> *(1+scale) -> +shift -> .type_as  (5 large [S,D] kernels)
            + hidden.float + attn*gate -> .type_as                      (4 large [S,D] kernels)
            norm2 likewise; FFN gelu is a torch op
  This version:
            ada_layernorm_fp16_io  -> 1 Triton kernel (fp32-stat LN + adaLN modulation)
            gate_mul_residual_bcast-> 1 in-place Triton kernel (residual += x * gate[D])
            gelu_inplace_fp16      -> 1 in-place FlashRT kernel (tanh gelu)

Key correctness details:
  * patch_embedding(...).transpose(1,2) produces a **non-contiguous** output; FlashRT's
    pointer-based kernels read contiguous memory, so the block entry must call
    .contiguous() (only the first block truly copies; subsequent blocks receive the
    contiguous output of the previous block, so contiguous() is a no-op).
  * The default ``FLASHRT_NORM_MODE=triton`` uses ``ada_layernorm_fp16_io`` which
    accumulates mean/var in fp32 across three passes -- bit-exact with the original
    FP32LayerNorm, unlike FlashRT's generic ``ada_layer_norm_fp16`` (the optional
    ``FLASHRT_NORM_MODE=fp16`` debug path, which fails fast if that symbol is absent).
  * gate_mul_residual_bcast takes a broadcast gate[D] vector, avoiding the [S,D]
    expand copy.
  * gelu_inplace_fp16 = tanh-approximate GELU (matches the FFN's approximate='tanh',
    verified exact).
"""

import math
import os
import logging
import torch
import torch.nn.functional as F
from flash_rt import flash_rt_kernels as kern
from ._kernels import ada_layernorm_fp16_io, rms_norm_fp32stat, gate_mul_residual_bcast
from ._attention import install_attention

# MiniMax-Remover fused FFN epilogue kernel (bias+gelu+quant → fp8).
# Opt-in module; if absent the FFN path falls back to the 3-kernel
# sequence (bias-add + gelu-inplace + quantise).
_fvk = None
try:
    from flash_rt import flash_rt_minimax_remover as _fvk
except ImportError:
    try:
        import flash_rt_minimax_remover as _fvk
    except ImportError:
        pass

# Backward-compat alias: the FP8 pipeline (``_fp8_pipeline``) installs the
# kernel attention processor via this name. The single source of truth for
# attention dispatch lives in ``_attention`` and is shared by both paths.
install_fa2_attention = install_attention

logger = logging.getLogger(__name__)

_FP16 = torch.float16

def install_fused_blocks(transformer, norm_mode=None, gelu_mode=None):
    """Replace each TransformerBlock.forward with the pure FlashRT kernel-fused version.

    norm_mode: 'fp16' (default, ada_layer_norm_fp16) | 'fp32' (original fp32 LayerNorm + modulation, for debugging)
    gelu_mode: 'inplace' (default, gelu_inplace_fp16) | 'torch' (original F.gelu, for debugging)
    """
    import os as _os
    if norm_mode is None:
        norm_mode = _os.environ.get("FLASHRT_NORM_MODE", "triton")
    if gelu_mode is None:
        gelu_mode = _os.environ.get("FLASHRT_GELU_MODE", "inplace")
    eps = float(transformer.blocks[0].norm1.eps)
    stream_of = lambda: torch.cuda.current_stream().cuda_stream

    def _ada_norm(self_hs, scale_v, shift_v, S, D):
        """Single-kernel fusion: fp32-statistics LayerNorm + adaLN modulation -> fp16.

        Why not use FlashRT's ada_layer_norm_fp16: its statistics are insufficiently
        precise on real diffusion latents, yielding only 41 dB end-to-end PSNR (vs 65 dB
        for the fp32 version). This Triton kernel accumulates mean/var in fp32 across
        three passes, bit-exact with the original FP32LayerNorm, while still being a
        single kernel. scale/shift stay fp32 (from temb.float()), and modulation is
        also done in fp32.
        """
        return ada_layernorm_fp16_io(self_hs, scale_v.view(D), shift_v.view(D), eps)

    def _ada_norm_flashrt_fp16(self_hs, scale_v, shift_v, S, D):
        if not hasattr(kern, "ada_layer_norm_fp16"):
            raise RuntimeError(
                "FLASHRT_NORM_MODE=fp16 requires the 'ada_layer_norm_fp16' kernel "
                "symbol, which is not present in this flash_rt_kernels build. "
                "Rebuild flash_rt_kernels, or use the default FLASHRT_NORM_MODE=triton "
                "(fp32-stat, reference-equivalent).")
        out = torch.empty(S, D, dtype=_FP16, device=self_hs.device)
        kern.ada_layer_norm_fp16(
            self_hs.data_ptr(),
            scale_v.view(D).to(_FP16).contiguous().data_ptr(),
            shift_v.view(D).to(_FP16).contiguous().data_ptr(),
            out.data_ptr(), S, D, eps, stream_of())
        return out

    _has_bgr = (_fvk is not None
                and hasattr(_fvk, "fp16_bias_gate_residual_bcast")
                and os.environ.get("FLASHRT_DISABLE_BIAS_GATE", "0") != "1")

    # Fused adaLN + fp8 quantise kernel — feeds Q/K/V from a shared fp8
    # tensor built with max(act_scale_q, act_scale_k, act_scale_v).  Saves
    # three activation-quantise passes and one full fp16 read of norm1.
    _fuse_ada_qkv = (_fvk is not None
                     and hasattr(_fvk, "fp16_ada_layernorm_quant_fp8")
                     and os.environ.get("FLASHRT_DISABLE_ADA_QKV", "0") != "1")

    # Fused norm2 + fp8 quantise — feeds FFN proj0 from fp8 directly,
    # eliminating the intermediate fp16 [S,D] write + read + quantise.
    _fuse_norm2_ffn = (_fvk is not None
                       and hasattr(_fvk, "fp16_ada_layernorm_quant_fp8")
                       and os.environ.get("FLASHRT_DISABLE_NORM2_FFN", "0") != "1")

    def _gate_residual_apply(hs, x_no_bias, bias, gate, S, D):
        """residual += (x + bias) * gate[D]  — one fused kernel if available,
        else falls back to add_bias_fp16 + gate_mul_residual_bcast."""
        if _has_bgr and bias is not None and (D & 7) == 0:
            gate_fp16 = gate.to(_FP16).contiguous().view(D)
            _fvk.fp16_bias_gate_residual_bcast(
                x_no_bias.data_ptr(), bias.data_ptr(),
                gate_fp16.data_ptr(), hs.data_ptr(),
                S, D, stream_of())
        else:
            if bias is not None:
                kern.add_bias_fp16(x_no_bias.data_ptr(), bias.data_ptr(),
                                   S, D, stream_of())
            gate_mul_residual_bcast(hs, x_no_bias, gate.view(D))

    def _get_qkv_shared_scale(block):
        """Cache & return the fp32[1] max(to_q, to_k, to_v).act_scale for
        the attn1 sub-module of `block`.  Persists across denoise steps
        (scales are frozen after calibration)."""
        cached = getattr(block, "_flashrt_qkv_shared_scale", None)
        if cached is not None:
            return cached
        attn = block.attn1
        s = torch.stack([attn.to_q.act_scale.view(1),
                         attn.to_k.act_scale.view(1),
                         attn.to_v.act_scale.view(1)]).max().view(1)
        s = s.to(torch.float32).contiguous()
        block._flashrt_qkv_shared_scale = s
        return s

    def block_forward(self, hidden_states, temb, rotary_emb):
        B, S, D = hidden_states.shape
        # Ensure contiguity at entry (first block truly copies; subsequent blocks are no-ops)
        hs = hidden_states.contiguous().view(S, D)

        (shift_msa, scale_msa, gate_msa,
         c_shift_msa, c_scale_msa, c_gate_msa) = (self.scale_shift_table + temb.float()).chunk(6, dim=1)

        # ── norm1 → attn ────────────────────────────────────────
        attn = self.attn1
        to_out0 = attn.to_out[0]
        _fuse_qkv = (_fuse_ada_qkv
                     and norm_mode != "fp16"
                     and hasattr(attn.to_q, "gemm_from_fp8_ext")
                     and hasattr(attn.to_k, "gemm_from_fp8_ext")
                     and hasattr(attn.to_v, "gemm_from_fp8_ext")
                     and not getattr(attn.to_q, "calibrating", False)
                     and (D & 7) == 0)
        _fuse_attn = (_has_bgr
                      and hasattr(to_out0, "gemm_no_bias")
                      and getattr(to_out0, "bias", None) is not None
                      and not getattr(to_out0, "calibrating", False)
                      and (D & 7) == 0)

        if _fuse_qkv:
            shared_scale = _get_qkv_shared_scale(self)
            # Ensure fp32 contiguous scale/shift vectors for the kernel.
            scale_f = scale_msa.contiguous().to(torch.float32).view(D)
            shift_f = shift_msa.contiguous().to(torch.float32).view(D)
            norm1_fp8 = torch.empty(S, D, dtype=torch.float8_e4m3fn,
                                    device=hs.device)
            _fvk.fp16_ada_layernorm_quant_fp8(
                hs.data_ptr(), scale_f.data_ptr(), shift_f.data_ptr(),
                shared_scale.data_ptr(), norm1_fp8.data_ptr(),
                S, D, eps, stream_of())
            attn_out = attn(hidden_states=hs.view(1, S, D),
                            rotary_emb=rotary_emb,
                            no_out_bias=_fuse_attn,
                            fp8_hidden=norm1_fp8,
                            fp8_scale=shared_scale).view(S, D)
        else:
            if norm_mode == "fp16":
                norm1_out = _ada_norm_flashrt_fp16(hs, scale_msa, shift_msa, S, D)
            else:
                norm1_out = _ada_norm(hs, scale_msa, shift_msa, S, D)
            attn_out = attn(hidden_states=norm1_out.view(1, S, D),
                            rotary_emb=rotary_emb,
                            no_out_bias=_fuse_attn).view(S, D)

        if _fuse_attn:
            _gate_residual_apply(hs, attn_out, to_out0.bias, gate_msa, S, D)
        else:
            gate_mul_residual_bcast(hs, attn_out, gate_msa.view(D))

        proj0 = self.ffn.net[0].proj
        proj1 = self.ffn.net[2]
        _can_fuse = (
            _fvk is not None
            and hasattr(proj0, "gemm_no_bias")
            and hasattr(proj1, "forward_from_fp8")
            and not getattr(proj1, "calibrating", False)
            and getattr(proj0, "bias", None) is not None
            and (proj0.out_features & 3) == 0)
        _fuse_ffn_down = (_has_bgr and _can_fuse
                          and hasattr(proj1, "gemm_no_bias_from_fp8")
                          and getattr(proj1, "bias", None) is not None
                          and (D & 7) == 0)

        _do_norm2_fp8 = (_fuse_norm2_ffn and _can_fuse
                         and hasattr(proj0, "gemm_no_bias_from_fp8")
                         and not getattr(proj0, "calibrating", False)
                         and (D & 7) == 0)

        if gelu_mode == "torch":
            if norm_mode == "fp16":
                norm2_out = _ada_norm_flashrt_fp16(hs, c_scale_msa, c_shift_msa, S, D)
            else:
                norm2_out = _ada_norm(hs, c_scale_msa, c_shift_msa, S, D)
            ff_out = self.ffn(norm2_out.view(1, S, D)).view(S, D)
            gate_mul_residual_bcast(hs, ff_out, c_gate_msa.view(D))
            return hs.view(1, S, D)

        if _do_norm2_fp8:
            c_scale_f = c_scale_msa.contiguous().to(torch.float32).view(D)
            c_shift_f = c_shift_msa.contiguous().to(torch.float32).view(D)
            norm2_fp8 = torch.empty(S, D, dtype=torch.float8_e4m3fn,
                                    device=hs.device)
            _fvk.fp16_ada_layernorm_quant_fp8(
                hs.data_ptr(), c_scale_f.data_ptr(), c_shift_f.data_ptr(),
                proj0.act_scale.data_ptr(), norm2_fp8.data_ptr(),
                S, D, eps, stream_of())
            raw = proj0.gemm_no_bias_from_fp8(norm2_fp8)
        elif _can_fuse:
            if norm_mode == "fp16":
                norm2_out = _ada_norm_flashrt_fp16(hs, c_scale_msa, c_shift_msa, S, D)
            else:
                norm2_out = _ada_norm(hs, c_scale_msa, c_shift_msa, S, D)
            raw = proj0.gemm_no_bias(norm2_out.view(1, S, D))
        else:
            if norm_mode == "fp16":
                norm2_out = _ada_norm_flashrt_fp16(hs, c_scale_msa, c_shift_msa, S, D)
            else:
                norm2_out = _ada_norm(hs, c_scale_msa, c_shift_msa, S, D)
            up = proj0(norm2_out.view(1, S, D))
            inner = up.shape[-1]
            _gelu_fn = kern.gelu_inplace if up.dtype == torch.bfloat16 else kern.gelu_inplace_fp16
            _gelu_fn(up.data_ptr(), S * inner, stream_of())
            ff_out = proj1(up).view(S, D)
            gate_mul_residual_bcast(hs, ff_out, c_gate_msa.view(D))
            return hs.view(1, S, D)

        inner = raw.shape[-1]
        up_fp8 = torch.empty(
            S, inner, dtype=torch.float8_e4m3fn, device=raw.device)
        _fvk.bias_gelu_quant_fp16_fp8(
            raw.data_ptr(), proj0.bias.data_ptr(),
            up_fp8.data_ptr(), proj1.act_scale.data_ptr(),
            S, inner, stream_of())
        if _fuse_ffn_down:
            ff_out = proj1.gemm_no_bias_from_fp8(up_fp8).view(S, D)
            _gate_residual_apply(hs, ff_out, proj1.bias, c_gate_msa, S, D)
            return hs.view(1, S, D)
        ff_out = proj1.forward_from_fp8(up_fp8).view(S, D)
        gate_mul_residual_bcast(hs, ff_out, c_gate_msa.view(D))
        return hs.view(1, S, D)

    block_cls = type(transformer.blocks[0])
    block_cls.forward = block_forward
    logger.info("  [FlashRT-Kern] block fusion: norm=%s gelu=%s", norm_mode, gelu_mode)
    return len(transformer.blocks)

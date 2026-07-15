"""FlashRT -- MiniMax-Remover kernel attention processor.

Replaces ``torch.nn.functional.scaled_dot_product_attention`` in every
attention block with a pointer-based kernel backend:

* ``FlashRTFA2Processor.__call__`` runs the QKV projection (through the
  NVFP4 Linears installed by ``_nvfp4_linear``), Q/K RMSNorm
  (fp32-stat Triton), interleaved RoPE on the native [B, S, H, D]
  layout (no transpose / copy), then the chosen attention kernel.

The attention backend is selected by ``FLASHRT_ATTN_MODE``:

  * ``sage`` / ``sage_auto``    -- SageAttention auto-dispatcher.
  * ``sage_fp8`` / ``sage2``    -- SageAttention QK-int8 + PV-fp8 CUDA
                                   (default; ~5x vs FA2, cos ~0.9993).
  * ``sage_fp16`` / ``sage1``   -- SageAttention QK-int8 + PV-fp16 CUDA.
  * ``sage_triton``             -- SageAttention QK-int8 + PV-fp16 Triton.
  * ``triton_fp8`` / ``triton_fp16`` -- self-contained Triton flash-attention
                                   (``_triton_flash_attn``); no external dep.
  * ``fa2``                     -- FlashRT FA2 (``flash_rt_fa2.fwd_fp16``).

``sageattention`` is an optional third-party dependency (pip); only the
selected backend is imported lazily. ``fa2`` uses FlashRT's own vendored
``flash_rt_fa2.so``. This module is the single source of truth for
attention dispatch -- shared by both the NVFP4 (``_manual_denoise``) and
FP8 (``_kern_block``) block paths. No MiniMax-Remover imports -- tensors
only.
"""

import math
import os

import torch

from ._kernels import rms_norm_fp32stat, rope_apply_bshd, freqs_to_cos_sin

# Optional MiniMax-Remover fused kernels — used when both are available.
_fvk = None
try:
    from flash_rt import flash_rt_minimax_remover as _fvk
except ImportError:
    try:
        import flash_rt_minimax_remover as _fvk  # type: ignore
    except ImportError:
        pass
_has_fused_rmsnorm_rope = (
    _fvk is not None and hasattr(_fvk, "fp16_rmsnorm_rope_bshd")
    and os.environ.get("FLASHRT_DISABLE_RMSNORM_ROPE", "0") != "1")

_has_fused_rmsnorm_rope_quant = (
    _fvk is not None
    and hasattr(_fvk, "fp16_rmsnorm_rope_quant_int8_q")
    and hasattr(_fvk, "fp16_rmsnorm_rope_quant_int8_k")
    and os.environ.get("FLASHRT_DISABLE_FUSED_QUANT", "0") != "1")

_SM89_COMPILE = None
_PER_CHANNEL_FP8 = None


def _get_sm89():
    global _SM89_COMPILE
    if _SM89_COMPILE is None:
        try:
            from sageattention import sm89_compile
            _SM89_COMPILE = sm89_compile
        except ImportError:
            _SM89_COMPILE = False
    return _SM89_COMPILE if _SM89_COMPILE is not False else None


def _get_per_channel_fp8():
    global _PER_CHANNEL_FP8
    if _PER_CHANNEL_FP8 is None:
        try:
            from sageattention.quant import per_channel_fp8
            _PER_CHANNEL_FP8 = per_channel_fp8
        except ImportError:
            _PER_CHANNEL_FP8 = False
    return _PER_CHANNEL_FP8 if _PER_CHANNEL_FP8 is not False else None

try:
    _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
except Exception:
    _NUM_SMS = 0


def _attention_mode():
    return os.environ.get("FLASHRT_ATTN_MODE", "sage_fp8").lower()


_SAGE = None


def _get_sage():
    global _SAGE
    if _SAGE is None:
        try:
            import sageattention as _m
        except ImportError as e:
            raise RuntimeError(
                "FLASHRT_ATTN_MODE=sage_* requires the 'sageattention' package "
                "(the default attention backend). Install the model extra:\n"
                "    pip install -e \".[minimax-remover]\"\n"
                "or switch to the dependency-light FlashRT FA2 backend:\n"
                "    FLASHRT_ATTN_MODE=fa2") from e
        _SAGE = _m
    return _SAGE


_TRITON_FA = None


def _get_triton_fa():
    """Lazy import the standalone Triton flash-attention (fp8 / fp16 variants)."""
    global _TRITON_FA
    if _TRITON_FA is None:
        from . import _triton_flash_attn as _m
        _TRITON_FA = _m
    return _TRITON_FA


def _sage_attn(q, k, v, scale, mode):
    """Dispatch to the requested SageAttention variant.

    All variants accept and return [B, S, H, D] (NHD) fp16.
    """
    sa = _get_sage()
    kw = dict(tensor_layout="NHD", is_causal=False, sm_scale=scale)
    if mode in ("sage", "sage_auto"):
        return sa.sageattn(q, k, v, **kw)
    if mode in ("sage_fp8", "sage2"):
        return sa.sageattn_qk_int8_pv_fp8_cuda(q, k, v, **kw)
    if mode in ("sage_fp16", "sage1"):
        return sa.sageattn_qk_int8_pv_fp16_cuda(q, k, v, **kw)
    if mode == "sage_triton":
        return sa.sageattn_qk_int8_pv_fp16_triton(q, k, v, **kw)
    return sa.sageattn(q, k, v, **kw)


def _fa2_attn(q, k, v, scale, lse_cache=None):
    """Run FlashRT's vendored FA2 backend on [B, S, H, D] fp16 tensors."""
    B, S, H, Dd = q.shape
    out = torch.empty_like(q)
    if lse_cache is None:
        lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    else:
        lse = lse_cache.get((B, S, H))
        if lse is None:
            lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
            lse_cache[(B, S, H)] = lse
    from flash_rt import flash_rt_fa2 as fa2
    qs, ks, vs, os_ = (q.stride(), k.stride(), v.stride(), out.stride())
    fa2.fwd_fp16(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        lse.data_ptr(), 0, 0,
        B, S, S, H, H, Dd,
        (qs[0], qs[1], qs[2]), (ks[0], ks[1], ks[2]), (vs[0], vs[1], vs[2]),
        (os_[0], os_[1], os_[2]),
        scale, _NUM_SMS, torch.cuda.current_stream().cuda_stream)
    return out


def attention_forward(q, k, v, scale, mode=None, *, lse_cache=None):
    """Dispatch MiniMax-Remover attention without importing unused backends.

    Single source of truth for the ``FLASHRT_ATTN_MODE`` routing; shared by
    both the NVFP4 (``_manual_denoise``) and FP8 (``_kern_block``) paths.
    """
    mode = _attention_mode() if mode is None else str(mode).lower()
    if mode.startswith("sage"):
        return _sage_attn(q, k, v, scale, mode)
    if mode in ("triton_fp8", "triton_fp16"):
        tfa = _get_triton_fa()
        return (tfa.flash_attn_fp8 if mode == "triton_fp8"
                else tfa.flash_attn_fp16)(q, k, v, scale)
    return _fa2_attn(q, k, v, scale, lse_cache)


class FlashRTFA2Processor:
    """Kernel attention processor for the native [B, S, H, D] layout.

    QKV projection goes through the installed NVFP4 Linears; Q/K
    RMSNorm uses the fp32-stat Triton kernel; RoPE is applied in place
    on the native layout (zero transpose / copy); the attention backend
    is chosen by ``FLASHRT_ATTN_MODE``. Per-shape cos/sin RoPE tables
    and FA2 log-sum-exp buffers are cached.
    """

    def __init__(self):
        self._lse_bufs = {}
        self._cos_sin = {}
        # Persistent [B*S] fp32 scratch for the fused rmsnorm+rope+int8-quant
        # Q/K kernels (avoids a per-call cudaMallocAsync in the hot path).
        self._rstd_bufs = {}

    def __call__(self, attn, hidden_states, rotary_emb=None,
                 attention_mask=None, encoder_hidden_states=None,
                 no_out_bias=False, fp8_hidden=None, fp8_scale=None):
        mode = _attention_mode()
        B, S, _ = hidden_states.shape
        H = attn.heads
        Dd = attn.inner_dim // H
        scale = 1.0 / math.sqrt(float(Dd))

        # ── Fused QKV entry from pre-quantised fp8 input ──
        # When the caller provides the fp8 output of the fused adaLN+quant
        # kernel plus the shared scale used to build it, drive Q/K/V from
        # that fp8 tensor directly — saves 3 activation-quantise passes
        # (one per Linear) and 1 fp16 read of norm1_out.  Both Linears
        # must expose gemm_from_fp8_ext (FlashRTFp8Linear post-calibration).
        _use_fp8 = (fp8_hidden is not None and fp8_scale is not None
                    and hasattr(attn.to_q, "gemm_from_fp8_ext")
                    and hasattr(attn.to_k, "gemm_from_fp8_ext")
                    and hasattr(attn.to_v, "gemm_from_fp8_ext"))
        cs = None
        if rotary_emb is not None:
            cs = self._cos_sin.get(S)
            if cs is None:
                cs = freqs_to_cos_sin(rotary_emb)
                self._cos_sin[S] = cs

        _fuse_qk = (_has_fused_rmsnorm_rope and cs is not None
                    and attn.norm_q is not None and attn.norm_k is not None
                    and (Dd & 7) == 0)
        # ── Fully-fused path: RMSNorm + RoPE + int8 quant → sm89 attn ──
        # Eliminates the fp16 intermediate between norm+rope and QK quantize.
        _fuse_quant = (_fuse_qk and _has_fused_rmsnorm_rope_quant
                       and mode in ("sage_fp8", "sage2")
                       and _get_sm89() is not None
                       and _get_per_channel_fp8() is not None)

        # Q/K GEMM: when the Q/K bias will be fused into the downstream
        # rmsnorm+rope+quant kernel (pre-norm add), fetch Q/K WITHOUT bias
        # to avoid adding it twice. V always keeps its bias (not normed).
        _qk_nobias = (_use_fp8 and _fuse_quant
                      and hasattr(attn.to_q, "gemm_from_fp8_ext_nobias"))
        if _use_fp8:
            if _qk_nobias:
                q = attn.to_q.gemm_from_fp8_ext_nobias(fp8_hidden, fp8_scale)
                k = attn.to_k.gemm_from_fp8_ext_nobias(fp8_hidden, fp8_scale)
            else:
                q = attn.to_q.gemm_from_fp8_ext(fp8_hidden, fp8_scale)
                k = attn.to_k.gemm_from_fp8_ext(fp8_hidden, fp8_scale)
            v = attn.to_v.gemm_from_fp8_ext(fp8_hidden, fp8_scale)
        else:
            q = attn.to_q(hidden_states)
            k = attn.to_k(hidden_states)
            v = attn.to_v(hidden_states)
        if _fuse_quant:
            if not q.is_contiguous():
                q = q.contiguous()
            if not k.is_contiguous():
                k = k.contiguous()
            stream = torch.cuda.current_stream().cuda_stream
            D = H * Dd

            num_groups_q = (S + 31) // 32
            num_groups_k = (S + 63) // 64
            q_int8 = torch.empty(B * S, D, device=q.device, dtype=torch.int8)
            k_int8 = torch.empty(B * S, D, device=q.device, dtype=torch.int8)
            q_scale = torch.empty(B, H, num_groups_q, device=q.device,
                                  dtype=torch.float32)
            k_scale = torch.empty(B, H, num_groups_k, device=q.device,
                                  dtype=torch.float32)

            # Q/K bias fused pre-norm when the nobias GEMM was used.
            q_bias_ptr = (attn.to_q.bias.data_ptr()
                          if (_qk_nobias and attn.to_q.bias is not None) else 0)
            k_bias_ptr = (attn.to_k.bias.data_ptr()
                          if (_qk_nobias and attn.to_k.bias is not None) else 0)

            # Reuse a persistent rstd scratch (B*S fp32) across Q/K calls so
            # the fused kernel does zero hot-path allocation. Q and K run
            # sequentially on the same stream, so one buffer serves both.
            rstd = self._rstd_bufs.get((B, S))
            if rstd is None or rstd.device != q.device:
                rstd = torch.empty(B * S, dtype=torch.float32, device=q.device)
                self._rstd_bufs[(B, S)] = rstd
            rstd_ptr = rstd.data_ptr()

            rc = _fvk.fp16_rmsnorm_rope_quant_int8_q(
                q.data_ptr(), attn.norm_q.weight.data_ptr(),
                q_bias_ptr,
                cs[0].data_ptr(), cs[1].data_ptr(),
                q_int8.data_ptr(), q_scale.data_ptr(),
                B, S, H, Dd, float(attn.norm_q.eps), 1.0,
                rstd_ptr, stream)
            if rc != 0:
                raise RuntimeError(
                    f"fp16_rmsnorm_rope_quant_int8_q failed rc={rc} "
                    f"(B={B} S={S} H={H} Dd={Dd})")
            rc = _fvk.fp16_rmsnorm_rope_quant_int8_k(
                k.data_ptr(), attn.norm_k.weight.data_ptr(),
                k_bias_ptr,
                cs[0].data_ptr(), cs[1].data_ptr(),
                0,  # no smooth_k (negligible impact, saves k.mean compute)
                k_int8.data_ptr(), k_scale.data_ptr(),
                B, S, H, Dd, float(attn.norm_k.eps), 1.0,
                rstd_ptr, stream)
            if rc != 0:
                raise RuntimeError(
                    f"fp16_rmsnorm_rope_quant_int8_k failed rc={rc} "
                    f"(B={B} S={S} H={H} Dd={Dd})")

            v = v.view(B, S, H, Dd)
            if not v.is_contiguous():
                v = v.contiguous()

            per_channel_fp8 = _get_per_channel_fp8()
            v_fp8, v_scale, _ = per_channel_fp8(
                v, tensor_layout="NHD", scale_max=2.25, smooth_v=False)

            q_int8 = q_int8.view(B, S, H, Dd)
            k_int8 = k_int8.view(B, S, H, Dd)
            out = torch.empty(B, S, H, Dd, device=q.device, dtype=q.dtype)

            sm89 = _get_sm89()
            sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
                q_int8, k_int8, v_fp8, out, q_scale, k_scale, v_scale,
                0, 0, 2, scale, 0)

            hidden_states = out.view(B, S, H * Dd)
            to_out0 = attn.to_out[0]
            if no_out_bias and hasattr(to_out0, "gemm_no_bias"):
                hidden_states = to_out0.gemm_no_bias(hidden_states)
            else:
                hidden_states = to_out0(hidden_states)
            return hidden_states

        if _fuse_qk:
            if not q.is_contiguous():
                q = q.contiguous()
            if not k.is_contiguous():
                k = k.contiguous()
            stream = torch.cuda.current_stream().cuda_stream
            _fvk.fp16_rmsnorm_rope_bshd(
                q.data_ptr(), attn.norm_q.weight.data_ptr(),
                cs[0].data_ptr(), cs[1].data_ptr(),
                B, S, H, Dd, float(attn.norm_q.eps), stream)
            _fvk.fp16_rmsnorm_rope_bshd(
                k.data_ptr(), attn.norm_k.weight.data_ptr(),
                cs[0].data_ptr(), cs[1].data_ptr(),
                B, S, H, Dd, float(attn.norm_k.eps), stream)
            q = q.view(B, S, H, Dd)
            k = k.view(B, S, H, Dd)
            v = v.view(B, S, H, Dd)
        else:
            if attn.norm_q is not None:
                q = rms_norm_fp32stat(q, attn.norm_q.weight, attn.norm_q.eps)
            if attn.norm_k is not None:
                k = rms_norm_fp32stat(k, attn.norm_k.weight, attn.norm_k.eps)

            q = q.view(B, S, H, Dd)
            k = k.view(B, S, H, Dd)
            v = v.view(B, S, H, Dd)

            if cs is not None:
                rope_apply_bshd(q, cs[0], cs[1])
                rope_apply_bshd(k, cs[0], cs[1])

        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        out = attention_forward(q, k, v, scale, mode,
                                lse_cache=self._lse_bufs)

        hidden_states = out.view(B, S, H * Dd)
        to_out0 = attn.to_out[0]
        if no_out_bias and hasattr(to_out0, "gemm_no_bias"):
            hidden_states = to_out0.gemm_no_bias(hidden_states)
        else:
            hidden_states = to_out0(hidden_states)
        return hidden_states


def install_attention(transformer):
    """Install the kernel attention processor on every transformer block."""
    proc = FlashRTFA2Processor()
    n = 0
    for block in transformer.blocks:
        block.attn1.processor = proc
        n += 1
    return n

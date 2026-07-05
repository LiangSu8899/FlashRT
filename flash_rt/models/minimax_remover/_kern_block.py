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

    def block_forward(self, hidden_states, temb, rotary_emb):
        B, S, D = hidden_states.shape
        # Ensure contiguity at entry (first block truly copies; subsequent blocks are no-ops)
        hs = hidden_states.contiguous().view(S, D)

        (shift_msa, scale_msa, gate_msa,
         c_shift_msa, c_scale_msa, c_gate_msa) = (self.scale_shift_table + temb.float()).chunk(6, dim=1)

        if norm_mode == "fp16":
            norm1_out = _ada_norm_flashrt_fp16(hs, scale_msa, shift_msa, S, D)
        else:
            norm1_out = _ada_norm(hs, scale_msa, shift_msa, S, D)
        attn_out = self.attn1(hidden_states=norm1_out.view(1, S, D), rotary_emb=rotary_emb).view(S, D)
        # Broadcast gate[D] (avoids [S,D] expand copy); fp16 in-place
        gate_mul_residual_bcast(hs, attn_out, gate_msa.view(D))

        if norm_mode == "fp16":
            norm2_out = _ada_norm_flashrt_fp16(hs, c_scale_msa, c_shift_msa, S, D)
        else:
            norm2_out = _ada_norm(hs, c_scale_msa, c_shift_msa, S, D)
        n2_3d = norm2_out.view(1, S, D)
        if gelu_mode == "torch":
            ff_out = self.ffn(n2_3d).view(S, D)
        else:
            up = self.ffn.net[0].proj(n2_3d)
            inner = up.shape[-1]
            _gelu_fn = kern.gelu_inplace if up.dtype == torch.bfloat16 else kern.gelu_inplace_fp16
            _gelu_fn(up.data_ptr(), S * inner, stream_of())
            ff_out = self.ffn.net[2](up).view(S, D)
        gate_mul_residual_bcast(hs, ff_out, c_gate_msa.view(D))
        return hs.view(1, S, D)

    block_cls = type(transformer.blocks[0])
    block_cls.forward = block_forward
    logger.info("  [FlashRT-Kern] block fusion: norm=%s gelu=%s", norm_mode, gelu_mode)
    return len(transformer.blocks)

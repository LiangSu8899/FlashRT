"""FlashVLA — RTX (consumer discrete GPU) Pi0 inference pipeline.

Framework-agnostic pipeline for Pi0 on consumer RTX GPUs (Blackwell SM120
/ Ada SM89, 5090 / 4090). Mirrors ``flash_vla/models/pi05/pipeline_rtx.py``
— shared vision/encoder paths; decoder is swapped for Pi0's variant:

  * Standard RMSNorm (no AdaRMSNorm / style modulation)
  * ``action_time_mlp`` replaces Pi0.5's ``time_mlp`` → AdaRMS conditioning
  * ``state_proj`` projects a continuous state vector into 1 prefix token
  * Decoder sequence length ``S_dec = Sa + 1`` (1 state + Sa actions, no
    padding); Pi0.5 uses ``S_dec = chunk_size = 10``
  * Cross-attention uses state masking — the state query (row 0) attends
    only to the encoder prefix + itself, while action queries see all
    ``S_dec`` decoder rows. Delegated to the attention backend via
    ``run("decoder", ..., state_nk=enc_seq + 1)``.

All activations are FP16 (IEEE 754 half, matching pi0_thor). FP8 E4M3
quantization on weights + activations for the large GEMMs (vision
attn/FFN, encoder attn/FFN, decoder attn/FFN).
"""

from __future__ import annotations

import ctypes
import logging

import numpy as np

from flash_vla.core.cuda_buffer import CudaBuffer
from flash_vla.core.cuda_graph import CUDAGraph

logger = logging.getLogger(__name__)


# Fixed Pi0 model dimensions (same Gemma backbones as Pi0.5)
VIS_L = 27
VIS_D = 1152
VIS_H = 4304
VIS_NH = 16
VIS_HD = 72
VIS_SEQ_PER_VIEW = 256
VIS_PATCH_FLAT = 14 * 14 * 3  # 588

ENC_L = 18
ENC_D = 2048
ENC_H = 16384
ENC_NH = 8
ENC_NKV = 1
ENC_HD = 256

DEC_L = 18
DEC_D = 1024
DEC_H = 4096
DEC_NH = 8
DEC_NKV = 1
DEC_HD = 256

ACTION_DIM = 32
CHUNK_SIZE_DEFAULT = 10
NUM_STEPS_DEFAULT = 10

# FP16 tagging. ``FP16`` is the 2-byte sizing placeholder for CudaBuffer
# allocations; ``FP16_NP`` is the real IEEE 754 FP16 dtype used for
# numeric staging arrays (ones vectors, RoPE cos/sin, etc). Pi0 on RTX
# runs the entire pipeline in FP16 to match pi0_thor's math precision
# — pi0's standard (non-Ada) RMSNorm makes activations sensitive to
# BF16's 7-bit mantissa over the 10-step flow-matching loop, degrading
# FP8 stability; FP16's 10-bit mantissa matches pi0_thor and pushes
# FP8 cos back into the >=0.998 range (vs ~0.97 under BF16).
FP16 = np.float16
FP16_NP = np.float16
FP8 = np.uint8
FP32 = np.float32


class Pi0Pipeline:
    """Pi0 inference pipeline for RTX (Blackwell / Ada) consumer GPUs.

    Args:
        gemm:         ``fvk.GemmRunner()`` cuBLASLt driver.
        fvk:          The ``flash_vla_kernels`` module.
        attn_backend: Attention backend (must support ``state_nk`` kwarg
                      on ``run("decoder", ...)`` — the torch flash_attn
                      backend implements this natively).
        weights:      Pointer dict — see ``_build_pipeline_weights`` in
                      ``flash_vla.frontends.torch.pi0_rtx``.
        num_views:    Number of observation camera views.
        max_prompt_len: Max tokenised prompt length.
        chunk_size:   Action chunk length Sa (default 10).
        use_fp8:      Enable FP8 E4M3 on vision / encoder GEMMs.
        use_fp8_decoder: Enable FP8 on decoder GEMMs (else BF16).
        num_steps:    Diffusion denoise steps (default 10).

    Expected weights dict keys:
        Vision BF16 / FP8:  identical to Pi05Pipeline.
        Encoder BF16 / FP8: identical to Pi05Pipeline (RMSNorm fold).
        Decoder BF16:
            state_proj_w / state_proj_b  — (32→1024),
            action_in_proj_w / action_in_proj_b  — (32→1024),
            action_time_mlp_in_wa_w      — (1024→1024) action half of
                                            the concat(action|time) MLP,
            action_time_mlp_out_w / _b   — (1024→1024) + bias,
            time_proj_all                — (num_steps, Sa, 1024) BF16
                                            — pre-computed time_emb @ W_t,
            decoder_final_norm_w         — (1024,) folded ``(1+w)``,
            decoder_action_out_proj_w    — (1024→32), pre-scaled by
                                            ``-1 / num_steps`` in the
                                            frontend,
            decoder_action_out_proj_b    — ditto.
        Decoder FP8 (fold ``(1 + rms_w)`` into QKV/GateUp like encoder):
            fp8.decoder_attn_qkv_w_{0..17}, fp8.decoder_attn_o_w_{0..17},
            fp8.decoder_ffn_gate_up_w_{0..17}, fp8.decoder_ffn_down_w_{0..17}
        Language:
            language_embeds_ptr — rebound per-prompt by frontend.
    """

    def __init__(self, gemm, fvk, attn_backend, weights, *,
                 num_views, max_prompt_len,
                 chunk_size=CHUNK_SIZE_DEFAULT,
                 use_fp8=True, use_fp8_decoder=True,
                 num_steps=NUM_STEPS_DEFAULT):
        self.gemm = gemm
        self.fvk = fvk
        # fvk.gmm_fp16 (cuBLAS HGEMM with beta support) needs an
        # FvkContext — used for the noise-path small GEMMs and the
        # fused action_out_proj accumulate (matches pi0_thor).
        self._ctx = fvk.FvkContext()
        self.attn = attn_backend
        self.weights = weights

        self.num_views = int(num_views)
        self.max_prompt_len = int(max_prompt_len)
        self.chunk_size = int(chunk_size)                 # Sa
        self.S_dec = self.chunk_size + 1                  # 1 state + Sa actions
        self.num_steps = int(num_steps)
        self.use_fp8 = bool(use_fp8)
        self.use_fp8_decoder = bool(use_fp8_decoder)

        # Derived sizes
        self.vision_seq = self.num_views * VIS_SEQ_PER_VIEW
        self.encoder_seq_len = self.vision_seq + self.max_prompt_len
        # KV cache holds encoder keys + S_dec decoder rows (1 state + Sa).
        self.total_kv = self.encoder_seq_len + self.S_dec

        # State-mask bound for Pi0 cross-attention (row 0 of dec_Q is
        # the state query; it may only attend to the encoder prefix
        # plus itself at position ``encoder_seq_len``).
        self._state_nk = self.encoder_seq_len + 1

        # Attention pointers (owned by attn_backend)
        self._attn_ptrs = attn_backend.get_ptrs()
        self._enc_kv_layer_stride = self._attn_ptrs["enc_k_layer_stride_bytes"]

        # Allocate internal buffers
        self.bufs = self._allocate_buffers()

        # RoPE table (max positions = encoder_seq_len + S_dec)
        self._build_rope_table()

        # valid_encoder_len placeholder — updated per forward
        _valid_len = np.array([self.vision_seq + 1], dtype=np.int32)
        self.bufs["valid_encoder_len"] = CudaBuffer.from_numpy(_valid_len)

        # Pre-allocated RMS norm "ones" weight vectors (REAL FP16 bits)
        _ones_2048 = np.ones(ENC_D, dtype=FP16_NP)
        _ones_1024 = np.ones(DEC_D, dtype=FP16_NP)
        self._rms_ones_enc = CudaBuffer.from_numpy(_ones_2048)
        self._rms_ones_dec = CudaBuffer.from_numpy(_ones_1024)

        # FP8 activation scratch + per-layer static scales
        self.fp8_act_scales = {}
        self.fp8_calibrated = False
        self._allocate_fp8_scratch()

        # CUDA graph state (set by record_infer_graph)
        self._graph = None
        self._graph_stream = None
        self._cudart = ctypes.CDLL("libcudart.so")

        # Pre-expand vision position embedding across num_views.
        self._build_pos_embed_expanded()

    # ══════════════════════════════════════════════════════════════════
    #   Buffer allocation
    # ══════════════════════════════════════════════════════════════════

    def _allocate_buffers(self):
        """Allocate all pipeline working buffers as CudaBuffer."""
        nv = self.num_views
        vs = self.vision_seq
        es = self.encoder_seq_len
        sd = self.S_dec
        sa = self.chunk_size
        B = {}

        # ── Vision (SigLIP) ── (identical to Pi0.5)
        B["observation_images_normalized"] = CudaBuffer.device_empty(
            nv * 224 * 224 * 3, FP16)
        B["vision_x"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_x_norm"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_QKV"] = CudaBuffer.device_empty(vs * 3 * VIS_D, FP16)
        B["vision_hidden"] = CudaBuffer.device_empty(vs * VIS_H, FP16)
        B["vision_pos_embed_expanded"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_patches"] = CudaBuffer.device_empty(vs * VIS_PATCH_FLAT, FP16)

        # ── Encoder (Gemma-2B) ── (identical to Pi0.5)
        B["encoder_rope_weights"] = CudaBuffer.device_empty(es * 2 * ENC_HD // 2, FP16)
        B["encoder_x"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        B["encoder_x_norm"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        B["encoder_QKV"] = CudaBuffer.device_empty(
            es * (ENC_NH + 2 * ENC_NKV) * ENC_HD, FP16)
        B["encoder_hidden"] = CudaBuffer.device_empty(es * ENC_H, FP16)
        B["encoder_gate_merged"] = CudaBuffer.device_empty(es * 2 * ENC_H, FP16)

        # ── Pi0 pre-decoder ──
        # Input: state vector (1, ACTION_DIM) bf16 — frontend writes here
        B["state_buf"] = CudaBuffer.device_empty(1 * ACTION_DIM, FP16)
        # state_proj output: (1, DEC_D)
        B["state_token"] = CudaBuffer.device_empty(1 * DEC_D, FP16)
        # action_time_mlp intermediate: (Sa, DEC_D)
        B["action_time_temp"] = CudaBuffer.device_empty(sa * DEC_D, FP16)

        # ── Decoder (Gemma-300M) ──
        B["decoder_rope_weights"] = CudaBuffer.device_empty(sd * 256, FP16)
        # decoder_x now holds (S_dec, DEC_D) = (Sa + 1, DEC_D)
        B["decoder_x"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["decoder_x_norm"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["decoder_QKV"] = CudaBuffer.device_empty(
            sd * (DEC_NH + 2 * DEC_NKV) * DEC_HD, FP16)
        B["decoder_hidden"] = CudaBuffer.device_empty(sd * DEC_H, FP16)
        B["decoder_gate_merged"] = CudaBuffer.device_empty(sd * 2 * DEC_H, FP16)
        # action output buf (Sa rows only — row 0/state is discarded)
        B["decoder_action_buf"] = CudaBuffer.device_empty(sa * ACTION_DIM, FP16)
        # diffusion noise (Sa, ACTION_DIM) — pipeline input/output
        B["diffusion_noise"] = CudaBuffer.device_empty(sa * ACTION_DIM, FP16)
        # scratch for attn output path (size = S_dec * DEC_D)
        B["dec_attn_proj"] = CudaBuffer.device_empty(sd * DEC_D, FP16)

        return B

    def _allocate_fp8_scratch(self):
        if not self.use_fp8:
            return
        B = self.bufs
        vs = self.vision_seq
        es = self.encoder_seq_len
        sd = self.S_dec

        # Vision FP8 scratch
        B["vis_act_fp8"] = CudaBuffer.device_zeros(vs * VIS_D, FP8)
        B["vis_act_fp8_large"] = CudaBuffer.device_zeros(vs * VIS_H, FP8)
        B["vis_act_scale"] = CudaBuffer.device_zeros(1, FP32)

        # Encoder FP8 scratch
        B["enc_act_fp8"] = CudaBuffer.device_zeros(es * ENC_D, FP8)
        B["enc_act_fp8_large"] = CudaBuffer.device_zeros(es * 2 * ENC_H, FP8)
        B["enc_act_scale"] = CudaBuffer.device_zeros(1, FP32)

        # Decoder FP8 scratch (sized for S_dec rows)
        B["dec_act_fp8"] = CudaBuffer.device_zeros(sd * DEC_D, FP8)
        B["dec_act_fp8_large"] = CudaBuffer.device_zeros(sd * 2 * DEC_H, FP8)
        B["dec_act_scale"] = CudaBuffer.device_zeros(1, FP32)
        # Pre-quantized context for O-projection (attn output (S_dec, 8*HD))
        B["dec_ctx_fp8"] = CudaBuffer.device_zeros(sd * DEC_NH * DEC_HD, FP8)

    # ══════════════════════════════════════════════════════════════════
    #   RoPE table
    # ══════════════════════════════════════════════════════════════════

    def _build_rope_table(self):
        """Build the FP16 interleaved cos/sin RoPE table."""
        max_pos = self.encoder_seq_len + self.S_dec
        inv_freq = 1.0 / (10000 ** (
            np.arange(0, 256, 2, dtype=np.float64) / 256))
        positions = np.arange(max_pos, dtype=np.float64)
        phase = positions[:, None] * inv_freq[None, :]
        cos = np.cos(phase).astype(FP16_NP)
        sin = np.sin(phase).astype(FP16_NP)
        interleaved = np.stack([cos, sin], axis=-1).reshape(max_pos, 256)
        self._rope_table_np = interleaved

        enc_rope_slice = interleaved[:self.encoder_seq_len]
        self.bufs["encoder_rope_weights"] = CudaBuffer.from_numpy(
            np.ascontiguousarray(enc_rope_slice))

        # Decoder RoPE: positions [encoder_seq_len .. encoder_seq_len + S_dec)
        dec_rope_slice = interleaved[
            self.encoder_seq_len: self.encoder_seq_len + self.S_dec]
        self.bufs["decoder_rope_weights"] = CudaBuffer.from_numpy(
            np.ascontiguousarray(dec_rope_slice))

    def _set_decoder_rope_for_prompt(self, prompt_len):
        """Update decoder_rope_weights for a new prompt length."""
        start = self.vision_seq + prompt_len
        end = start + self.S_dec
        self.bufs["decoder_rope_weights"].upload(
            np.ascontiguousarray(self._rope_table_np[start:end]))

    def _build_pos_embed_expanded(self):
        pos_src_ptr = self.weights["vision_position_embedding"]
        per_view_nbytes = VIS_SEQ_PER_VIEW * VIS_D * 2
        dst_buf = self.bufs["vision_pos_embed_expanded"]
        assert dst_buf.nbytes == self.num_views * per_view_nbytes
        for v in range(self.num_views):
            self._cudart.cudaMemcpy(
                ctypes.c_void_p(dst_buf.ptr.value + v * per_view_nbytes),
                ctypes.c_void_p(pos_src_ptr),
                per_view_nbytes, 3)
        self._cudart.cudaDeviceSynchronize()

    # ══════════════════════════════════════════════════════════════════
    #   Helpers
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _p(buf):
        return buf.ptr.value

    def _weight_fp8(self, name):
        return self.weights["fp8"][name]

    def _enc_kv_layer_ptrs(self, layer, offset_tokens=0):
        k_base = self._attn_ptrs["enc_K"]
        v_base = self._attn_ptrs["enc_V"]
        layer_stride = self._enc_kv_layer_stride
        token_offset_bytes = offset_tokens * ENC_NKV * ENC_HD * 2
        return (k_base + layer * layer_stride + token_offset_bytes,
                v_base + layer * layer_stride + token_offset_bytes)

    def _fp8_scale_buf(self, name):
        buf = self.fp8_act_scales.get(name)
        if buf is None:
            buf = CudaBuffer.device_zeros(1, FP32)
            self.fp8_act_scales[name] = buf
        return buf

    def _pick_fp8_scratch(self, weight_name, act_n):
        B = self.bufs
        if (weight_name.startswith("vision_")
                or weight_name == "vision_projector_w"):
            small = B["vis_act_fp8"]
            large = B["vis_act_fp8_large"]
            scratch_scale = B["vis_act_scale"]
        elif weight_name.startswith("encoder_"):
            small = B["enc_act_fp8"]
            large = B["enc_act_fp8_large"]
            scratch_scale = B["enc_act_scale"]
        else:
            small = B["dec_act_fp8"]
            large = B["dec_act_fp8_large"]
            scratch_scale = B["dec_act_scale"]
        buf = small if act_n <= (small.nbytes // 1) else large
        return buf.ptr.value, scratch_scale.ptr.value

    def _fp8_gemm(self, act_bf16_ptr, act_n, weight_name,
                  out_bf16_ptr, M, N, K, stream):
        fvk = self.fvk
        w_fp8_ptr, w_scale_ptr = self._weight_fp8(weight_name)
        act_fp8_ptr, _ = self._pick_fp8_scratch(weight_name, act_n)
        if self.fp8_calibrated and weight_name in self.fp8_act_scales:
            static_scale_ptr = self.fp8_act_scales[weight_name].ptr.value
            fvk.quantize_fp8_static_fp16(
                act_bf16_ptr, act_fp8_ptr, static_scale_ptr, act_n,
                stream=stream)
            self.fvk.fp8_gemm_descale_fp16(
                act_fp8_ptr, w_fp8_ptr, out_bf16_ptr,
                M, N, K, static_scale_ptr, w_scale_ptr, stream)
        else:
            layer_scale = self._fp8_scale_buf(weight_name)
            fvk.quantize_fp8_device_fp16(
                act_bf16_ptr, act_fp8_ptr, layer_scale.ptr.value, act_n,
                stream=stream)
            self.fvk.fp8_gemm_descale_fp16(
                act_fp8_ptr, w_fp8_ptr, out_bf16_ptr,
                M, N, K, layer_scale.ptr.value, w_scale_ptr, stream)

    def _fp8_gemm_fused(self, act_fp8_ptr, weight_name,
                        out_bf16_ptr, M, N, K, act_scale_ptr, stream):
        w_fp8_ptr, w_scale_ptr = self._weight_fp8(weight_name)
        self.fvk.fp8_gemm_descale_fp16(
            act_fp8_ptr, w_fp8_ptr, out_bf16_ptr,
            M, N, K, act_scale_ptr, w_scale_ptr, stream)

    def _bias_add_bf16(self, x_ptr, bias_ptr, seq, dim, stream):
        if not hasattr(self, "_bias_zero_buf"):
            nbytes = max(
                self.vision_seq * 3 * VIS_D,
                self.vision_seq * VIS_H,
            )
            self._bias_zero_buf = CudaBuffer.device_zeros(nbytes, FP16)
        self.fvk.bias_residual_fp16(
            x_ptr, self._bias_zero_buf.ptr.value, bias_ptr,
            seq, dim, stream=stream)

    # ══════════════════════════════════════════════════════════════════
    #   Phase A: Vision (SigLIP) — identical to Pi0.5
    # ══════════════════════════════════════════════════════════════════

    def vision_encoder(self, stream=0):
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        seq = self.vision_seq
        nv = self.num_views

        fvk.patch_im2col(
            B["observation_images_normalized"].ptr.value,
            B["vision_patches"].ptr.value,
            nv, stream)
        gemm.fp16_nn(
            B["vision_patches"].ptr.value,
            W["vision_patch_embedding_w"],
            B["vision_x"].ptr.value,
            seq, VIS_D, VIS_PATCH_FLAT, stream=stream)
        fvk.bias_residual_fp16(
            B["vision_x"].ptr.value,
            B["vision_pos_embed_expanded"].ptr.value,
            W["vision_patch_embedding_b"],
            seq, VIS_D, stream=stream)

        use_fp8 = (self.use_fp8
                   and "vision_attn_qkv_w_0" in self.weights.get("fp8", {}))
        for i in range(VIS_L):
            self._vision_layer(i, seq, use_fp8, stream)

    def _vision_layer(self, i, seq, use_fp8, stream):
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        attn_ptrs = self._attn_ptrs

        fvk.layer_norm_fp16(
            B["vision_x"].ptr.value,
            W["vision_pre_attn_norm_w"][i], W["vision_pre_attn_norm_b"][i],
            B["vision_x_norm"].ptr.value,
            seq, VIS_D, 1e-5, stream=stream)

        if use_fp8:
            self._fp8_gemm(
                B["vision_x_norm"].ptr.value, seq * VIS_D,
                f"vision_attn_qkv_w_{i}",
                B["vision_QKV"].ptr.value,
                seq, 3 * VIS_D, VIS_D, stream)
        else:
            gemm.fp16_nn(
                B["vision_x_norm"].ptr.value, W["vision_attn_qkv_w"][i],
                B["vision_QKV"].ptr.value,
                seq, 3 * VIS_D, VIS_D, stream=stream)
        self._bias_add_bf16(
            B["vision_QKV"].ptr.value, W["vision_attn_qkv_b"][i],
            seq, 3 * VIS_D, stream)

        fvk.qkv_split_fp16(
            B["vision_QKV"].ptr.value,
            attn_ptrs["vis_Q"], attn_ptrs["vis_K"], attn_ptrs["vis_V"],
            seq, VIS_D, VIS_D, VIS_D, stream=stream)

        vis_o_ptr = self.attn.run(
            "siglip", i, q_seq=VIS_SEQ_PER_VIEW, stream=stream)

        if use_fp8:
            self._fp8_gemm(
                vis_o_ptr, seq * VIS_D,
                f"vision_attn_o_w_{i}",
                B["vision_x_norm"].ptr.value,
                seq, VIS_D, VIS_D, stream)
        else:
            gemm.fp16_nn(
                vis_o_ptr, W["vision_attn_o_w"][i],
                B["vision_x_norm"].ptr.value,
                seq, VIS_D, VIS_D, stream=stream)
        fvk.bias_residual_fp16(
            B["vision_x"].ptr.value, B["vision_x_norm"].ptr.value,
            W["vision_attn_o_b"][i], seq, VIS_D, stream=stream)

        fvk.layer_norm_fp16(
            B["vision_x"].ptr.value,
            W["vision_pre_ffn_norm_w"][i], W["vision_pre_ffn_norm_b"][i],
            B["vision_x_norm"].ptr.value,
            seq, VIS_D, 1e-5, stream=stream)

        if use_fp8:
            self._fp8_gemm(
                B["vision_x_norm"].ptr.value, seq * VIS_D,
                f"vision_ffn_up_w_{i}",
                B["vision_hidden"].ptr.value,
                seq, VIS_H, VIS_D, stream)
        else:
            gemm.fp16_nn(
                B["vision_x_norm"].ptr.value, W["vision_ffn_up_w"][i],
                B["vision_hidden"].ptr.value,
                seq, VIS_H, VIS_D, stream=stream)
        self._bias_add_bf16(
            B["vision_hidden"].ptr.value, W["vision_ffn_up_b"][i],
            seq, VIS_H, stream)
        fvk.gelu_inplace_fp16(B["vision_hidden"].ptr.value, seq * VIS_H,
                         stream=stream)

        if use_fp8:
            self._fp8_gemm(
                B["vision_hidden"].ptr.value, seq * VIS_H,
                f"vision_ffn_down_w_{i}",
                B["vision_x_norm"].ptr.value,
                seq, VIS_D, VIS_H, stream)
        else:
            gemm.fp16_nn(
                B["vision_hidden"].ptr.value, W["vision_ffn_down_w"][i],
                B["vision_x_norm"].ptr.value,
                seq, VIS_D, VIS_H, stream=stream)
        fvk.bias_residual_fp16(
            B["vision_x"].ptr.value, B["vision_x_norm"].ptr.value,
            W["vision_ffn_down_b"][i], seq, VIS_D, stream=stream)

    # ══════════════════════════════════════════════════════════════════
    #   Phase B: Gemma-2B encoder — identical to Pi0.5
    # ══════════════════════════════════════════════════════════════════

    def transformer_encoder(self, stream=0):
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        seq = self.encoder_seq_len
        vs = self.vision_seq
        use_fp8 = self.use_fp8

        fvk.layer_norm_fp16(
            B["vision_x"].ptr.value,
            W["vision_final_norm_w"], W["vision_final_norm_b"],
            B["vision_x_norm"].ptr.value,
            vs, VIS_D, 1e-5, stream=stream)

        if use_fp8 and "vision_projector_w" in self.weights.get("fp8", {}):
            self._fp8_gemm(
                B["vision_x_norm"].ptr.value, vs * VIS_D,
                "vision_projector_w",
                B["encoder_x"].ptr.value,
                vs, ENC_D, VIS_D, stream)
        else:
            gemm.fp16_nn(
                B["vision_x_norm"].ptr.value,
                W["encoder_multi_modal_projector_w"],
                B["encoder_x"].ptr.value,
                vs, ENC_D, VIS_D, stream=stream)
        self._bias_add_bf16(
            B["encoder_x"].ptr.value,
            W["encoder_multi_modal_projector_b"],
            vs, ENC_D, stream)

        fused = use_fp8 and self.fp8_calibrated
        for i in range(ENC_L):
            self._encoder_layer(i, seq, fuse_b1=(i > 0 and fused),
                                stream=stream)

    def _encoder_layer(self, i, seq, fuse_b1, stream):
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        attn_ptrs = self._attn_ptrs
        fused = self.use_fp8 and self.fp8_calibrated

        qkv_name = f"encoder_attn_qkv_w_{i}"
        if fused:
            act_scale_ptr = self.fp8_act_scales[qkv_name].ptr.value
            if fuse_b1:
                fvk.residual_add_rms_norm_fp8_noweight_fp16(
                    B["encoder_x"].ptr.value,
                    B["encoder_x_norm"].ptr.value,
                    B["enc_act_fp8"].ptr.value,
                    seq, ENC_D, act_scale_ptr, stream)
            else:
                fvk.rms_norm_fp8_noweight_fp16(
                    B["encoder_x"].ptr.value,
                    B["enc_act_fp8"].ptr.value,
                    seq, ENC_D, act_scale_ptr, stream)
            self._fp8_gemm_fused(
                B["enc_act_fp8"].ptr.value, qkv_name,
                B["encoder_QKV"].ptr.value,
                seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D,
                act_scale_ptr, stream)
        elif self.use_fp8:
            fvk.rms_norm_fp16(
                B["encoder_x"].ptr.value, self._rms_ones_enc.ptr.value,
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, 1e-6, stream=stream)
            self._fp8_gemm(
                B["encoder_x_norm"].ptr.value, seq * ENC_D,
                f"encoder_attn_qkv_w_{i}",
                B["encoder_QKV"].ptr.value,
                seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D, stream)
        else:
            fvk.rms_norm_fp16(
                B["encoder_x"].ptr.value, self._rms_ones_enc.ptr.value,
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, 1e-6, stream=stream)
            gemm.fp16_nn(
                B["encoder_x_norm"].ptr.value, W["encoder_attn_qkv_w"][i],
                B["encoder_QKV"].ptr.value,
                seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D, stream=stream)

        # _enc_kv_layer_ptrs returns pre-offset pointers for this
        # (layer, token_offset), so kc_offset=0. kc_stride = ENC_HD
        # (per-token K/V width, 1 KV head × HD).
        k_ptr, v_ptr = self._enc_kv_layer_ptrs(i, offset_tokens=0)
        enc_qkv_stride = (ENC_NH + 2 * ENC_NKV) * ENC_HD
        fvk.qkv_split_rope_kvcache_fp16(
            B["encoder_QKV"].ptr.value,
            B["encoder_rope_weights"].ptr.value,
            attn_ptrs["enc_Q"],
            k_ptr, v_ptr,
            seq, ENC_NH * ENC_HD, ENC_NKV * ENC_HD, ENC_HD,
            enc_qkv_stride, 0, ENC_HD, stream)

        if i == ENC_L - 1:
            return

        enc_o_ptr = self.attn.run("encoder", i, q_seq=seq, stream=stream)

        if self.use_fp8:
            self._fp8_gemm(
                enc_o_ptr, seq * ENC_D,
                f"encoder_attn_o_w_{i}",
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, ENC_D, stream)
        else:
            gemm.fp16_nn(
                enc_o_ptr, W["encoder_attn_o_w"][i],
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, ENC_D, stream=stream)

        gu_name = f"encoder_ffn_gate_up_w_{i}"
        if fused:
            act_scale_gu = self.fp8_act_scales[gu_name].ptr.value
            fvk.residual_add_rms_norm_fp8_noweight_fp16(
                B["encoder_x"].ptr.value, B["encoder_x_norm"].ptr.value,
                B["enc_act_fp8"].ptr.value,
                seq, ENC_D, act_scale_gu, stream)
            self._fp8_gemm_fused(
                B["enc_act_fp8"].ptr.value, gu_name,
                B["encoder_gate_merged"].ptr.value,
                seq, 2 * ENC_H, ENC_D, act_scale_gu, stream)
        elif self.use_fp8:
            fvk.residual_add_fp16(
                B["encoder_x"].ptr.value, B["encoder_x_norm"].ptr.value,
                seq * ENC_D, stream=stream)
            fvk.rms_norm_fp16(
                B["encoder_x"].ptr.value, self._rms_ones_enc.ptr.value,
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, 1e-6, stream=stream)
            self._fp8_gemm(
                B["encoder_x_norm"].ptr.value, seq * ENC_D,
                f"encoder_ffn_gate_up_w_{i}",
                B["encoder_gate_merged"].ptr.value,
                seq, 2 * ENC_H, ENC_D, stream)
        else:
            fvk.residual_add_fp16(
                B["encoder_x"].ptr.value, B["encoder_x_norm"].ptr.value,
                seq * ENC_D, stream=stream)
            fvk.rms_norm_fp16(
                B["encoder_x"].ptr.value, self._rms_ones_enc.ptr.value,
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, 1e-6, stream=stream)
            gemm.fp16_nn(
                B["encoder_x_norm"].ptr.value, W["encoder_ffn_gate_w"][i],
                B["encoder_gate_merged"].ptr.value,
                seq, ENC_H, ENC_D, stream=stream)
            gemm.fp16_nn(
                B["encoder_x_norm"].ptr.value, W["encoder_ffn_up_w"][i],
                B["encoder_hidden"].ptr.value,
                seq, ENC_H, ENC_D, stream=stream)

        down_name = f"encoder_ffn_down_w_{i}"
        if fused:
            act_scale_down = self.fp8_act_scales[down_name].ptr.value
            fvk.gate_geglu_merged_fp8_fp16(
                B["encoder_gate_merged"].ptr.value,
                B["enc_act_fp8_large"].ptr.value,
                seq, ENC_H, act_scale_down, stream=stream)
            self._fp8_gemm_fused(
                B["enc_act_fp8_large"].ptr.value, down_name,
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, ENC_H, act_scale_down, stream)
        elif self.use_fp8:
            fvk.gate_geglu_merged_fp16(
                B["encoder_gate_merged"].ptr.value,
                B["encoder_hidden"].ptr.value,
                seq, ENC_H, stream=stream)
            self._fp8_gemm(
                B["encoder_hidden"].ptr.value, seq * ENC_H,
                f"encoder_ffn_down_w_{i}",
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, ENC_H, stream)
        else:
            fvk.gate_geglu_fp16(
                B["encoder_gate_merged"].ptr.value,
                B["encoder_hidden"].ptr.value,
                B["encoder_hidden"].ptr.value,
                seq * ENC_H, stream=stream)
            gemm.fp16_nn(
                B["encoder_hidden"].ptr.value, W["encoder_ffn_down_w"][i],
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, ENC_H, stream=stream)

        if not fused:
            fvk.residual_add_fp16(
                B["encoder_x"].ptr.value, B["encoder_x_norm"].ptr.value,
                seq * ENC_D, stream=stream)

    # ══════════════════════════════════════════════════════════════════
    #   Phase C-pre: Pi0 state projection + per-step action_time_mlp
    # ══════════════════════════════════════════════════════════════════

    def _state_project(self, stream):
        """Project observation state (1, 32) FP16 → state_token (1, DEC_D) FP16.

        Matches pi0_thor's gmm_fp16 + add_bias_fp16 noise-path convention.
        """
        B = self.bufs
        W = self.weights
        fvk = self.fvk
        fvk.gmm_fp16(
            self._ctx,
            B["state_buf"].ptr.value,
            W["state_proj_w"],
            B["state_token"].ptr.value,
            1, DEC_D, ACTION_DIM, 0.0, stream)
        fvk.add_bias_fp16(
            B["state_token"].ptr.value, W["state_proj_b"],
            1, DEC_D, stream)

    def _assemble_decoder_x(self, step, stream):
        """Build decoder_x = [state_token; action_time_mlp(noise, t_step)].

        Mirrors pi0_thor's per-step pre-decoder exactly: small cuBLAS
        HGEMMs (gmm_fp16) + FP16 bias + fused_add_silu_fp16 all on FP16
        data, no BF16 round-trips.
        """
        B = self.bufs
        W = self.weights
        fvk = self.fvk
        sa = self.chunk_size

        # 1) state_token → decoder_x[0]   (D2D copy, fp16 = 2 bytes)
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(B["decoder_x"].ptr.value),
            ctypes.c_void_p(B["state_token"].ptr.value),
            DEC_D * 2, 3, stream)

        # 2) action_in_proj(noise) → decoder_x[1:S_dec]
        x_action_ptr = B["decoder_x"].ptr.value + DEC_D * 2
        fvk.gmm_fp16(
            self._ctx,
            B["diffusion_noise"].ptr.value,
            W["decoder_action_in_proj_w"],
            x_action_ptr,
            sa, DEC_D, ACTION_DIM, 0.0, stream)
        fvk.add_bias_fp16(
            x_action_ptr, W["decoder_action_in_proj_b"],
            sa, DEC_D, stream)

        # 3) action_time_mlp_in (W_a half) + silu(⋅ + time_proj[s]) + out
        fvk.gmm_fp16(
            self._ctx,
            x_action_ptr,
            W["action_time_mlp_in_wa_w"],
            B["action_time_temp"].ptr.value,
            sa, DEC_D, DEC_D, 0.0, stream)
        time_proj_ptr = W["time_proj_all"] + step * sa * DEC_D * 2
        fvk.fused_add_silu_fp16(
            B["action_time_temp"].ptr.value,
            time_proj_ptr,
            sa * DEC_D, stream)
        fvk.gmm_fp16(
            self._ctx,
            B["action_time_temp"].ptr.value,
            W["action_time_mlp_out_w"],
            x_action_ptr,
            sa, DEC_D, DEC_D, 0.0, stream)
        fvk.add_bias_fp16(
            x_action_ptr, W["action_time_mlp_out_b"],
            sa, DEC_D, stream)

    # ══════════════════════════════════════════════════════════════════
    #   Phase C: Pi0 decoder (18 layers × 10 steps, flow matching)
    # ══════════════════════════════════════════════════════════════════

    def transformer_decoder(self, stream=0):
        fvk = self.fvk
        W = self.weights
        B = self.bufs
        enc_seq = self.encoder_seq_len
        sd = self.S_dec
        sa = self.chunk_size
        gemm = self.gemm

        for step in range(self.num_steps):
            # Build decoder_x for this denoise step.
            self._assemble_decoder_x(step, stream)

            # 18 decoder layers
            for i in range(DEC_L):
                self._decoder_layer(i, enc_seq, sd, stream)

            # Final RMSNorm with weight (Pi0 — standard RMSNorm, not AdaRMS)
            fvk.rms_norm_fp16(
                B["decoder_x"].ptr.value,
                W["decoder_final_norm_w"],
                B["decoder_x_norm"].ptr.value,
                sd, DEC_D, 1e-6, stream=stream)

            # Action out projection: fused accumulate into noise via
            # gmm_fp16 with beta=1.0 — matches pi0_thor's two-step
            # `noise += xn_action @ aow` + `noise += aob`. FP32
            # internal accumulation, one FP16 cast on write; no
            # intermediate BF16/FP16 round-trip through a temp buffer.
            # (Weights pre-scaled by -1/num_steps in the frontend.)
            x_out_action_ptr = B["decoder_x_norm"].ptr.value + DEC_D * 2
            fvk.gmm_fp16(
                self._ctx, x_out_action_ptr,
                W["decoder_action_out_proj_w"],
                B["diffusion_noise"].ptr.value,
                sa, ACTION_DIM, DEC_D, 1.0, stream)
            fvk.add_bias_fp16(
                B["diffusion_noise"].ptr.value,
                W["decoder_action_out_proj_b"],
                sa, ACTION_DIM, stream)

    def _decoder_layer(self, i, enc_seq, sd, stream):
        """One Pi0 decoder layer — encoder-pattern + state-masked cross-attn."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        attn_ptrs = self._attn_ptrs
        fused = self.use_fp8_decoder and self.fp8_calibrated

        # C1: RMSNorm(decoder_x) → FP8 (or BF16 fallback) → QKV GEMM
        qkv_name = f"decoder_attn_qkv_w_{i}"
        if fused:
            act_scale_qkv = self.fp8_act_scales[qkv_name].ptr.value
            fvk.rms_norm_fp8_noweight_fp16(
                B["decoder_x"].ptr.value,
                B["dec_act_fp8"].ptr.value,
                sd, DEC_D, act_scale_qkv, stream)
            self._fp8_gemm_fused(
                B["dec_act_fp8"].ptr.value, qkv_name,
                B["decoder_QKV"].ptr.value,
                sd, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D,
                act_scale_qkv, stream)
        elif self.use_fp8_decoder:
            fvk.rms_norm_fp16(
                B["decoder_x"].ptr.value, self._rms_ones_dec.ptr.value,
                B["decoder_x_norm"].ptr.value,
                sd, DEC_D, 1e-6, stream=stream)
            self._fp8_gemm(
                B["decoder_x_norm"].ptr.value, sd * DEC_D,
                qkv_name,
                B["decoder_QKV"].ptr.value,
                sd, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D, stream)
        else:
            fvk.rms_norm_fp16(
                B["decoder_x"].ptr.value, self._rms_ones_dec.ptr.value,
                B["decoder_x_norm"].ptr.value,
                sd, DEC_D, 1e-6, stream=stream)
            gemm.fp16_nn(
                B["decoder_x_norm"].ptr.value, W["decoder_attn_qkv_w"][i],
                B["decoder_QKV"].ptr.value,
                sd, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D, stream=stream)

        # C2: QKV split + RoPE. Decoder K/V write into enc cache at enc_seq.
        # Pre-offset pointers (layer + token offset), so kc_offset=0.
        k_ptr, v_ptr = self._enc_kv_layer_ptrs(i, offset_tokens=enc_seq)
        dec_qkv_stride = (DEC_NH + 2 * DEC_NKV) * DEC_HD
        fvk.qkv_split_rope_kvcache_fp16(
            B["decoder_QKV"].ptr.value,
            B["decoder_rope_weights"].ptr.value,
            attn_ptrs["dec_Q"],
            k_ptr, v_ptr,
            sd, DEC_NH * DEC_HD, DEC_NKV * DEC_HD, DEC_HD,
            dec_qkv_stride, 0, DEC_HD, stream)

        # C3: Cross-attention with Pi0 state mask. Row 0 (state) may only
        # attend to the encoder prefix plus itself; action rows attend
        # to the full KV.
        dec_o_ptr = self.attn.run(
            "decoder", i,
            q_seq=sd,
            kv_seq=enc_seq + sd,
            stream=stream,
            state_nk=self._state_nk,
        )

        # C4: Attn output projection (FP8 or BF16) → dec_attn_proj
        if self.use_fp8_decoder:
            self._fp8_gemm(
                dec_o_ptr, sd * DEC_NH * DEC_HD,
                f"decoder_attn_o_w_{i}",
                B["dec_attn_proj"].ptr.value,
                sd, DEC_D, DEC_NH * DEC_HD, stream)
        else:
            gemm.fp16_nn(
                dec_o_ptr, W["decoder_attn_o_w"][i],
                B["dec_attn_proj"].ptr.value,
                sd, DEC_D, DEC_NH * DEC_HD, stream=stream)

        # C5: residual_add + RMSNorm(decoder_x + attn_proj) → FP8, then
        # FFN gate+up. For the final layer we also produce the same FP8
        # rail (next step's C1 re-normalises decoder_x anyway).
        gu_name = f"decoder_ffn_gate_up_w_{i}"
        if fused:
            act_scale_gu = self.fp8_act_scales[gu_name].ptr.value
            fvk.residual_add_rms_norm_fp8_noweight_fp16(
                B["decoder_x"].ptr.value, B["dec_attn_proj"].ptr.value,
                B["dec_act_fp8"].ptr.value,
                sd, DEC_D, act_scale_gu, stream)
            self._fp8_gemm_fused(
                B["dec_act_fp8"].ptr.value, gu_name,
                B["decoder_gate_merged"].ptr.value,
                sd, 2 * DEC_H, DEC_D, act_scale_gu, stream)
        elif self.use_fp8_decoder:
            fvk.residual_add_fp16(
                B["decoder_x"].ptr.value, B["dec_attn_proj"].ptr.value,
                sd * DEC_D, stream=stream)
            fvk.rms_norm_fp16(
                B["decoder_x"].ptr.value, self._rms_ones_dec.ptr.value,
                B["decoder_x_norm"].ptr.value,
                sd, DEC_D, 1e-6, stream=stream)
            self._fp8_gemm(
                B["decoder_x_norm"].ptr.value, sd * DEC_D,
                gu_name,
                B["decoder_gate_merged"].ptr.value,
                sd, 2 * DEC_H, DEC_D, stream)
        else:
            fvk.residual_add_fp16(
                B["decoder_x"].ptr.value, B["dec_attn_proj"].ptr.value,
                sd * DEC_D, stream=stream)
            fvk.rms_norm_fp16(
                B["decoder_x"].ptr.value, self._rms_ones_dec.ptr.value,
                B["decoder_x_norm"].ptr.value,
                sd, DEC_D, 1e-6, stream=stream)
            gemm.fp16_nn(
                B["decoder_x_norm"].ptr.value, W["decoder_ffn_gate_w"][i],
                B["decoder_gate_merged"].ptr.value,
                sd, DEC_H, DEC_D, stream=stream)
            gemm.fp16_nn(
                B["decoder_x_norm"].ptr.value, W["decoder_ffn_up_w"][i],
                B["decoder_hidden"].ptr.value,
                sd, DEC_H, DEC_D, stream=stream)

        # C6: SiLU(gate) * up → hidden → Down GEMM
        down_name = f"decoder_ffn_down_w_{i}"
        if fused:
            act_scale_down = self.fp8_act_scales[down_name].ptr.value
            fvk.gate_geglu_merged_fp8_fp16(
                B["decoder_gate_merged"].ptr.value,
                B["dec_act_fp8_large"].ptr.value,
                sd, DEC_H, act_scale_down, stream=stream)
            self._fp8_gemm_fused(
                B["dec_act_fp8_large"].ptr.value, down_name,
                B["dec_attn_proj"].ptr.value,
                sd, DEC_D, DEC_H, act_scale_down, stream)
        elif self.use_fp8_decoder:
            fvk.gate_geglu_merged_fp16(
                B["decoder_gate_merged"].ptr.value,
                B["decoder_hidden"].ptr.value,
                sd, DEC_H, stream=stream)
            self._fp8_gemm(
                B["decoder_hidden"].ptr.value, sd * DEC_H,
                down_name,
                B["dec_attn_proj"].ptr.value,
                sd, DEC_D, DEC_H, stream)
        else:
            fvk.gate_geglu_fp16(
                B["decoder_gate_merged"].ptr.value,
                B["decoder_hidden"].ptr.value,
                B["decoder_hidden"].ptr.value,
                sd * DEC_H, stream=stream)
            gemm.fp16_nn(
                B["decoder_hidden"].ptr.value, W["decoder_ffn_down_w"][i],
                B["dec_attn_proj"].ptr.value,
                sd, DEC_D, DEC_H, stream=stream)

        # C7: FFN residual: decoder_x += dec_attn_proj
        fvk.residual_add_fp16(
            B["decoder_x"].ptr.value, B["dec_attn_proj"].ptr.value,
            sd * DEC_D, stream=stream)

    # ══════════════════════════════════════════════════════════════════
    #   Full pipeline + calibration + graph
    # ══════════════════════════════════════════════════════════════════

    def run_pipeline(self, stream=0):
        """Run state_proj → vision → encoder → decoder end-to-end."""
        self._copy_lang_embeds_to_encoder_x(stream=stream)
        self._state_project(stream)
        self.vision_encoder(stream)
        self.transformer_encoder(stream)
        self.transformer_decoder(stream)

    def calibrate_fp8(self):
        if not self.use_fp8 or self.fp8_calibrated:
            return
        if len(self.fp8_act_scales) > 0:
            self.fp8_calibrated = True
            logger.info(
                "FP8 reuse-from-forward: %d scales already populated",
                len(self.fp8_act_scales))
            return
        self.fp8_calibrated = False
        self.run_pipeline(stream=0)
        self._cudart.cudaDeviceSynchronize()
        self.fp8_calibrated = True
        logger.info("FP8 calibrated: %d activation scales collected",
                    len(self.fp8_act_scales))

    def autotune_gemms(self):
        # FP16 pipeline: gemm.fp16_nn uses cuBLASLt's cached-descriptor
        # path (heuristic algorithm picked on first call and reused),
        # and FP8 GEMMs go through fvk.fp8_gemm_descale_fp16 (CUTLASS,
        # no cuBLASLt autotune handle). Neither needs an explicit warm-
        # up call through GemmRunner; this method is a no-op kept for
        # API compatibility with the Pi0.5 RTX pipeline structure.
        pass

    def record_infer_graph(self, external_stream_int=None):
        if self.use_fp8 and not self.fp8_calibrated:
            self.calibrate_fp8()
        self.autotune_gemms()

        self._graph = CUDAGraph()
        if external_stream_int is None:
            stream = self._graph.create_stream()
            stream_int = stream.value or 0
            stream_handle = stream
        else:
            stream_int = int(external_stream_int)
            stream_handle = ctypes.c_void_p(stream_int)
        self._graph_stream = stream_handle

        for _ in range(3):
            self.run_pipeline(stream=stream_int)
        self._cudart.cudaStreamSynchronize(stream_handle)

        self._graph.begin_capture(stream_handle)
        self.run_pipeline(stream=stream_int)
        self._graph.end_capture(stream_handle)
        self._cudart.cudaStreamSynchronize(stream_handle)
        logger.info("CUDA Graph captured for Pi0Pipeline")

    # ══════════════════════════════════════════════════════════════════
    #   Public API
    # ══════════════════════════════════════════════════════════════════

    @property
    def input_images_buf(self):
        return self.bufs["observation_images_normalized"]

    @property
    def input_noise_buf(self):
        return self.bufs["diffusion_noise"]

    @property
    def input_state_buf(self):
        """Pipeline input: observation state (1, 32) bf16. Pi0-specific."""
        return self.bufs["state_buf"]

    @property
    def input_encoder_x_buf(self):
        return self.bufs["encoder_x"]

    def set_language_embeds(self, lang_embeds_np):
        prompt_len = lang_embeds_np.shape[0]
        assert prompt_len <= self.max_prompt_len, (
            f"prompt_len {prompt_len} > max_prompt_len {self.max_prompt_len}")
        assert lang_embeds_np.shape[1] == ENC_D

        arr = np.ascontiguousarray(lang_embeds_np)
        self._lang_embeds_buf = CudaBuffer.from_numpy(arr)
        self._current_prompt_len = prompt_len

        self._set_decoder_rope_for_prompt(prompt_len)
        self._copy_lang_embeds_to_encoder_x()

    def _copy_lang_embeds_to_encoder_x(self, stream=0):
        if not hasattr(self, "_lang_embeds_buf"):
            return
        start_byte = self.vision_seq * ENC_D * 2
        dst_ptr = self.bufs["encoder_x"].ptr.value + start_byte
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(dst_ptr),
            self._lang_embeds_buf.ptr,
            self._lang_embeds_buf.nbytes, 3, stream)

    def forward(self):
        """Replay the captured graph (or fall back to ``run_pipeline``).

        Frontend must write inputs to ``input_images_buf``,
        ``input_noise_buf`` and ``input_state_buf`` before calling;
        after return, ``input_noise_buf`` holds the final actions.
        """
        if self._graph is not None:
            self._graph.replay(self._graph_stream)
            self._cudart.cudaStreamSynchronize(self._graph_stream)
        else:
            self.run_pipeline(stream=0)
            self._cudart.cudaDeviceSynchronize()
        return self.bufs["diffusion_noise"].ptr.value

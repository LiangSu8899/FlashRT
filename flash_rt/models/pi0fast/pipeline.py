"""FlashVLA -- Thor SM110 + RTX 5090 SM120 Pipeline for Pi0-FAST.

Autoregressive token generation (NOT diffusion).
Single Gemma 2B model (no separate action expert).

Functions:
    prefill_forward_pi0fast     Prefill: 18-layer forward, populate KV cache
    decode_step_pi0fast_bf16    Single token decode (BF16 residual)
    prefill_calibrate_pi0fast   Calibrate FP8 scales during prefill
    siglip_forward_sm120        SigLIP 27-layer vision encoder, SM120
                                variant using decomposed FP8 GEMM +
                                elementwise epilogues (the SM100 fused
                                epilogues fp8_nn_bias / _res / _gelu_bias
                                are not supported by SM120 cuBLASLt).
"""

import math
import ctypes
import numpy as np

from flash_vla.hardware.thor.shared_primitives import (
    _gpu_copy, _gpu_zero, _gpu_sync,
    _d2h_float, _d2h_floats,
    _measure_scale_gpu,
    _gpu_alloc, _gpu_free,
)

_crt = ctypes.CDLL('libcudart.so')


def _measure_scale_gpu_bf16(fvk_mod, bf16_ptr, n_elements, d_scale_ptr,
                             d_fp8_scratch, stream=0):
    """BF16 variant of _measure_scale_gpu.

    The base _measure_scale_gpu calls quantize_fp8_device_fp16 which
    interprets bytes as __half. For BF16 buffers (Pi0-FAST residual stream),
    those bytes are wildly misinterpreted (e.g., real value 45.0 reads as
    ~3.0 in FP16, giving an amax ~58x too small → calibration scale ~58x too
    small → FP8 saturates → catastrophic precision loss).

    Use the BF16 binding (`quantize_fp8_device`) which correctly typed-casts
    to __nv_bfloat16. Output FP8 is discarded; we only care about d_scale.
    """
    fvk_mod.quantize_fp8_device(bf16_ptr, d_fp8_scratch, d_scale_ptr,
                                 n_elements, stream)


# ==================================================================
# Prefill Forward (18 layers, ALL layers complete — no last-layer skip)
# ==================================================================

def prefill_forward_pi0fast(gemm, fvk, bufs, weights, dims, stream=0):
    """Full prefill forward: ALL 18 layers, BF16 residual stream.

    Uses BF16 for residual buffer (x) to handle activations > 65504 (FP16 max).
    Pi0-FAST Gemma 2B residual reaches ~569K at deep layers (verified in JAX bf16).
    GEMM outputs use BF16 variants for O proj and Down proj (fg path).
    QKV/attention/KV-cache stay FP16 (values always < 1000, safe).

    SM dispatch:
      SM100/SM110 (Thor):  CUTLASS FP8 (cutlass_fp8_sq / _t1 / _wide_bf16out),
                            alpha = act_scale * w_scale precomputed host-side.
      SM120 (RTX 5090):    cuBLASLt FP8 (fp8_gemm_descale_fp16 / _bf16out) via
                            device-side (act_scale_ptr, w_scale_ptr) pair.
      The branch is selected per-call by probing ``hasattr(fvk, 'cutlass_fp8_sq')``;
      Thor path is byte-for-byte unchanged.

    Args:
        gemm: GemmRunner
        fvk: flash_vla_kernels module
        bufs: dict — x (BF16!), x_fp8, qkv (FP16), logits, attn_out (FP16),
                     o_fp8, gate (FP16), hid_fp8, fg (BF16!), xn (BF16!), ctx
        weights: dict — always required:
                    rope, Kc, Vc, final_norm_w, act_scales
                 SM100 (CUTLASS path) requires:
                    qkv_w[L], o_w[L], gate_w[L], down_w[L], alpha_host (list len=L*4)
                 SM120 (cuBLASLt path) requires:
                    qkv_w_flat, o_w_flat, gate_w_flat, down_w_flat  (flat [K,N] buffers)
                    w_scales  (device float32 ptr, len=L*4)
        dims: dict — Se, D, H, NH, HD, L, total_keys_max
    """
    Se = dims['Se']
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    L = dims['L']
    total_keys_max = dims['total_keys_max']
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))

    x = bufs['x']          # BF16 residual stream
    x_fp8 = bufs['x_fp8']
    qkv = bufs['qkv']      # FP16 (QKV values safe for FP16)
    logits = bufs['logits']
    attn_out = bufs['attn_out']  # FP16
    o_fp8 = bufs['o_fp8']
    gate = bufs['gate']     # FP16 (GELU input, safe)
    hid_fp8 = bufs['hid_fp8']
    fg = bufs['fg']         # BF16 (GEMM output, may exceed FP16 range)

    act_scales = weights['act_scales']

    # SM dispatch — probe once, hoist out of the layer loop
    _use_sm100 = hasattr(fvk, 'cutlass_fp8_sq')
    if _use_sm100:
        alpha_host = weights['alpha_host']
        qkv_w_list  = weights['qkv_w']
        o_w_list    = weights['o_w']
        gate_w_list = weights['gate_w']
        down_w_list = weights['down_w']
    else:
        w_scales    = weights['w_scales']
        qkv_w_flat  = weights['qkv_w_flat']
        o_w_flat    = weights['o_w_flat']
        gate_w_flat = weights['gate_w_flat']
        down_w_flat = weights['down_w_flat']

    for l in range(L):
        last = (l == L - 1)
        kv_elem_off = l * total_keys_max * HD

        as_qkv = act_scales + (l * 4 + 0) * 4
        as_o   = act_scales + (l * 4 + 1) * 4
        as_gu  = act_scales + (l * 4 + 2) * 4
        as_d   = act_scales + (l * 4 + 3) * 4

        # 1. RMSNorm(BF16 x) → FP8
        fvk.rms_norm_fp8_noweight_bf16(x, x_fp8, Se, D, as_qkv, stream)

        # 2. QKV GEMM → FP16 (QKV values always < 500, safe for FP16)
        if _use_sm100:
            fvk.cutlass_fp8_sq(x_fp8, qkv_w_list[l], qkv,
                               Se, 2560, D, alpha_host[l * 4 + 0], 0.0, stream)
        else:
            fvk.fp8_gemm_descale_fp16(x_fp8, qkv_w_flat + l * D * 2560, qkv,
                                       Se, 2560, D,
                                       as_qkv, w_scales + (l * 4 + 0) * 4, stream)

        # 3. Split + RoPE + KV cache (FP16)
        fvk.qkv_split_rope_kvcache_fp16(
            qkv, weights['rope'], attn_out,
            weights['Kc'], weights['Vc'],
            Se, Q_dim, K_dim, HD, 2560, kv_elem_off, HD, stream)

        # 4. Attention (FP16)
        K_ptr = weights['Kc'] + kv_elem_off * 2
        V_ptr = weights['Vc'] + kv_elem_off * 2
        fvk.attention_qkv_fp16(bufs['ctx'], attn_out, K_ptr, V_ptr,
                                logits, attn_out, Se, Se, NH, HD, attn_scale, stream)

        # 5. O proj → BF16 (output may exceed FP16 at deep layers)
        fvk.quantize_fp8_static_fp16(attn_out, o_fp8, as_o, Se * D, stream)
        if _use_sm100:
            fvk.cutlass_fp8_sq_bf16out(o_fp8, o_w_list[l], fg,
                                        Se, D, D, alpha_host[l * 4 + 1], 0.0, stream)
        else:
            fvk.fp8_gemm_descale_bf16out(o_fp8, o_w_flat + l * D * D, fg,
                                          Se, D, D,
                                          as_o, w_scales + (l * 4 + 1) * 4, stream)

        # 6. Residual(BF16) + RMSNorm → FP8
        fvk.residual_add_rms_norm_fp8_noweight_bf16(x, fg, x_fp8, Se, D, as_gu, stream)

        # 7. Gate+Up GEMM → FP16 (GELU input, values always safe)
        if _use_sm100:
            fvk.cutlass_fp8_t1(x_fp8, gate_w_list[l], gate,
                               Se, H * 2, D, alpha_host[l * 4 + 2], 0.0, stream)
        else:
            fvk.fp8_gemm_descale_fp16(x_fp8, gate_w_flat + l * D * H * 2, gate,
                                       Se, H * 2, D,
                                       as_gu, w_scales + (l * 4 + 2) * 4, stream)

        # 8. GELU → FP8 (FP16 gate input, FP32 internal, FP8 output — safe)
        fvk.gate_geglu_merged_fp8_fp16(gate, hid_fp8, Se, H, as_d, stream)

        # 9. Down GEMM → BF16 (output may reach 100K+ at deep layers)
        if _use_sm100:
            fvk.cutlass_fp8_wide_bf16out(hid_fp8, down_w_list[l], fg,
                                          Se, D, H, alpha_host[l * 4 + 3], 0.0, stream)
        else:
            fvk.fp8_gemm_descale_bf16out(hid_fp8, down_w_flat + l * H * D, fg,
                                          Se, D, H,
                                          as_d, w_scales + (l * 4 + 3) * 4, stream)

        # 10. Residual(BF16) + prep next layer
        if not last:
            as_next = act_scales + ((l + 1) * 4 + 0) * 4
            fvk.residual_add_rms_norm_fp8_noweight_bf16(x, fg, x_fp8, Se, D, as_next, stream)
        else:
            fvk.residual_add(x, fg, Se * D, stream)  # BF16 residual_add

    # Final RMSNorm with weight → BF16 output
    fvk.rms_norm(x, weights['final_norm_w'], bufs['xn'],
                 Se, D, 1e-6, stream)


# ==================================================================
# Decode Step (1 token, 18 layers, KV cache read+write)
# ==================================================================

def decode_step_pi0fast(ctx, fvk, bufs, weights, dims, step, stream=0):
    """Single-token decode through 18 layers with KV cache.

    Args:
        ctx: FvkContext
        fvk: flash_vla_kernels module
        bufs: dict — x, x_fp8, qkv, logits, attn_out, o_fp8, gate, hid_fp8, fg, xn
        weights: dict — qkv_w[L], o_w[L], gate_w[L], down_w[L], rope_base,
                        Kc, Vc, final_norm_w, act_scales, w_scales
        dims: dict — D, H, NH, HD, L, prefill_len, total_keys_max
        step: current decode step index (0, 1, 2, ...)
        stream: CUDA stream
    """
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    L = dims['L']
    prefill_len = dims['prefill_len']
    total_keys_max = dims['total_keys_max']
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))

    # KV cache: prefill fills [0, prefill_len), decode writes at pos = prefill_len + step
    # Attention must see all keys including the just-written token
    S_kv = prefill_len + step + 1  # [0, prefill_len + step] inclusive

    x = bufs['x']
    x_fp8 = bufs['x_fp8']
    qkv = bufs['qkv']
    logits = bufs['logits']
    attn_out = bufs['attn_out']
    o_fp8 = bufs['o_fp8']
    gate = bufs['gate']
    hid_fp8 = bufs['hid_fp8']
    fg = bufs['fg']

    act_scales = weights['act_scales']
    w_scales = weights['w_scales']

    # RoPE pointer for this position
    # Training uses contiguous positions [0, ..., seq_len-1]. Prefill occupies [0, prefill_len).
    # Decode tokens continue at [prefill_len, prefill_len+1, ...].
    # (Original JAX inference uses prefill_len+step+1 due to left_to_right_align artifact,
    #  but we follow the training convention for correctness.)
    pos = prefill_len + step
    rope_ptr = weights['rope_base'] + pos * 256 * 2  # byte offset, fp16

    for l in range(L):
        last = (l == L - 1)

        as_qkv = act_scales + (l * 4 + 0) * 4
        ws_qkv = w_scales + (l * 4 + 0) * 4
        as_o = act_scales + (l * 4 + 1) * 4
        ws_o = w_scales + (l * 4 + 1) * 4
        as_gu = act_scales + (l * 4 + 2) * 4
        ws_gu = w_scales + (l * 4 + 2) * 4
        as_d = act_scales + (l * 4 + 3) * 4
        ws_d = w_scales + (l * 4 + 3) * 4

        # 1. RMSNorm -> FP8
        fvk.rms_norm_fp8_noweight_fp16(x, x_fp8, 1, D, as_qkv, stream)

        # 2. QKV GEMM [1,D] x [D,2560]
        qw_ptr = weights['qkv_w_flat'] + l * D * 2560
        fvk.fp8_gemm_descale_fp16(x_fp8, qw_ptr, qkv, 1, 2560, D,
                                   as_qkv, ws_qkv, stream)

        # 3. Split + RoPE + KV cache write at position (prefill_len + step)
        kv_write_off = l * total_keys_max * HD + pos * HD
        fvk.qkv_split_rope_kvcache_fp16(
            qkv, rope_ptr, attn_out,
            weights['Kc'], weights['Vc'],
            1, Q_dim, K_dim, HD, 2560,
            kv_write_off, HD, stream)

        # 4. Attention: Q[1] x K[S_kv]
        #    S_kv may be odd (prefill_len + step + 1) → use padded variant
        K_ptr = weights['Kc'] + l * total_keys_max * HD * 2
        V_ptr = weights['Vc'] + l * total_keys_max * HD * 2
        if S_kv % 2 == 0:
            fvk.attention_qkv_fp16(ctx, attn_out, K_ptr, V_ptr,
                                    logits, attn_out,
                                    1, S_kv, NH, HD, attn_scale, stream)
        else:
            fvk.attention_qkv_fp16_padded(ctx, attn_out, K_ptr, V_ptr,
                                           logits, attn_out,
                                           1, S_kv, NH, HD, attn_scale, stream)

        # 5. Quantize attn -> FP8 + O proj
        fvk.quantize_fp8_static_fp16(attn_out, o_fp8, as_o, 1 * D, stream)
        ow_ptr = weights['o_w_flat'] + l * NH * HD * D
        fvk.fp8_gemm_descale_fp16(o_fp8, ow_ptr, fg, 1, D, NH * HD,
                                   as_o, ws_o, stream)

        # 6. Residual + RMSNorm -> FP8
        as_gu_ptr = act_scales + (l * 4 + 2) * 4
        fvk.residual_add_rms_norm_fp8_noweight_fp16(x, fg, x_fp8,
                                                      1, D, as_gu_ptr, stream)

        # 7. Gate+Up GEMM [1,D] x [D,2H]
        gw_ptr = weights['gate_w_flat'] + l * D * H * 2
        fvk.fp8_gemm_descale_fp16(x_fp8, gw_ptr, gate, 1, H * 2, D,
                                   as_gu, ws_gu, stream)

        # 8. GELU(gate) * up -> FP8
        fvk.gate_geglu_merged_fp8_fp16(gate, hid_fp8, 1, H, as_d, stream)

        # 9. Down GEMM [1,H] x [H,D]
        dw_ptr = weights['down_w_flat'] + l * H * D
        fvk.fp8_gemm_descale_fp16(hid_fp8, dw_ptr, fg, 1, D, H,
                                   as_d, ws_d, stream)

        # 10. Residual
        if not last:
            as_next = act_scales + ((l + 1) * 4 + 0) * 4
            fvk.residual_add_rms_norm_fp8_noweight_fp16(x, fg, x_fp8,
                                                          1, D, as_next, stream)
        else:
            fvk.residual_add_fp16(x, fg, 1 * D, stream)

    # Final RMSNorm with weight
    fvk.rms_norm_fp16(x, weights['final_norm_w'], bufs['xn'],
                      1, D, 1e-6, stream)


# ==================================================================
# Decode Step BF16 — Pi0-FAST hidden state can reach ~569K, needs BF16
# residual stream. Mirrors decode_step_pi0fast structure but uses BF16
# kernels at the residual chain (x, fg, xn) and the new bf16out cuBLASLt
# GEMM for the two GEMMs whose outputs feed back into the residual (O and
# Down). QKV / Gate+Up GEMMs keep FP16 output because their consumers
# (qkv_split_rope, GELU) only have FP16 variants and their outputs are
# always within FP16 range in Gemma 2B.
# ==================================================================

def decode_step_pi0fast_bf16(ctx, fvk, bufs, weights, dims, step, stream=0):
    """BF16 residual variant of decode_step_pi0fast.

    Required buffer dtypes (allocated by caller):
        x       : BF16 [1, D]      — residual stream
        x_fp8   : uint8 [D]
        qkv     : FP16 [1, 2560]   — consumed by qkv_split_rope_kvcache_fp16
        attn_out: FP16 [1, NH*HD]  — consumed by attention_qkv_fp16
        logits  : FP16 [NH, S_kv]  — softmax scratch
        o_fp8   : uint8 [NH*HD]
        gate    : FP16 [1, 2H]     — consumed by gate_geglu_merged_fp8_fp16
        hid_fp8 : uint8 [H]
        fg      : BF16 [1, D]      — GEMM output, added back to residual
        xn      : BF16 [1, D]      — final RMSNorm output

    SM120-only buffer (ignored on Thor):
        fg_scratch : BF16 [1, D]   — accumulator scratch for the split-K
                                       Down GEMM workaround (see down_proj
                                       code below for rationale).

    weights dict additionally needs:
        final_norm_w : BF16 weight (use rms_norm BF16 template, not rms_norm_fp16)

    All other args identical to decode_step_pi0fast (dims, step, etc.).
    """
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    L = dims['L']
    prefill_len = dims['prefill_len']
    total_keys_max = dims['total_keys_max']
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))

    S_kv = prefill_len + step + 1

    x = bufs['x']            # BF16
    x_fp8 = bufs['x_fp8']
    qkv = bufs['qkv']        # FP16
    logits = bufs['logits']
    attn_out = bufs['attn_out']  # FP16
    o_fp8 = bufs['o_fp8']
    gate = bufs['gate']      # FP16
    hid_fp8 = bufs['hid_fp8']
    fg = bufs['fg']          # BF16
    # fg_scratch is required only when the SM120 split-K Down GEMM
    # workaround is active. Thor does not read it.
    fg_scratch = bufs.get('fg_scratch', 0)

    # SM120 cuBLASLt LtMatmul FP8 has a heuristic failure for
    # [M<=~128, N=D, K=H=16384]: it returns zeros silently. The symptom is
    # first decode token correct (from prefill) and all subsequent tokens
    # wrong (because Down proj output is zero). Split the K=16384 axis into
    # 4 chunks of K=4096 — K=4096 is inside the working region — run 4
    # sub-GEMMs with matching weight slices and sum the BF16 partials
    # into fg. Verified cos=0.999995 vs ground-truth FP8 GEMM at this shape.
    #
    # Activated only on SM120 (detected by absence of cutlass_fp8_wide, the
    # same probe the pipeline uses elsewhere). Thor uses the direct single
    # GEMM as before.
    _need_down_split = not hasattr(fvk, 'cutlass_fp8_wide')

    act_scales = weights['act_scales']
    w_scales = weights['w_scales']

    pos = prefill_len + step
    rope_ptr = weights['rope_base'] + pos * 256 * 2  # byte offset, fp16

    for l in range(L):
        last = (l == L - 1)

        as_qkv = act_scales + (l * 4 + 0) * 4
        ws_qkv = w_scales + (l * 4 + 0) * 4
        as_o = act_scales + (l * 4 + 1) * 4
        ws_o = w_scales + (l * 4 + 1) * 4
        as_gu = act_scales + (l * 4 + 2) * 4
        ws_gu = w_scales + (l * 4 + 2) * 4
        as_d = act_scales + (l * 4 + 3) * 4
        ws_d = w_scales + (l * 4 + 3) * 4

        # 1. RMSNorm(BF16 x) → FP8 (noweight; weight already fused into qkv_w)
        fvk.rms_norm_fp8_noweight_bf16(x, x_fp8, 1, D, as_qkv, stream)

        # 2. QKV GEMM [1, D] × [D, 2560] → FP16 qkv (values always small for Gemma 2B)
        qw_ptr = weights['qkv_w_flat'] + l * D * 2560
        fvk.fp8_gemm_descale_fp16(x_fp8, qw_ptr, qkv, 1, 2560, D,
                                   as_qkv, ws_qkv, stream)

        # 3. Split + RoPE + KV cache write at position (prefill_len + step)
        kv_write_off = l * total_keys_max * HD + pos * HD
        fvk.qkv_split_rope_kvcache_fp16(
            qkv, rope_ptr, attn_out,
            weights['Kc'], weights['Vc'],
            1, Q_dim, K_dim, HD, 2560,
            kv_write_off, HD, stream)

        # 4. Attention Q[1] × K[S_kv] (FP16, KV cache stays FP16)
        K_ptr = weights['Kc'] + l * total_keys_max * HD * 2
        V_ptr = weights['Vc'] + l * total_keys_max * HD * 2
        if S_kv % 2 == 0:
            fvk.attention_qkv_fp16(ctx, attn_out, K_ptr, V_ptr,
                                    logits, attn_out,
                                    1, S_kv, NH, HD, attn_scale, stream)
        else:
            fvk.attention_qkv_fp16_padded(ctx, attn_out, K_ptr, V_ptr,
                                           logits, attn_out,
                                           1, S_kv, NH, HD, attn_scale, stream)

        # 5. Quantize attn → FP8 + O proj → BF16 fg (writes into residual addend)
        fvk.quantize_fp8_static_fp16(attn_out, o_fp8, as_o, 1 * D, stream)
        ow_ptr = weights['o_w_flat'] + l * NH * HD * D
        fvk.fp8_gemm_descale_bf16out(o_fp8, ow_ptr, fg, 1, D, NH * HD,
                                      as_o, ws_o, stream)

        # 6. Residual(BF16 x) + Residual(BF16 fg) → BF16 sum + RMSNorm → FP8
        fvk.residual_add_rms_norm_fp8_noweight_bf16(x, fg, x_fp8,
                                                      1, D, as_gu, stream)

        # 7. Gate+Up GEMM [1, D] × [D, 2H] → FP16 gate
        #    cutlass_fp8_wide is 1.41x faster at M=1 (0.259 vs 0.366 ms)
        if 'gate_w_list' in weights:
            fvk.cutlass_fp8_wide(x_fp8, weights['gate_w_list'][l], gate,
                                  1, H * 2, D,
                                  weights['alpha_host'][l * 4 + 2], 0.0, stream)
        else:
            gw_ptr = weights['gate_w_flat'] + l * D * H * 2
            fvk.fp8_gemm_descale_fp16(x_fp8, gw_ptr, gate, 1, H * 2, D,
                                       as_gu, ws_gu, stream)

        # 8. GELU(gate) × up → FP8
        fvk.gate_geglu_merged_fp8_fp16(gate, hid_fp8, 1, H, as_d, stream)

        # 9. Down GEMM [1, H] × [H, D] → BF16 fg
        dw_ptr = weights['down_w_flat'] + l * H * D
        if _need_down_split:
            # SM120 split-K workaround — see _need_down_split comment above.
            # K=H=16384 is split into 4 chunks of 4096 (inside the working
            # cuBLASLt FP8 region). Weight is [K, N] row-major so each chunk
            # of K advances the weight pointer by chunk_k * N bytes (FP8 = 1B).
            chunks = 4
            chunk_k = H // chunks  # 4096 for Gemma 2B
            # c=0 writes directly into fg (no add)
            fvk.fp8_gemm_descale_bf16out(hid_fp8, dw_ptr, fg,
                                          1, D, chunk_k,
                                          as_d, ws_d, stream)
            for c in range(1, chunks):
                act_p = hid_fp8 + c * chunk_k
                w_p   = dw_ptr + c * chunk_k * D
                fvk.fp8_gemm_descale_bf16out(act_p, w_p, fg_scratch,
                                              1, D, chunk_k,
                                              as_d, ws_d, stream)
                fvk.residual_add(fg, fg_scratch, 1 * D, stream)
        else:
            fvk.fp8_gemm_descale_bf16out(hid_fp8, dw_ptr, fg,
                                          1, D, H,
                                          as_d, ws_d, stream)

        # 10. Residual + next-layer prep
        if not last:
            as_next = act_scales + ((l + 1) * 4 + 0) * 4
            fvk.residual_add_rms_norm_fp8_noweight_bf16(x, fg, x_fp8,
                                                          1, D, as_next, stream)
        else:
            fvk.residual_add(x, fg, 1 * D, stream)  # BF16 residual_add (Pi0-FAST safe)

    # Final RMSNorm (BF16 weight) → BF16 xn
    fvk.rms_norm(x, weights['final_norm_w'], bufs['xn'],
                 1, D, 1e-6, stream)


# ==================================================================
# Prefill Calibration (ALL 18 layers — unlike encoder_forward_calibrate)
# ==================================================================

def prefill_calibrate_pi0fast(gemm, fvk_mod, bufs, weights, dims,
                               calib_scales_ptr, stream=0):
    """Calibrate FP8 scales for Pi0-FAST prefill. ALL 18 layers, mixed mode.

    Mixed calibration: each layer measures amax in FP16, then runs FP8 with
    that scale. The FP8 output feeds into next layer, matching inference behavior.
    Same approach as encoder_forward_calibrate but for ALL 18 layers.

    SM dispatch:
      SM100/SM110: CUTLASS FP8 GEMM with alpha = act_scale * w_scale (host).
      SM120:       cuBLASLt fp8_gemm_descale_{fp16,bf16out} with device
                   (act_scale_ptr, w_scale_ptr). Since calibration measures
                   act_scale into a device buffer already (cs_qkv / cs_o /
                   cs_gu / cs_down), the SM120 path just hands that device
                   pointer to cuBLASLt — strictly fewer D2H syncs.
    """
    Se = dims['Se']
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    L = dims['L']
    total_keys_max = dims['total_keys_max']
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))

    x = bufs['x']
    x_fp8 = bufs['x_fp8']
    qkv = bufs['qkv']
    logits = bufs['logits']
    attn_out = bufs['attn_out']
    o_fp8 = bufs['o_fp8']
    gate = bufs['gate']
    hidden = bufs['hidden']
    hid_fp8 = bufs['hid_fp8']
    fg = bufs['fg']

    w_scales_dev = weights['w_scales']
    ws_host = _d2h_floats(w_scales_dev, L * 4)

    # SM dispatch — probe once
    _use_sm100 = hasattr(fvk_mod, 'cutlass_fp8_sq')
    if not _use_sm100:
        qkv_w_flat  = weights['qkv_w_flat']
        o_w_flat    = weights['o_w_flat']
        gate_w_flat = weights['gate_w_flat']
        down_w_flat = weights['down_w_flat']

    norm_scratch = bufs['norm_scratch']
    x_scratch = bufs['x_scratch']
    calib_buf = bufs['calib_buf']
    d_scale = bufs['d_scale']
    fp8_scratch = bufs['fp8_scratch']
    ones_buf = bufs['ones']
    _gpu_zero(calib_buf, L * 4 * 4, stream)

    for l in range(L):
        # 1. RMSNorm(BF16) → FP8 (measure via norm_scratch — BF16 buffer!)
        fvk_mod.rms_norm(x, ones_buf, norm_scratch, Se, D, 1e-6, stream)  # BF16 rms_norm
        _measure_scale_gpu_bf16(fvk_mod, norm_scratch, Se * D, d_scale, fp8_scratch, stream)
        _gpu_sync(stream)
        cs_qkv = calib_buf + (l * 4 + 0) * 4
        ws_qkv = w_scales_dev + (l * 4 + 0) * 4
        _gpu_copy(cs_qkv, d_scale, 4, stream)
        fvk_mod.rms_norm_fp8_noweight_bf16(x, x_fp8, Se, D, cs_qkv, stream)

        # 2. QKV GEMM (FP8, with measured scale) — output FP16 (safe)
        if _use_sm100:
            as_qkv = _d2h_float(d_scale)
            alpha_qkv = float(np.float32(as_qkv) * np.float32(ws_host[l * 4 + 0]))
            fvk_mod.cutlass_fp8_sq(x_fp8, weights['qkv_w'][l], qkv,
                                    Se, 2560, D, alpha_qkv, 0.0, stream)
        else:
            fvk_mod.fp8_gemm_descale_fp16(x_fp8, qkv_w_flat + l * D * 2560, qkv,
                                           Se, 2560, D,
                                           cs_qkv, ws_qkv, stream)

        # 3. Split + RoPE + KV cache
        kv_off = l * total_keys_max * HD
        fvk_mod.qkv_split_rope_kvcache_fp16(qkv, weights['rope'], attn_out,
                                             weights['Kc'], weights['Vc'],
                                             Se, Q_dim, K_dim, HD, 2560,
                                             kv_off, HD, stream)

        # 4. Attention (FP16)
        K_ptr = weights['Kc'] + kv_off * 2
        V_ptr = weights['Vc'] + kv_off * 2
        fvk_mod.attention_qkv_fp16(bufs['ctx'], attn_out, K_ptr, V_ptr,
                                    logits, attn_out,
                                    Se, Se, NH, HD, attn_scale, stream)

        # 5. O proj: measure → FP8 → GEMM with BF16 output
        _measure_scale_gpu(fvk_mod, attn_out, Se * Q_dim, d_scale, fp8_scratch, stream)
        _gpu_sync(stream)
        cs_o = calib_buf + (l * 4 + 1) * 4
        ws_o = w_scales_dev + (l * 4 + 1) * 4
        _gpu_copy(cs_o, d_scale, 4, stream)
        fvk_mod.quantize_fp8_static_fp16(attn_out, o_fp8, cs_o, Se * D, stream)
        if _use_sm100:
            as_o = _d2h_float(d_scale)
            alpha_o = float(np.float32(as_o) * np.float32(ws_host[l * 4 + 1]))
            fvk_mod.cutlass_fp8_sq_bf16out(o_fp8, weights['o_w'][l], fg,
                                            Se, D, D, alpha_o, 0.0, stream)
        else:
            fvk_mod.fp8_gemm_descale_bf16out(o_fp8, o_w_flat + l * D * D, fg,
                                              Se, D, D,
                                              cs_o, ws_o, stream)

        # 6. Residual(BF16) + RMSNorm → measure → FP8 (norm_scratch is BF16!)
        _gpu_copy(x_scratch, x, Se * D * 2, stream)  # 2 bytes for BF16
        fvk_mod.residual_add(x_scratch, fg, Se * D, stream)  # BF16 residual_add
        fvk_mod.rms_norm(x_scratch, ones_buf, norm_scratch, Se, D, 1e-6, stream)  # BF16 rms_norm
        _measure_scale_gpu_bf16(fvk_mod, norm_scratch, Se * D, d_scale, fp8_scratch, stream)
        _gpu_sync(stream)
        cs_gu = calib_buf + (l * 4 + 2) * 4
        ws_gu = w_scales_dev + (l * 4 + 2) * 4
        _gpu_copy(cs_gu, d_scale, 4, stream)
        fvk_mod.residual_add_rms_norm_fp8_noweight_bf16(x, fg, x_fp8,
                                                          Se, D, cs_gu, stream)

        # 7. Gate+Up GEMM (FP8)
        if _use_sm100:
            as_gu = _d2h_float(d_scale)
            alpha_gu = float(np.float32(as_gu) * np.float32(ws_host[l * 4 + 2]))
            fvk_mod.cutlass_fp8_t1(x_fp8, weights['gate_w'][l], gate,
                                    Se, H * 2, D, alpha_gu, 0.0, stream)
        else:
            fvk_mod.fp8_gemm_descale_fp16(x_fp8, gate_w_flat + l * D * H * 2, gate,
                                           Se, H * 2, D,
                                           cs_gu, ws_gu, stream)

        # 8. GELU*up → measure amax in FP32 → FP8
        # gate_geglu_merged_fp16 can overflow FP16 in mixed calibration
        # (FP8 error accumulation causes larger activations at deep layers).
        # Measure amax in FP32 via PyTorch to avoid FP16 overflow.
        import torch as _torch
        _gate_t = _torch.empty(Se, H * 2, dtype=_torch.float16, device='cuda')
        _crt.cudaMemcpyAsync(ctypes.c_void_p(_gate_t.data_ptr()),
                             ctypes.c_void_p(gate), ctypes.c_size_t(Se * H * 2 * 2),
                             3, ctypes.c_void_p(stream))
        _gpu_sync(stream)
        _g = _gate_t[:, :H].float()
        _u = _gate_t[:, H:].float()
        _gelu = _g / (1.0 + _torch.exp(-1.5957691216057308 * _g * (1.0 + 0.044715 * _g * _g)))
        _down_scale = max(float((_gelu * _u).abs().max().item()) / 448.0, 1e-12)
        _scale_dev = _torch.tensor([_down_scale], dtype=_torch.float32, device='cuda')
        cs_down = calib_buf + (l * 4 + 3) * 4
        ws_d = w_scales_dev + (l * 4 + 3) * 4
        _crt.cudaMemcpyAsync(ctypes.c_void_p(cs_down),
                             ctypes.c_void_p(_scale_dev.data_ptr()),
                             ctypes.c_size_t(4), 3, ctypes.c_void_p(stream))
        _gpu_sync(stream)
        fvk_mod.gate_geglu_merged_fp8_fp16(gate, hid_fp8, Se, H, cs_down, stream)

        # 9. Down GEMM → BF16 output
        if _use_sm100:
            as_down = _d2h_float(cs_down)
            alpha_down = float(np.float32(as_down) * np.float32(ws_host[l * 4 + 3]))
            fvk_mod.cutlass_fp8_wide_bf16out(hid_fp8, weights['down_w'][l], fg,
                                              Se, D, H, alpha_down, 0.0, stream)
        else:
            fvk_mod.fp8_gemm_descale_bf16out(hid_fp8, down_w_flat + l * H * D, fg,
                                              Se, D, H,
                                              cs_down, ws_d, stream)

        # 10. Residual (BF16)
        fvk_mod.residual_add(x, fg, Se * D, stream)

    _gpu_copy(calib_scales_ptr, calib_buf, L * 4 * 4, stream)
    _gpu_sync(stream)


# ====================================================================
# SigLIP Vision Encoder -- SM120 (RTX 5090) variant
# ====================================================================

def siglip_forward_sm120(gemm, fvk, bufs, weights, dims, stream=0):
    """SigLIP 27-layer vision encoder, SM120 decomposed FP8 path.

    Numerically equivalent to :func:`pipeline_pi05.siglip_forward` but
    replaces the SM100-era fused FP8 epilogue kernels
    (``gemm.fp8_nn_bias`` / ``fp8_nn_bias_res`` / ``fp8_nn_gelu_bias``,
    all FP8 -> FP16 with epilogue) which are rejected by cuBLASLt on
    SM120 with ``CUBLAS_STATUS_NOT_SUPPORTED`` (error 15). The
    decomposed path uses ``fp8_gemm_descale_fp16`` (always available)
    plus explicit ``add_bias_fp16`` / ``bias_residual_fp16`` /
    ``gelu_inplace_fp16`` kernels, keeping the buffer layout identical
    (FP16 residual stream, matches thor siglip_forward).

    Attention: ``fmha_strided_full`` delegates to a separately-built
    ``libfmha_fp16_strided.so`` whose kernels are SM100-only (Blackwell
    TMA warpspecialized). On SM120 consumer silicon we call
    ``torch.nn.functional.scaled_dot_product_attention`` instead — it
    picks the Flash backend automatically and is CUDA-graph-capturable
    via PyTorch's stream-ordered caching allocator. For this reason the
    frontend must pass persistent torch tensor handles for the qkv /
    attn buffers via ``bufs['qkv_t']`` and ``bufs['attn_t']``.

    Note: the FP8 input for every GEMM comes from ``layer_norm_fp8`` /
    ``quantize_fp8_static_fp16`` with ``unit_scale`` (= 1.0), so the
    effective act_scale passed to cuBLASLt is unit_scale and the only
    per-layer descale factor is the per-weight scale ``w_scales[l*4+k]``.

    Buffer reuse: the FP16 ``attn_out`` buffer ([S, D]) is reused as the
    post-GEMM scratch for O proj and Down proj, since by the time those
    GEMMs run its earlier role (FMHA output, quantized attention source)
    is done. This keeps the SigLIP memory footprint identical to the
    SM100 path.

    Args:
        gemm: GemmRunner (unused on SM120 path — kept for interface
              parity with siglip_forward).
        fvk: flash_vla_kernels module.
        bufs: dict — x, x_fp8, qkv, attn_out, hidden, hid_fp8 (FP16
              raw pointers, same contract as pipeline_pi05.siglip_forward);
              plus SM120-only keys ``qkv_t`` (torch.Tensor view of qkv
              buffer, shape [S, 3*D], fp16, cuda) and ``attn_t``
              (torch.Tensor view of attn_out buffer, shape [S, D], fp16,
              cuda) — both must be persistent (no alloc per call).
        weights: dict — ln_attn_w[L], ln_attn_b[L], qkv_w[L], qkv_b[L],
              o_w[L], o_b[L], ln_ffn_w[L], ln_ffn_b[L], up_w[L], up_b[L],
              down_w[L], down_b[L], unit_scale (device float32 = 1.0),
              w_scales_dev (device float32, len = L*4, layer-major
              ordering: qkv, o, up, down).
        dims: dict — S, D, H, NH, HD, L, num_views, seq_per_view.
    """
    import torch as _torch
    import torch.nn.functional as _F
    S = dims['S']
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    L = dims['L']
    nv = dims['num_views']
    spv = dims['seq_per_view']  # 256

    x = bufs['x']
    x_fp8 = bufs['x_fp8']
    qkv = bufs['qkv']
    attn_out = bufs['attn_out']
    hidden = bufs['hidden']
    hid_fp8 = bufs['hid_fp8']

    # SM120-only: persistent torch tensor handles for the attention step
    # (F.scaled_dot_product_attention requires Tensor input). Both views
    # alias the same device memory as the qkv / attn_out raw pointers.
    qkv_t = bufs['qkv_t']        # [S, 3*D], fp16, cuda
    attn_t = bufs['attn_t']      # [S, D],   fp16, cuda

    unit_scale = weights['unit_scale']   # device float32 ptr, value 1.0
    w_scales_dev = weights['w_scales_dev']

    sdpa_scale = 1.0 / math.sqrt(float(HD))

    for l in range(L):
        ws_qkv  = w_scales_dev + (l * 4 + 0) * 4
        ws_o    = w_scales_dev + (l * 4 + 1) * 4
        ws_up   = w_scales_dev + (l * 4 + 2) * 4
        ws_down = w_scales_dev + (l * 4 + 3) * 4

        # 1. Attention LayerNorm -> FP8 (unit_scale implicit in kernel)
        fvk.layer_norm_fp8(x, x_fp8,
                            weights['ln_attn_w'][l], weights['ln_attn_b'][l],
                            S, D, 1e-6, stream)

        # 2. QKV GEMM (FP8 -> FP16, no epilogue) then separate bias add
        fvk.fp8_gemm_descale_fp16(x_fp8, weights['qkv_w'][l], qkv,
                                    S, 3 * D, D,
                                    unit_scale, ws_qkv, stream)
        fvk.add_bias_fp16(qkv, weights['qkv_b'][l], S, 3 * D, stream)

        # 3. Per-view self-attention via torch SDPA.
        # qkv_t layout: [S=nv*spv, 3*D] where the trailing 3*D is the
        # concatenation [Q_all_heads | K_all_heads | V_all_heads] — same
        # as fp8_nn_bias output layout. Reshape to [nv, spv, 3, NH, HD]
        # to isolate view batch and split Q/K/V, then transpose to
        # [nv, NH, spv, HD] (canonical SDPA layout) and call the fused
        # SDPA kernel (Flash backend auto-selected by torch on SM120).
        qkv_r = qkv_t.view(nv, spv, 3, NH, HD)
        q_ = qkv_r[:, :, 0].transpose(1, 2)  # [nv, NH, spv, HD]
        k_ = qkv_r[:, :, 1].transpose(1, 2)
        v_ = qkv_r[:, :, 2].transpose(1, 2)
        out_ = _F.scaled_dot_product_attention(
            q_, k_, v_, is_causal=False, scale=sdpa_scale)  # [nv, NH, spv, HD]
        # Back to [nv, spv, NH, HD] -> [S, NH*HD] via .reshape
        attn_t.view(nv, spv, NH, HD).copy_(out_.transpose(1, 2))

        # 4. Quantize attention output -> FP8 (scale = 1.0)
        fvk.quantize_fp8_static_fp16(attn_out, x_fp8, unit_scale,
                                       S * D, stream)

        # 5. O proj: FP8 GEMM -> FP16 scratch (reuse attn_out) + bias_res
        fvk.fp8_gemm_descale_fp16(x_fp8, weights['o_w'][l], attn_out,
                                    S, D, D,
                                    unit_scale, ws_o, stream)
        # x += attn_out + o_b
        fvk.bias_residual_fp16(x, attn_out, weights['o_b'][l],
                                S, D, stream)

        # 6. FFN LayerNorm -> FP8
        fvk.layer_norm_fp8(x, x_fp8,
                            weights['ln_ffn_w'][l], weights['ln_ffn_b'][l],
                            S, D, 1e-6, stream)

        # 7. Up GEMM -> FP16 hidden + bias + GELU
        fvk.fp8_gemm_descale_fp16(x_fp8, weights['up_w'][l], hidden,
                                    S, H, D,
                                    unit_scale, ws_up, stream)
        fvk.add_bias_fp16(hidden, weights['up_b'][l], S, H, stream)
        fvk.gelu_inplace_fp16(hidden, S * H, stream)

        # 8. Quantize FFN hidden -> FP8 (unit scale)
        fvk.quantize_fp8_static_fp16(hidden, hid_fp8, unit_scale,
                                       S * H, stream)

        # 9. Down GEMM -> FP16 scratch (reuse attn_out) + bias_res
        fvk.fp8_gemm_descale_fp16(hid_fp8, weights['down_w'][l], attn_out,
                                    S, D, H,
                                    unit_scale, ws_down, stream)
        fvk.bias_residual_fp16(x, attn_out, weights['down_b'][l],
                                S, D, stream)

    # x[S, D] now contains final SigLIP output (matches siglip_forward)

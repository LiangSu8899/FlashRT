"""FlashVLA -- Thor SM110 Pipeline for Pi0.

Pi0 decoder forward pass and calibration.
Adapted from Pi0.5 decoder (pipeline_pi05.py) with key differences:
  - Standard RMSNorm instead of AdaRMSNorm
  - action_time_mlp + state_proj instead of time_mlp + AdaRMS conditioning
  - Decoder sequence length S_dec = Sa + 1 (1 state + Sa actions, no padding)

Functions:
    decoder_forward_pi0        -- Pi0 decoder inference (static FP8)
    decoder_forward_calibrate_pi0 -- Pi0 decoder FP8 scale calibration
"""

import math
import ctypes
import numpy as np

from flash_rt.hardware.thor.shared_primitives import (
    _gpu_copy, _gpu_zero, _gpu_sync,
    _d2h_float, _d2h_floats,
    _measure_scale_gpu,
    _gpu_alloc, _gpu_free,
)

_crt = ctypes.CDLL('libcudart.so')


# ==================================================================
# Pi0 Decoder (18 layers, 10 diffusion steps, static FP8)
# ==================================================================

def decoder_forward_pi0(ctx, fvk, bufs, weights, dims, stream=0, *, attn=None):
    """Full Pi0 decoder forward pass with static FP8.

    Uses standard RMSNorm (no AdaRMSNorm), action_time_mlp for noise
    conditioning, and encoder-style per-layer quantization.

    Args:
        ctx: FvkContext (C++ object with cuBLAS handle)
        fvk: flash_rt_kernels module
        bufs: dict of GPU buffer pointers (uintptr_t)
            noise, x, xn, temp, action_buf, gate, qkv, logits,
            attn_out, hid, fg, xn_fp8, hid_fp8, ctx_fp8, state_token
        weights: dict of GPU buffer pointers
            ain_w, ain_b, wa_w, atmlp_out_w, atmlp_out_b, time_proj_all,
            qw (flat), Kc, Vc, ow (flat), gw (flat), dw (flat),
            aow, aob, final_norm_w, rope, w_scales, act_scales
        dims: dict
            Sa, S_dec, D, H, NH, HD, steps, layers, enc_seq, total_keys
        stream: CUDA stream (int)
    """
    Sa = dims['Sa']
    S_dec = dims['S_dec']
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    steps = dims['steps']
    layers = dims['layers']
    enc_seq = dims['enc_seq']
    total_keys = dims['total_keys']
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))

    # Buffer pointers
    noise = bufs['noise']
    x = bufs['x']
    xn = bufs['xn']
    temp = bufs['temp']
    action_buf = bufs['action_buf']
    gate = bufs['gate']
    qkv = bufs['qkv']
    logits = bufs['logits']
    attn_out = bufs['attn_out']
    fg = bufs['fg']
    xn_fp8 = bufs['xn_fp8']
    hid_fp8 = bufs['hid_fp8']
    ctx_fp8 = bufs['ctx_fp8']
    state_token = bufs['state_token']

    # Weight pointers
    ain_w = weights['ain_w']
    ain_b = weights['ain_b']
    wa_w = weights['wa_w']
    atmlp_out_w = weights['atmlp_out_w']
    atmlp_out_b = weights['atmlp_out_b']
    time_proj_all = weights['time_proj_all']
    qw = weights['qw']
    Kc = weights['Kc']
    Vc = weights['Vc']
    ow = weights['ow']
    gw = weights['gw']
    dw = weights['dw']
    aow = weights['aow']
    aob = weights['aob']
    final_norm_w = weights['final_norm_w']
    rope = weights['rope']
    w_scales = weights['w_scales']
    act_scales = weights['act_scales']

    for s in range(steps):
        # ── Step preprocessing: action_time_mlp + assemble x ──

        # Copy state_token -> x[0] (constant across steps)
        _crt.cudaMemcpyAsync(ctypes.c_void_p(x),
                             ctypes.c_void_p(state_token),
                             ctypes.c_size_t(1 * D * 2), 3,
                             ctypes.c_void_p(stream))

        # action_in_proj(noise) -> x[1:S_dec] directly
        x_action = x + 1 * D * 2
        fvk.gmm_fp16(ctx, noise, ain_w, x_action, Sa, D, 32, 0.0, stream)
        fvk.add_bias_fp16(x_action, ain_b, Sa, D, stream)

        # action_time_mlp (FP16 GEMMs — small [10,1024]×[1024,1024])
        fvk.gmm_fp16(ctx, x_action, wa_w, temp, Sa, D, D, 0.0, stream)
        time_proj_ptr = time_proj_all + s * Sa * D * 2
        fvk.fused_add_silu_fp16(temp, time_proj_ptr, Sa * D, stream)
        fvk.gmm_fp16(ctx, temp, atmlp_out_w, x_action, Sa, D, D, 0.0, stream)
        fvk.add_bias_fp16(x_action, atmlp_out_b, Sa, D, stream)

        # ── Per-layer (encoder pattern with cross-attention) ──
        for l in range(layers):
            act_scale_qkv = act_scales + (l * 4 + 0) * 4
            w_scale_qkv = w_scales + (l * 4 + 0) * 4

            # a. RMSNorm(x) -> FP8
            fvk.rms_norm_fp8_noweight_fp16(x, xn_fp8, S_dec, D,
                                           act_scale_qkv, stream)

            # b. QKV GEMM
            qw_ptr = qw + l * D * 2560
            fvk.fp8_gemm_descale_fp16(xn_fp8, qw_ptr, qkv, S_dec, 2560, D,
                                      act_scale_qkv, w_scale_qkv, stream)

            # c. RoPE + KV cache
            kv_offset = l * total_keys * HD + enc_seq * HD
            fvk.qkv_split_rope_kvcache_fp16(qkv, rope, attn_out, Kc, Vc,
                                            S_dec, Q_dim, K_dim, HD, 2560,
                                            kv_offset, HD, stream)

            # d. Cross-attention (single call with state masking)
            # State token sees enc_seq+1 keys; action tokens see all total_keys.
            # Handled by state_masked kernel: mask state rows [enc_seq+1:] to -inf.
            state_nk = enc_seq + 1
            if attn is not None:
                attn.run("decoder", l, q_seq=S_dec, kv_seq=total_keys,
                         stream=stream, state_nk=state_nk)
            else:
                K_ptr = Kc + l * total_keys * HD * 2
                V_ptr = Vc + l * total_keys * HD * 2
                fvk.attention_qkv_fp16_state_masked(ctx, attn_out, K_ptr, V_ptr,
                                                    logits, attn_out,
                                                    S_dec, total_keys, NH, HD,
                                                    state_nk, attn_scale, stream)

            # e. Quantize + O proj
            act_scale_o = act_scales + (l * 4 + 1) * 4
            w_scale_o = w_scales + (l * 4 + 1) * 4
            fvk.quantize_fp8_static_fp16(attn_out, ctx_fp8, act_scale_o,
                                         S_dec * NH * HD, stream)
            ow_ptr = ow + l * NH * HD * D
            fvk.fp8_gemm_descale_fp16(ctx_fp8, ow_ptr, fg, S_dec, D, NH * HD,
                                      act_scale_o, w_scale_o, stream)

            # f. Residual + RMSNorm -> FP8
            act_scale_gu = act_scales + (l * 4 + 2) * 4
            fvk.residual_add_rms_norm_fp8_noweight_fp16(x, fg, xn_fp8,
                                                        S_dec, D,
                                                        act_scale_gu, stream)

            # g. Gate+Up GEMM
            w_scale_gu = w_scales + (l * 4 + 2) * 4
            gw_ptr = gw + l * D * H * 2
            fvk.fp8_gemm_descale_fp16(xn_fp8, gw_ptr, fg, S_dec, H * 2, D,
                                      act_scale_gu, w_scale_gu, stream)

            # h. GELU -> FP8
            act_scale_down = act_scales + (l * 4 + 3) * 4
            fvk.gate_geglu_merged_fp8_fp16(fg, hid_fp8, S_dec, H,
                                              act_scale_down, stream)

            # i. Down GEMM
            w_scale_down = w_scales + (l * 4 + 3) * 4
            dw_ptr = dw + l * H * D
            fvk.fp8_gemm_descale_fp16(hid_fp8, dw_ptr, fg, S_dec, D, H,
                                      act_scale_down, w_scale_down, stream)

            # j. Residual (+ RMSNorm for next layer, or plain residual for last)
            if l < layers - 1:
                act_scale_next = act_scales + ((l + 1) * 4 + 0) * 4
                fvk.residual_add_rms_norm_fp8_noweight_fp16(
                    x, fg, xn_fp8, S_dec, D, act_scale_next, stream)
            else:
                fvk.residual_add_fp16(x, fg, S_dec * D, stream)

        # ── Post-processing ──

        # 9. Final RMSNorm with weight
        fvk.rms_norm_fp16(x, final_norm_w, xn, S_dec, D, 1e-6, stream)

        # 10. Action output (skip state token, only tokens [1:Sa+1])
        xn_action = xn + 1 * D * 2  # byte offset to skip row 0
        fvk.gmm_fp16(ctx, xn_action, aow, noise, Sa, 32, D, 1.0, stream)
        fvk.add_bias_fp16(noise, aob, Sa, 32, stream)


# ==================================================================
# Pi0 Decoder Calibration
# ==================================================================

def decoder_forward_calibrate_pi0(ctx, fvk_mod, bufs, weights, dims,
                                  calib_scales_ptr, stream=0):
    """Calibrate Pi0 decoder FP8 scales. Framework-agnostic (pure pointers).

    Follows encoder calibration pattern:
      - For each quantization point: FP16 kernel -> measure amax on GPU
        -> compute scale -> FP8 kernel with that scale
      - Uses rms_norm_fp16 + _measure_scale_gpu -> rms_norm_fp8_noweight_fp16
      - Uses residual_add_fp16 + rms_norm_fp16 + _measure_scale_gpu
        -> residual_add_rms_norm_fp8_noweight_fp16

    Args:
        ctx: FvkContext (C++ object with cuBLAS handle)
        fvk_mod: flash_rt_kernels module
        bufs: dict of GPU buffer pointers (uintptr_t)
            noise, x, xn, temp, action_buf, gate, qkv, logits, attn_out,
            hid, fg, xn_fp8, hid_fp8, ctx_fp8, state_token,
            calib_buf, d_scale, hidden_scratch, fp8_scratch, ones
        weights: dict of GPU buffer pointers
            ain_w, ain_b, wa_w, atmlp_out_w, atmlp_out_b, time_proj_all,
            qw (flat), Kc, Vc, ow (flat), gw (flat), dw (flat),
            aow, aob, final_norm_w, rope, w_scales
        dims: dict
            Sa, S_dec, D, H, NH, HD, steps, layers, enc_seq, total_keys
        calib_scales_ptr: output device pointer for calibrated scales
        stream: CUDA stream (int)
    """
    Sa = dims['Sa']
    S_dec = dims['S_dec']
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    steps = dims['steps']
    layers = dims['layers']
    enc_seq = dims['enc_seq']
    total_keys = dims['total_keys']
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))

    # Buffer pointers
    noise = bufs['noise']
    x = bufs['x']
    xn = bufs['xn']
    temp = bufs['temp']
    action_buf = bufs['action_buf']
    gate = bufs['gate']
    qkv = bufs['qkv']
    logits = bufs['logits']
    attn_out = bufs['attn_out']
    fg = bufs['fg']
    xn_fp8 = bufs['xn_fp8']
    hid_fp8 = bufs['hid_fp8']
    ctx_fp8 = bufs['ctx_fp8']
    state_token = bufs['state_token']

    # Calibration scratch buffers
    calib_buf = bufs['calib_buf']           # layers*4 float32
    d_scale = bufs['d_scale']               # 1 float32
    hidden_scratch = bufs['hidden_scratch'] # S_dec*H fp16
    fp8_scratch = bufs['fp8_scratch']       # S_dec*max(D,H) fp8
    ones_buf = bufs['ones']                 # D fp16 (all 1.0)
    norm_scratch = bufs['norm_scratch']  # S_dec*D fp16
    x_scratch = bufs['x_scratch']       # S_dec*D fp16

    # Weight pointers
    ain_w = weights['ain_w']
    ain_b = weights['ain_b']
    wa_w = weights['wa_w']                           # FP8 weight
    atmlp_out_w = weights['atmlp_out_w']             # FP8 weight
    atmlp_out_b = weights['atmlp_out_b']
    atmlp_out_w = weights['atmlp_out_w']
    atmlp_out_b = weights['atmlp_out_b']
    time_proj_all = weights['time_proj_all']
    qw = weights['qw']
    Kc = weights['Kc']
    Vc = weights['Vc']
    ow = weights['ow']
    gw = weights['gw']
    dw = weights['dw']
    aow = weights['aow']
    aob = weights['aob']
    final_norm_w = weights['final_norm_w']
    rope = weights['rope']
    w_scales = weights['w_scales']

    # Read w_scales to host
    ws_host = _d2h_floats(w_scales, layers * 4)

    _gpu_zero(calib_buf, layers * 4 * 4, stream)

    for s in range(steps):
        # ── Step preprocessing: action_time_mlp (identical to inference) ──

        # action_in_proj + action_time_mlp (FP16) + assemble x
        _crt.cudaMemcpyAsync(ctypes.c_void_p(x),
                             ctypes.c_void_p(state_token),
                             ctypes.c_size_t(1 * D * 2), 3,
                             ctypes.c_void_p(stream))
        x_action = x + 1 * D * 2
        fvk_mod.gmm_fp16(ctx, noise, ain_w, x_action, Sa, D, 32, 0.0, stream)
        fvk_mod.add_bias_fp16(x_action, ain_b, Sa, D, stream)
        fvk_mod.gmm_fp16(ctx, x_action, wa_w, temp, Sa, D, D, 0.0, stream)
        time_proj_ptr = time_proj_all + s * Sa * D * 2
        fvk_mod.fused_add_silu_fp16(temp, time_proj_ptr, Sa * D, stream)
        fvk_mod.gmm_fp16(ctx, temp, atmlp_out_w, x_action, Sa, D, D, 0.0, stream)
        fvk_mod.add_bias_fp16(x_action, atmlp_out_b, Sa, D, stream)

        # ── Per-layer calibration (encoder pattern) ──
        for l in range(layers):

            # a. RMSNorm FP16 -> measure amax -> scale
            fvk_mod.rms_norm_fp16(x, ones_buf, norm_scratch, S_dec, D,
                                  1e-6, stream)
            _gpu_sync(stream)
            _measure_scale_gpu(fvk_mod, norm_scratch, S_dec * D,
                               d_scale, fp8_scratch, stream)
            _gpu_sync(stream)
            as_qkv = _d2h_float(d_scale)
            cs_qkv = calib_buf + (l * 4 + 0) * 4
            _gpu_copy(cs_qkv, d_scale, 4, stream)

            # RMSNorm -> FP8 with calibrated scale
            fvk_mod.rms_norm_fp8_noweight_fp16(x, xn_fp8, S_dec, D,
                                               cs_qkv, stream)

            # b. QKV GEMM (with descale)
            ws_qkv = w_scales + (l * 4 + 0) * 4
            qw_ptr = qw + l * D * 2560
            fvk_mod.fp8_gemm_descale_fp16(xn_fp8, qw_ptr, qkv, S_dec, 2560, D,
                                          cs_qkv, ws_qkv, stream)

            # c. RoPE + KV cache
            kv_offset = l * total_keys * HD + enc_seq * HD
            fvk_mod.qkv_split_rope_kvcache_fp16(qkv, rope, attn_out, Kc, Vc,
                                                S_dec, Q_dim, K_dim, HD, 2560,
                                                kv_offset, HD, stream)

            # d. Attention (single call with state masking)
            K_ptr = Kc + l * total_keys * HD * 2
            V_ptr = Vc + l * total_keys * HD * 2

            state_nk = enc_seq + 1
            fvk_mod.attention_qkv_fp16_state_masked(ctx, attn_out, K_ptr, V_ptr,
                                                    logits, attn_out,
                                                    S_dec, total_keys, NH, HD,
                                                    state_nk, attn_scale, stream)

            # e. O proj: measure attn amax -> FP8 -> GEMM
            _measure_scale_gpu(fvk_mod, attn_out, S_dec * NH * HD,
                               d_scale, fp8_scratch, stream)
            _gpu_sync(stream)
            as_o = _d2h_float(d_scale)
            cs_o = calib_buf + (l * 4 + 1) * 4
            _gpu_copy(cs_o, d_scale, 4, stream)
            ws_o = w_scales + (l * 4 + 1) * 4
            fvk_mod.quantize_fp8_static_fp16(attn_out, ctx_fp8, cs_o,
                                             S_dec * NH * HD, stream)
            ow_ptr = ow + l * NH * HD * D
            fvk_mod.fp8_gemm_descale_fp16(ctx_fp8, ow_ptr, fg, S_dec, D,
                                          NH * HD, cs_o, ws_o, stream)

            # f. Residual + RMSNorm -> FP8
            #    Two-pass: residual_add FP16 + rms_norm FP16 -> measure -> fused FP8
            _gpu_copy(x_scratch, x, S_dec * D * 2, stream)
            fvk_mod.residual_add_fp16(x_scratch, fg, S_dec * D, stream)
            fvk_mod.rms_norm_fp16(x_scratch, ones_buf, norm_scratch,
                                  S_dec, D, 1e-6, stream)
            _measure_scale_gpu(fvk_mod, norm_scratch, S_dec * D,
                               d_scale, fp8_scratch, stream)
            _gpu_sync(stream)
            as_gu = _d2h_float(d_scale)
            cs_gu = calib_buf + (l * 4 + 2) * 4
            _gpu_copy(cs_gu, d_scale, 4, stream)
            fvk_mod.residual_add_rms_norm_fp8_noweight_fp16(
                x, fg, xn_fp8, S_dec, D, cs_gu, stream)

            # g. Gate+Up GEMM
            ws_gu = w_scales + (l * 4 + 2) * 4
            gw_ptr = gw + l * D * H * 2
            fvk_mod.fp8_gemm_descale_fp16(xn_fp8, gw_ptr, fg, S_dec, H * 2, D,
                                          cs_gu, ws_gu, stream)

            # h. GELU FP16 -> measure -> FP8
            fvk_mod.gate_geglu_merged_fp16(fg, hidden_scratch, S_dec, H,
                                              stream)
            _measure_scale_gpu(fvk_mod, hidden_scratch, S_dec * H,
                               d_scale, fp8_scratch, stream)
            _gpu_sync(stream)
            as_down = _d2h_float(d_scale)
            cs_down = calib_buf + (l * 4 + 3) * 4
            _gpu_copy(cs_down, d_scale, 4, stream)
            fvk_mod.gate_geglu_merged_fp8_fp16(fg, hid_fp8, S_dec, H,
                                                  cs_down, stream)

            # i. Down GEMM
            ws_down = w_scales + (l * 4 + 3) * 4
            dw_ptr = dw + l * H * D
            fvk_mod.fp8_gemm_descale_fp16(hid_fp8, dw_ptr, fg, S_dec, D, H,
                                          cs_down, ws_down, stream)

            # j. Residual (+ prep next layer)
            if l < layers - 1:
                # Residual + measure for next layer's first RMSNorm
                fvk_mod.residual_add_fp16(x, fg, S_dec * D, stream)
            else:
                fvk_mod.residual_add_fp16(x, fg, S_dec * D, stream)

        # ── Post-processing (FP16, no calibration needed) ──

        # Final RMSNorm with weight
        fvk_mod.rms_norm_fp16(x, final_norm_w, xn, S_dec, D, 1e-6, stream)

        # Action output (skip state token, only tokens [1:Sa+1])
        xn_action = xn + 1 * D * 2
        fvk_mod.gmm_fp16(ctx, xn_action, aow, noise, Sa, 32, D, 1.0, stream)
        fvk_mod.add_bias_fp16(noise, aob, Sa, 32, stream)

    # Copy calibrated scales to output
    _gpu_copy(calib_scales_ptr, calib_buf, layers * 4 * 4, stream)
    _gpu_sync(stream)

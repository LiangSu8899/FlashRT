"""FlashVLA new-model template — pipeline (model-specific compute).

Copy this file to `flash_vla/models/<mymodel>/pipeline_<hw>.py`
(`<hw>` = `thor` or `rtx`). This is **the bulk of your hand-written
work** — it's where you translate your model's `forward()` into a
sequence of `fvk.*` kernel calls.

# WHAT YOU TRANSLATE
=====================

This is the most concrete mapping in the template. For every line
in your reference PyTorch model's forward, write the equivalent
kernel call here.

Your reference PyTorch (one encoder layer):
    x_norm = F.rms_norm(x, weight=norm_w, eps=1e-6)
    qkv = F.linear(x_norm, qkv_weight)
    q, k, v = qkv.chunk(3, dim=-1)
    q, k = apply_rope(q, k, rope_freqs)
    attn = F.scaled_dot_product_attention(q, k, v)
    h = F.linear(attn, out_proj_weight)
    x = x + h
    x_norm = F.rms_norm(x, weight=post_norm_w, eps=1e-6)
    gate, up = F.linear(x_norm, gate_up_weight).chunk(2, dim=-1)
    ffn_out = F.linear(F.silu(gate) * up, down_weight)
    x = x + ffn_out

Equivalent FlashVLA kernel sequence:
    fvk.rms_norm_fp16(x, norm_w, x_norm, S, D, eps, stream)        # ← norm
    fvk.quantize_fp8_static(x_norm, x_norm_fp8, ...)               # ← quantize before fp8 GEMM
    fvk.gemm_fp8_fp16(x_norm_fp8, qkv_w_fp8, qkv,                  # ← QKV proj
                      M=S, N=3*D, K=D, alpha=alpha, ...)
    fvk.qkv_split_rope_kvcache_fp16(qkv, q, Kc, Vc, rope_table,    # ← split + RoPE + KV write
                                    S, NH, NUM_KV_HEADS, HD, layer_idx, stream)
    attn.run("encoder", layer_idx, q_seq=S, stream=stream)         # ← attention dispatch
    fvk.gemm_fp8_fp16(attn_out_fp8, out_proj_w_fp8, h, ...)        # ← out proj
    fvk.add_inplace_fp16(x, h, S * D, stream)                      # ← residual
    fvk.rms_norm_fp16(x, post_norm_w, x_norm, S, D, eps, stream)
    fvk.quantize_fp8_static(...)
    fvk.gemm_fp8_fp16(x_norm_fp8, gate_up_w_fp8, gate_up, ...)
    fvk.silu_mul_split_fp8_fp16(gate_up, gate_up + H, ffn_hid, S, H, stream)  # ← gated activation
    fvk.gemm_fp8_fp16(ffn_hid_fp8, down_w_fp8, ffn_out, ...)
    fvk.add_inplace_fp16(x, ffn_out, S * D, stream)

The 1:1 correspondence is the whole game. If you can do this layer
mapping for one encoder layer, the rest is 90% loop unrolling.

# TWO FUNCTIONS PER STAGE
==========================

For each forward stage (encoder, decoder, vision tower), write
**two** functions:

1. `<stage>_forward(...)` — the production path. Uses FP8 kernels
   with pre-computed scales. This gets captured into the CUDA Graph.
2. `<stage>_forward_calibrate(...)` — the FP8 calibration twin.
   Same compute as `_forward` but BF16 / no-quant, AND it
   measures activation scales at every quantization point and
   stores them in the `act_scales` dict. Run once during warm-up
   per real-data sample to populate scales, then deleted.

Why two functions instead of an `if calibrate:` flag? Because
forward must produce zero Python overhead for graph capture, and
because the calibration path needs `_measure_scale_gpu` calls that
the forward path must skip. See pi05/pipeline_thor.py for the
canonical pair.

# WHO CALLS THIS
=================

The frontend (frontend.py STEP 5) calls this in two contexts:
- During calibration: calls `<stage>_forward_calibrate` once per
  real-data sample to populate `act_scales`.
- During graph capture: enters a CUDA Graph capture stream, then
  calls `<stage>_forward` once. The graph then replays for every
  subsequent `infer()`.
"""

import math


# ──────────────────────────────────────────────────────────────────
# STEP 1: Production forward (gets captured into CUDA Graph)
# ──────────────────────────────────────────────────────────────────

def encoder_forward(ctx, fvk, bufs, weights, dims, stream=0, *, attn=None):
    """Run the prompt+vision encoder once, populating Kc/Vc KV cache.

    Args:
        ctx: FvkContext (C++ object holding cuBLAS handle)
        fvk: flash_vla_kernels module (pybind layer)
        bufs: dict of GPU buffer pointers (uintptr_t).
              Allocated once in the frontend, reused across calls.
              Required keys: x, x_norm, qkv, attn_out, ffn_hid, residual
              (and any model-specific intermediates)
        weights: dict of GPU weight pointers + scales
                 (loaded by load_weights() in weights_spec.py)
        dims: dict of model dimensions (S, D, H, NH, NUM_KV_HEADS, HD, num_layers, ...)
        stream: CUDA stream (int)
        attn: AttentionBackend instance (provides attn.run(site, layer_idx, q_seq=S))

    Side effects:
        Writes Kc/Vc KV cache buffers (consumed by decoder cross-attn).
    """
    S = dims["S"]              # current prompt+vision token count
    D = dims["D"]              # hidden dim
    H = dims["H"]              # FFN intermediate dim
    NH = dims["NH"]            # num query heads
    NUM_KV = dims["NUM_KV"]    # num KV heads (GQA)
    HD = dims["HD"]            # head dim
    num_layers = dims["num_layers"]
    eps = 1e-6                 # TODO: replace with your model's norm epsilon

    # TODO: pull buffer pointers out of `bufs` once; keep them local
    # for readability. Pulling inside the loop adds dict-lookup overhead
    # to graph capture (one-time, but lots of it).
    x = bufs["x"]
    x_norm = bufs["x_norm"]
    qkv_buf = bufs["qkv"]
    attn_out = bufs["attn_out"]
    ffn_hid = bufs["ffn_hid"]

    for layer_idx in range(num_layers):
        # --- attention block ---
        # TODO: copy your reference's actual op sequence here.
        # Below is the canonical Pi0.5 / Gemma-style pattern.

        # Pre-norm
        fvk.rms_norm_fp16(
            x, weights[("encoder", layer_idx, "input_norm_w")],
            x_norm, S, D, eps, stream,
        )

        # Quantize for FP8 GEMM (act_scale comes from calibration)
        fvk.quantize_fp8_static(
            x_norm, bufs["x_norm_fp8"], S * D,
            weights["scales"][("encoder", layer_idx, "qkv_act")],
            stream,
        )

        # Fused QKV projection (uses the stacked q+k+v weight from weights_spec.py)
        alpha = weights["scales"][("encoder", layer_idx, "qkv_act")] * \
                weights["scales"][("encoder", layer_idx, "qkv_w")]
        # IMPORTANT: alpha must be float32, not float64. See
        # docs/calibration.md §2.3 for the bug it causes if you forget.
        # Most loaders return float, but np.float32(act) * np.float32(w)
        # is the safe pattern.
        fvk.gemm_fp8_fp16(
            ctx, bufs["x_norm_fp8"],
            weights["fp8"][("encoder", layer_idx, "qkv_w")],
            qkv_buf,
            S, (NH + 2 * NUM_KV) * HD, D,
            alpha, stream,
        )

        # Split QKV, apply RoPE, write into KV cache.
        # The kernel does all three in one launch — saves bandwidth.
        fvk.qkv_split_rope_kvcache_fp16(
            qkv_buf,
            attn.get_slot_ptrs("encoder", layer_idx)["Q"],
            weights["Kc"], weights["Vc"],
            weights["rope_table"],
            S, NH, NUM_KV, HD, layer_idx, stream,
        )

        # Dispatch attention via the backend (FA2 on RTX, FMHA on Thor)
        attn_ptr = attn.run("encoder", layer_idx, q_seq=S, stream=stream)

        # Output projection
        fvk.quantize_fp8_static(attn_ptr, bufs["attn_out_fp8"], S * D, ..., stream)
        alpha_out = ...   # same alpha pattern
        fvk.gemm_fp8_fp16(
            ctx, bufs["attn_out_fp8"],
            weights["fp8"][("encoder", layer_idx, "out_proj_w")],
            attn_out, S, D, NH * HD, alpha_out, stream,
        )

        # Residual add (in-place into x)
        fvk.add_inplace_fp16(x, attn_out, S * D, stream)

        # --- FFN block ---
        # Same pattern: norm -> quantize -> fused gate+up GEMM ->
        # silu+mul -> down GEMM -> residual.
        # TODO: write out the FFN block analogous to the attention block above.
        # The fused GEGLU/SiLU activation is `silu_mul_split_fp8_fp16`
        # (TRUE SiLU; for Pi0.5/Gemma's GEGLU/tanh-approx-GELU use
        # `gate_geglu_fp16` instead — pick by your reference's activation).
        ...


def decoder_forward(ctx, fvk, bufs, weights, dims, stream=0, *, attn=None):
    """Diffusion-style action decoder.

    For Pi0.5 / Pi0: 10 diffusion steps over self-attn (action tokens)
    + cross-attn (into encoder Kc/Vc).

    Loop structure:
        for step_idx in range(num_steps):
            # 1. Per-step embedding (timestep + state)
            # 2. Per-layer self-attn over action tokens (Sa = action horizon, ~10)
            # 3. Per-layer cross-attn into encoder KV
            # 4. Per-layer FFN
            # 5. Output projection -> velocity field
            # 6. Update noise: x = x + velocity * dt
    """
    # TODO: write this following the same pattern as encoder_forward.
    # See pi05/pipeline_thor.py decoder_forward (lines 30-280) for the
    # canonical implementation, including AdaRMSNorm style modulation
    # and the diffusion step embedding.
    ...


# ──────────────────────────────────────────────────────────────────
# STEP 2: Calibration twins (run once per real-data sample)
# ──────────────────────────────────────────────────────────────────

def encoder_forward_calibrate(ctx, fvk, bufs, weights, dims, act_scales, stream=0, *, attn=None):
    """Calibration variant of encoder_forward.

    Identical compute to encoder_forward but:
    1. Runs all GEMMs in BF16 (no quantize_fp8_static calls)
    2. At every quantization point, calls _measure_scale_gpu(...) to
       record the activation amax for that tensor
    3. Writes the measured scales into act_scales[(site, layer_idx, slot)]

    After this runs once per real-data sample, the frontend's
    calibration code applies a percentile reduction over samples
    (see flash_vla.core.calibration.accumulate_amax) and writes the
    final scales into the production weights dict.
    """
    from flash_vla.hardware.thor.shared_primitives import _measure_scale_gpu  # rtx version is identical
    # TODO: implement following the same control flow as encoder_forward.
    # At every fvk.quantize_fp8_static call site in encoder_forward,
    # replace with:
    #   amax = _measure_scale_gpu(x_norm, S * D, stream)
    #   act_scales[(site, layer_idx, slot)] = amax
    #   # then: do bf16 GEMM in place of the fp8 GEMM
    #   gemm.bf16_nn(ctx, x_norm, weight_bf16, qkv_buf, S, ..., stream)
    ...


def decoder_forward_calibrate(...):
    """Same shape as encoder_forward_calibrate, applied to decoder."""
    ...


# ──────────────────────────────────────────────────────────────────
# DONE-CHECKLIST (verify before moving to frontend.py)
# ──────────────────────────────────────────────────────────────────
# - [ ] Every kernel call uses the slot keys defined in weights_spec.py
#       (no inline `weights["my_key_typo"]` strings).
# - [ ] Every fp8 GEMM has its `alpha = act_scale * weight_scale`
#       computed in float32 (not implicit float64). The ergonomic
#       pattern is `np.float32(act) * np.float32(w)` at load time
#       and storing the f32 product as the alpha.
# - [ ] No torch.empty(), .cpu(), .numpy(), .item(), F.* anywhere
#       inside *_forward. Allocate everything in the frontend's
#       _load_weights and pass pointers in via `bufs`.
# - [ ] Every quantize_fp8_static call in *_forward has a matching
#       _measure_scale_gpu call in *_forward_calibrate.
# - [ ] decoder_forward correctly reads from the same Kc/Vc that
#       encoder_forward wrote (your decoder cross-attn site must use
#       the encoder's KV cache — verify the pointer aliasing).

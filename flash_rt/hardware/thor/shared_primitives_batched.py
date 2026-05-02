"""FlashVLA — Thor SM110 model-agnostic primitives, B>=1 batched variants.

Companion to :mod:`flash_vla.hardware.thor.shared_primitives` which
holds the B=1 hot path (the production single-sample inference).
This module isolates the B>=1 batched kernel orchestrations so the
B=1 file stays small and easy to reason about — the non-batched
single-sample inference is the main-line product, the batched path
is opt-in (used by the fused-CFG B=2 pipeline and future RL-rollout
B>2 paths).

Mirrors the model-layer split between
:mod:`flash_vla.models.pi05.pipeline_thor` (B=1) and
:mod:`flash_vla.models.pi05.pipeline_thor_batched` (B>=1), and the
RTX layout's cfg / cfg_batched split.

Functions:
    encoder_forward_b2  — Paligemma encoder forward at B>=1
"""

import math


def encoder_forward_b2(gemm, fvk, bufs, weights, dims, stream=0, *,
                       attn=None, B=2):
    """Batched encoder forward for ``B`` independent samples.

    Stage 2 of the Thor batched-CFG port. The kernel inventory split is:

      * **Flat-elementwise** kernels (``rms_norm_fp8_noweight_fp16``,
        ``residual_add_rms_norm_fp8_noweight_fp16``, ``quantize_fp8_static_fp16``,
        ``gate_geglu_merged_fp8_fp16``) auto-scale to ``M = B*Se``
        — they consume / produce a flat ``[B*Se, D]`` buffer.
      * **GEMMs** (``cutlass_fp8_sq``, ``cutlass_fp8_t1``,
        ``cutlass_fp8_wide``) likewise scale via their first dim arg
        (``M = B*Se``).
      * **Per-token-indexed ops** (``qkv_split_rope_kvcache_fp16`` and
        ``attention_qkv_fp16``) are NOT batch-aware on Thor today —
        the rope buffer is single-Se, the KV cache layer offset has
        no batch axis. We resolve this **at the python level** by
        looping over ``b ∈ [0, B)`` per layer and adjusting:
          - QKV input pointer:        ``qkv + b * Se * 2560 * 2``
          - Q output pointer:         ``attn_out + b * Se * Q_dim * 2``
          - Per-sample Kc / Vc slab:  ``weights['Kc_b2'][b]`` /
                                       ``weights['Vc_b2'][b]``
        The kernel itself is unchanged — same symbol Thor already
        ships, same per-layer-stride convention. No new SM110 kernels
        required.

    Buffer contract (``B`` is the leading axis or fold):

      bufs: same key set as
        :func:`flash_vla.hardware.thor.shared_primitives.encoder_forward`;
        shapes are flat ``B*Se`` along the row dim. The frontend
        allocates a fresh set of ``_b2``-suffixed buffers and hands
        them in here.

      weights: same as ``encoder_forward`` plus ``Kc_b2`` / ``Vc_b2``
        — lists of length ``B``, each entry a device pointer to that
        sample's ``[La * total_keys * HD]`` flat KV slab. ``Kc`` /
        ``Vc`` (the B=1 keys) are ignored.

      dims: same as ``encoder_forward``. ``Se`` is per-sample
        sequence length (NOT B*Se).

    Args:
        attn: Optional :class:`flash_vla.hardware.thor.attn_backend.ThorFlashAttnBackend`.
            **Not used in Stage 2** — the backend's encoder slot is
            single-batch; Stage 2 calls ``fvk.attention_qkv_fp16``
            directly per-sample. A future Stage extends the backend
            to handle B>1.
    """
    Se = dims['Se']
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    L = dims['L']
    total_keys = dims['total_keys']
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))
    BSe = B * Se

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
    alpha_host = weights['alpha_host']

    # Per-sample KV slab device pointers (one per b, one ptr per slab).
    Kc_b2 = weights['Kc_b2']
    Vc_b2 = weights['Vc_b2']
    if len(Kc_b2) != B or len(Vc_b2) != B:
        raise ValueError(
            f"Kc_b2/Vc_b2 must each have B={B} entries; "
            f"got {len(Kc_b2)} / {len(Vc_b2)}")

    # Byte strides for the inline per-sample loop. fp16 = 2 bytes.
    qkv_stride_bytes = Se * 2560 * 2
    attn_q_stride_bytes = Se * Q_dim * 2

    for l in range(L):
        last = (l == L - 1)

        as_qkv = act_scales + (l * 4 + 0) * 4
        as_o   = act_scales + (l * 4 + 1) * 4
        as_gu  = act_scales + (l * 4 + 2) * 4
        as_d   = act_scales + (l * 4 + 3) * 4

        # ── 1. RMSNorm → FP8 (flat elementwise, M = B*Se) ──
        fvk.rms_norm_fp8_noweight_fp16(x, x_fp8, BSe, D, as_qkv, stream)

        # ── 2. QKV GEMM (M = B*Se, output (B*Se, 2560)) ──
        fvk.cutlass_fp8_sq(x_fp8, weights['qkv_w'][l], qkv,
                           BSe, 2560, D, alpha_host[l * 4 + 0], 0.0, stream)

        # ── 3+4. QKV split + RoPE + KV cache write (per-sample inline) ──
        # Kernel is not batch-aware: rope is length Se, kv cache has no
        # batch axis. Loop B times, one kernel call per sample, with
        # adjusted offsets for QKV input / Q output / KV slab.
        kv_elem_off = l * total_keys * HD
        for b in range(B):
            fvk.qkv_split_rope_kvcache_fp16(
                qkv + b * qkv_stride_bytes,
                weights['rope'],
                attn_out + b * attn_q_stride_bytes,
                Kc_b2[b], Vc_b2[b],
                Se, Q_dim, K_dim, HD, 2560,
                kv_elem_off, HD, stream)

        if not last:
            # ── 5. Attention (per-sample inline) ──
            for b in range(B):
                K_ptr = Kc_b2[b] + kv_elem_off * 2  # byte offset (fp16)
                V_ptr = Vc_b2[b] + kv_elem_off * 2
                fvk.attention_qkv_fp16(
                    bufs['ctx'],
                    attn_out + b * attn_q_stride_bytes,
                    K_ptr, V_ptr,
                    logits,  # scratch: reused across samples
                    attn_out + b * attn_q_stride_bytes,
                    Se, Se, NH, HD, attn_scale, stream)

            # ── 6. Quantize attn → FP8 (flat) + O proj GEMM (M=B*Se) ──
            fvk.quantize_fp8_static_fp16(attn_out, o_fp8, as_o, BSe * D, stream)
            fvk.cutlass_fp8_sq(o_fp8, weights['o_w'][l], fg,
                               BSe, D, D, alpha_host[l * 4 + 1], 0.0, stream)

            # ── 7. Residual + RMSNorm → FP8 (flat, M=B*Se) ──
            fvk.residual_add_rms_norm_fp8_noweight_fp16(x, fg, x_fp8,
                                                          BSe, D, as_gu, stream)

            # ── 8. Gate+Up merged GEMM (M=B*Se) ──
            fvk.cutlass_fp8_t1(x_fp8, weights['gate_w'][l], gate,
                               BSe, H * 2, D, alpha_host[l * 4 + 2], 0.0, stream)

            # ── 9. GELU(gate) × up → FP8 (flat, M=B*Se*H) ──
            fvk.gate_geglu_merged_fp8_fp16(gate, hid_fp8, BSe, H,
                                               as_d, stream)

            # ── 10. Down GEMM (M=B*Se) ──
            fvk.cutlass_fp8_wide(hid_fp8, weights['down_w'][l], fg,
                                  BSe, D, H, alpha_host[l * 4 + 3], 0.0, stream)

            # ── 11. Residual + RMSNorm → FP8 for next layer (flat) ──
            as_next = act_scales + ((l + 1) * 4 + 0) * 4
            fvk.residual_add_rms_norm_fp8_noweight_fp16(x, fg, x_fp8,
                                                          BSe, D, as_next, stream)

    # x[B*Se, D] now contains the encoder output for both samples,
    # contiguous: rows [0:Se] = sample 0, rows [Se:2*Se] = sample 1.

"""FlashRT — Weight Transformer.

Transforms raw checkpoint weights into engine-ready format.
All operations are pure numpy — no framework dependency.

Transformations:
  1. QKV merge: separate Q, K, V → concatenated QKV tensor
  2. Head interleave: HF contiguous layout → interleaved for RoPE
  3. RMSNorm fusion: w_fused = w * (1 + norm_scale)
  4. FP8 quantize: float32 → E4M3 uint8 + per-tensor scale
  5. Layout transpose: (out_dim, in_dim) → (in_dim, out_dim) for GEMM
  6. Time embedding: precompute sinusoidal embeddings for diffusion steps
"""

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def interleave_qk(w: np.ndarray, num_heads: int) -> np.ndarray:
    """Interleave Q/K head dims for RoPE compatibility.

    HF layout: [h0_0, h0_1, ..., h0_255, h1_0, ...]
    RoPE layout: [h0_0, h0_128, h0_1, h0_129, ..., h1_0, ...]
    """
    out_dim, in_dim = w.shape
    head_dim = out_dim // num_heads
    return (w.reshape(num_heads, head_dim, in_dim)
             .reshape(num_heads, 2, head_dim // 2, in_dim)
             .transpose(0, 2, 1, 3)
             .reshape(out_dim, in_dim))


def quantize_fp8_e4m3(w: np.ndarray):
    """Quantize float32/float16 → FP8 E4M3 (uint8) with per-tensor scale.

    Returns (fp8_bytes: uint8 array, scale: float).
    Scale = amax / 448.0 (E4M3 max representable = 448).

    Uses ml_dtypes.float8_e4m3fn (JAX ecosystem, bit-exact with torch).
    Falls back to torch or pure numpy if ml_dtypes unavailable.
    """
    w_f32 = w.astype(np.float32)
    amax = np.abs(w_f32).max()
    scale = max(float(amax) / 448.0, 1e-12)
    w_scaled = np.clip(w_f32 / scale, -448.0, 448.0)

    # Prefer ml_dtypes (JAX ecosystem, no torch dependency, bit-exact with torch)
    try:
        import ml_dtypes
        fp8_arr = w_scaled.astype(ml_dtypes.float8_e4m3fn)
        fp8_bytes = fp8_arr.view(np.uint8).reshape(w.shape)
        return fp8_bytes, scale
    except ImportError:
        pass

    # Fallback: torch
    try:
        import torch
        t = torch.from_numpy(w_scaled).to(torch.float8_e4m3fn)
        fp8_bytes = t.view(torch.uint8).numpy().reshape(w.shape)
        return fp8_bytes, scale
    except (ImportError, RuntimeError, TypeError):
        pass

    # Last resort: pure numpy approximation (truncates, ~49% LSB error)
    fp8_bytes = _numpy_to_fp8_e4m3(w_scaled)
    return fp8_bytes, scale


def _numpy_to_fp8_e4m3(w: np.ndarray) -> np.ndarray:
    """Pure numpy FP8 E4M3 conversion (approximate)."""
    sign = (w < 0).astype(np.uint8) << 7
    aw = np.abs(w)

    # FP8 E4M3: bias=7, max=448, min_subnormal=2^-9
    bits = np.zeros_like(w, dtype=np.uint8)

    # Use float32 bit manipulation for conversion
    f32_bits = aw.view(np.float32).view(np.uint32)
    f32_exp = ((f32_bits >> 23) & 0xFF).astype(np.int32) - 127  # unbias
    f32_mant = (f32_bits >> 20) & 0x7  # top 3 mantissa bits

    fp8_exp = np.clip(f32_exp + 7, 0, 15).astype(np.uint8)  # rebias to E4M3

    # Normal values
    normal_mask = (aw >= 2**-6) & (aw <= 448.0)
    bits[normal_mask] = (fp8_exp[normal_mask] << 3) | f32_mant[normal_mask].astype(np.uint8)

    # Clamp to max
    bits[aw > 448.0] = 0x7E  # max finite E4M3

    return sign | bits


def compute_rope_table(max_seq: int, head_dim: int = 256,
                       base: float = 10000.0) -> np.ndarray:
    """Compute RoPE cos/sin table in interleaved format.

    Returns (max_seq, head_dim) with [cos0, sin0, cos1, sin1, ...] per position.
    """
    half_dim = head_dim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half_dim, dtype=np.float64) * 2 / head_dim))
    positions = np.arange(max_seq, dtype=np.float64)
    phase = positions[:, None] * inv_freq[None, :]  # (max_seq, half_dim)
    cos_t = np.cos(phase).astype(np.float16)
    sin_t = np.sin(phase).astype(np.float16)
    # Interleave: [cos0, sin0, cos1, sin1, ...]
    table = np.stack([cos_t, sin_t], axis=-1).reshape(max_seq, head_dim)
    return table


def compute_time_embeddings(num_steps: int = 10, dim: int = 1024) -> np.ndarray:
    """Compute sinusoidal time embeddings for diffusion steps.

    Returns (num_steps, dim) float16 embeddings.
    t = 1.0, 0.9, 0.8, ..., 0.1 (descending from noise to clean).
    """
    fraction = np.linspace(0, 1, dim // 2)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    scaling = 1.0 / period * 2 * math.pi

    embeds = []
    for step in range(num_steps):
        t_val = 1.0 - step / num_steps
        sin_input = scaling * t_val
        emb = np.concatenate([np.sin(sin_input), np.cos(sin_input)])
        embeds.append(emb.astype(np.float16))
    return np.stack(embeds)


def _to_fp16(w):
    """Cast to float16 via bfloat16 intermediate — matches safetensors bf16→f16 rounding.
    Orbax stores float32; safetensors stores bfloat16. Without this,
    f32→f16 and bf16→f16 give 85% different fp16 values (1 ULP diffs)."""
    import ml_dtypes
    if w.dtype == np.float32 or w.dtype == np.float64:
        return w.astype(ml_dtypes.bfloat16).astype(np.float16)
    return w.astype(np.float16)


def transform_jax_weights(raw: dict) -> dict:
    """Transform JAX Orbax weights to engine format.

    Input: flat dict with JAX key names (51 keys, layers stacked).
    Output: flat dict with engine key names, per-layer, transposed for GEMM.

    Orbax stores float32; safetensors stores bfloat16. To get bit-exact FP8 weights,
    truncate all raw weights to bfloat16 precision first (matching safetensors source).
    """
    import ml_dtypes
    raw = {k: v.astype(ml_dtypes.bfloat16).astype(np.float32) if v.dtype == np.float32 else v
           for k, v in raw.items()}

    out = {}

    # ── Vision (SigLIP) ──
    # Patch embedding: JAX (14,14,3,1152) — already correct layout
    # All weights cast to float16 to match engine expectation (Thor loads as fp16)
    out["vision.patch_embed.weight"] = raw["PaliGemma.img.embedding.kernel"].astype(np.float16)
    out["vision.patch_embed.bias"] = raw["PaliGemma.img.embedding.bias"].astype(np.float16)
    out["vision.pos_embed"] = raw["PaliGemma.img.pos_embedding"].squeeze(0).astype(np.float16)

    # Vision layers (stacked (27, ...))
    for i in range(27):
        pfx = f"vision.layer.{i}"
        out[f"{pfx}.attn_norm.weight"] = raw["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.scale"][i].astype(np.float16)
        out[f"{pfx}.attn_norm.bias"] = raw["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.bias"][i].astype(np.float16)
        out[f"{pfx}.ffn_norm.weight"] = raw["PaliGemma.img.Transformer.encoderblock.LayerNorm_1.scale"][i].astype(np.float16)
        out[f"{pfx}.ffn_norm.bias"] = raw["PaliGemma.img.Transformer.encoderblock.LayerNorm_1.bias"][i].astype(np.float16)

        # Q/K/V: JAX (27, 1152, 16, 72) → flatten heads → (1152, 1152)
        q_w = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.kernel"][i]
        k_w = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.key.kernel"][i]
        v_w = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.value.kernel"][i]
        q_b = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.bias"][i]
        k_b = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.key.bias"][i]
        v_b = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.value.bias"][i]
        # (1152, 16, 72) → (1152, 1152) then merge QKV
        q_2d = q_w.reshape(1152, -1)  # (1152, 1152)
        k_2d = k_w.reshape(1152, -1)  # (1152, 1152)
        v_2d = v_w.reshape(1152, -1)  # (1152, 1152)
        # QKV merged: (1152, 3456), transposed to (3456, 1152) for engine
        out[f"{pfx}.qkv.weight"] = np.concatenate([q_2d, k_2d, v_2d], axis=1).T.astype(np.float16)
        out[f"{pfx}.qkv.bias"] = np.concatenate([q_b.reshape(-1), k_b.reshape(-1), v_b.reshape(-1)]).astype(np.float16)

        # O: JAX (16, 72, 1152) → (1152, 1152)
        o_w = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.out.kernel"][i]
        out[f"{pfx}.o.weight"] = o_w.reshape(-1, 1152).T.astype(np.float16)  # (1152, 1152)
        out[f"{pfx}.o.bias"] = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.out.bias"][i].astype(np.float16)

        # FFN up: JAX (1152, 4304) — already correct
        out[f"{pfx}.ffn_up.weight"] = raw["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.kernel"][i].T.astype(np.float16)
        out[f"{pfx}.ffn_up.bias"] = raw["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.bias"][i].astype(np.float16)
        out[f"{pfx}.ffn_down.weight"] = raw["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.kernel"][i].T.astype(np.float16)
        out[f"{pfx}.ffn_down.bias"] = raw["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.bias"][i].astype(np.float16)

    out["vision.final_norm.weight"] = raw["PaliGemma.img.Transformer.encoder_norm.scale"].astype(np.float16)
    out["vision.final_norm.bias"] = raw["PaliGemma.img.Transformer.encoder_norm.bias"].astype(np.float16)
    out["vision.projector.weight"] = raw["PaliGemma.img.head.kernel"].T.astype(np.float16)  # (1152,2048) → (2048,1152) — wait, check
    out["vision.projector.bias"] = raw["PaliGemma.img.head.bias"].astype(np.float16)

    # ── Encoder (Gemma 2B, 18 layers) ──
    out["encoder.embedding"] = raw["PaliGemma.llm.embedder.input_embedding"].astype(np.float16)

    for i in range(18):
        pfx = f"encoder.layer.{i}"
        # RMSNorm scale (used for fusion: w *= (1+scale))
        attn_scale = raw["PaliGemma.llm.layers.pre_attention_norm.scale"][i]  # (2048,)
        ffn_scale = raw["PaliGemma.llm.layers.pre_ffw_norm.scale"][i]  # (2048,)
        out[f"{pfx}.attn_norm.scale"] = attn_scale
        out[f"{pfx}.ffn_norm.scale"] = ffn_scale

        fuse_attn = 1.0 + attn_scale  # (2048,)
        fuse_ffn = 1.0 + ffn_scale    # (2048,)

        # Q: JAX (N, D, H) = (8, 2048, 256)
        #   einsum "BTD,NDH->BTNH" ≡ HF (N*H, D) with matmul x @ w.T
        #   JAX→HF: w.transpose(0,2,1).reshape(N*H, D) = (2048, 2048)
        q_w = raw["PaliGemma.llm.layers.attn.q_einsum.w"][i]  # (8, 2048, 256)
        q_2d = q_w.transpose(0, 2, 1).reshape(-1, q_w.shape[1])  # (N*H, D) = (2048, 2048)
        q_2d = interleave_qk(q_2d, 8) * fuse_attn[None, :]

        # KV: JAX (2, K, D, H) = (2, 1, 2048, 256)
        #   k: (K, D, H) → (K*H, D) = (256, 2048)
        kv_w = raw["PaliGemma.llm.layers.attn.kv_einsum.w"][i]  # (2, 1, 2048, 256)
        k_2d = kv_w[0].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])  # (K*H, D) = (256, 2048)
        v_2d = kv_w[1].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])  # (K*H, D) = (256, 2048)
        k_2d = interleave_qk(k_2d, 1) * fuse_attn[None, :]
        v_2d = v_2d * fuse_attn[None, :]

        # QKV merged: (2560, 2048)
        qkv = np.concatenate([q_2d, k_2d, v_2d], axis=0).astype(np.float16)
        out[f"{pfx}.qkv.weight"] = qkv

        # O: JAX (N, H, D) = (8, 256, 2048)
        #   JAX→HF: reshape(N*H, D).T = (2048, 2048)
        o_w = raw["PaliGemma.llm.layers.attn.attn_vec_einsum.w"][i]  # (8, 256, 2048)
        out[f"{pfx}.o.weight"] = o_w.reshape(-1, o_w.shape[-1]).T.astype(np.float16)  # (2048, 2048)

        # Gate+Up: JAX (2, 2048, 16384) → gate (16384, 2048), up (16384, 2048)
        gu_w = raw["PaliGemma.llm.layers.mlp.gating_einsum"][i]  # (2, 2048, 16384)
        gate_w = (gu_w[0].T * fuse_ffn[None, :]).astype(np.float16)  # (16384, 2048)
        up_w = (gu_w[1].T * fuse_ffn[None, :]).astype(np.float16)    # (16384, 2048)
        # Merged: (32768, 2048)
        out[f"{pfx}.gate_up.weight"] = np.concatenate([gate_w, up_w], axis=0)

        # Down: JAX (16384, 2048) — NO transpose, matches safetensors (2048, 16384)
        # Wait: JAX shape is (16384, 2048) but safetensors is (2048, 16384)?
        # safetensors down_proj.weight = (2048, 16384) (HF: out, in)
        # JAX mlp.linear = (16384, 2048) → need to transpose
        out[f"{pfx}.down.weight"] = raw["PaliGemma.llm.layers.mlp.linear"][i].T.astype(np.float16)  # → (2048, 16384)

    # ── Decoder (Expert, 18 layers) ──
    for i in range(18):
        pfx = f"decoder.layer.{i}"
        # AdaRMSNorm modulation
        out[f"{pfx}.attn_mod.weight"] = raw["PaliGemma.llm.layers.pre_attention_norm_1.Dense_0.kernel"][i].astype(np.float16)
        out[f"{pfx}.attn_mod.bias"] = raw["PaliGemma.llm.layers.pre_attention_norm_1.Dense_0.bias"][i].astype(np.float16)
        out[f"{pfx}.ffn_mod.weight"] = raw["PaliGemma.llm.layers.pre_ffw_norm_1.Dense_0.kernel"][i].astype(np.float16)
        out[f"{pfx}.ffn_mod.bias"] = raw["PaliGemma.llm.layers.pre_ffw_norm_1.Dense_0.bias"][i].astype(np.float16)

        # Q: JAX (N, D, H) = (8, 1024, 256)
        #   JAX→HF: transpose(0,2,1).reshape(N*H, D) = (2048, 1024)
        q_w = raw["PaliGemma.llm.layers.attn.q_einsum_1.w"][i]  # (8, 1024, 256)
        q_2d = q_w.transpose(0, 2, 1).reshape(-1, q_w.shape[1])  # (2048, 1024)
        q_2d = interleave_qk(q_2d, 8)

        # KV: JAX (2, K, D, H) = (2, 1, 1024, 256)
        kv_w = raw["PaliGemma.llm.layers.attn.kv_einsum_1.w"][i]
        k_2d = kv_w[0].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])  # (256, 1024)
        v_2d = kv_w[1].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])  # (256, 1024)
        k_2d = interleave_qk(k_2d, 1)

        # QKV: cat → (2560, 1024) → .T → (1024, 2560) for engine
        qkv = np.concatenate([q_2d, k_2d, v_2d], axis=0).T.astype(np.float16)
        out[f"{pfx}.qkv.weight"] = qkv  # (1024, 2560)

        # O: JAX (N, H, D) = (8, 256, 1024)
        #   reshape(N*H, D) = (2048, 1024) — directly matches production
        o_w = raw["PaliGemma.llm.layers.attn.attn_vec_einsum_1.w"][i]
        out[f"{pfx}.o.weight"] = o_w.reshape(-1, o_w.shape[-1]).astype(np.float16)  # (2048, 1024)

        # Gate+Up: JAX (2, 1024, 4096) → cat along axis=0 then transpose
        # safetensors: cat([gate(4096,1024), up(4096,1024)], 0).T = (1024, 8192)
        gu_w = raw["PaliGemma.llm.layers.mlp_1.gating_einsum"][i]
        gate_w = gu_w[0].T.astype(np.float16)  # (1024, 4096) → T → (4096, 1024)
        up_w = gu_w[1].T.astype(np.float16)
        out[f"{pfx}.gate_up.weight"] = np.concatenate([gate_w, up_w], axis=0).T.astype(np.float16)  # (8192,1024).T = (1024,8192)

        # Down: JAX (4096, 1024) — NO transpose, matches safetensors .T
        out[f"{pfx}.down.weight"] = raw["PaliGemma.llm.layers.mlp_1.linear"][i].astype(np.float16)

    # Decoder final norm
    out["decoder.final_mod.weight"] = raw["PaliGemma.llm.final_norm_1.Dense_0.kernel"].astype(np.float16)
    out["decoder.final_mod.bias"] = raw["PaliGemma.llm.final_norm_1.Dense_0.bias"].astype(np.float16)

    # ── Action projections ──
    # JAX kernel (32,1024), safetensors weight (1024,32) → .T → (32,1024)
    # So JAX kernel shape already matches production .t() result
    out["action.in_proj.weight"] = raw["action_in_proj.kernel"].astype(np.float16)  # (32, 1024) — matches prod
    out["action.in_proj.bias"] = raw["action_in_proj.bias"].astype(np.float16)
    # out_proj: JAX (1024,32), prod = HF(32,1024).T*(−1/10) = (1024,32)*(−1/10)
    num_steps = 10
    out["action.out_proj.weight"] = (raw["action_out_proj.kernel"] * (-1.0 / num_steps)).astype(np.float16)  # (1024,32)
    out["action.out_proj.bias"] = (raw["action_out_proj.bias"] * (-1.0 / num_steps)).astype(np.float16)
    # time MLP: JAX kernel (D,D) used as x@w, HF weight (D,D) used as x@w.T
    # For square matrices HF_w = JAX_kernel.T, but production keeps HF layout
    # Production: time_mlp_in_w = g('time_mlp_in.weight') — no transpose
    # safetensors 'time_mlp_in.weight' = (1024, 1024)
    # JAX 'time_mlp_in.kernel' = (1024, 1024) but w is used as x@w (not x@w.T)
    # So JAX_kernel = HF_weight.T → to get HF layout, transpose
    out["time.mlp_in.weight"] = raw["time_mlp_in.kernel"].T.astype(np.float16)
    out["time.mlp_in.bias"] = raw["time_mlp_in.bias"].astype(np.float16)
    out["time.mlp_out.weight"] = raw["time_mlp_out.kernel"].T.astype(np.float16)
    out["time.mlp_out.bias"] = raw["time_mlp_out.bias"].astype(np.float16)

    logger.info(f"Transformed {len(out)} engine weights from JAX format")
    return out


def transform_safetensors_weights(raw: dict) -> dict:
    """Transform safetensors weights to engine format.

    Input: flat dict with HF key names (849 keys, per-layer).
    Output: flat dict with engine key names, same transforms as JAX path.
    """
    out = {}
    vp = 'paligemma_with_expert.paligemma.model.vision_tower.vision_model'
    ep = 'paligemma_with_expert.paligemma.model.language_model.layers'
    dp = 'paligemma_with_expert.gemma_expert.model.layers'

    # ── Vision ──
    pe_w = raw[f'{vp}.embeddings.patch_embedding.weight']  # (1152,3,14,14)
    out["vision.patch_embed.weight"] = pe_w.transpose(2, 3, 1, 0).astype(np.float16)  # (14,14,3,1152)
    out["vision.patch_embed.bias"] = raw[f'{vp}.embeddings.patch_embedding.bias'].astype(np.float16)
    out["vision.pos_embed"] = raw[f'{vp}.embeddings.position_embedding.weight'].astype(np.float16)

    for i in range(27):
        pfx = f"vision.layer.{i}"
        lp = f'{vp}.encoder.layers.{i}'
        q_w = raw[f'{lp}.self_attn.q_proj.weight']
        k_w = raw[f'{lp}.self_attn.k_proj.weight']
        v_w = raw[f'{lp}.self_attn.v_proj.weight']
        # QKV merged: (3456, 1152) = (out_dim, in_dim) — for cuBLASLt NN GEMM
        qkv_w = np.concatenate([q_w, k_w, v_w], axis=0).astype(np.float16)
        out[f"{pfx}.qkv.weight"] = qkv_w

        q_b = raw[f'{lp}.self_attn.q_proj.bias']
        k_b = raw[f'{lp}.self_attn.k_proj.bias']
        v_b = raw[f'{lp}.self_attn.v_proj.bias']
        out[f"{pfx}.qkv.bias"] = np.concatenate([q_b, k_b, v_b]).astype(np.float16)

        out[f"{pfx}.o.weight"] = raw[f'{lp}.self_attn.out_proj.weight'].astype(np.float16)
        out[f"{pfx}.o.bias"] = raw[f'{lp}.self_attn.out_proj.bias'].astype(np.float16)
        out[f"{pfx}.attn_norm.weight"] = raw[f'{lp}.layer_norm1.weight'].astype(np.float16)
        out[f"{pfx}.attn_norm.bias"] = raw[f'{lp}.layer_norm1.bias'].astype(np.float16)
        out[f"{pfx}.ffn_norm.weight"] = raw[f'{lp}.layer_norm2.weight'].astype(np.float16)
        out[f"{pfx}.ffn_norm.bias"] = raw[f'{lp}.layer_norm2.bias'].astype(np.float16)
        out[f"{pfx}.ffn_up.weight"] = raw[f'{lp}.mlp.fc1.weight'].astype(np.float16)  # (4304, 1152)
        out[f"{pfx}.ffn_up.bias"] = raw[f'{lp}.mlp.fc1.bias'].astype(np.float16)
        out[f"{pfx}.ffn_down.weight"] = raw[f'{lp}.mlp.fc2.weight'].astype(np.float16)  # (1152, 4304)
        out[f"{pfx}.ffn_down.bias"] = raw[f'{lp}.mlp.fc2.bias'].astype(np.float16)

    out["vision.final_norm.weight"] = raw[f'{vp}.post_layernorm.weight'].astype(np.float16)
    out["vision.final_norm.bias"] = raw[f'{vp}.post_layernorm.bias'].astype(np.float16)
    mp = 'paligemma_with_expert.paligemma.model.multi_modal_projector.linear'
    out["vision.projector.weight"] = raw[f'{mp}.weight'].astype(np.float16)
    out["vision.projector.bias"] = raw[f'{mp}.bias'].astype(np.float16)

    # ── Encoder ──
    out["encoder.embedding"] = raw['paligemma_with_expert.paligemma.lm_head.weight'].astype(np.float16)

    for i in range(18):
        pfx = f"encoder.layer.{i}"
        attn_scale = raw[f'{ep}.{i}.input_layernorm.weight']
        ffn_scale = raw[f'{ep}.{i}.post_attention_layernorm.weight']
        out[f"{pfx}.attn_norm.scale"] = attn_scale
        out[f"{pfx}.ffn_norm.scale"] = ffn_scale

        fuse_attn = 1.0 + attn_scale
        fuse_ffn = 1.0 + ffn_scale

        q_w = interleave_qk(raw[f'{ep}.{i}.self_attn.q_proj.weight'], 8) * fuse_attn[None, :]
        k_w = interleave_qk(raw[f'{ep}.{i}.self_attn.k_proj.weight'], 1) * fuse_attn[None, :]
        v_w = raw[f'{ep}.{i}.self_attn.v_proj.weight'] * fuse_attn[None, :]
        out[f"{pfx}.qkv.weight"] = np.concatenate([q_w, k_w, v_w], axis=0).astype(np.float16)
        out[f"{pfx}.o.weight"] = raw[f'{ep}.{i}.self_attn.o_proj.weight'].astype(np.float16)

        gate_w = (raw[f'{ep}.{i}.mlp.gate_proj.weight'] * fuse_ffn[None, :]).astype(np.float16)
        up_w = (raw[f'{ep}.{i}.mlp.up_proj.weight'] * fuse_ffn[None, :]).astype(np.float16)
        out[f"{pfx}.gate_up.weight"] = np.concatenate([gate_w, up_w], axis=0)
        out[f"{pfx}.down.weight"] = raw[f'{ep}.{i}.mlp.down_proj.weight'].astype(np.float16)

    # ── Decoder ──
    for i in range(18):
        pfx = f"decoder.layer.{i}"
        out[f"{pfx}.attn_mod.weight"] = raw[f'{dp}.{i}.input_layernorm.dense.weight'].T.astype(np.float16)
        out[f"{pfx}.attn_mod.bias"] = raw[f'{dp}.{i}.input_layernorm.dense.bias'].astype(np.float16)
        out[f"{pfx}.ffn_mod.weight"] = raw[f'{dp}.{i}.post_attention_layernorm.dense.weight'].T.astype(np.float16)
        out[f"{pfx}.ffn_mod.bias"] = raw[f'{dp}.{i}.post_attention_layernorm.dense.bias'].astype(np.float16)

        q_w = interleave_qk(raw[f'{dp}.{i}.self_attn.q_proj.weight'], 8)
        k_w = interleave_qk(raw[f'{dp}.{i}.self_attn.k_proj.weight'], 1)
        v_w = raw[f'{dp}.{i}.self_attn.v_proj.weight']
        out[f"{pfx}.qkv.weight"] = np.concatenate([q_w, k_w, v_w], axis=0).T.astype(np.float16)
        out[f"{pfx}.o.weight"] = raw[f'{dp}.{i}.self_attn.o_proj.weight'].T.astype(np.float16)

        out[f"{pfx}.gate_up.weight"] = np.concatenate([
            raw[f'{dp}.{i}.mlp.gate_proj.weight'],
            raw[f'{dp}.{i}.mlp.up_proj.weight']
        ], axis=0).T.astype(np.float16)
        out[f"{pfx}.down.weight"] = raw[f'{dp}.{i}.mlp.down_proj.weight'].T.astype(np.float16)

    dp_full = 'paligemma_with_expert.gemma_expert.model'
    out["decoder.final_mod.weight"] = raw[f'{dp_full}.norm.dense.weight'].T.astype(np.float16)
    out["decoder.final_mod.bias"] = raw[f'{dp_full}.norm.dense.bias'].astype(np.float16)

    # Action weights: safetensors (out,in), engine uses .t() → (in,out)
    out["action.in_proj.weight"] = raw['action_in_proj.weight'].T.astype(np.float16)  # (1024,32) → (32,1024)
    out["action.in_proj.bias"] = raw['action_in_proj.bias'].astype(np.float16)
    # out_proj bakes in diffusion dt: *= -1/num_steps
    num_steps = 10
    out["action.out_proj.weight"] = (raw['action_out_proj.weight'].T * (-1.0 / num_steps)).astype(np.float16)
    out["action.out_proj.bias"] = (raw['action_out_proj.bias'] * (-1.0 / num_steps)).astype(np.float16)
    out["time.mlp_in.weight"] = raw['time_mlp_in.weight'].astype(np.float16)  # (1024,1024)
    out["time.mlp_in.bias"] = raw['time_mlp_in.bias'].astype(np.float16)
    out["time.mlp_out.weight"] = raw['time_mlp_out.weight'].astype(np.float16)
    out["time.mlp_out.bias"] = raw['time_mlp_out.bias'].astype(np.float16)

    logger.info(f"Transformed {len(out)} engine weights from safetensors format")
    return out


def transform_jax_weights_pi0(raw: dict) -> dict:
    """Transform JAX Orbax weights to engine format — Pi0 variant.

    Key differences from Pi0.5 (transform_jax_weights):
      - Decoder uses standard RMSNorm: fuse (1+scale) into QKV/GateUp (like encoder)
      - No AdaRMSNorm modulation weights (attn_mod, ffn_mod, final_mod)
      - action_time_mlp replaces time_mlp (split W into action/time parts)
      - state_proj weights added
      - Final norm is standard RMSNorm with weight (not Dense modulation)

    Vision + Encoder sections are identical to Pi0.5.
    """
    import ml_dtypes
    raw = {k: v.astype(ml_dtypes.bfloat16).astype(np.float32) if v.dtype == np.float32 else v
           for k, v in raw.items()}

    out = {}

    # ── Vision (SigLIP) — identical to Pi0.5 ──
    out["vision.patch_embed.weight"] = raw["PaliGemma.img.embedding.kernel"].astype(np.float16)
    out["vision.patch_embed.bias"] = raw["PaliGemma.img.embedding.bias"].astype(np.float16)
    out["vision.pos_embed"] = raw["PaliGemma.img.pos_embedding"].squeeze(0).astype(np.float16)

    for i in range(27):
        pfx = f"vision.layer.{i}"
        out[f"{pfx}.attn_norm.weight"] = raw["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.scale"][i].astype(np.float16)
        out[f"{pfx}.attn_norm.bias"] = raw["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.bias"][i].astype(np.float16)
        out[f"{pfx}.ffn_norm.weight"] = raw["PaliGemma.img.Transformer.encoderblock.LayerNorm_1.scale"][i].astype(np.float16)
        out[f"{pfx}.ffn_norm.bias"] = raw["PaliGemma.img.Transformer.encoderblock.LayerNorm_1.bias"][i].astype(np.float16)

        q_w = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.kernel"][i]
        k_w = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.key.kernel"][i]
        v_w = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.value.kernel"][i]
        q_b = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.bias"][i]
        k_b = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.key.bias"][i]
        v_b = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.value.bias"][i]
        q_2d = q_w.reshape(1152, -1)
        k_2d = k_w.reshape(1152, -1)
        v_2d = v_w.reshape(1152, -1)
        out[f"{pfx}.qkv.weight"] = np.concatenate([q_2d, k_2d, v_2d], axis=1).T.astype(np.float16)
        out[f"{pfx}.qkv.bias"] = np.concatenate([q_b.reshape(-1), k_b.reshape(-1), v_b.reshape(-1)]).astype(np.float16)

        o_w = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.out.kernel"][i]
        out[f"{pfx}.o.weight"] = o_w.reshape(-1, 1152).T.astype(np.float16)
        out[f"{pfx}.o.bias"] = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.out.bias"][i].astype(np.float16)

        out[f"{pfx}.ffn_up.weight"] = raw["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.kernel"][i].T.astype(np.float16)
        out[f"{pfx}.ffn_up.bias"] = raw["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.bias"][i].astype(np.float16)
        out[f"{pfx}.ffn_down.weight"] = raw["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.kernel"][i].T.astype(np.float16)
        out[f"{pfx}.ffn_down.bias"] = raw["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.bias"][i].astype(np.float16)

    out["vision.final_norm.weight"] = raw["PaliGemma.img.Transformer.encoder_norm.scale"].astype(np.float16)
    out["vision.final_norm.bias"] = raw["PaliGemma.img.Transformer.encoder_norm.bias"].astype(np.float16)
    out["vision.projector.weight"] = raw["PaliGemma.img.head.kernel"].T.astype(np.float16)
    out["vision.projector.bias"] = raw["PaliGemma.img.head.bias"].astype(np.float16)

    # ── Encoder (Gemma 2B, 18 layers) — identical to Pi0.5 ──
    out["encoder.embedding"] = raw["PaliGemma.llm.embedder.input_embedding"].astype(np.float16)

    for i in range(18):
        pfx = f"encoder.layer.{i}"
        attn_scale = raw["PaliGemma.llm.layers.pre_attention_norm.scale"][i]
        ffn_scale = raw["PaliGemma.llm.layers.pre_ffw_norm.scale"][i]
        out[f"{pfx}.attn_norm.scale"] = attn_scale
        out[f"{pfx}.ffn_norm.scale"] = ffn_scale

        fuse_attn = 1.0 + attn_scale
        fuse_ffn = 1.0 + ffn_scale

        q_w = raw["PaliGemma.llm.layers.attn.q_einsum.w"][i]
        q_2d = q_w.transpose(0, 2, 1).reshape(-1, q_w.shape[1])
        q_2d = interleave_qk(q_2d, 8) * fuse_attn[None, :]

        kv_w = raw["PaliGemma.llm.layers.attn.kv_einsum.w"][i]
        k_2d = kv_w[0].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])
        v_2d = kv_w[1].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])
        k_2d = interleave_qk(k_2d, 1) * fuse_attn[None, :]
        v_2d = v_2d * fuse_attn[None, :]

        qkv = np.concatenate([q_2d, k_2d, v_2d], axis=0).astype(np.float16)
        out[f"{pfx}.qkv.weight"] = qkv

        o_w = raw["PaliGemma.llm.layers.attn.attn_vec_einsum.w"][i]
        out[f"{pfx}.o.weight"] = o_w.reshape(-1, o_w.shape[-1]).T.astype(np.float16)

        gu_w = raw["PaliGemma.llm.layers.mlp.gating_einsum"][i]
        gate_w = (gu_w[0].T * fuse_ffn[None, :]).astype(np.float16)
        up_w = (gu_w[1].T * fuse_ffn[None, :]).astype(np.float16)
        out[f"{pfx}.gate_up.weight"] = np.concatenate([gate_w, up_w], axis=0)

        out[f"{pfx}.down.weight"] = raw["PaliGemma.llm.layers.mlp.linear"][i].T.astype(np.float16)

    # ── Decoder (Expert, 18 layers) — Pi0: RMSNorm fusion (NOT AdaRMSNorm) ──
    Da = 1024  # decoder hidden dim
    for i in range(18):
        pfx = f"decoder.layer.{i}"

        # Fuse input_layernorm (standard RMSNorm) into QKV
        attn_scale = raw["PaliGemma.llm.layers.pre_attention_norm_1.scale"][i]  # (1024,)
        fa = 1.0 + attn_scale

        q_w = raw["PaliGemma.llm.layers.attn.q_einsum_1.w"][i]  # (8, 1024, 256)
        q_2d = q_w.transpose(0, 2, 1).reshape(-1, q_w.shape[1])  # (2048, 1024)
        q_2d = interleave_qk(q_2d, 8) * fa[None, :]

        kv_w = raw["PaliGemma.llm.layers.attn.kv_einsum_1.w"][i]  # (2, 1, 1024, 256)
        k_2d = kv_w[0].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])  # (256, 1024)
        v_2d = kv_w[1].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])  # (256, 1024)
        k_2d = interleave_qk(k_2d, 1) * fa[None, :]
        v_2d = v_2d * fa[None, :]

        # QKV: (2560, 1024) → .T → (1024, 2560) for engine
        qkv = np.concatenate([q_2d, k_2d, v_2d], axis=0).T.astype(np.float16)
        out[f"{pfx}.qkv.weight"] = qkv  # (1024, 2560)

        # O: (8, 256, 1024) → (2048, 1024)
        o_w = raw["PaliGemma.llm.layers.attn.attn_vec_einsum_1.w"][i]
        out[f"{pfx}.o.weight"] = o_w.reshape(-1, o_w.shape[-1]).astype(np.float16)  # (2048, 1024)

        # Fuse post_attention_layernorm into GateUp
        ffn_scale = raw["PaliGemma.llm.layers.pre_ffw_norm_1.scale"][i]  # (1024,)
        ff = 1.0 + ffn_scale

        gu_w = raw["PaliGemma.llm.layers.mlp_1.gating_einsum"][i]  # (2, 1024, 4096)
        gate_w = (gu_w[0].T * ff[None, :]).astype(np.float16)  # (4096, 1024)
        up_w = (gu_w[1].T * ff[None, :]).astype(np.float16)    # (4096, 1024)
        out[f"{pfx}.gate_up.weight"] = np.concatenate([gate_w, up_w], axis=0).T.astype(np.float16)  # (1024, 8192)

        # Down: (4096, 1024) — no transpose, matches engine layout
        out[f"{pfx}.down.weight"] = raw["PaliGemma.llm.layers.mlp_1.linear"][i].astype(np.float16)

    # Final norm: standard RMSNorm weight (not Dense modulation)
    final_scale = raw["PaliGemma.llm.final_norm_1.scale"]  # (1024,)
    out["decoder.final_norm.weight"] = (1.0 + final_scale).astype(np.float16)

    # ── Action projections ──
    out["action.in_proj.weight"] = raw["action_in_proj.kernel"].astype(np.float16)  # (32, 1024)
    out["action.in_proj.bias"] = raw["action_in_proj.bias"].astype(np.float16)
    num_steps = 10
    out["action.out_proj.weight"] = (raw["action_out_proj.kernel"] * (-1.0 / num_steps)).astype(np.float16)
    out["action.out_proj.bias"] = (raw["action_out_proj.bias"] * (-1.0 / num_steps)).astype(np.float16)

    # ── action_time_mlp: split W_full into W_action and W_time ──
    # JAX kernel (2*Da, Da) = (in, out). First Da rows = action, last Da rows = time.
    atmlp_in_kernel = raw["action_time_mlp_in.kernel"]  # (2048, 1024)
    out["action_time_mlp.wa"] = atmlp_in_kernel[:Da, :].astype(np.float16)       # (Da, Da) action → CudaBuffer GEMM
    out["action_time_mlp.wt"] = atmlp_in_kernel[Da:, :].T.astype(np.float16)     # (Da, Da) time → numpy precompute
    out["action_time_mlp.in_bias"] = raw["action_time_mlp_in.bias"].astype(np.float16)
    out["action_time_mlp.out_w"] = raw["action_time_mlp_out.kernel"].astype(np.float16)  # (Da, Da)
    out["action_time_mlp.out_bias"] = raw["action_time_mlp_out.bias"].astype(np.float16)

    # ── state_proj ──
    out["state_proj.weight"] = raw["state_proj.kernel"].astype(np.float16)  # (32, 1024)
    out["state_proj.bias"] = raw["state_proj.bias"].astype(np.float16)

    logger.info(f"Transformed {len(out)} engine weights from JAX format (Pi0)")
    return out


def transform_jax_weights_pi0fast(raw: dict) -> dict:
    """Transform JAX Orbax weights to engine format — Pi0-FAST variant.

    Pi0-FAST is a single Gemma 2B model (autoregressive, not diffusion).
    No separate action expert decoder — the encoder IS the full model.
    No action_in/out_proj, state_proj, or action_time_mlp.

    Vision + Encoder sections are identical to Pi0/Pi0.5.
    New: embedding table for input lookup + output logit projection.
    """
    import ml_dtypes
    raw = {k: v.astype(ml_dtypes.bfloat16).astype(np.float32) if v.dtype == np.float32 else v
           for k, v in raw.items()}

    out = {}

    # ── Vision (SigLIP) — identical to Pi0 ──
    out["vision.patch_embed.weight"] = raw["PaliGemma.img.embedding.kernel"].astype(np.float16)
    out["vision.patch_embed.bias"] = raw["PaliGemma.img.embedding.bias"].astype(np.float16)
    out["vision.pos_embed"] = raw["PaliGemma.img.pos_embedding"].squeeze(0).astype(np.float16)

    for i in range(27):
        pfx = f"vision.layer.{i}"
        out[f"{pfx}.attn_norm.weight"] = raw["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.scale"][i].astype(np.float16)
        out[f"{pfx}.attn_norm.bias"] = raw["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.bias"][i].astype(np.float16)
        out[f"{pfx}.ffn_norm.weight"] = raw["PaliGemma.img.Transformer.encoderblock.LayerNorm_1.scale"][i].astype(np.float16)
        out[f"{pfx}.ffn_norm.bias"] = raw["PaliGemma.img.Transformer.encoderblock.LayerNorm_1.bias"][i].astype(np.float16)

        q_w = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.kernel"][i]
        k_w = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.key.kernel"][i]
        v_w = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.value.kernel"][i]
        q_b = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.bias"][i]
        k_b = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.key.bias"][i]
        v_b = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.value.bias"][i]
        q_2d = q_w.reshape(1152, -1)
        k_2d = k_w.reshape(1152, -1)
        v_2d = v_w.reshape(1152, -1)
        out[f"{pfx}.qkv.weight"] = np.concatenate([q_2d, k_2d, v_2d], axis=1).T.astype(np.float16)
        out[f"{pfx}.qkv.bias"] = np.concatenate([q_b.reshape(-1), k_b.reshape(-1), v_b.reshape(-1)]).astype(np.float16)

        o_w = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.out.kernel"][i]
        out[f"{pfx}.o.weight"] = o_w.reshape(-1, 1152).T.astype(np.float16)
        out[f"{pfx}.o.bias"] = raw["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.out.bias"][i].astype(np.float16)

        out[f"{pfx}.ffn_up.weight"] = raw["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.kernel"][i].T.astype(np.float16)
        out[f"{pfx}.ffn_up.bias"] = raw["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.bias"][i].astype(np.float16)
        out[f"{pfx}.ffn_down.weight"] = raw["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.kernel"][i].T.astype(np.float16)
        out[f"{pfx}.ffn_down.bias"] = raw["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.bias"][i].astype(np.float16)

    out["vision.final_norm.weight"] = raw["PaliGemma.img.Transformer.encoder_norm.scale"].astype(np.float16)
    out["vision.final_norm.bias"] = raw["PaliGemma.img.Transformer.encoder_norm.bias"].astype(np.float16)
    out["vision.projector.weight"] = raw["PaliGemma.img.head.kernel"].T.astype(np.float16)
    out["vision.projector.bias"] = raw["PaliGemma.img.head.bias"].astype(np.float16)

    # ── Encoder / LLM (Gemma 2B, 18 layers) — identical to Pi0 encoder ──
    # This IS the full model for Pi0-FAST (no separate decoder)
    out["encoder.embedding"] = raw["PaliGemma.llm.embedder.input_embedding"].astype(np.float16)

    for i in range(18):
        pfx = f"encoder.layer.{i}"
        attn_scale = raw["PaliGemma.llm.layers.pre_attention_norm.scale"][i]
        ffn_scale = raw["PaliGemma.llm.layers.pre_ffw_norm.scale"][i]

        fuse_attn = 1.0 + attn_scale
        fuse_ffn = 1.0 + ffn_scale

        q_w = raw["PaliGemma.llm.layers.attn.q_einsum.w"][i]
        q_2d = q_w.transpose(0, 2, 1).reshape(-1, q_w.shape[1])
        q_2d = interleave_qk(q_2d, 8) * fuse_attn[None, :]

        kv_w = raw["PaliGemma.llm.layers.attn.kv_einsum.w"][i]
        k_2d = kv_w[0].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])
        v_2d = kv_w[1].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])
        k_2d = interleave_qk(k_2d, 1) * fuse_attn[None, :]
        v_2d = v_2d * fuse_attn[None, :]

        qkv = np.concatenate([q_2d, k_2d, v_2d], axis=0).astype(np.float16)
        out[f"{pfx}.qkv.weight"] = qkv

        o_w = raw["PaliGemma.llm.layers.attn.attn_vec_einsum.w"][i]
        out[f"{pfx}.o.weight"] = o_w.reshape(-1, o_w.shape[-1]).T.astype(np.float16)

        gu_w = raw["PaliGemma.llm.layers.mlp.gating_einsum"][i]
        gate_w = (gu_w[0].T * fuse_ffn[None, :]).astype(np.float16)
        up_w = (gu_w[1].T * fuse_ffn[None, :]).astype(np.float16)
        out[f"{pfx}.gate_up.weight"] = np.concatenate([gate_w, up_w], axis=0)

        out[f"{pfx}.down.weight"] = raw["PaliGemma.llm.layers.mlp.linear"][i].T.astype(np.float16)

    # ── Final norm (standard RMSNorm with weight) ──
    final_scale = raw["PaliGemma.llm.final_norm.scale"]
    out["encoder.final_norm.weight"] = (1.0 + final_scale).astype(np.float16)

    logger.info(f"Transformed {len(out)} engine weights from JAX format (Pi0-FAST)")
    return out

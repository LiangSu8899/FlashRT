"""Pure-PyTorch reference for Qwen3-VL M-RoPE.

Mirrors transformers ``Qwen3VLTextRotaryEmbedding`` + ``apply_rotary_pos_emb``
exactly, so the CUDA kernel can be validated against this oracle without
depending on the transformers package being importable at test time.

Spec sourced from transformers 4.57.1
``models/qwen3_vl/modeling_qwen3_vl.py`` lines 109-113, 278-334, 358-382.

Math (verified against HF):

    1. inv_freq[i] = 1.0 / (rope_theta ** (2i / head_dim))   for i in [0, head_dim/2)
    2. For each axis a in {T=0, H=1, W=2}:
         freqs[a, b, s, i] = position_ids[a, b, s] * inv_freq[i]
       shape: (3, B, S, head_dim/2)
    3. apply_interleaved_mrope(freqs, mrope_section=[t,h,w]) ->
         freqs_t[..., 0:3*t:3] = freqs[0, ..., 0:3*t:3]    # T-axis at offset 0, stride 3
         freqs_t[..., 1:3*h:3] = freqs[1, ..., 1:3*h:3]    # H-axis at offset 1, stride 3
         freqs_t[..., 2:3*w:3] = freqs[2, ..., 2:3*w:3]    # W-axis at offset 2, stride 3
       (All other positions in [0, head_dim/2) keep the T-axis values.)
    4. emb = cat(freqs_t, freqs_t, dim=-1)                    # shape (B, S, head_dim)
       cos = emb.cos() * attention_scaling
       sin = emb.sin() * attention_scaling
    5. rotate_half(x) = cat(-x[..., D/2:], x[..., :D/2], dim=-1)
       q_rot = q * cos + rotate_half(q) * sin                 # broadcast over heads
       k_rot = k * cos + rotate_half(k) * sin
"""

from __future__ import annotations

import torch


def compute_inv_freq(head_dim: int, rope_theta: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """1.0 / theta^(2i/D) for i in [0, D/2). Shape: (head_dim/2,)."""
    half = head_dim // 2
    return 1.0 / (rope_theta ** (torch.arange(0, half, dtype=dtype, device=device) / half))


def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Interleave the 3 per-axis frequency tensors into one.

    Args:
        freqs: shape (3, B, S, head_dim/2). [0]=T, [1]=H, [2]=W.
        mrope_section: [t_dims, h_dims, w_dims], typically [24, 20, 20].

    Returns:
        Interleaved tensor of shape (B, S, head_dim/2).
    """
    out = freqs[0].clone()
    for axis, offset in enumerate((1, 2), start=1):
        length = mrope_section[axis] * 3
        idx = slice(offset, length, 3)
        out[..., idx] = freqs[axis][..., idx]
    return out


def build_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    rope_theta: float,
    mrope_section: list[int],
    attention_scaling: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (cos, sin) tables for M-RoPE.

    Args:
        position_ids: shape (3, B, S) — T,H,W axes per token. Or (B, S), in
            which case it is broadcast to all 3 axes.
        head_dim: e.g. 128.
        rope_theta: e.g. 5_000_000.
        mrope_section: e.g. [24, 20, 20] (sum * 2 must equal head_dim/2 ... no, sum * 3 covers head_dim/2).
        attention_scaling: 1.0 for default rope_type.

    Returns:
        cos, sin: each shape (B, S, head_dim).
    """
    if position_ids.ndim == 2:
        position_ids = position_ids[None].expand(3, -1, -1)
    assert position_ids.shape[0] == 3, "position_ids must have a leading axis of 3 (T,H,W)"

    inv_freq = compute_inv_freq(head_dim, rope_theta, device=position_ids.device, dtype=dtype)
    # freqs: (3, B, S, head_dim/2) = (3, B, 1, head_dim/2) * (3, B, S, 1)
    freqs = inv_freq[None, None, None, :] * position_ids[..., None].to(dtype)
    freqs_t = apply_interleaved_mrope(freqs, mrope_section)
    emb = torch.cat([freqs_t, freqs_t], dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos.to(dtype), sin.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard RoPE rotate_half: cat(-x[D/2:], x[:D/2], dim=-1)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply M-RoPE to q,k.

    Args:
        q: (B, NHQ, S, head_dim).
        k: (B, NHKV, S, head_dim).
        cos, sin: (B, S, head_dim) — already-interleaved per ``build_cos_sin``.
        unsqueeze_dim: 1 (broadcast cos/sin along the heads axis).

    Returns:
        q_rot, k_rot: same shapes as inputs.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def mrope_full(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int = 128,
    rope_theta: float = 5_000_000.0,
    mrope_section: list[int] | None = None,
    attention_scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-shot helper: build (cos, sin) and apply to (q, k)."""
    if mrope_section is None:
        mrope_section = [24, 20, 20]
    cos, sin = build_cos_sin(position_ids, head_dim, rope_theta, mrope_section, attention_scaling, dtype=q.dtype)
    return apply_rotary_pos_emb(q, k, cos, sin)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, NHQ, NHKV, S, HD = 1, 16, 8, 64, 128
    q = torch.randn(B, NHQ, S, HD, dtype=torch.float32)
    k = torch.randn(B, NHKV, S, HD, dtype=torch.float32)
    pos = torch.stack([
        torch.arange(S),
        torch.arange(S),
        torch.arange(S),
    ])[None].expand(3, B, -1)
    q_rot, k_rot = mrope_full(q, k, pos)
    assert q_rot.shape == q.shape, q_rot.shape
    assert k_rot.shape == k.shape, k_rot.shape
    print("mrope_ref smoke OK", q_rot.norm().item(), k_rot.norm().item())

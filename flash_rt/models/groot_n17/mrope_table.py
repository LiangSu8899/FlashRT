"""Host-side M-RoPE cos/sin table construction for GROOT N1.7.

Built once per ``set_prompt`` (we know the prompt + vision token layout at
that point). Stored as a device tensor and reused as input to the existing
Qwen3 RoPE kernel during every replay.

This module is pure-PyTorch (CPU/GPU agnostic). It does NOT call any
flash_rt_kernels symbol — the kernel consumes the tensors produced here
unchanged.

Math source: ``transformers.models.qwen3_vl.modeling_qwen3_vl``
* ``Qwen3VLTextRotaryEmbedding``        (cos/sin construction)
* ``Qwen3VLModel.get_rope_index``       (3-axis position_ids derivation)

Both have been replicated in ``tests/_helpers/groot_n17/mrope_ref.py`` and
validated bit-exact against HF (cos=1.0, max_diff=0.0 on fp32 random Q/K).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RopeConfig:
    """Qwen3-VL M-RoPE constants (locked by the N1.7 ckpt config)."""

    head_dim: int = 128
    rope_theta: float = 5_000_000.0
    mrope_section: tuple[int, int, int] = (24, 20, 20)  # T, H, W axes
    attention_scaling: float = 1.0
    spatial_merge_size: int = 2  # Qwen3-VL ViT spatial merge


def compute_inv_freq(cfg: RopeConfig, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """1.0 / theta^(2i/D). Shape: (head_dim/2,)."""
    half = cfg.head_dim // 2
    return 1.0 / (cfg.rope_theta ** (torch.arange(0, half, dtype=dtype, device=device) / half))


def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section) -> torch.Tensor:
    """Interleave the 3 per-axis frequency tensors into one.

    Args:
        freqs: shape ``(3, B, S, head_dim/2)``. ``[0]=T, [1]=H, [2]=W``.
        mrope_section: ``[t_dims, h_dims, w_dims]``.

    Returns:
        Interleaved tensor of shape ``(B, S, head_dim/2)``.
    """
    out = freqs[0].clone()
    for axis, offset in enumerate((1, 2), start=1):
        length = mrope_section[axis] * 3
        idx = slice(offset, length, 3)
        out[..., idx] = freqs[axis][..., idx]
    return out


def build_position_ids_for_segments(
    *,
    segment_lengths: list[int],
    segment_kinds: list[str],
    segment_grids: list[tuple[int, int, int] | None],
    cfg: RopeConfig,
    device=None,
) -> torch.Tensor:
    """Replicate ``Qwen3VLModel.get_rope_index`` for a flat token sequence
    composed of named segments.

    Args:
        segment_lengths: number of tokens in each segment.
        segment_kinds:   ``"text"`` or ``"image"``. Lengths must equal
                         segment_lengths.
        segment_grids:   for image segments, ``(t, h, w)`` patch grid (raw,
                         pre-spatial-merge); for text segments, ``None``.
        cfg:             RopeConfig.
        device:          device for the output tensor.

    Returns:
        position_ids of shape ``(3, 1, S)`` where ``S = sum(segment_lengths)``.
    """
    assert len(segment_lengths) == len(segment_kinds) == len(segment_grids), (
        "segment_* must align"
    )

    pieces: list[torch.Tensor] = []
    st_idx = 0
    for L, kind, grid in zip(segment_lengths, segment_kinds, segment_grids):
        if kind == "text":
            # Text: identical sequential indices on all 3 axes.
            arange = torch.arange(L, dtype=torch.long, device=device)
            pieces.append(arange.view(1, -1).expand(3, -1) + st_idx)
            st_idx += L
        elif kind == "image":
            assert grid is not None, "image segment must have a (t,h,w) grid"
            t, h, w = grid
            llm_grid_t = t
            llm_grid_h = h // cfg.spatial_merge_size
            llm_grid_w = w // cfg.spatial_merge_size
            assert llm_grid_t * llm_grid_h * llm_grid_w == L, (
                f"image segment length {L} does not match grid "
                f"({llm_grid_t}, {llm_grid_h}, {llm_grid_w}) post-spatial-merge"
            )
            t_index = (
                torch.arange(llm_grid_t, dtype=torch.long, device=device)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h, dtype=torch.long, device=device)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w, dtype=torch.long, device=device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            pieces.append(torch.stack([t_index, h_index, w_index]) + st_idx)
            st_idx += max(llm_grid_t, llm_grid_h, llm_grid_w)
        else:
            raise ValueError(f"unknown segment kind: {kind!r}")

    pos = torch.cat(pieces, dim=-1)  # (3, S)
    return pos.unsqueeze(1)  # (3, 1, S)


def build_cos_sin_tables(
    position_ids: torch.Tensor,
    cfg: RopeConfig,
    *,
    dtype: torch.dtype = torch.float16,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (cos, sin) of shape ``(S, head_dim)`` for the M-RoPE-rotated path.

    Args:
        position_ids: ``(3, 1, S)`` (T, H, W axes per token; B=1 inference).
        cfg:          RopeConfig.
        dtype:        output dtype (default fp16, matches kernel input).
        device:       output device.

    Returns:
        ``cos``, ``sin`` tensors of shape ``(S, head_dim)``.
        Layout is **split-half** (matches the existing
        ``rope_rotate_half_fp16`` kernel: cos[d] applies to dim ``d``, the
        first ``head_dim/2`` of ``cos`` is the interleaved T/H/W block, and
        the second ``head_dim/2`` is a duplicate — exactly as HF Qwen3VL).
    """
    if position_ids.ndim == 2:
        position_ids = position_ids.unsqueeze(1)
    assert position_ids.shape[0] == 3, "position_ids must have leading T/H/W axis of 3"

    inv_freq = compute_inv_freq(cfg, device=position_ids.device, dtype=torch.float32)
    # freqs: (3, B, S, head_dim/2)
    freqs = inv_freq[None, None, None, :] * position_ids[..., None].to(torch.float32)
    freqs_t = apply_interleaved_mrope(freqs, cfg.mrope_section)
    emb = torch.cat([freqs_t, freqs_t], dim=-1)
    cos = (emb.cos() * cfg.attention_scaling).to(dtype)
    sin = (emb.sin() * cfg.attention_scaling).to(dtype)
    if device is not None:
        cos = cos.to(device)
        sin = sin.to(device)
    # Squeeze batch dim (we know B=1 at inference).
    return cos.squeeze(0), sin.squeeze(0)


__all__ = [
    "RopeConfig",
    "build_position_ids_for_segments",
    "build_cos_sin_tables",
    "apply_interleaved_mrope",
    "compute_inv_freq",
]

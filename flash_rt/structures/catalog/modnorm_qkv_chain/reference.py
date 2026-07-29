"""Plain PyTorch reference for the modulated-norm to QKV chain."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def modnorm_qkv_chain_ref(
    x: torch.Tensor,
    cond: torch.Tensor,
    w_cond: torch.Tensor,
    b_cond: torch.Tensor,
    w_q: torch.Tensor,
    b_q: torch.Tensor,
    w_k: torch.Tensor,
    b_k: torch.Tensor,
    w_v: torch.Tensor,
    b_v: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm, two-way modulation, then three sibling projections."""
    scale, shift = F.linear(F.silu(cond), w_cond, b_cond).chunk(2, dim=-1)
    normed = F.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype)
    normed = normed * (1 + scale[:, None]) + shift[:, None]
    return (
        F.linear(normed, w_q, b_q),
        F.linear(normed, w_k, b_k),
        F.linear(normed, w_v, b_v),
    )

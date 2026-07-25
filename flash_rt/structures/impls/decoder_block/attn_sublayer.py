"""The attention sublayer with no layout churn between its parts.

Three bound structures already sit inside a host attention module: the
packed projections, the rotary embedding, the fused attention core. Each
is faster than what it replaced, and between them the host still pays for
a layout it does not need. The host lays q/k/v out as ``(B, H, S, D)``
because that is what eager SDPA wants; the fused kernel wants
``(B, S, H, D)``, which is exactly what the projections' own output view
already is. So the host transposes, the rotary embedding runs on the
transposed layout, and the core transposes back and makes it
contiguous — two cancelling transposes plus the copies around them, per
projection, per layer, per step.

None of the three seams can see that, because each is bound inside the
module that owns the layout. The sublayer boundary can: run the packed
projections, view their output as ``(B, S, H, D)``, apply the rotary
embedding on that layout (the ``unsqueeze`` axis moves, the arithmetic
does not), and hand it straight to the kernel.

The rotary form is family-specific (half-split against interleaved), so
it enters as a callable from the host-family adapter; everything else
here is the generic pre-norm attention sublayer.
"""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half-split rotation (Llama/Gemma/Qwen convention)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class PackedAttnSublayer(nn.Module):
    """Packed projections -> rotary -> fused core -> output projection."""

    def __init__(self, attn: nn.Module, core, *, scale: float,
                 rotate=rotate_half):
        super().__init__()
        self.attn = attn
        self.core = core
        self.scale = scale
        self.rotate = rotate

    def forward(self, x: torch.Tensor, position_embeddings=None, **kw):
        a = self.attn
        bsz, seq, _ = x.shape
        hd = a.head_dim
        # the host's own call order is the data dependency the packed
        # projection relies on: the first call runs the GEMM, the others
        # read its stash
        q = a.q_proj(x).view(bsz, seq, -1, hd)
        k = a.k_proj(x).view(bsz, seq, -1, hd)
        v = a.v_proj(x).view(bsz, seq, -1, hd)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = cos.unsqueeze(2).to(q.dtype)
            sin = sin.unsqueeze(2).to(q.dtype)
            q = q * cos + self.rotate(q) * sin
            k = k * cos + self.rotate(k) * sin
        out = self.core.forward_suffix(q, k, v, scale=self.scale)
        return a.o_proj(out.reshape(bsz, seq, -1))


def bind_attn_sublayer(attn: nn.Module, core, *, rotate=rotate_half):
    """Compose one attention sublayer around an already-bound core.

    Returns ``None`` rather than raising when the host module or the core
    is missing a part: the block then keeps the host's own attention, so
    this can only add coverage.
    """
    if core is None or not hasattr(core, "forward_suffix"):
        return None
    for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim"):
        if not hasattr(attn, attr):
            return None
    # the sublayer's projections produce the new tokens only, so the
    # packed plan's suffix has to be exactly those. A host that carries
    # its own KV cache into attention breaks that equality, and there the
    # host's attention module stays.
    plan = getattr(core, "plan", None)
    if plan is None or plan.suffix_len != getattr(core, "seq_q", -1):
        return None
    scale = getattr(attn, "scaling", None)
    if scale is None:
        scale = getattr(attn, "scale", None)
    if scale is None:
        return None
    return PackedAttnSublayer(attn, core, scale=float(scale),
                              rotate=rotate)

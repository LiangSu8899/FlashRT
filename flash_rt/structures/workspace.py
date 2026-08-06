"""Shared workspaces: seat scratch lives in a pool, not in every seat.

A pack's sibling stash, a producer's residual scratch, a wire's packed
buffer — their lifetimes all sit inside one layer's forward. Sequential
layers therefore never need their own copies: every seat that asks for
the same (shape, dtype, device, tag) receives the *same* tensor, and
the pool's footprint is one layer's worth instead of layers x tokens.
Capture-compatible by the same argument that makes memory pools
capture-compatible: the graph records fixed pointers, and same-stream
sequential lifetimes never overlap.

Two lease kinds:
- ``scratch``: contents are call-transient; the seat must write before
  it reads (packs and producers already do).
- ``ones``: constant-filled; shared freely and never written.

The pool is also the accounting surface: :func:`report` returns bytes
held and the reuse count per tag — the memory column of the receipt.
"""

from __future__ import annotations

import torch

_POOL: dict[tuple, torch.Tensor] = {}
_LEASES: dict[str, int] = {}


def lease(shape, dtype, device, *, tag: str,
          fill: str = "scratch") -> torch.Tensor:
    """One shared tensor for every seat asking this (shape, tag)."""
    key = (tuple(shape), dtype, str(device), tag, fill)
    buf = _POOL.get(key)
    if buf is None:
        buf = torch.zeros(*shape, dtype=dtype, device=device)
        if fill == "ones":
            buf.fill_(1)
        _POOL[key] = buf
    _LEASES[tag] = _LEASES.get(tag, 0) + 1
    return buf


def report() -> dict:
    """Bytes held and lease counts — the receipt's memory column."""
    by_tag: dict[str, int] = {}
    for (shape, dtype, _dev, tag, _fill), buf in _POOL.items():
        by_tag[tag] = by_tag.get(tag, 0) + buf.numel() * buf.element_size()
    return {"held_bytes": sum(by_tag.values()), "by_tag": by_tag,
            "leases": dict(_LEASES)}


def clear() -> None:
    """Drop every pooled buffer (between hosts, or in tests)."""
    _POOL.clear()
    _LEASES.clear()

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

import weakref

import torch

_POOL: dict[tuple, torch.Tensor] = {}
_LEASES: dict[str, int] = {}
# exclusive leases are owned by their seats, not the pool; the receipt
# tracks them through weak references so bytes leave the ledger the
# moment a detached seat releases its buffer
_EXCLUSIVE: dict[str, list] = {}


def lease(shape, dtype, device, *, tag: str,
          fill: str = "scratch",
          exclusive: bool = False) -> torch.Tensor:
    """One shared tensor for every seat asking this (shape, tag).

    The pool's safety argument is single-stream sequential execution:
    layer i's scratch is dead before layer i+1 writes it *because they
    share one stream*. A caller leasing from a non-default stream is
    outside that argument, and the pool refuses rather than corrupt
    silently — multi-stream serving needs per-lane isolation first.

    ``exclusive=True`` allocates a private tensor instead of joining
    the shared slab, still counted in the pool's receipt. It exists
    for *state*, not scratch: a buffer whose consumer is the host may
    be retained past the tick (a KV cache holding a reader's view was
    the measured failure — every layer's write clobbered every other
    layer's cached slice, with every ledger clean). Sharing such a
    buffer requires the immediacy of consumption as a verified fact;
    absent that fact, state is exclusive.
    """
    if (torch.cuda.is_available()
            and str(device).startswith("cuda")
            and torch.cuda.current_stream()
            != torch.cuda.default_stream()):
        raise RuntimeError(
            "workspace: leasing from a non-default CUDA stream — the "
            "shared pool's lifetime argument only covers single-stream "
            "sequential execution; isolate per-stream lanes first")
    if exclusive:
        buf = torch.zeros(*shape, dtype=dtype, device=device)
        if fill == "ones":
            buf.fill_(1)
        _LEASES[tag] = _LEASES.get(tag, 0) + 1
        _EXCLUSIVE.setdefault(tag, []).append(weakref.ref(buf))
        return buf
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
    """Bytes held and lease counts — the receipt's memory column.

    ``held_bytes`` covers both the shared pool and every live exclusive
    lease; ``exclusive_by_tag`` breaks the latter out so the receipt
    shows what stayed private and what the pool actually deduplicated.
    """
    by_tag: dict[str, int] = {}
    for (shape, dtype, _dev, tag, _fill), buf in _POOL.items():
        by_tag[tag] = by_tag.get(tag, 0) + buf.numel() * buf.element_size()
    exclusive_by_tag: dict[str, int] = {}
    for tag, refs in _EXCLUSIVE.items():
        live = [ref() for ref in refs]
        refs[:] = [ref for ref, buf in zip(refs, live) if buf is not None]
        held = sum(buf.numel() * buf.element_size()
                   for buf in live if buf is not None)
        if held:
            exclusive_by_tag[tag] = held
    return {"held_bytes": sum(by_tag.values())
            + sum(exclusive_by_tag.values()),
            "by_tag": by_tag,
            "exclusive_by_tag": exclusive_by_tag,
            "leases": dict(_LEASES)}


def clear() -> None:
    """Drop every pooled buffer (between hosts, or in tests)."""
    _POOL.clear()
    _LEASES.clear()
    _EXCLUSIVE.clear()

"""Transactional module attachment — mechanism only, no policy.

Swaps host modules for bound structure implementations atomically: every
staged swap is applied only if all of them can be applied, and a handle
restores the originals exactly. Which modules to swap, and whether an
implementation passed its gates, is decided by the caller before
staging; this layer refuses partial application by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


def resolve_parent(root: torch.nn.Module, path: str) -> tuple[torch.nn.Module, str]:
    """Resolve the parent module and attribute name for a dotted path."""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    attr = parts[-1]
    leaf = parent[int(attr)] if attr.isdigit() else getattr(parent, attr)
    if not isinstance(leaf, torch.nn.Module):
        raise TypeError(f"{path!r} does not resolve to a torch.nn.Module")
    return parent, attr


@dataclass
class AttachHandle:
    """Restores the originals of one committed attachment."""

    _entries: list[tuple[torch.nn.Module, str, torch.nn.Module]]
    records: Mapping[str, Any] = field(default_factory=dict)
    active: bool = True

    def detach(self) -> None:
        """Restore every swapped module. Idempotent."""
        if not self.active:
            return
        for parent, attr, original in reversed(self._entries):
            setattr(parent, attr, original)
        self.active = False


def attach(
    root: torch.nn.Module,
    swaps: Mapping[str, torch.nn.Module],
    *,
    records: Mapping[str, Any] | None = None,
) -> AttachHandle:
    """Atomically replace the modules at ``swaps`` paths.

    All paths are resolved and validated before the first swap is
    applied; any resolution failure leaves the model untouched. Callers
    pass the qualification ``records`` that justified the attachment so
    the handle carries its own evidence.
    """
    if not swaps:
        raise ValueError("no swaps staged")
    staged: list[tuple[torch.nn.Module, str, torch.nn.Module, torch.nn.Module]] = []
    for path, replacement in swaps.items():
        parent, attr = resolve_parent(root, path)
        staged.append((parent, attr, getattr(parent, attr), replacement))

    entries = []
    try:
        for parent, attr, original, replacement in staged:
            setattr(parent, attr, replacement)
            entries.append((parent, attr, original))
    except Exception:
        for parent, attr, original in reversed(entries):
            setattr(parent, attr, original)
        raise
    return AttachHandle(_entries=entries, records=dict(records or {}))

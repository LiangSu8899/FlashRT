"""Transactional module attachment — mechanism only, no policy.

Swaps host modules for bound structure implementations atomically: every
staged swap is applied only if all of them can be applied, and a handle
restores the originals exactly. Which modules to swap, and whether an
implementation passed its gates, is decided by the caller before
staging; this layer refuses partial application by construction.

The handle is also where the attachment answers for itself at runtime.
Every swapped-in structure carries a guard recording the form it was
calibrated for (:mod:`flash_rt.structures.guard`); attaching gives each
guard its path and a way to restore the host module, and
``handle.report()`` reads them back. An attachment whose report shows
fallbacks is one that is not running the structures it claims to — that
has to be visible here, because nothing downstream can tell the
difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from .guard import GUARD_ATTR, SeamGuard, collect as _collect_guards


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


def _get(parent: torch.nn.Module, attr: str) -> torch.nn.Module:
    return parent[int(attr)] if attr.isdigit() else getattr(parent, attr)


def _set(parent: torch.nn.Module, attr: str,
         module: torch.nn.Module) -> None:
    if attr.isdigit():
        parent[int(attr)] = module
    else:
        setattr(parent, attr, module)


@dataclass
class AttachHandle:
    """Restores the originals of one committed attachment."""

    _entries: list[tuple[torch.nn.Module, str, torch.nn.Module]]
    records: Mapping[str, Any] = field(default_factory=dict)
    active: bool = True
    _guards: dict[str, SeamGuard] = field(default_factory=dict)
    _revert: list[Any] = field(default_factory=list)

    def detach(self) -> None:
        """Restore every swapped module. Idempotent.

        Also runs any ``revert`` callables the caller handed over. Some
        seams are not modules at paths — an adapter that patches a
        library-level function — and restoring only what ``setattr`` can
        reach would leave those live while reporting the model as restored.
        """
        if not self.active:
            return
        for parent, attr, original in reversed(self._entries):
            _set(parent, attr, original)
        for undo in reversed(self._revert):
            undo()
        self._revert.clear()
        for guard in self._guards.values():
            guard.release_site()
        self.active = False

    # ---- what actually ran -----------------------------------------

    def report(self) -> dict[str, dict[str, Any]]:
        """Per-seam ledger: calls, fallbacks, and why.

        The unit of truth for "did this attachment run what it says". A
        seam with ``fallbacks`` above zero ran the host module for that
        many calls; one with ``detached`` set gave up and put the host
        module back. Tests assert on this rather than assuming it.

        Counts are eager-only, and necessarily so: inside a compiled or
        captured region the kernel runs without re-entering Python, so
        there is nothing left to count. A captured host therefore reports
        zero calls for the seams inside its graph — read that as "this
        ledger does not cover the graph", not as "the seam did not run".
        The graph's own parity check is what covers it.
        """
        return {site: guard.entry()
                for site, guard in sorted(self._guards.items())}

    def summary(self) -> dict[str, Any]:
        """The report reduced to what a caller has to act on.

        ``seams_never_called`` means "no eager call reached this seam",
        which for a captured host is every seam in the graph. It is a
        finding on an eager host and noise on a captured one.
        """
        entries = self.report()
        fell_back = sorted(site for site, e in entries.items()
                           if e["fallbacks"])
        return {
            "seams": len(entries),
            "guarded_calls": sum(e["calls"] for e in entries.values()),
            "fallbacks": sum(e["fallbacks"] for e in entries.values()),
            "seams_fell_back": fell_back,
            "seams_self_detached": sorted(
                site for site, e in entries.items() if e["detached"]),
            "seams_never_called": sorted(
                site for site, e in entries.items() if not e["calls"]),
            "clean": not fell_back,
        }

    def raise_on_fallback(self) -> None:
        """Assertion helper: fail loudly if any seam fell back.

        For tests, and for callers who would rather not ship a run whose
        seams quietly reverted. Reads the ledger, so it costs nothing in
        the hot path.
        """
        summary = self.summary()
        if summary["clean"]:
            return
        detail = {site: entry["last_reason"]
                  for site, entry in self.report().items()
                  if entry["fallbacks"]}
        raise RuntimeError(
            f"{len(detail)} seam(s) fell back to the host module during "
            f"this attachment ({summary['fallbacks']} call(s) total): "
            f"{detail}")


def attach(
    root: torch.nn.Module,
    swaps: Mapping[str, torch.nn.Module],
    *,
    records: Mapping[str, Any] | None = None,
    on_guard_fail: str = "fallback",
    allow_training: bool = False,
    observe: Mapping[str, torch.nn.Module] | None = None,
    revert: Any = None,
) -> AttachHandle:
    """Atomically replace the modules at ``swaps`` paths.

    All paths are resolved and validated before the first swap is
    applied; any resolution failure leaves the model untouched. Callers
    pass the qualification ``records`` that justified the attachment so
    the handle carries its own evidence.

    ``on_guard_fail`` selects what a seam does when it is called outside
    the form it was bound for: ``"fallback"`` runs the host module and
    counts it in the ledger, ``"raise"`` refuses immediately. Development
    and CI should use ``"raise"`` — a fallback a test tolerates is a
    fallback nobody reads.

    Attaching to a model in training mode is refused: these
    implementations pack and quantise the host weights once at bind time,
    so a gradient step would update the host copy and leave the kernel's
    stale. Pass ``allow_training=True`` for a forward-only pass that
    happens to sit inside a training script.

    ``observe`` names modules that carry a guard but are not swapped here
    — an adapter's routed seam. They are reported in the ledger and never
    installed. ``revert`` collects undo callables for host mutations made
    before this call, so ``detach`` restores those too.
    """
    if not swaps:
        raise ValueError("no swaps staged")
    if on_guard_fail not in ("fallback", "raise"):
        raise ValueError(
            f"on_guard_fail must be 'fallback' or 'raise', "
            f"got {on_guard_fail!r}")
    if root.training and not allow_training:
        raise ValueError(
            "structures.attach: the model is in training mode. These "
            "implementations pack and quantise the host weights once at "
            "bind time, so training through them would update the host "
            "copy and leave the kernel's stale. Call model.eval() first, "
            "or pass allow_training=True for a forward-only pass.")

    staged: list[tuple[torch.nn.Module, str, torch.nn.Module,
                       torch.nn.Module]] = []
    for path, replacement in swaps.items():
        parent, attr = resolve_parent(root, path)
        staged.append((parent, attr, _get(parent, attr), replacement))

    entries: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []
    try:
        for parent, attr, original, replacement in staged:
            _set(parent, attr, replacement)
            entries.append((parent, attr, original))
    except Exception:
        for parent, attr, original in reversed(entries):
            _set(parent, attr, original)
        raise

    # ---- give every guard its site and its own way out ----
    # keyed by guard identity as well as by site: one module can be
    # reachable by two routes (a composed block holds the host block, so a
    # core hanging off the host's attention is found twice), and counting
    # it twice would overstate how many seams an attachment has
    guards: dict[str, SeamGuard] = {}
    seen: set[int] = set()
    # named seams first: an adapter knows what its routed seam should be
    # called, and the same object found by walking a composed structure
    # would otherwise claim it under an incidental path
    for site, module in (observe or {}).items():
        for child, guard in _collect_guards(module):
            if id(guard) in seen:
                continue
            seen.add(id(guard))
            key = site if not child else f"{site}::{child}"
            guard.bind_site(key, restore=None, mode=on_guard_fail)
            guards[key] = guard
    for (parent, attr, original), (path, replacement) in zip(
            entries, swaps.items()):
        for child, guard in _collect_guards(replacement):
            if id(guard) in seen:
                continue
            seen.add(id(guard))
            site = path if not child else f"{path}::{child}"
            # only the module actually swapped in at a path can restore
            # itself; a guard held inside a composed structure reports but
            # cannot exit on its own
            restore = ((lambda p=parent, a=attr, o=original: _set(p, a, o))
                       if not child else None)
            guard.bind_site(site, restore=restore, mode=on_guard_fail)
            guards[site] = guard

    return AttachHandle(_entries=entries, records=dict(records or {}),
                        _guards=guards, _revert=list(revert or ()))


__all__ = ["AttachHandle", "attach", "resolve_parent", "GUARD_ATTR"]

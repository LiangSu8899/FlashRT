"""Region adjudication: structure-level winners are receipts, not order.

A region is a span of the host larger than one seat — a DiT block's
fused chain versus its seat-by-seat composition, a single-stream
attention takeover versus factored routes. Hardware wants different
shapes here: the same pipeline's fastest form on one device is a fused
launch chain, on another the seated composition. Until now the winner
at this level was whichever adapter registered first — a global
decision nobody signed. This module gives regions the pipeline the
precision bands already have: candidates declare factual
prerequisites, a production-form measurement run records the winner
per (device, region), and every tier consumes the receipt.

The tier discipline this module enforces:

- **Automatic reads receipts only** — author pin, then the decision
  cache, then the seated floor. It never experiments at bind: a cold
  box runs seated (correct, possibly not full speed) until a
  measurement run records a winner. A receipt that names a candidate
  this box cannot qualify falls through to seated with the reason on
  the trail — a stale or foreign receipt degrades speed, never
  correctness.
- **Explicit is maximum host replacement** — it pins winners at the
  region key and claims regions discovery refuses; it consumes the
  same candidate set, so anything it proves the automatic tier can
  later inherit through the cache.
- **Every candidate is assembled from structure primitives** — seats,
  producers, workspace leases, guards. A form that replaces a forward
  wholesale with hand-written code has no seat here: it cannot carry
  the ledger, the fallback contract, or the revert path, so there is
  nothing for a receipt to certify.

Measurement itself is not in this module on purpose. A region winner
is a captured-form end-to-end number (seat-level micro-timing was
refuted in both directions); the measuring harness lives with the
family that owns the region and writes its result through
:func:`record`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from . import decisions

#: the floor candidate, always in play: the region stays with the seat
#: scan's per-seat composition. Every resolution can land here, so the
#: worst case of a wrong, stale, or missing receipt is today's
#: behavior, never a crash and never a half-bound region.
SEATED = "seated"


@dataclass
class RegionCandidate:
    """One structural form a region can take, with its prerequisites.

    ``missing`` reports the factual prerequisites this box does not
    meet — hub symbols absent, shapes outside the form's band, a
    memory plan the allocator refuses. An empty report qualifies the
    candidate; anything else disqualifies it and the report lands on
    the trail verbatim. ``bind`` routes the form onto one region
    occurrence and returns the adapter-result dict ({observed, revert,
    ...}); it runs under the attach transaction like any adapter.
    """

    name: str
    missing: Callable[[], Sequence[str]] = lambda: ()
    bind: Callable[..., Any] | None = None
    #: tie-break inside a measurement's win margin: lower = higher
    #: precision (same convention as form adjudication)
    precision_rank: int = 0


@dataclass
class RegionFamily:
    """A region kind: how to find its occurrences, and its forms.

    ``identify`` matches structural signatures on the module graph
    (shape of the subtree, never class or model names) and returns the
    qualified names of the region roots it claims.
    """

    family: str
    identify: Callable[[Any], Sequence[str]]
    candidates: list[RegionCandidate] = field(default_factory=list)

    def candidate(self, name: str) -> RegionCandidate | None:
        for cand in self.candidates:
            if cand.name == name:
                return cand
        return None


_FAMILIES: dict[str, RegionFamily] = {}


def register_region_family(fam: RegionFamily) -> None:
    """Register (or replace) the family owning a region kind."""
    _FAMILIES[fam.family] = fam


def family(name: str) -> RegionFamily:
    return _FAMILIES[name]


def registered() -> tuple[RegionFamily, ...]:
    return tuple(_FAMILIES.values())


def _pin_env(fam: str) -> str:
    return "FRT_REGION_" + re.sub(r"[^A-Za-z0-9]", "_", fam).upper()


def resolve(family_name: str, *,
            notes: dict | None = None) -> tuple[str, str]:
    """Which form does this box run for this region kind, and why.

    Author pin > decision cache > seated, with one discipline on top:
    a pinned or cached name must be a known candidate that qualifies
    on this box *right now* — otherwise it falls through, the reason
    lands on the trail, and the next source is consulted. The return
    is ``(winner, source)`` where source is one of ``pin``, ``cache``,
    ``default``; the full fall-through trail goes to
    ``notes["regions"]``.
    """
    fam = _FAMILIES[family_name]
    fell_through: list[dict] = []
    winner, source = SEATED, "default"
    pin = os.environ.get(_pin_env(family_name))
    cached = decisions.lookup(f"region:{family_name}")
    for src, name in (("pin", pin), ("cache", cached)):
        if not name:
            continue
        if name == SEATED:
            winner, source = SEATED, src
            break
        cand = fam.candidate(name)
        if cand is None:
            fell_through.append(
                {"source": src, "name": name,
                 "reason": "unknown_candidate"})
            continue
        gaps = list(cand.missing())
        if gaps:
            fell_through.append(
                {"source": src, "name": name,
                 "reason": f"missing: {', '.join(map(str, gaps))}"})
            continue
        winner, source = name, src
        break
    if notes is not None:
        notes.setdefault("regions", []).append(
            {"family": family_name, "winner": winner, "source": source,
             "fell_through": fell_through})
    return winner, source


def record(family_name: str, winner: str, times_ms: dict):
    """Write a measured region winner into the decision cache.

    The receipt is what the automatic tier will obey unquestioningly,
    so the door is strict: the family must be registered and the
    winner must be ``seated`` or one of its declared candidates — a
    typo'd measurement run must fail here, at the writer, not poison
    every later bind at the reader.
    """
    fam = _FAMILIES[family_name]
    if winner != SEATED and fam.candidate(winner) is None:
        raise ValueError(
            f"'{winner}' is not a candidate of region family "
            f"'{family_name}' (have: "
            f"{[c.name for c in fam.candidates]} + '{SEATED}')")
    return decisions.record(f"region:{family_name}", winner, times_ms)

"""Quantisation schemes: how statistics become per-seam decisions.

A scheme owns exactly two questions and nothing else:

1. **What statistic does each calibration point need?**
   (:meth:`QuantScheme.statistics`) — amax for FP8-style static scales, a
   per-channel second moment for imatrix-style weight quantisation, or
   ``None`` for formats that quantise dynamically in-kernel and need no
   calibration at that point (this repo's NVFP4 activation path computes
   per-block scale factors at runtime). The statistic *discipline* is not
   the scheme's to change: per-sample reduction then a cross-sample
   percentile, one vector held per sample, never activations.

2. **Given the reduced statistics, what happens at each seam?**
   (:meth:`QuantScheme.decide`) — bind with these values, or keep the
   host module ("this layer stays at host precision" is a decision, not
   a failure).

What a scheme does **not** own: bytes. Scale-factor memory layouts,
sub-normal handling in packed formats, kernel selection, M-dispatch
tables — all execution detail, owned by the impl variant that consumes
the decision. The same decision can be executed by different kernels;
that boundary is what keeps schemes portable across backends.

Schemes are registered by name and selected at the door::

    structures.auto_swaps(model, forward, scheme="fp8_static")

Registering a scheme adds no calibration entry point: the calibration
axis (``forward`` / ``samples``) is fixed, and a scheme only declares
what to measure along it and consumes the result.
"""

from __future__ import annotations

import statistics as _stats
from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = ["PointStat", "Decision", "QuantScheme", "Fp8Static",
           "register", "get", "names", "validate_request"]

#: statistics the collector can currently execute. Granularities other
#: than per-tensor (per-channel, per-block16) are part of the declared
#: interface — NVFP4 weight scale factors are per-16-block, imatrix is
#: per-channel — but the collector does not measure them yet, so a
#: scheme requesting one fails loudly at plan time instead of silently
#: getting per-tensor numbers with the wrong shape.
_EXECUTABLE = {("amax", "tensor"), (None, "tensor")}


@dataclass(frozen=True)
class PointStat:
    """What one calibration point should measure.

    ``stat`` is ``"amax"`` (this repo's static-scale statistic),
    ``"second_moment"``, ``"histogram"``, or ``None`` — ``None`` means
    the format quantises this point dynamically at runtime and wants no
    calibration data at all. ``granularity`` is ``"tensor"``,
    ``"channel"`` or ``"block16"``.
    """

    stat: str | None = "amax"
    granularity: str = "tensor"


@dataclass
class Decision:
    """What :meth:`QuantScheme.decide` hands back.

    ``keep_host`` lists seam paths that stay on the host module at host
    precision — a first-class outcome, recorded in the plan notes, not a
    refusal. ``reasons`` says why, per path, so the receipt can print it.
    """

    keep_host: tuple[str, ...] = ()
    reasons: Mapping[str, str] = field(default_factory=dict)


class QuantScheme:
    """Base scheme: amax everywhere, bind everything.

    Subclass and override the two methods; do not add entry points.
    """

    name = "base"

    def statistics(self, points: Sequence) -> dict[str, PointStat]:
        """Per point key (``"path|name"``): what to measure there."""
        return {f"{p.path}|{p.name}": PointStat() for p in points}

    def decide(self, report: Mapping[str, Mapping[str, float]]) -> Decision:
        """``report`` is per seam path: its points' reduced statistics."""
        return Decision()


class Fp8Static(QuantScheme):
    """The default: static per-tensor FP8, exactly the shipped behaviour.

    ``keep_outliers`` turns the house scale-ceiling diagnostic into a
    decision: seams owning a point whose reduced amax sits more than
    ``keep_outliers`` times above the median of all points stay at host
    precision. The criterion is the one ``check_scale_ceiling`` already
    warns with (20.0 there); this consumes it instead of only saying it.
    ``None`` (the default) keeps nothing and binds identically to the
    behaviour before schemes existed.
    """

    name = "fp8_static"

    def __init__(self, keep_outliers: float | None = None) -> None:
        self.keep_outliers = keep_outliers

    def decide(self, report: Mapping[str, Mapping[str, float]]) -> Decision:
        if not self.keep_outliers or not report:
            return Decision()
        values = [v for pts in report.values() for v in pts.values()
                  if v is not None and v > 0]
        if not values:
            return Decision()
        median = _stats.median(values)
        keep, reasons = [], {}
        for seam_path, pts in report.items():
            worst = max(((k, v) for k, v in pts.items() if v is not None),
                        key=lambda kv: kv[1], default=None)
            if worst is not None and worst[1] > self.keep_outliers * median:
                keep.append(seam_path)
                reasons[seam_path] = (
                    f"{worst[0]} amax {worst[1]:.4g} > "
                    f"{self.keep_outliers:g}x median {median:.4g}; "
                    f"kept at host precision")
        return Decision(keep_host=tuple(keep), reasons=reasons)


_REGISTRY: dict[str, QuantScheme] = {}


def register(name: str, scheme: QuantScheme) -> None:
    """Register a scheme instance under ``name`` (last write wins)."""
    _REGISTRY[name] = scheme


def get(name: str) -> QuantScheme:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown quantisation scheme {name!r}; "
                       f"registered: {sorted(_REGISTRY)}") from None


def names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def validate_request(request: Mapping[str, PointStat]) -> None:
    """Refuse loudly what the collector cannot measure yet.

    A scheme asking for a per-block or per-channel statistic must not
    silently receive per-tensor numbers — wrong-shaped scales bind and
    run, and the error surfaces as accuracy nobody can trace. The wall
    stays until the collector grows that granularity.
    """
    bad = {key: ps for key, ps in request.items()
           if (ps.stat, ps.granularity) not in _EXECUTABLE}
    if bad:
        k, ps = next(iter(bad.items()))
        raise NotImplementedError(
            f"scheme requests ({ps.stat}, {ps.granularity}) at {k} "
            f"(and {len(bad) - 1} more point(s)); the collector currently "
            f"measures only per-tensor amax. Extending it is the "
            f"supported path — do not fall back to per-tensor silently.")


register("fp8_static", Fp8Static())
register("fp8_static_keep_outliers", Fp8Static(keep_outliers=20.0))

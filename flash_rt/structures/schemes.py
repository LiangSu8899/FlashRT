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
           "NoQuant", "W8A16Decode", "W4A16Decode",
           "register", "get", "names", "resolve_auto", "validate_request"]

#: statistics the collector can currently execute. Granularities other
#: than per-tensor (per-channel, per-block16) are part of the declared
#: interface — NVFP4 weight scale factors are per-16-block, imatrix is
#: per-channel — but the collector does not measure them yet, so a
#: scheme requesting one fails loudly at plan time instead of silently
#: getting per-tensor numbers with the wrong shape.
_EXECUTABLE = {("amax", "tensor"), (None, "tensor"),
               ("amax", "channel"), ("second_moment", "channel")}


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
    ``formats`` routes a seam to a named impl variant instead of the
    structure's default (``"w8a16_static"`` on a ``decoder_ffn`` seam
    binds the weight-only path). A seam routed to a non-default format
    is excluded from FP8 seam negotiation — a chain shares one scale and
    one wire dtype, and a member in another format has neither. An
    unknown format fails loudly at bind time.
    """

    keep_host: tuple[str, ...] = ()
    reasons: Mapping[str, str] = field(default_factory=dict)
    formats: Mapping[str, str] = field(default_factory=dict)


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


class W8A16Decode(QuantScheme):
    """Weight-only INT8, activations untouched — the decode-band recipe.

    Needs no calibration data at all (quantisation is per-output-channel
    on weights, done at bind time), so every point declares ``None``.
    Routes ``decoder_ffn`` seams to the ``w8a16_static`` impl, whose own
    M-dispatch sends decode shapes to the kernel and prefill back to the
    host. Other structures stay at host precision: this scheme is the
    decode recipe, not a whole-host FP8 replacement.

    A ``decoder_ffn`` seam is recognised by the point its spec declares
    (``act_after_mul`` — the gated activation), which is
    backend-independent by construction.
    """

    name = "w8a16_decode"
    _format = "w8a16_static"

    def statistics(self, points: Sequence) -> dict[str, PointStat]:
        return {f"{p.path}|{p.name}": PointStat(None) for p in points}

    def decide(self, report: Mapping[str, Mapping[str, float]]) -> Decision:
        formats, keep = {}, []
        for seam_path, pts in report.items():
            if any(k.endswith("|act_after_mul") for k in pts):
                formats[seam_path] = self._format
            else:
                keep.append(seam_path)
        return Decision(keep_host=tuple(keep),
                        reasons={p: f"{self.name} binds decoder_ffn only"
                                 for p in keep},
                        formats=formats)


class W4A16Decode(W8A16Decode):
    """Weight-only NVFP4 (E2M1 packed + block scale factors) twin of
    :class:`W8A16Decode` — same decode band, same M-dispatch, half the
    weight bytes. The ``flashrt/weight-only-ffn`` package quantises
    weights per 16-element block at bind time; activations stay BF16,
    so like the INT8 twin it needs no calibration data.
    """

    name = "w4a16_decode"
    _format = "w4a16_static"


class NoQuant(QuantScheme):
    """Quantisation off: every quantised seam stays at host precision.

    This is the explicit off-switch, not a degraded mode. Structures
    that are pure fusion (the attention core, cadence buffers) never
    consult a scheme decision and attach as usual — a BF16/FP16 host
    under this scheme still gets every fusion structure, it just gets
    no quantised GEMMs. Zero calibration, zero kernel dependencies.
    """

    name = "none"

    def statistics(self, points: Sequence) -> dict[str, PointStat]:
        return {f"{p.path}|{p.name}": PointStat(None) for p in points}

    def decide(self, report: Mapping[str, Mapping[str, float]]) -> Decision:
        keep = tuple(report)
        return Decision(keep_host=keep,
                        reasons={p: "quantisation off (scheme 'none')"
                                 for p in keep})


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


def resolve_auto() -> str:
    """Resolve the ``"auto"`` profile: highest performance this device
    can execute, from the registered names.

    FP8-capable hardware (SM >= 89) gets ``fp8_static`` — bit-identical
    to the behaviour before ``auto`` existed. Anything else gets
    ``none``: fusion structures still attach, quantised seams stay at
    host precision, and the receipt records why. The resolution table is
    deliberately one function so a future profile that measures faster
    (an FP4 mix, say) is promoted by editing exactly one line.
    """
    try:
        from flash_rt.core.utils.hardware import supports_fp8
        fp8 = bool(supports_fp8())
    except Exception:
        fp8 = False
    return "fp8_static" if fp8 else "none"


register("fp8_static", Fp8Static())
register("fp8_static_keep_outliers", Fp8Static(keep_outliers=20.0))
register("w8a16_decode", W8A16Decode())
register("w4a16_decode", W4A16Decode())
register("none", NoQuant())

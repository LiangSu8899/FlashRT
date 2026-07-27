"""Auto-assembly: discover seams, calibrate them in one pass, bind them.

This is the distribution layer. Given a host model and a way to run it,
it finds every structure seam (:mod:`.discover`), captures exactly the
calibration each one needs in a single forward pass, and binds each
through its library impl. The caller gets a ``path -> module`` swap map
and any outside-cadence update functions — the same thing the hand
recipes produced, derived from the model object rather than written by
hand. A host integrates by importing and calling; it writes no
per-seam scaffolding.

The calibration each structure needs, captured structure-aware:
  linear_proj / qkv_pack : the shared input the projection(s) see, and
                           its per-tensor amax (the static act scale)
  adaln_producer         : the (cond, style) pairs the conditioning
                           projection emits across the tick, for the
                           step table and its fingerprint locator
  decoder_ffn / vision_ffn : the normed input the MLP sees

Seam negotiation is resolved here: when an adaln_producer feeds a
sibling qkv_pack under the same parent, the producer emits fp8 and the
pack takes the shared act scale, skipping its own input quantization.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch

from .discover import (Seam, _resolve, discover, group_families,
                       seam_weights)
from .points import Collector, Point, resolve as resolve_points

_FP8 = torch.float8_e4m3fn
_FP8_CHAIN_MAX_ROWS = 256  # fp8 producer chain qualifies at denoise
                          # M (bandwidth-bound); large-M prefill skips

# Host-family attention adapters. Attention seams are not a static
# module pattern — where the attention math actually runs is
# host-specific (a function in one host, a processor in another), so
# auto-discovery of the attention_core structure is delegated to
# registered adapters. Each adapter, given the model and a way to run
# it, returns (swaps, update) or None (this host is not its family).
_ATTENTION_ADAPTERS: list = []


def register_attention_adapter(adapter) -> None:
    """Register a host-family attention adapter (callable)."""
    _ATTENTION_ADAPTERS.append(adapter)


# Per-structure binders registered from impls, consulted before the
# built-in routing in :func:`_bind_auto`. New structures land by
# registering here from their own module instead of editing the routing
# function — parallel additions then touch disjoint files. A binder is
# ``f(model, seam, cap, *, points, fmt) -> module | None`` with the same
# refusal contract as ``_bind_auto`` (raise ``ValueError`` with the
# reason; return ``None`` for "host keeps its path").
_STRUCTURE_BINDERS: dict[str, Any] = {}


def register_structure_binder(structure: str, binder) -> None:
    """Route ``structure`` seams to ``binder`` (last write wins)."""
    _STRUCTURE_BINDERS[structure] = binder


@dataclass
class AutoPlan:
    """Discovered + calibrated swaps, ready to stage."""

    swaps: dict[str, torch.nn.Module] = field(default_factory=dict)
    updates: list[Callable[[], None]] = field(default_factory=list)
    seams: list[Seam] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    #: modules that carry a guard but are not swapped at a path — an
    #: adapter's routed seam. Reported by the attachment's ledger, never
    #: installed by it, so a seam that cannot be swapped can still be
    #: counted instead of being invisible.
    observed: dict[str, torch.nn.Module] = field(default_factory=dict)
    #: undo callables for host mutations that had to happen while the plan
    #: was being built rather than when it was attached (a patched
    #: module-level function). Handed to ``attach`` so ``detach`` really
    #: does give back the model that came in.
    revert: list[Callable[[], None]] = field(default_factory=list)
    #: ``(enable, disable)`` pairs for those same non-module seams, so a
    #: gate can put the host back for the baseline arm without unbinding
    #: anything. A seam that cannot be turned off cannot be measured.
    toggles: list[tuple[Callable[[], None], Callable[[], None]]] = field(
        default_factory=list)
    #: ``flash_rt.core.precision_spec.ModelPrecisionSpec`` for the scales
    #: this plan baked in — the repo's introspection format, not a private
    #: one, so ``plan.precision_spec`` reads like ``rt.precision_spec``
    precision_spec: Any = None

    def enable_routed(self) -> None:
        for on, _ in self.toggles:
            on()

    def disable_routed(self) -> None:
        for _, off in self.toggles:
            off()

    def revert_all(self) -> None:
        """Undo the plan-time host mutations. Idempotent per adapter."""
        for undo in reversed(self.revert):
            undo()
        self.revert.clear()
        self.observed.clear()


def _spec_points(seam) -> tuple[str, ...]:
    """The point names this seam's structure spec declares."""
    from .registry import load

    try:
        return tuple(load(seam.structure).calibration.get("points", ()))
    except Exception:                                   # noqa: BLE001
        return ()


def _consumer_point(seam) -> tuple[str, str]:
    """Where a negotiated consumer's input is observed.

    The producer's output *is* this tensor, so one amax serves both sides
    of the chain — which is why the pair can share a static scale at all.
    """
    if seam.structure == "qkv_pack":
        return (seam.path + "." + (seam.pack_attrs or ("q_proj",))[0], "x")
    if seam.structure == "decoder_ffn":
        return (seam.path, "x_after_norm")
    return (seam.path, "x")


def auto_swaps(
    model: torch.nn.Module,
    forward: Callable[..., Any] | Sequence[Callable[[], Any]],
    *,
    structures: tuple[str, ...] = ("decoder_ffn", "vision_ffn",
                                   "qkv_pack", "adaln_producer",
                                   "linear_proj", "norm_fused",
                                   "attention_core", "decoder_block"),
    negotiate_fp8: bool = True,
    prefix_cadence: bool = False,
    observations: Iterable[Any] | None = None,
    percentile: float = 99.9,
    max_samples: int | None = None,
    scheme: str | Any = "auto",
    verbose: bool = False,
) -> AutoPlan:
    """Discover, calibrate in one pass, and bind every applicable seam.

    The calibration arguments are the repo's, not this layer's: the names,
    the defaults and the meaning of ``percentile`` / ``max_samples`` /
    ``observations`` are ``flash_rt.api.FlashRT.calibrate``'s, and the
    reduction is ``flash_rt.core.calibration.accumulate_amax``. A second
    vocabulary for the same thing is the one thing this layer must not
    add.

        auto_swaps(model, forward)                        # one sample
        auto_swaps(model, [f0, f1, f2])                    # one thunk each
        auto_swaps(model, feed, observations=dataset)      # feed(obs) per obs
        auto_swaps(model, feed, observations=ds, percentile=95.0)

    ``scheme`` names a registered quantisation scheme (:mod:`.schemes`):
    what statistic each point needs, and per seam whether to bind or keep
    the host at host precision. The default ``"auto"`` resolves to the
    highest-performing profile the device can execute — on FP8-capable
    hardware that is ``fp8_static``, bit-identical to the behaviour this
    layer shipped with; elsewhere it is ``none`` (fusion structures
    attach, quantised seams stay at host precision). Explicit selection
    (``scheme="none"``, ``scheme="w4a16_decode"``, ...) overrides. It
    adds no calibration entry.

    ``forward`` is always "run the host once"; with ``observations`` it
    takes one observation. That indirection is this layer's only
    difference from a frontend's ``calibrate``, and it exists because a
    host here is an arbitrary ``nn.Module`` with no common observation
    contract — not because the calibration standard differs.

    ``prefix_cadence`` declares that the caller will run ``plan.updates``
    whenever the observation changes. Structures that hold per-observation
    host state — the attention core keeps the prefix keys and values — are
    only offered when it is set, because without the refresh they attend to
    whatever the calibration saw. Leaving it off is the accurate default.

    On ``percentile``: it reduces *across* samples. Within one sample the
    reduction is a max over every call, which is required rather than
    chosen — see ``docs/calibration.md`` §4.2. And note §10's own caveat
    that at small N a 99.9 percentile barely clips at all (it interpolates
    between the top two ranks); with N ≤ 64 and suspected outliers, pass a
    lower one.
    """

    def say(msg: str) -> None:
        if verbose:
            print(f"[autobuild] {msg}", flush=True)

    thunks, source = _calibration_thunks(forward, None, observations)
    if max_samples is not None and len(thunks) > max_samples:
        thunks = thunks[:max_samples]
    plan_notes_calibration: dict[str, Any] = {}
    plan_refusals: list[tuple[str, str]] = []

    seams = discover(model, structures)
    # a packed group owns its q/k/v; linear_proj keeps only what the
    # pack does not take (the output projection), so the two structures
    # compose instead of fighting over the same module
    packed = {s.path + "." + a for s in seams
              if s.structure == "qkv_pack" for a in (s.pack_attrs or ())}
    seams = [s for s in seams
             if not (s.structure == "linear_proj" and s.path in packed)]
    say(f"discovered {len(seams)} seam(s)")
    adapter_only = not seams and "attention_core" in structures
    if not seams and not adapter_only:
        return AutoPlan()

    # ---- one calibration pass, structure-aware capture ----
    # Activation scales go through the house two-level statistic: a max
    # over every call inside one sample (docs/calibration.md §4.2 — a
    # flow-matching host runs every step inside one forward and per-step
    # scales crashed the compiler), then accumulate_amax's percentile
    # across samples. Nothing holds an activation tensor for this: each
    # point is one float, measured where the spec says it is
    # (:mod:`.points`).
    caps: dict[str, dict[str, Any]] = {}
    hooks = []
    all_points: list[Point] = []
    seam_points: dict[str, list[Point]] = {}
    for seam in seams:
        try:
            pts = resolve_points(seam, _spec_points(seam))
        except ValueError as refusal:
            plan_refusals.append((seam.path, str(refusal)[:120]))
            continue
        seam_points[seam.path] = pts
        all_points.extend(pts)

    from . import schemes as _schemes
    auto_resolved = isinstance(scheme, str) and scheme == "auto"
    if auto_resolved:
        scheme = _schemes.resolve_auto()
        say(f"scheme auto -> {scheme}")
    scheme_obj = (_schemes.get(scheme) if isinstance(scheme, str)
                  else scheme)
    # loud wall before any calibration work: a scheme asking for a
    # granularity the collector cannot measure must not silently get
    # per-tensor numbers of the wrong shape
    stat_request = scheme_obj.statistics(tuple(all_points))
    _schemes.validate_request(stat_request)
    collector = Collector(points=all_points, request=dict(stat_request))

    # observed call order across the whole calibration pass. Anything
    # that has to know which seam runs first (a stream-scoped buffer
    # needs a writer, and the writer has to be the one the host calls
    # first) reads it from here rather than assuming the module tree's
    # order matches the forward's.
    call_order = itertools.count()

    def cap_cond(path):
        def hook(module, args, out):
            cap = caps[path]
            if "order" not in cap:
                cap["order"] = next(call_order)
            cap.setdefault("pairs", []).append(
                (args[0].detach().clone(), out.detach().clone()))
            return None
        return hook

    def cap_shape(path):
        # a block seam needs no tensors of its own, only the host's
        # return convention (bare tensor or 1-tuple)
        def hook(module, args, kwargs, out):
            caps[path]["returns_tuple"] = isinstance(out, tuple)
            return None
        return hook

    # the amax points are hooked by the collector; only the two
    # content/observation captures need their own hooks here, and neither
    # is a statistic: a step table is memoised host output, a return
    # convention is one boolean
    for seam in seams:
        caps[seam.path] = {}
        target = _resolve(model, seam.path)
        if seam.structure == "decoder_block":
            hooks.append(target.register_forward_hook(
                cap_shape(seam.path), with_kwargs=True))
        elif seam.structure == "adaln_producer":
            hooks.append(getattr(target, seam.cond_attr)
                         .register_forward_hook(cap_cond(seam.path)))
    hooks.extend(collector.hooks(lambda path: _resolve(model, path)))

    if hooks:
        # the calibration pass is a transaction over the host: if a thunk
        # raises, the hooks come off and no plan is returned. Removing
        # them only on the success path leaves a failed calibration's
        # hooks on the model, where they keep firing into a dict nobody
        # reads and slow down every later forward for reasons that are
        # nowhere in sight.
        try:
            with torch.no_grad():
                for thunk in thunks:
                    thunk()
                    # one vector per sample, so the percentile across them
                    # is possible at all
                    collector.end_sample()
        finally:
            for h in hooks:
                h.remove()
        plan_notes_calibration = dict(
            collector.reduce(percentile, verbose=verbose,
                             label=f"structures_N{len(thunks)}"),
            source=source)
        say(f"calibration pass done ({len(thunks)} sample(s) from "
            f"{source}, {plan_notes_calibration['points']} point(s), "
            f"{plan_notes_calibration['method']}"
            + (f" p={percentile}" if len(thunks) > 1 else "") + ")")

    # ---- the scheme turns statistics into decisions. Keep-host is a
    # first-class outcome recorded in the receipt, not a refusal: the
    # seam is healthy, the scheme chose host precision for it. ----
    scheme_report = {
        path: {f"{pt.path}|{pt.name}": collector.amax(pt.path, pt.name)
               for pt in pts}
        for path, pts in seam_points.items()}
    decision = scheme_obj.decide(scheme_report)
    scheme_note: dict[str, Any] = {
        "name": getattr(scheme_obj, "name", type(scheme_obj).__name__)}
    if auto_resolved:
        scheme_note["auto"] = True
    if decision.keep_host:
        kept = set(decision.keep_host)
        seams = [s for s in seams if s.path not in kept]
        scheme_note["keep_host"] = {
            p: decision.reasons.get(p, "") for p in sorted(kept)}
        say(f"scheme {scheme_note['name']}: {len(kept)} seam(s) kept at "
            f"host precision")
    formats: dict[str, str] = dict(decision.formats or {})
    if formats:
        scheme_note["formats"] = dict(sorted(formats.items()))
        say(f"scheme {scheme_note['name']}: {len(formats)} seam(s) "
            f"routed to a non-default format")

    # ---- fp8 seam negotiation: the load-bearing structure combination.
    # A single kernel need not win alone (fp8 qkv at M=50 is marginal,
    # fa2 in a bf16 stack loses); the *chain* wins — an adaln producer
    # that emits fp8 lets the qkv pack skip its own input quantization
    # and hands a clean fp8 seam down to the attention core. Bind the
    # producer→pack pair together with one shared act scale wherever a
    # producer feeds a pack under the same parent layer. ----
    act_scales: dict[str, torch.Tensor] = {}
    chain_rows: dict[str, int] = {}
    negotiated: dict[str, dict[str, Seam]] = {}
    if negotiate_fp8:
        by_parent: dict[str, dict[str, Seam]] = {}
        for seam in seams:
            if formats.get(seam.path):
                # a chain shares one scale and one wire dtype; a member
                # routed to another format has neither, so it binds
                # standalone through its own impl instead
                continue
            layer = _layer_of(seam.path)
            if seam.structure == "adaln_producer":
                # a layer has two producer→consumer seams: the norm
                # before attention feeds the projections, the norm after
                # it feeds the MLP. Both can hand fp8 downstream.
                slot = ("producer" if _feeds_attention(seam.path)
                        else "producer_ffn")
                by_parent.setdefault(layer, {})[slot] = seam
            elif seam.structure == "qkv_pack":
                by_parent.setdefault(layer, {})["pack"] = seam
            elif seam.structure == "decoder_ffn":
                by_parent.setdefault(layer, {})["ffn"] = seam
        # the chain wins at small M (denoise): fp8 is bandwidth-bound and
        # pays there, while a large-M prefill GEMM is compute-bound and
        # fp8 buys little — and an fp8 producer feeding a big compiled
        # prefill region is where the triton fp8 codegen chokes. Qualify
        # on the calibrated row count, not on host names.
        dev = next(model.parameters()).device
        blocks = {s.path for s in seams if s.structure == "decoder_block"}
        for lay, g in by_parent.items():
            # the attention pack is always negotiated. The FFN chain is
            # negotiated only where a decoder_block owns the boundary,
            # and the reason is the boundary rather than the kernel: at
            # the norm seam the fused producer costs a kernel
            # (gate_residual, +180 launches) plus its style
            # materialization (+180) to save the FFN's own input
            # quantize (-180) — measured net +0.17ms, so it is refused
            # there. Inside a block the same kernel *replaces* the
            # host's gated residual add instead of adding to it, which
            # is the whole point of owning the block.
            # Both chains need the block boundary, and for the same
            # reason: a negotiated producer emits FP8, and only a caller
            # that owns the block consumes it. Bound at the norm boundary
            # the *host* is the consumer, and the host expects its norm to
            # return a compute dtype — handed FP8 it keeps going and the
            # output is garbage (measured 0.24 output match, and NaN on a
            # neighbouring configuration) with nothing to see, because
            # every shape and dtype is inside its contract. The FFN chain
            # was already gated this way; the attention chain was not.
            if lay not in blocks:
                continue
            pairs = [("producer", "pack"), ("producer_ffn", "ffn")]
            keep = {}
            for p_slot, c_slot in pairs:
                if p_slot not in g or c_slot not in g:
                    continue
                c_path, c_name = _consumer_point(g[c_slot])
                amax = collector.amax(c_path, c_name)
                rows_seen = collector.row_profile(c_path, c_name)
                rows = rows_seen[len(rows_seen) // 2] if rows_seen else 1 << 30
                if amax is None or rows > _FP8_CHAIN_MAX_ROWS:
                    continue
                # the consumer's input == the producer's output; its amax
                # is the one static scale both sides share
                keep[p_slot], keep[c_slot] = g[p_slot], g[c_slot]
                act_scales[f"{lay}|{c_slot}"] = torch.tensor(
                    [max(amax / 448.0, 1e-8)], device=dev)
                chain_rows[f"{lay}|{c_slot}"] = rows
            if keep:
                negotiated[lay] = keep

    # ---- the negotiated chain binds as one unit ----
    # producer and consumer must agree on the seam dtype: a pack bound
    # for fp8 input whose producer failed to bind would be handed BF16,
    # and the host would silently grow a quantize fused into whatever
    # produced it. Bind the pair together, or leave both on BF16.
    plan = AutoPlan(seams=seams)
    plan.notes["scheme"] = scheme_note
    handled: set[str] = set()
    for lay, g in negotiated.items():
        for p_slot, c_slot in (("producer", "pack"),
                               ("producer_ffn", "ffn")):
            if p_slot not in g or c_slot not in g:
                continue
            p_seam, c_seam = g[p_slot], g[c_slot]
            p_cap = caps.get(p_seam.path, {})
            if not p_cap.get("pairs"):
                continue
            try:
                pair = _bind_negotiated(
                    model, p_seam, c_seam, p_cap, collector,
                    act_scales[f"{lay}|{c_slot}"],
                    chain_rows[f"{lay}|{c_slot}"], plan)
            except (ValueError, RuntimeError) as refusal:
                plan.notes.setdefault("refused", []).append(
                    (f"{lay} [{c_slot} chain]", str(refusal)[:80]))
                continue
            plan.swaps.update(pair)
            handled.update({p_seam.path, c_seam.path})
    plan.notes["negotiated_layers"] = sorted(
        lay for lay, g in negotiated.items()
        if any(sm.path in handled for sm in g.values()))

    # ---- bind the remaining seams individually ----
    for name, members in group_families(seams).items():
        for seam in members:
            if seam.path in handled:
                continue
            cap = caps.get(seam.path, {})
            try:
                bound = _bind_auto(model, seam, cap, plan, act_scales,
                                   negotiate_fp8, points=collector,
                                   fmt=formats.get(seam.path))
            except (ValueError, RuntimeError) as refusal:
                plan.notes.setdefault("refused", []).append(
                    (seam.path, str(refusal)[:80]))
                continue
            if bound is None:
                continue
            if isinstance(bound, dict):
                plan.swaps.update(bound)
            else:
                plan.swaps[seam.path] = bound
    # ---- attention_core: host-family adapters (fa2 seam) ----
    if "attention_core" in structures:
        from . import adapters as _adapters  # noqa: F401 (registers)
        for adapter in _ATTENTION_ADAPTERS:
            try:
                # the adapter needs "run the host once", which is what a
                # thunk is. Handing it the caller's callable breaks the
                # sample entry, where that callable takes a sample —
                # the whole point of normalising the three ways in was
                # that nothing downstream should see the difference
                result = adapter(model, thunks[0],
                                 prefix_cadence=prefix_cadence)
            except (ValueError, RuntimeError) as refusal:
                plan.notes.setdefault("refused", []).append(
                    ("attention_core", str(refusal)[:80]))
                continue
            if result is None:
                continue
            # an adapter may hand back a third element for the parts of
            # its seam that are not modules at paths: how to undo them,
            # and what to report
            att_swaps, update = result[0], result[1]
            extras = result[2] if len(result) > 2 else {}
            plan.swaps.update(att_swaps)
            plan.observed.update(extras.get("observed", {}))
            plan.revert.extend(extras.get("revert", ()))
            if extras.get("toggle") is not None:
                plan.toggles.append(extras["toggle"])
            if update is not None:
                plan.updates.append(update)
            plan.notes["attention_adapter"] = type(adapter).__name__ \
                if hasattr(adapter, "__name__") else str(adapter)
            break

    # ---- one step-scoped style materialisation per conditioning stream
    # Every adaptive-norm producer on one stream resolves the same step,
    # so the whole stream's styles are fixed for the step's duration.
    # Materialising them once beats materialising them per call by the
    # launch count, which is what that work actually costs. Runs before
    # the block assembly: a block holds its producers directly and drops
    # them from the swap map, so afterwards they are no longer findable
    # here.
    _attach_brokers(caps, plan, say)

    # ---- decoder_block: compose the bound sublayers into one block ----
    # last, because it is assembled from what the region structures
    # produced. The swaps it absorbs are dropped from the plan: the
    # block holds those modules directly, and a swap that also targeted
    # the host child would leave two live copies of the same seam.
    for seam in (s for s in seams if s.structure == "decoder_block"):
        try:
            block = _bind_block(model, seam, caps.get(seam.path, {}), plan)
        except (ValueError, RuntimeError) as refusal:
            plan.notes.setdefault("refused", []).append(
                (seam.path + " [block]", str(refusal)[:80]))
            continue
        if block is None:
            continue
        for child in _BLOCK_OWNED:
            plan.swaps.pop(seam.path + "." + child, None)
        plan.swaps[seam.path] = block

    # what discovery took on trust, for the seams that actually bound. An
    # assumption that reaches the model without reaching the receipt is
    # indistinguishable from something that was checked.
    assumed = [(s.path, note) for s in seams if s.assumptions
               and s.path in plan.swaps for note in s.assumptions]
    if assumed:
        plan.notes["assumed"] = assumed
        say(f"{len(assumed)} seam(s) carry an assumption the parity gate "
            f"has to check (see notes['assumed'])")

    if plan_notes_calibration:
        # the calibration method is part of the result, not a detail of
        # how it was produced: a parity band means something different
        # depending on how much of the distribution it was scaled from
        plan.notes["calibration"] = plan_notes_calibration
        # and the receipt itself is the repo's, so a structures attachment
        # answers ``precision_spec`` the same way a frontend does
        from .points import precision_spec as _spec
        plan.precision_spec = _spec(collector, plan_notes_calibration)
    if plan_refusals:
        plan.notes.setdefault("refused", []).extend(plan_refusals)
    say(f"bound {len(plan.swaps)} seam(s), "
        f"{len(plan.notes.get('refused', []))} refused")
    return plan


def _attach_brokers(caps, plan, say) -> None:
    from .impls.adaln_producer import AdaLNProducer, bind_style_broker

    groups: dict[tuple, list] = {}
    for path, module in plan.swaps.items():
        if not isinstance(module, AdaLNProducer):
            continue
        cap = caps.get(path, {})
        order = cap.get("order")
        if order is None or not cap.get("pairs"):
            continue
        # one broker per (stream, style width, row count): producers
        # that differ in any of those cannot share a buffer
        key = (_stream_key(cap["pairs"]), int(module.styles.shape[-1]),
               int(module.resid.shape[0]))
        groups.setdefault(key, []).append((order, path, module))

    for key, members in groups.items():
        # the writer is the producer the host calls first, taken from the
        # observed order of the calibration pass — not from the module
        # tree's order, which need not match the forward's
        members.sort(key=lambda entry: entry[0])
        try:
            broker = bind_style_broker([m for _, _, m in members], key[2])
        except (ValueError, RuntimeError) as refusal:
            plan.notes.setdefault("refused", []).append(
                (f"style_broker[{key[1]}x{key[2]}]", str(refusal)[:80]))
            continue
        if broker is None:
            continue
        plan.notes.setdefault("brokers", []).append(
            {"slots": broker.slots, "rows": key[2], "width": key[1],
             "writer": members[0][1]})
        say(f"style broker: {broker.slots} producer(s) share one "
            f"step-scoped materialisation (writer {members[0][1]})")


_BLOCK_OWNED = ("input_layernorm", "post_attention_layernorm", "mlp")


def _cond_kw(host) -> str:
    """The keyword the host threads its conditioning through."""
    import inspect
    try:
        params = list(inspect.signature(host.forward).parameters)
    except (TypeError, ValueError):
        params = []
    for name in ("adarms_cond", "cond", "temb", "emb"):
        if name in params:
            return name
    return "adarms_cond"


def _bind_block(model, seam, cap, plan):
    """Assemble one decoder_block from its already-bound sublayers."""
    from .impls.decoder_block import bind_decoder_block

    prod_in = plan.swaps.get(seam.path + ".input_layernorm")
    prod_out = plan.swaps.get(seam.path + ".post_attention_layernorm")
    ffn = plan.swaps.get(seam.path + ".mlp")
    if prod_in is None or prod_out is None or ffn is None:
        # a sublayer that did not bind leaves the host block intact:
        # the block structure adds composition, it does not substitute
        # for the region seams it is made of
        return None
    host = _resolve(model, seam.path)
    # the attention sublayer is family-specific (where the attention runs
    # and which rotary form it uses), so it comes from the same adapters
    # that bound the attention core. None keeps the host's attention
    # module, which is the pre-block behaviour.
    attn = None
    for adapter in _ATTENTION_ADAPTERS:
        builder = getattr(adapter, "sublayer", None)
        if builder is None:
            continue
        attn = builder(host)
        if attn is not None:
            break
    if attn is not None:
        _alias_kv_region(plan, seam.path, attn)
    return bind_decoder_block(
        host, prod_in, prod_out, ffn, cond_kw=_cond_kw(host),
        returns_tuple=bool(cap.get("returns_tuple")), attn=attn)


def _alias_kv_region(plan, path: str, sublayer) -> None:
    """Let the packed projections write into the core's packed KV region.

    Both sides can express this (see ``beta.joins``); the qualification
    is that nothing transforms the tensor in between. Value goes straight
    from the projection to the kernel and qualifies. Key does not on this
    family: a rotary embedding runs after the projection, so aliasing it
    would leave untransformed keys in the packed region — writing the
    transformed ones back is the copy this was meant to remove. Hosts
    without a rotary step qualify for both; the attribute is general and
    the qualification is per join.
    """
    from .impls.qkv_pack import PackedLinear

    head = plan.swaps.get(path + ".self_attn.q_proj")
    core = getattr(sublayer, "core", None)
    if not isinstance(head, PackedLinear) or core is None:
        return
    if not hasattr(core, "alias_suffix"):
        return
    _, v_region = core.alias_suffix(key=False, value=True)
    if v_region is None:
        return
    try:
        head.alias_stash(2, v_region)          # sibling order q, k, v
    except (ValueError, RuntimeError) as refusal:
        core._alias_v = False
        plan.notes.setdefault("refused", []).append(
            (path + " [kv alias]", str(refusal)[:80]))
        return
    plan.notes.setdefault("aliased_kv", []).append(path)

    # The joint q|k view is deliberately not enabled here. q and k are
    # one contiguous run of the packed output and take the same rotary
    # arithmetic, so one pass over the pair should replace two — and
    # measured, it replaces nothing: the rotary kernels keep their exact
    # launch counts (180/162/63) because the compiler splits the merged
    # pass back apart, fusing it into each consumer (q is made
    # contiguous for the kernel, k is copied into the packed region).
    # Paired timing: -0.014 ms on 23.1, which is below the margin this
    # stack ships at. Expressing "do this once" in tensor ops does not
    # survive a compiler that re-derives its fusion boundaries from the
    # consumers; the style broker only survived by being opaque, and an
    # opaque wrapper here would be worse, since the rotary would then run
    # as several eager ops instead of one fused kernel. Merging these
    # needs a rotary kernel, not a rearrangement. The capability stays
    # on the impl for a host where the arithmetic is not launch-bound.


def _bind_auto(model, seam, cap, plan, act_scales, negotiate_fp8,
               points=None, fmt=None):
    """Route one seam to its impl with the calibrated scales.

    ``points`` is the reduced collector: every scale an impl needs is one
    float looked up by (path, spec point name). No activation tensors are
    threaded through here, because none are needed — the two scales that
    used to be recomputed from held inputs are measured at the GEMM whose
    input they are (:mod:`.points`).

    ``fmt`` is the scheme's per-seam format routing. ``None`` binds the
    structure's default impl; a named format binds that variant instead,
    and a name with no variant for this structure fails loudly — the
    scheme author's error surfaces at bind time, not as accuracy.
    """
    from .impls.decoder_ffn import fp8_static as ffn_impl
    from .impls.vision_ffn import fp8_static as vis_impl

    def scale(name, path=None):
        return None if points is None else points.scale(path or seam.path,
                                                        name)

    custom = _STRUCTURE_BINDERS.get(seam.structure)
    if custom is not None:
        return custom(model, seam, cap, points=points, fmt=fmt)

    if fmt and seam.structure != "decoder_ffn":
        raise ValueError(f"scheme routed {seam.structure} to format "
                         f"{fmt!r}, which has no impl variant here")

    if seam.structure == "decoder_ffn":
        if fmt in ("w8a16_static", "w4a16_static"):
            if fmt == "w8a16_static":
                from .impls.decoder_ffn import w8a16_static as wq_impl
            else:
                from .impls.decoder_ffn import w4a16_static as wq_impl

            # two callers, two layout conventions: ``seam_weights``
            # serves the fp8 impl transposed ([D, F]); these binders are
            # checkpoint-native ([F, D]) and their dim check passes with
            # the names swapped, so handing them the transposed dict
            # binds a guard with k = F and every call falls back.
            # Transpose back here, at the seam between the conventions.
            w = seam_weights(model, seam)
            w = dict(w,
                     w_gate=w["w_gate"].t().contiguous(),
                     w_up=w["w_up"].t().contiguous(),
                     w_down=w["w_down"].t().contiguous())
            return wq_impl.bind_mlp_seam(
                w, variant=seam.variant,
                original=_resolve(model, seam.path))
        if fmt not in (None, "fp8_static"):
            raise ValueError(f"scheme routed decoder_ffn to format "
                             f"{fmt!r}, which has no impl variant here")
        in_s = scale("x_after_norm")
        hid_s = scale("act_after_mul", seam.path + ".down_proj")
        if in_s is None or hid_s is None:
            return None
        return ffn_impl.bind_mlp_seam(
            seam_weights(model, seam), variant=seam.variant,
            input_scale=in_s, hidden_scale=hid_s,
            original=_resolve(model, seam.path))

    if seam.structure == "vision_ffn":
        fc2 = (seam.fc_attrs or ("fc1", "fc2"))[1]
        in_s = scale("x_after_norm")
        hid_s = scale("hidden_after_act", seam.path + "." + fc2)
        if in_s is None or hid_s is None:
            return None
        return vis_impl.bind_mlp_seam(
            seam_weights(model, seam), input_scale=in_s,
            hidden_scale=hid_s, original=_resolve(model, seam.path))

    if seam.structure == "norm_fused":
        from .impls.norm_fused import bind_norm_fused
        return bind_norm_fused(
            _resolve(model, seam.path),
            host_dtypes=(None if points is None
                         else points.seen_dtypes(seam.path, "x")))

    if seam.structure == "linear_proj":
        in_s = scale("x")
        if in_s is None:
            return None
        from .impls.linear_proj import fp8_static as proj_impl
        return proj_impl.bind_proj_seam(
            seam_weights(model, seam), input_scale=in_s,
            row_profile=points.row_profile(seam.path, "x"),
            original=_resolve(model, seam.path))

    if seam.structure == "qkv_pack":
        from .impls.qkv_pack import bind_attn_block, bind_qkv_pack
        first = (seam.pack_attrs or ("q_proj",))[0]
        amax = None if points is None else points.amax(
            seam.path + "." + first, "x")
        if amax is None:
            return None
        block = _resolve(model, seam.path)
        rows = points.row_profile(seam.path + "." + first, "x")
        cap = dict(cap or {}, rows=rows[len(rows) // 2] if rows else 1)
        act_scale = torch.tensor(
            [max(amax / 448.0, 1e-8)],
            device=getattr(block, first).weight.device)
        if seam.variant.get("bind") == "module":
            # the whole block: packed projections *and* the attention
            # compute dtype (hosts that run SDPA in fp32 pay for it)
            return {seam.path: bind_attn_block(
                block, act_scale, rows=cap["rows"],
                sdpa_dtype=torch.bfloat16)}
        mods = [getattr(block, a) for a in seam.pack_attrs]
        parts = bind_qkv_pack(mods, act_scale, rows=cap["rows"],
                              in_dtype="bf16_fused_quant")
        return {seam.path + "." + a: m
                for a, m in zip(seam.pack_attrs, parts)}

    if seam.structure == "adaln_producer":
        from .impls.adaln_producer import (bind_adaln_producer,
                                           bind_style_table)
        if not cap.get("pairs"):
            return None
        norm = _resolve(model, seam.path)
        proj = getattr(norm, seam.cond_attr)
        key = _stream_key(cap["pairs"])
        loc = plan.notes.setdefault("_locators", {}).get(key)
        table = bind_style_table(proj, cap["pairs"], locator=loc)
        plan.notes["_locators"][key] = table.locator
        return {seam.path + "." + seam.cond_attr: table}

    return None


class _Eager(torch.nn.Module):
    """Wrap a module so its forward runs outside the compiled region.

    An fp8-emitting seam's arithmetic, if traced by inductor, gets fused
    into fp8 math (illegal on sm120 triton) — and the quantize even
    reaches back across the boundary, so inductor casts the host's own
    gated residual to fp8 to feed it. The hand recipes never hit this
    because the whole denoise block froze to eager. A swapped-in module
    does not inherit that freezing, so fp8 seams declare it. Overriding
    the instance ``forward`` is not enough (dynamo inlines the class
    forward); the disable must sit on a class method, which is what this
    wrapper provides. The kernels are opaque either way, so eager here
    is a graph break, not real work.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    @torch._dynamo.disable
    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)


def _eager(module):
    return _Eager(module)


def _bind_negotiated(model, p_seam, k_seam, p_cap, points, scale, rows,
                     plan):
    """Bind an fp8 producer and the pack it feeds as one chain.

    This is the combination the structure layer exists for: neither half
    is worth much alone (a small-M fp8 projection barely beats BF16, a
    producer that only reshapes styles saves nothing), but together the
    producer's fused quantize removes the consumer's input quantization
    entirely and hands a clean fp8 seam downstream.
    """
    from .impls.adaln_producer import bind_adaln_producer
    from .impls.qkv_pack import bind_qkv_pack

    norm = _resolve(model, p_seam.path)
    consumer = _resolve(model, k_seam.path)
    key = _stream_key(p_cap["pairs"])
    loc = plan.notes.setdefault("_locators", {}).get(key)
    dim, form = _adaln_form(p_cap, points, p_seam)
    prod = bind_adaln_producer(
        norm, p_cap["pairs"], act_scale=scale, rows=rows,
        dim=dim, locator=loc, norm=form)
    plan.notes["_locators"][key] = prod.locator

    swaps = {p_seam.path: prod}
    if k_seam.structure == "decoder_ffn":
        from .impls.decoder_ffn import fp8_static as ffn_impl
        # the input scale is the one the producer upstream will quantize
        # with — the same number, because the producer's output is this
        # consumer's input; the hidden scale is measured at the down
        # projection whose input it is
        w = seam_weights(model, k_seam)
        bound = ffn_impl.bind_mlp_seam(
            w, variant={**k_seam.variant, "in_dtype": "fp8_static"},
            input_scale=float(scale.item()),
            hidden_scale=points.scale(k_seam.path + ".down_proj",
                                      "act_after_mul"),
            original=consumer)
        swaps[k_seam.path] = bound
        return swaps
    mods = [getattr(consumer, a) for a in k_seam.pack_attrs]
    parts = bind_qkv_pack(mods, scale, rows=rows,
                          in_dtype="fp8_static")
    swaps.update({k_seam.path + "." + a: m
                  for a, m in zip(k_seam.pack_attrs, parts)})
    return swaps


def _calibration_thunks(forward, frames, samples):
    """Turn the three ways of asking for calibration into one list.

    They differ only in where a frame's input comes from, so they end as
    the same thing: a list of callables, each of which runs the host
    once. Keeping them one axis is what stops "how much calibration" and
    "how to run the host" from becoming two interfaces that can disagree.
    """
    if samples is not None:
        if not callable(forward):
            raise ValueError(
                "auto_swaps: with samples=, forward takes one sample")
        taken = list(samples) if frames is None else [
            s for _, s in zip(range(max(1, frames)), samples)]
        if not taken:
            raise ValueError("auto_swaps: samples is empty")
        return [(lambda s=s: forward(s)) for s in taken], "samples"
    if isinstance(forward, (list, tuple)):
        if not forward:
            raise ValueError("auto_swaps: no forward thunks given")
        if frames is not None and frames != len(forward):
            raise ValueError(
                f"auto_swaps: {len(forward)} thunk(s) given but "
                "observations= and a thunk list are alternatives, "
                "not a pair; the thunks decide")
        return list(forward), "thunks"
    if not callable(forward):
        raise ValueError("auto_swaps: forward must be callable")
    return [forward] * max(1, frames or 1), "forward"


def _adaln_form(cap, points, seam) -> tuple[int, str]:
    """Read the producer's width and form off the calibration.

    Both were assumed before: the form was hard-coded to rms and the
    width taken as ``style_width // 3``. That holds only where the style
    carries three parts. A host whose adaptive norm emits (scale, shift)
    — the layer form — got bound as rms at two thirds of its real width,
    and the plan built cleanly and then could not run. It took a second
    host and an actual forward to see it, because nothing on the way
    there had to disagree.

    The norm's own input says how wide it is, and the ratio to the style
    says which form it is. Neither is a guess.
    """
    dim = points.width(seam.path, "x")
    if dim is None:
        raise ValueError(
            "adaln_producer: the norm's own input was never observed, so "
            "the form cannot be told from the style width alone")
    style_width = int(cap["pairs"][0][1].shape[-1])
    if style_width == 3 * dim:
        return dim, "rms"           # scale, shift, gate
    if style_width == 2 * dim:
        return dim, "layer"         # scale, shift
    raise ValueError(
        f"adaln_producer: style width {style_width} is neither two nor "
        f"three times the norm width {dim} — the modulation is a shape "
        "this structure does not model")


def _stream_key(pairs) -> str:
    """Identify the conditioning stream a producer was calibrated on.

    Locators were keyed by seam family, which gives every family its own
    lookup even when they all read the same conditioning — the two norms
    of one block among them. Keying by the observed conditioning instead
    shares one locator across the whole stream. It is safe by
    construction rather than by convention: the key is a digest of the
    conditioning rows themselves, so two seams share a locator only when
    they saw byte-identical inputs, and identical inputs resolve to
    identical indices whichever seam built the table.
    """
    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    for cond, _ in pairs:
        c = cond.detach().reshape(-1, cond.shape[-1]).to(torch.float32)
        digest.update(c.cpu().numpy().tobytes())
    return digest.hexdigest()


def _layer_of(path: str) -> str:
    """The parent layer key: a.layers.7.self_attn -> a.layers.7."""
    import re
    m = re.search(r"(.*\.layers\.\d+)\.", path)
    return m.group(1) if m else path.rsplit(".", 1)[0]


def _feeds_attention(path: str) -> bool:
    """An adaln producer that feeds attention (input_layernorm) rather
    than the MLP (post_attention_layernorm)."""
    leaf = path.rsplit(".", 1)[-1]
    return "input" in leaf or leaf in ("norm1", "ln1")

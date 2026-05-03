"""Declarative weight-loader scaffolding (stage 7.1).

This module defines the **protocols and data classes** for the
``WEIGHT_SPEC`` abstraction described in ``docs/v2/stage7_weight_loader.md``.
No concrete Source/Sink/Transform implementations live here yet — they
land in stage 7.2 (torch) and 7.5 (jax). The module is pure-python, has
no runtime dependencies on frontends, and is not yet imported by any
frontend.

Shape (three layers):

    WeightSource  →  TransformPipeline  →  WeightSink
      (read)           (.T / fuse / quant)    (store)

Core types:

    * ``WeightSource``  protocol   — ``get(key) -> array-like``
    * ``Transform``     protocol   — ``apply(tensor, ctx) -> tensor``
    * ``WeightSink``    protocol   — ``store(tensor, scale=None)``
    * ``Item``          dataclass  — one logical weight: key + transforms + sink
    * ``LayerBlock``    dataclass  — ``prefix_fmt`` × ``num_layers`` of ``Item``s
    * ``ModelWeightSpec`` dataclass — top-level: blocks + singletons + buffers + dims
    * ``LoaderContext`` dataclass  — runtime scratch passed through transforms
    * ``WeightLoader``  class      — executes a spec against a source+target

The runner itself is a stub in 7.1 (raises ``NotImplementedError`` on
``run``) — it becomes real in 7.2 once the torch adapter lands. Writing
the protocols first lets us unit-test the shape in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


# ════════════════════════════════════════════════════════════════════
#  Protocols
# ════════════════════════════════════════════════════════════════════

@runtime_checkable
class WeightSource(Protocol):
    """Read a weight by string key.

    Concrete implementations in later stages:
      * ``SafetensorsSource``  (stage 7.2) — wraps ``safe_open``
      * ``OrbaxDictSource``    (stage 7.5) — wraps ``engine_w`` dict
    """

    def get(self, key: str) -> Any: ...

    def has(self, key: str) -> bool: ...


@runtime_checkable
class Transform(Protocol):
    """Pure function from tensor to tensor.

    ``ctx`` is mutable scratch (see ``LoaderContext``) — transforms that
    read other keys (e.g. ``FuseNorm``) use ``ctx.source``; transforms
    that write scales append to ``ctx.scales[scale_into]``.
    """

    def apply(self, tensor: Any, ctx: "LoaderContext") -> Any: ...


@runtime_checkable
class WeightSink(Protocol):
    """Consume a transformed tensor (and optional per-item scale).

    Concrete sinks in later stages:
      * ``Attr(name)``            — ``setattr(target, name, tensor)``
      * ``TensorList(name)``      — append to ``target.<name>`` list
      * ``FlatCat(name)``         — collect; finalize with ``torch.cat``
      * ``CudaBufferJax(idx, cache=...)``  (stage 7.5)
    """

    def store(self, tensor: Any, *, scale: float | None = None) -> None: ...

    def finalize(self) -> None:
        """Optional post-pass (e.g. ``FlatCat`` calls ``torch.cat`` here)."""
        ...


# ════════════════════════════════════════════════════════════════════
#  Spec data classes
# ════════════════════════════════════════════════════════════════════

@dataclass
class Item:
    """One logical weight in a spec.

    Attributes
    ----------
    name : str
        Human-readable tag for diagnostics + error messages.
    key : str | CompositeKey
        Either a raw checkpoint key (with optional ``{prefix}`` /
        ``{i}`` placeholders), or a composite source descriptor (e.g.
        ``Cat(...)``, ``FusedQKV(...)``) — those classes implement a
        ``resolve(source, ctx) -> tensor`` method and are introduced in
        stage 7.2.
    transforms : list[Transform]
        Applied in order to the fetched tensor.
    sink : WeightSink
        Where the final tensor goes.
    scale_into : str | None
        If a transform in the pipeline produces a scale (i.e. ``Quant``),
        the scale is appended to ``ctx.scales[scale_into]`` in spec order.
        ``None`` means scale is discarded (non-quantized item).
    """

    name: str
    key: Any
    transforms: list[Any] = field(default_factory=list)
    sink: Any = None
    scale_into: str | None = None


@dataclass
class LayerBlock:
    """A ``num_layers`` × ``items`` rectangular loop.

    Equivalent to::

        for i in range(num_layers):
            prefix = prefix_fmt.format(i=i)
            for item in items:
                load item with "{prefix}" substituted
    """

    prefix_fmt: str
    num_layers: int
    items: list[Item]
    # Optional symbolic name for diagnostics (e.g. "siglip", "encoder").
    name: str = ""


@dataclass
class BufferSpec:
    """Pre-allocated device buffer (not a weight, but part of load-time setup).

    Populated in stage 7.9 (spec gains BufferSpec support); left here as
    a placeholder so spec files can already reference it.
    """

    attr: str
    shape: tuple                      # symbolic dims resolved via spec.dims
    dtype: str                        # "fp16" | "uint8" | "fp32"
    init: str = "empty"               # "empty" | "zeros"


@dataclass
class ModelWeightSpec:
    """Top-level spec for one frontend's weight loading.

    A frontend declares exactly one ``WEIGHT_SPEC = ModelWeightSpec(...)``
    at class scope and its ``_load_weights`` body becomes::

        WeightLoader(source, target=self, spec=self.WEIGHT_SPEC).run()
    """

    framework: str                    # "torch" | "jax"
    blocks: list[LayerBlock] = field(default_factory=list)
    singletons: list[Item] = field(default_factory=list)
    buffers: list[BufferSpec] = field(default_factory=list)
    # Symbolic dim table used by BufferSpec.shape; resolved at run-time.
    dims: dict[str, Any] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════
#  Runtime context + runner
# ════════════════════════════════════════════════════════════════════

@dataclass
class LoaderContext:
    """Mutable scratch passed through each transform.

    ``source`` is the active ``WeightSource`` (transforms like ``FuseNorm``
    read extra keys from it).  ``prefix`` and ``layer_idx`` are set by the
    runner for each ``LayerBlock`` iteration. ``scales`` accumulates quant
    scales keyed by ``Item.scale_into``; the runner hands these back to
    the target frontend at the end.
    """

    source: WeightSource
    target: Any
    prefix: str = ""
    layer_idx: int = -1
    scales: dict[str, list[float]] = field(default_factory=dict)
    # Free-form scratch for transforms that cache intermediates.
    scratch: dict[str, Any] = field(default_factory=dict)

    def subkey(self, template: str) -> str:
        """Substitute ``{prefix}`` and ``{i}`` in a key template.

        Composite sources (Cat / FusedQKV / FusedGateUp) call this to
        resolve per-layer key templates without reaching into module
        internals.
        """
        if "{" not in template:
            return template
        return template.format(prefix=self.prefix, i=self.layer_idx)


class WeightLoader:
    """Executes a ``ModelWeightSpec`` against a source+target.

    Stage 7.1 ships the class skeleton only. ``run()`` is wired in stage
    7.2 once torch Source/Sink/Transform concretes exist.
    """

    def __init__(
        self,
        source: WeightSource,
        *,
        target: Any,
        spec: ModelWeightSpec,
    ) -> None:
        if not isinstance(source, WeightSource):
            raise TypeError(
                f"WeightLoader: source must implement WeightSource protocol "
                f"(got {type(source).__name__})"
            )
        if not isinstance(spec, ModelWeightSpec):
            raise TypeError(
                f"WeightLoader: spec must be ModelWeightSpec "
                f"(got {type(spec).__name__})"
            )
        self.source = source
        self.target = target
        self.spec = spec

    def run(self) -> LoaderContext:
        """Execute the spec. Returns the final ``LoaderContext``.

        Framework-agnostic: the runner only looks for a ``.resolve`` method
        on composite keys and a ``.apply`` / ``.store`` / ``.finalize``
        duck-type on transforms/sinks. Torch and JAX concretes live in
        ``torch_weights`` / ``jax_weights``.
        """
        ctx = LoaderContext(source=self.source, target=self.target)
        _reset_sinks(self.spec)

        for singleton in self.spec.singletons:
            ctx.prefix = ""
            ctx.layer_idx = -1
            _run_item(singleton, ctx)

        for block in self.spec.blocks:
            for i in range(block.num_layers):
                ctx.prefix = block.prefix_fmt.format(i=i)
                ctx.layer_idx = i
                for item in block.items:
                    _run_item(item, ctx)

        _finalize_sinks(self.spec)

        # Publish scale lists onto the target so frontends can wrap them
        # into device tensors without reaching into ``ctx.scales``.
        for name, values in ctx.scales.items():
            setattr(self.target, name, values)

        return ctx


# ════════════════════════════════════════════════════════════════════
#  Runner internals (framework-agnostic)
# ════════════════════════════════════════════════════════════════════

def _resolve_source(key: Any, ctx: "LoaderContext"):
    if isinstance(key, str):
        return ctx.source.get(ctx.subkey(key))
    if hasattr(key, "resolve"):
        return key.resolve(ctx)
    raise TypeError(
        f"Unsupported Item.key type: {type(key).__name__}. "
        f"Expected str or object with .resolve(ctx)."
    )


def _run_item(item: "Item", ctx: "LoaderContext") -> None:
    ctx.scratch.pop("_pending_scale", None)
    x = _resolve_source(item.key, ctx)
    for tr in item.transforms:
        x = tr.apply(x, ctx)
    scale = ctx.scratch.pop("_pending_scale", None)
    if (
        hasattr(item.sink, "_bind")
        and getattr(item.sink, "_bound_target", None) is None
    ):
        item.sink._bind(ctx.target)
    item.sink.store(x, scale=scale)
    if scale is not None and item.scale_into is not None:
        ctx.scales.setdefault(item.scale_into, []).append(scale)


def _reset_sinks(spec: "ModelWeightSpec") -> None:
    for it in spec.singletons:
        _reset_one_sink(it.sink)
    for b in spec.blocks:
        for it in b.items:
            _reset_one_sink(it.sink)


def _reset_one_sink(sink) -> None:
    if hasattr(sink, "_bound_target"):
        sink._bound_target = None
    if hasattr(sink, "_list"):
        sink._list = None
    if hasattr(sink, "_parts"):
        sink._parts = None
    if hasattr(sink, "_stored"):
        sink._stored = False


def _finalize_sinks(spec: "ModelWeightSpec") -> None:
    seen: set[int] = set()
    for it in spec.singletons:
        _finalize_one(it.sink, seen)
    for b in spec.blocks:
        for it in b.items:
            _finalize_one(it.sink, seen)


def _finalize_one(sink, seen: set[int]) -> None:
    if id(sink) in seen:
        return
    seen.add(id(sink))
    if hasattr(sink, "finalize"):
        sink.finalize()

"""StagePlan — producer-side declaration of a model subgraph DAG.

The producer owns graph capture and therefore owns graph cuts. This helper is
deliberately small: it names stages, points each stage at an exported graph,
declares the replay stream name, and resolves dependencies. The frozen model
runtime ABI still only receives graph indices + dependency indices; stream
placement lives on the graph descriptor and is part of deployment identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Stage:
    """One producer-declared stage.

    ``record`` is optional setup-time metadata for frontends that use the same
    object to drive capture. Export only consumes ``graph``, ``stream`` and
    ``after``; it never calls ``record``.
    """

    name: str
    graph: str | None = None
    stream: str = "main"
    after: Sequence[str | int] = ()
    record: Any = None

    def graph_name(self) -> str:
        return self.graph or self.name


@dataclass(frozen=True)
class StagePlan:
    """Explicit producer-owned graph cut plan.

    The default is one full stage over one graph. Multi-stage plans are valid
    only when the producer has captured/adopted matching graph names.
    """

    stages: Sequence[Stage]
    name: str = "custom"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def full(cls, graph: str = "infer", *, stage: str = "infer",
             stream: str = "main") -> "StagePlan":
        return cls((Stage(stage, graph=graph, stream=stream),), name="full")

    @classmethod
    def context_action(cls, *, stream: str = "main") -> "StagePlan":
        return cls((
            Stage("context", graph="context", stream=stream),
            Stage("action", graph="decode_only", stream=stream,
                  after=("context",)),
        ), name="context_action")

    @classmethod
    def chain(cls, name: str, graphs: Sequence[str], *, stream: str = "main",
              stage_names: Sequence[str] | None = None,
              metadata: Mapping[str, Any] | None = None) -> "StagePlan":
        """Build a linear graph chain.

        Useful for structural plans such as VLM->ViT->DiT->action_expert or
        denoise_0_4->denoise_5_9. Each stage depends on the previous stage.
        """
        if stage_names is not None and len(stage_names) != len(graphs):
            raise ValueError("stage_names length must match graphs")
        stages = []
        prev = None
        for i, graph in enumerate(graphs):
            stage = stage_names[i] if stage_names is not None else graph
            after = () if prev is None else (prev,)
            stages.append(Stage(stage, graph=graph, stream=stream,
                                after=after))
            prev = stage
        return cls(tuple(stages), name=name, metadata=metadata or {})

    @classmethod
    def from_value(cls, value: "StagePlan | str | None") -> "StagePlan":
        if value is None or value == "full":
            return cls.full()
        if value == "context_action":
            return cls.context_action()
        if isinstance(value, StagePlan):
            return value
        raise ValueError(f"unknown stage plan {value!r}")

    def validate(self, graph_names: Iterable[str] = (),
                 stream_names: Iterable[str] = ()) -> None:
        if not self.stages:
            raise ValueError("stage plan must contain at least one stage")
        seen: dict[str, int] = {}
        graphs = set(graph_names)
        streams = set(stream_names)
        for i, st in enumerate(self.stages):
            if not st.name:
                raise ValueError("stage name must be non-empty")
            if st.name in seen:
                raise ValueError(f"duplicate stage name {st.name!r}")
            seen[st.name] = i
            if graphs and st.graph_name() not in graphs:
                raise ValueError(
                    f"stage {st.name!r} references unknown graph "
                    f"{st.graph_name()!r}")
            if streams and st.stream not in streams:
                raise ValueError(
                    f"stage {st.name!r} references unknown stream "
                    f"{st.stream!r}")
            for dep in st.after:
                dep_i = dep if isinstance(dep, int) else seen.get(dep, -1)
                if dep_i < 0 or dep_i >= i:
                    raise ValueError(
                        f"stage {st.name!r} dependency {dep!r} must refer to "
                        "an earlier stage")

    def to_stage_specs(self, export_module) -> list:
        """Return ``flash_rt.runtime.export.StageSpec`` records."""
        self.validate()
        index = {st.name: i for i, st in enumerate(self.stages)}
        specs = []
        for st in self.stages:
            after = tuple(dep if isinstance(dep, int) else index[dep]
                          for dep in st.after)
            specs.append(export_module.StageSpec(st.graph_name(), after))
        return specs

    def manifest(self) -> dict[str, Any]:
        stages = [
            {
                "name": st.name,
                "graph": st.graph_name(),
                "stream": st.stream,
                "after": list(st.after),
            }
            for st in self.stages
        ]
        return {
            "name": self.name,
            "metadata": dict(self.metadata),
            "stages": stages,
        }


StagePlanFactory = Callable[..., StagePlan]
_REGISTRY: dict[tuple[str | None, str], StagePlanFactory] = {}


def _key(name: str, model: str | None) -> tuple[str | None, str]:
    if not name:
        raise ValueError("stage plan name must be non-empty")
    return (model, name)


def register_stage_plan(name: str, plan: StagePlan | StagePlanFactory, *,
                        model: str | None = None,
                        replace: bool = False) -> None:
    """Register a named producer-side stage plan.

    ``model=None`` registers a global plan. Model-specific plans shadow
    globals. ``plan`` may be a StagePlan instance or a factory accepting
    keyword arguments and returning a StagePlan.
    """
    k = _key(name, model)
    if not replace and k in _REGISTRY:
        raise ValueError(f"stage plan already registered: {name!r}"
                         f" model={model!r}")
    if isinstance(plan, StagePlan):
        _REGISTRY[k] = lambda **_: plan
    elif callable(plan):
        _REGISTRY[k] = plan
    else:
        raise TypeError("plan must be a StagePlan or callable factory")


def resolve_stage_plan(value: StagePlan | str | None, *,
                       model: str | None = None, **kwargs: Any) -> StagePlan:
    """Resolve a StagePlan object or registered name.

    Lookup order for names: model-specific first, then global. ``None`` means
    "full". Extra keyword arguments are passed to registered factories.
    """
    if isinstance(value, StagePlan):
        return value
    name = "full" if value is None else value
    if not isinstance(name, str):
        raise TypeError("stage_plan must be None, a string, or StagePlan")
    for k in (_key(name, model), _key(name, None)):
        factory = _REGISTRY.get(k)
        if factory is not None:
            return factory(**kwargs)
    try:
        return StagePlan.from_value(name)
    except ValueError as e:
        available = ", ".join(list_stage_plans(model=model)) or "<none>"
        raise ValueError(
            f"unknown stage plan {name!r} for model={model!r}; "
            f"available: {available}") from e


def list_stage_plans(*, model: str | None = None) -> list[str]:
    names = {name for m, name in _REGISTRY if m is None or m == model}
    return sorted(names)


register_stage_plan("full", StagePlan.full(), replace=True)


__all__ = [
    "Stage", "StagePlan", "StagePlanFactory",
    "register_stage_plan", "resolve_stage_plan", "list_stage_plans",
]

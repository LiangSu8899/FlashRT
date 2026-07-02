"""Producer-side subgraph capture hook utilities.

The main model pipeline stays the source of truth for the default full graph.
Optional subgraph packages register hooks before graph capture; those hooks may
capture/adopt additional graph handles and register them for export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


CaptureHook = Callable[[object, object, int], None]


@dataclass(frozen=True)
class ExportGraphRecord:
    name: str
    graph: object
    stream: str = "main"
    variants: Sequence[int] = (0,)


@dataclass
class CapturedGraphRecord:
    name: str
    cuda_graph: object
    exec_name: str
    stream: str = "main"
    variants: Sequence[int] = (0,)
    materialized: bool = False


def register_capture_hook(pipeline: object, hook: CaptureHook) -> None:
    hooks = getattr(pipeline, "_flashrt_subgraph_capture_hooks", None)
    if hooks is None:
        hooks = []
        setattr(pipeline, "_flashrt_subgraph_capture_hooks", hooks)
    if hook not in hooks:
        hooks.append(hook)
    stream_handle = getattr(pipeline, "_graph_stream", None)
    if getattr(pipeline, "_graph", None) is not None and stream_handle is not None:
        stream_int = int(getattr(stream_handle, "value", stream_handle) or 0)
        hook(pipeline, stream_handle, stream_int)


def register_frontend_capture_hook(frontend: object,
                                   installer: Callable[[object], None]) -> None:
    hooks = getattr(frontend, "_flashrt_subgraph_frontend_hooks", None)
    if hooks is None:
        hooks = []
        setattr(frontend, "_flashrt_subgraph_frontend_hooks", hooks)
    if installer not in hooks:
        hooks.append(installer)
    apply_frontend_capture_hooks(frontend)


def apply_frontend_capture_hooks(frontend: object) -> None:
    pipeline = getattr(frontend, "pipeline", None)
    if pipeline is None:
        return
    for installer in list(getattr(frontend, "_flashrt_subgraph_frontend_hooks", ())):
        installer(pipeline)


def run_capture_hooks(pipeline: object, stream_handle: object,
                      stream_int: int) -> None:
    for hook in list(getattr(pipeline, "_flashrt_subgraph_capture_hooks", ())):
        hook(pipeline, stream_handle, stream_int)


def capture_graph(pipeline: object, stream_handle: object, body: Callable[[], None],
                  *, warmups: int = 3) -> object:
    """Capture a graph using the same CUDAGraph class as the producer.

    ``pipeline._graph`` must already exist; hooks run after the default full
    graph has been captured. ``body`` is responsible for launching on the same
    stream represented by ``stream_handle``.
    """
    if getattr(pipeline, "_graph", None) is None:
        raise RuntimeError("capture_graph requires the producer full graph first")
    graph = type(pipeline._graph)()
    for _ in range(warmups):
        body()
    pipeline._cudart.cudaStreamSynchronize(stream_handle)
    graph.begin_capture(stream_handle)
    body()
    graph.end_capture(stream_handle)
    pipeline._cudart.cudaStreamSynchronize(stream_handle)
    return graph


def _stream_name(stream: str | int) -> str:
    if isinstance(stream, str):
        if not stream:
            raise ValueError("stream name must be non-empty")
        return stream
    if stream == 0:
        return "main"
    raise ValueError(
        "subgraph stream must be a StreamSpec name; integer 0 is accepted "
        "only as a backward-compatible alias for 'main'")


def register_export_graph(pipeline: object, name: str, graph: object, *,
                          stream: str | int = "main",
                          variants: Sequence[int] = (0,)) -> None:
    variants = tuple(variants)
    if not variants:
        raise ValueError("register_export_graph requires at least one variant")
    records = getattr(pipeline, "_flashrt_subgraph_export_graphs", None)
    if records is None:
        records = []
        setattr(pipeline, "_flashrt_subgraph_export_graphs", records)
    records[:] = [r for r in records if r.name != name]
    records.append(ExportGraphRecord(name, graph, _stream_name(stream),
                                     variants))


def register_captured_graph(pipeline: object, name: str, cuda_graph: object, *,
                            exec_name: str | None = None,
                            stream: str | int = "main",
                            variants: Sequence[int] = (0,)) -> None:
    variants = tuple(variants)
    if len(variants) != 1:
        raise ValueError(
            "register_captured_graph accepts one captured CUDA graph for one "
            "variant; use register_export_graph for a pre-built multi-variant "
            "exec graph")
    records = getattr(pipeline, "_flashrt_subgraph_captured_graphs", None)
    if records is None:
        records = []
        setattr(pipeline, "_flashrt_subgraph_captured_graphs", records)
    records[:] = [r for r in records if r.name != name]
    rec = CapturedGraphRecord(
        name=name,
        cuda_graph=cuda_graph,
        exec_name=exec_name or name,
        stream=_stream_name(stream),
        variants=variants,
    )
    records.append(rec)
    materialize_captured_graphs(pipeline)


def materialize_captured_graphs(pipeline: object) -> None:
    ctx = getattr(pipeline, "_exec_ctx", None)
    if ctx is None:
        return
    for rec in getattr(pipeline, "_flashrt_subgraph_captured_graphs", ()):
        if rec.materialized:
            continue
        exec_graph = ctx.graph(rec.exec_name, len(rec.variants))
        for idx, key in enumerate(rec.variants):
            exec_graph.adopt(idx, rec.cuda_graph._graph_exec.value)
        register_export_graph(pipeline, rec.name, exec_graph,
                              stream=rec.stream, variants=rec.variants)
        rec.materialized = True


def export_graph_records(pipeline: object) -> list[ExportGraphRecord]:
    return list(getattr(pipeline, "_flashrt_subgraph_export_graphs", ()))


__all__ = [
    "CaptureHook",
    "CapturedGraphRecord",
    "ExportGraphRecord",
    "capture_graph",
    "export_graph_records",
    "materialize_captured_graphs",
    "register_captured_graph",
    "register_capture_hook",
    "apply_frontend_capture_hooks",
    "register_frontend_capture_hook",
    "register_export_graph",
    "run_capture_hooks",
]

"""Capture door: ``structures.capture(fn, ...)`` — graph a hot stage.

The schedule-structure counterpart of the region doors. ``fn`` is the
hot stage of a cond_iter pipeline (a denoise loop, a decode step chain);
``capture`` warms it, records it into a CUDA graph on a side stream, and
returns a runtime whose ``replay()`` re-executes the whole stage as one
launch. Declared ``windows`` are the tensors you may rewrite between
replays (noise, observations, condition buffers) — replay reads their
current contents in place, which is what makes the graph a reusable
stage rather than a frozen trace.

Gates are built in: ``reference`` (an eager thunk producing the same
output under the same window contents) certifies parity, and the replay
is timed against eager ``fn``; a capture that is not both accurate and
net-faster raises ``CaptureRefused`` — the host keeps its eager path,
never a silently wrong or slower graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from .gates import parity_metrics


class CaptureRefused(RuntimeError):
    """The captured stage failed its parity or net-win gate."""


def _time_ms(fn: Callable[[], Any], warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@dataclass
class CapturedStage:
    """A replayable hot stage with declared swap windows."""

    graph: torch.cuda.CUDAGraph
    stream: torch.cuda.Stream
    output: Any
    windows: Mapping[str, torch.Tensor]
    certification: dict[str, Any] = field(default_factory=dict)

    def replay(self, sync: bool = True) -> Any:
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        if sync:
            torch.cuda.synchronize()
        return self.output

    def write(self, name: str, value: torch.Tensor) -> None:
        """Rewrite one declared window in place."""
        self.windows[name].copy_(value)

    def export(self, *, ports, regions=(), stages=None, roles=None,
               identity=None, manifest_extra=None):
        """One call from this captured stage to ``frt_model_runtime_v1``.

        The declared windows become the contract's boundary windows
        (name -> device pointer + bytes); ``ports`` reference them by
        name exactly as in
        :func:`flash_rt.structures.provider.export_captured_runtime`.
        This is the absorption edge: a stage captured out of any torch
        host becomes a runtime the FlashRT serving mechanisms (Nexus
        tick, capsule snapshot/restore) consume like a whitebox one.
        """
        from .provider import export_captured_runtime

        window_map = {
            name: (t.data_ptr(), t.numel() * t.element_size())
            for name, t in self.windows.items()}
        return export_captured_runtime(
            stream_handle=self.stream.cuda_stream,
            graphs=[("infer", self.graph.raw_cuda_graph_exec())],
            windows=window_map,
            ports=ports, regions=regions,
            stages=tuple(stages) if stages else ("infer",),
            roles=roles, identity=identity,
            manifest_extra=manifest_extra)


def capture(
    fn: Callable[[], Any],
    *,
    windows: Mapping[str, torch.Tensor] | None = None,
    reference: Callable[[], torch.Tensor] | None = None,
    output_of: Callable[[Any], torch.Tensor] | None = None,
    warmup: int = 3,
    gate_cos: float = 0.999,
    min_speedup: float = 1.02,
    verbose: bool = True,
) -> CapturedStage:
    """Capture ``fn`` into a replayable, gated stage.

    ``fn`` must be graph-safe: fixed shapes, no host-side branching on
    tensor values, all varying inputs read from the declared ``windows``
    buffers. ``reference`` runs the host's own eager path under the same
    window contents; parity is judged between its output and the
    replayed stage output. Pass ``gate_cos=0`` / ``min_speedup=0`` to
    skip a gate explicitly (recorded in the certification).
    """

    def say(msg: str) -> None:
        if verbose:
            print(f"[structures] {msg}", flush=True)

    windows = dict(windows or {})
    stream = torch.cuda.Stream()
    with torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(max(1, warmup)):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = fn()
        torch.cuda.synchronize()
    say(f"stage captured ({len(windows)} window(s))")

    stage = CapturedStage(graph=graph, stream=stream, output=output,
                          windows=windows)
    pick = output_of or (lambda out: out)

    cert: dict[str, Any] = {"windows": sorted(windows),
                            "gate_cos": gate_cos,
                            "min_speedup": min_speedup}
    if reference is not None and gate_cos:
        with torch.no_grad():
            want = reference().detach().float().cpu()
        got = pick(stage.replay()).detach().float().cpu()
        cos = parity_metrics(got, want)["cosine"]
        cert["parity_cos"] = round(cos, 7)
        if cos < gate_cos:
            raise CaptureRefused(
                f"captured stage parity cos {cos:.6f} < {gate_cos} vs "
                "eager reference — check that every varying input is a "
                "declared window (in-graph RNG never matches eager)")
        say(f"parity vs eager: cos={cos:.6f}")
    if min_speedup:
        eager_ms = _time_ms(lambda: fn())
        # replays run on the side stream: the timing events must be
        # recorded on that same stream or they only measure enqueue time
        for _ in range(5):
            stage.replay(sync=False)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        with torch.cuda.stream(stream):
            start.record()
        for _ in range(20):
            stage.replay(sync=False)
        with torch.cuda.stream(stream):
            end.record()
        torch.cuda.synchronize()
        replay_ms = start.elapsed_time(end) / 20
        cert["eager_ms"] = round(eager_ms, 3)
        cert["replay_ms"] = round(replay_ms, 3)
        cert["speedup"] = round(eager_ms / replay_ms, 4)
        if eager_ms / replay_ms < min_speedup:
            raise CaptureRefused(
                f"captured stage is not a net win "
                f"({eager_ms / replay_ms:.3f}x vs eager, margin "
                f"{min_speedup}) — host keeps its eager path")
        say(f"net win: {eager_ms:.2f} -> {replay_ms:.2f} ms "
            f"({eager_ms / replay_ms:.3f}x)")
    stage.certification = cert
    return stage

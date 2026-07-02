"""RuntimeExport — package a captured FlashRT model as ``frt_runtime_export_v1``.

The phase-1 PRODUCER of the runtime-export ABI (runtime/include/flashrt/runtime.h):
the Python frontend captures graphs and allocates buffers as it does today, then
assembles one C struct a native consumer (e.g. a capsule/state host) adopts.
Setup/dev bridge only — after the hand-off, the hot path is native replay; this
process merely stays resident to keep the CUDA graphs and buffers alive.

The canonical identity string and its fingerprint are computed by the C builder
(one implementation, one hashing rule) — never in Python.

Build the native modules first (standalone, like exec/):
    cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release
    cmake --build runtime/build -j
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _import_native():
    try:
        import _flashrt_runtime as _c  # noqa: F401
        return _c
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    candidate = os.path.join(repo, "runtime", "build")
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)
    try:
        import _flashrt_runtime as _c  # noqa: F401
        return _c
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Could not import _flashrt_runtime. Build it first:\n"
            "  cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build runtime/build -j"
        ) from e


_c = _import_native()

# Role / region-flag masks (ABI-frozen values, re-exported from the C module).
ROLE_INPUT = int(_c.ROLE_INPUT)
ROLE_OUTPUT = int(_c.ROLE_OUTPUT)
ROLE_STATE = int(_c.ROLE_STATE)
ROLE_SCRATCH = int(_c.ROLE_SCRATCH)
REGION_SNAPSHOT = int(_c.REGION_SNAPSHOT)
REGION_RESTORE = int(_c.REGION_RESTORE)
REGION_DEFAULT = REGION_SNAPSHOT | REGION_RESTORE

_ROLE_NAMES = {
    "input": ROLE_INPUT, "output": ROLE_OUTPUT,
    "state": ROLE_STATE, "scratch": ROLE_SCRATCH,
}

# Model-runtime enums (ABI-frozen values, re-exported from the C module).
MODALITY = {
    "tensor": int(_c.MOD_TENSOR), "image": int(_c.MOD_IMAGE),
    "text": int(_c.MOD_TEXT), "state": int(_c.MOD_STATE),
    "action": int(_c.MOD_ACTION), "audio": int(_c.MOD_AUDIO),
    "depth": int(_c.MOD_DEPTH), "force": int(_c.MOD_FORCE),
}
DTYPE = {
    "u8": int(_c.DTYPE_U8), "f32": int(_c.DTYPE_F32), "f16": int(_c.DTYPE_F16),
    "bf16": int(_c.DTYPE_BF16), "i32": int(_c.DTYPE_I32),
    "i64": int(_c.DTYPE_I64),
}
LAYOUT = {
    "flat": int(_c.LAYOUT_FLAT), "hwc": int(_c.LAYOUT_HWC),
    "nhwc": int(_c.LAYOUT_NHWC), "chw": int(_c.LAYOUT_CHW),
    "nchw": int(_c.LAYOUT_NCHW),
}
DIRECTION = {"in": int(_c.PORT_IN), "out": int(_c.PORT_OUT)}
UPDATE = {
    "swap": int(_c.PORT_SWAP), "staged": int(_c.PORT_STAGED),
    "setup": int(_c.PORT_SETUP),
}


def _enum(table: Mapping[str, int], value: int | str) -> int:
    return value if isinstance(value, int) else table[value]


def _role_mask(role: int | str | Sequence[str]) -> int:
    """Accept an int mask, a name ("input"), or names ("input", "output")."""
    if isinstance(role, int):
        return role
    if isinstance(role, str):
        role = [role]
    mask = 0
    for r in role:
        mask |= _ROLE_NAMES[r]
    return mask


@dataclass
class StreamSpec:
    name: str
    stream_id: int                 # frt_ctx-scoped id (Ctx.stream / Ctx.wrap_stream)
    priority: int = 0
    native_handle: int = 0         # raw backend stream handle (e.g. cudaStream_t int)


@dataclass
class GraphSpec:
    name: str
    graph: Any                     # exec-binding Graph (has .raw())
    default_key: int = 0
    keys: Sequence[int] = (0,)
    stream: str = "main"           # StreamSpec.name this graph replays on by default


@dataclass
class BufferSpec:
    name: str
    buffer: Any                    # exec-binding Buffer (has .raw() / .nbytes())
    role: int | str | Sequence[str] = "input"


@dataclass
class RegionSpec:
    name: str
    buffer: Any                    # exec-binding Buffer (has .raw() / .nbytes())
    offset: int = 0
    nbytes: int | None = None      # None = whole buffer
    flags: int = REGION_DEFAULT


@dataclass
class PortSpec:
    """One dynamic input/output of the tick (see flashrt/model_runtime.h).

    ``update`` is the load-bearing field: "swap" ports are raw device windows
    the host writes directly; "staged" ports go through the producer's
    ``set_input``/``get_output`` callables; "setup" is illegal inside a tick.
    """

    name: str
    modality: int | str            # MODALITY name or value
    dtype: int | str = "bf16"      # device-side tensor dtype
    layout: int | str = "flat"
    direction: int | str = "in"
    update: int | str = "swap"
    required: bool = False
    shape: Sequence[int] = ()
    cadence_hz: int = 0
    buffer: Any = None             # exec-binding Buffer for the SWAP window
    offset: int = 0
    nbytes: int | None = None      # None = whole buffer (when buffer is set)


@dataclass
class StageSpec:
    """One schedulable stage: an export graph + dependencies on earlier
    stages (indices). Declared order is the sequential ``step`` order."""

    graph: str                     # GraphSpec.name
    after: Sequence[int] = ()


@dataclass
class RuntimeExport:
    """A finished export. ``ptr`` is the ``frt_runtime_export_v1*`` to hand to a
    native consumer. The export holds one reference; this object anchors every
    Python object behind the handles for as long as it (or any native retain)
    lives."""

    ptr: int
    fingerprint: int
    identity: str
    manifest: str | None
    _anchor: Any = field(repr=False, default=None)

    def counts(self) -> dict:
        return dict(_c.export_counts(self.ptr))

    def release(self) -> None:
        """Drop the producer's reference (native retains keep it alive)."""
        if self.ptr:
            _c.export_release(self.ptr)
            self.ptr = 0


@dataclass
class ModelRuntime:
    """A finished model runtime. ``ptr`` is the ``frt_model_runtime_v1*`` a
    native consumer adopts (it retains/releases only this object; the export
    reference is internal). ``export_ptr`` points at the embedded export."""

    ptr: int
    export_ptr: int
    fingerprint: int
    identity: str
    manifest: str | None
    _anchor: Any = field(repr=False, default=None)

    def ports(self) -> list:
        return list(_c.model_ports(self.ptr))

    def stages(self) -> list:
        return list(_c.model_stages(self.ptr))

    def release(self) -> None:
        """Drop the producer's reference (native retains keep it alive)."""
        if self.ptr:
            _c.model_release(self.ptr)
            self.ptr = 0


class _Anchor:
    """Keeps the exec-binding wrappers (Ctx/Graph/Buffer) and the producer's
    owner object alive for the lifetime of the export. This is the object the
    C holder references; its destruction (GIL-safe, from any thread) is the
    release path."""

    def __init__(self, objs):
        self._objs = objs


def build_export(
    ctx: Any,
    *,
    streams: Sequence[StreamSpec],
    graphs: Sequence[GraphSpec],
    buffers: Sequence[BufferSpec] = (),
    regions: Sequence[RegionSpec] = (),
    identity: Mapping[str, str],
    manifest_extra: Mapping[str, Any] | None = None,
    owner: Any = None,
) -> RuntimeExport:
    """Assemble an ``frt_runtime_export_v1`` from exec-binding objects.

    - ``ctx``: the exec-binding Ctx (must outlive the export — it is anchored).
    - ``identity``: canonical identity pairs, emitted in the given order. Must
      include everything that makes stored state deployment-bound: a weights
      digest, quant mode, kernel version, arch. Structural identity (graph
      names, region layout) is appended by the C builder automatically.
    - ``manifest_extra``: merged into the auto-generated discovery manifest.
    - ``owner``: the producer object to keep alive (e.g. the model pipeline).
    """
    b, anchor, manifest_json = _assemble(
        ctx, streams=streams, graphs=graphs, buffers=buffers, regions=regions,
        ports=(), stages=(), identity=identity, manifest_extra=manifest_extra,
        owner=owner)
    ptr = b.finish(anchor)
    return RuntimeExport(
        ptr=ptr,
        fingerprint=int(_c.export_fingerprint(ptr)),
        identity=_c.export_identity(ptr),
        manifest=manifest_json,
        _anchor=anchor,
    )


def build_model_runtime(
    ctx: Any,
    *,
    streams: Sequence[StreamSpec],
    graphs: Sequence[GraphSpec],
    buffers: Sequence[BufferSpec] = (),
    regions: Sequence[RegionSpec] = (),
    ports: Sequence[PortSpec] = (),
    stages: Sequence[StageSpec] = (),
    identity: Mapping[str, str],
    manifest_extra: Mapping[str, Any] | None = None,
    owner: Any = None,
    set_input=None,
    get_output=None,
    prepare=None,
    step=None,
) -> ModelRuntime:
    """Assemble an ``frt_model_runtime_v1``: an export plus the dynamic-IO
    contract (ports, stage DAG, optional verb callables) under one identity —
    a port-schema change changes the fingerprint.

    Verb callables (all optional; absent verbs report unsupported):
      ``set_input(port, payload: bytes, stream) -> int``,
      ``get_output(port, stream) -> bytes``,
      ``prepare(graph, key) -> int``, ``step() -> int``.
    They run under GIL-acquiring trampolines, so a native consumer may call
    them from any thread. SWAP ports need no callable — hosts write the
    declared buffer window directly.
    """
    b, anchor, manifest_json = _assemble(
        ctx, streams=streams, graphs=graphs, buffers=buffers, regions=regions,
        ports=ports, stages=stages, identity=identity,
        manifest_extra=manifest_extra, owner=owner)
    anchor._objs.append((set_input, get_output, prepare, step))
    ptr = b.finish_model(anchor, set_input=set_input, get_output=get_output,
                         prepare=prepare, step=step)
    export_ptr = int(_c.model_export_ptr(ptr))
    return ModelRuntime(
        ptr=ptr,
        export_ptr=export_ptr,
        fingerprint=int(_c.export_fingerprint(export_ptr)),
        identity=_c.export_identity(export_ptr),
        manifest=manifest_json,
        _anchor=anchor,
    )


def _assemble(ctx, *, streams, graphs, buffers, regions, ports, stages,
              identity, manifest_extra, owner):
    if not streams:
        raise ValueError("at least one stream is required")
    stream_ids = {s.name: s.stream_id for s in streams}
    graph_index = {g.name: i for i, g in enumerate(graphs)}

    b = _c.Builder(ctx.raw())
    for s in streams:
        b.add_stream(s.name, s.stream_id, s.priority, s.native_handle)
    for g in graphs:
        if g.stream not in stream_ids:
            raise ValueError(f"graph {g.name!r} references unknown stream {g.stream!r}")
        b.add_graph(g.name, g.graph.raw(), g.default_key, list(g.keys),
                    stream_ids[g.stream])
    for buf in buffers:
        b.add_buffer(buf.name, buf.buffer.raw(), buf.buffer.nbytes(),
                     _role_mask(buf.role))
    for r in regions:
        nbytes = r.buffer.nbytes() if r.nbytes is None else r.nbytes
        b.add_region(r.name, r.buffer.raw(), r.offset, nbytes, r.flags)
    for p in ports:
        buffer_raw = p.buffer.raw() if p.buffer is not None else 0
        nbytes = p.nbytes
        if nbytes is None:
            nbytes = p.buffer.nbytes() if p.buffer is not None else 0
        b.add_port(p.name, _enum(MODALITY, p.modality), _enum(DTYPE, p.dtype),
                   _enum(LAYOUT, p.layout), _enum(DIRECTION, p.direction),
                   _enum(UPDATE, p.update), int(bool(p.required)),
                   list(p.shape), p.cadence_hz, buffer_raw, p.offset, nbytes)
    for st in stages:
        if st.graph not in graph_index:
            raise ValueError(f"stage references unknown graph {st.graph!r}")
        b.add_stage(graph_index[st.graph], list(st.after))
    for k, v in identity.items():
        b.add_identity(str(k), str(v))

    manifest = {
        "streams": [{"name": s.name, "priority": s.priority} for s in streams],
        "graphs": [{"name": g.name, "default_key": g.default_key,
                    "keys": list(g.keys), "stream": g.stream} for g in graphs],
        "buffers": [{"name": buf.name, "bytes": buf.buffer.nbytes(),
                     "role": _role_mask(buf.role)} for buf in buffers],
        "capsule_regions": [{"name": r.name, "offset": r.offset,
                             "bytes": (r.buffer.nbytes() if r.nbytes is None else r.nbytes),
                             "flags": r.flags} for r in regions],
    }
    if ports:
        manifest["ports"] = [
            {"name": p.name, "modality": _enum(MODALITY, p.modality),
             "dtype": _enum(DTYPE, p.dtype), "layout": _enum(LAYOUT, p.layout),
             "direction": _enum(DIRECTION, p.direction),
             "update": _enum(UPDATE, p.update), "required": bool(p.required),
             "shape": list(p.shape), "cadence_hz": p.cadence_hz}
            for p in ports]
    if stages:
        manifest["stages"] = [
            {"graph": st.graph, "after": list(st.after)} for st in stages]
    if manifest_extra:
        manifest.update(dict(manifest_extra))
    manifest_json = json.dumps(manifest, sort_keys=True)
    b.set_manifest(manifest_json)

    anchor = _Anchor([ctx, [g.graph for g in graphs],
                      [buf.buffer for buf in buffers],
                      [r.buffer for r in regions],
                      [p.buffer for p in ports if p.buffer is not None],
                      owner])
    return b, anchor, manifest_json


__all__ = [
    "RuntimeExport", "ModelRuntime",
    "StreamSpec", "GraphSpec", "BufferSpec", "RegionSpec", "PortSpec",
    "StageSpec",
    "build_export", "build_model_runtime",
    "ROLE_INPUT", "ROLE_OUTPUT", "ROLE_STATE", "ROLE_SCRATCH",
    "REGION_SNAPSHOT", "REGION_RESTORE", "REGION_DEFAULT",
    "MODALITY", "DTYPE", "LAYOUT", "DIRECTION", "UPDATE",
]

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
    if not streams:
        raise ValueError("at least one stream is required")
    stream_ids = {s.name: s.stream_id for s in streams}

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
    if manifest_extra:
        manifest.update(dict(manifest_extra))
    manifest_json = json.dumps(manifest, sort_keys=True)
    b.set_manifest(manifest_json)

    anchor = _Anchor([ctx, [g.graph for g in graphs],
                      [buf.buffer for buf in buffers],
                      [r.buffer for r in regions], owner])
    ptr = b.finish(anchor)
    return RuntimeExport(
        ptr=ptr,
        fingerprint=int(_c.export_fingerprint(ptr)),
        identity=_c.export_identity(ptr),
        manifest=manifest_json,
        _anchor=anchor,
    )


__all__ = [
    "RuntimeExport", "StreamSpec", "GraphSpec", "BufferSpec", "RegionSpec",
    "build_export",
    "ROLE_INPUT", "ROLE_OUTPUT", "ROLE_STATE", "ROLE_SCRATCH",
    "REGION_SNAPSHOT", "REGION_RESTORE", "REGION_DEFAULT",
]

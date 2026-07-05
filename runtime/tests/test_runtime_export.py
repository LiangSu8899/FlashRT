"""Runtime-export acceptance — model-free, one trivial captured graph.

Verifies the producer side of frt_runtime_export_v1:
  1. struct ABI layout: read the raw struct back via a ctypes mirror and check
     every field against what the builder was given (catches packing drift);
  2. fingerprint: deterministic across builds, sensitive to identity pairs and
     to capsule-region ORDER (restore matches by position), insensitive to the
     manifest;
  3. lifetime: consumer retain keeps the Python anchor alive after the
     producer releases; final release frees it (observed via weakref);
  4. the exported graph replays through the exported handles.

Run from the repo root (after building exec/ and runtime/):
    PYTHONPATH=.:./exec/build:./runtime/build python runtime/tests/test_runtime_export.py
"""

import ctypes
import gc
import weakref

import _flashrt_exec as ex
import _flashrt_runtime as rt

from flash_rt.runtime.export import (
    BufferSpec, GraphSpec, RegionSpec, StreamSpec, build_export,
    REGION_DEFAULT, ROLE_INPUT, ROLE_OUTPUT,
)

CHECKS = []


def check(name, ok):
    CHECKS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


# --- ctypes mirror of the v1 ABI (must match flashrt/runtime.h exactly) ---
class StreamDesc(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("stream_id", ctypes.c_int),
                ("priority", ctypes.c_int), ("native_handle", ctypes.c_void_p)]


class GraphDesc(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("handle", ctypes.c_void_p),
                ("default_key", ctypes.c_uint64),
                ("keys", ctypes.POINTER(ctypes.c_uint64)),
                ("n_keys", ctypes.c_uint64), ("stream_id", ctypes.c_int)]


class BufferDesc(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("handle", ctypes.c_void_p),
                ("bytes", ctypes.c_uint64), ("role", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32)]


class RegionDesc(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("buffer", ctypes.c_void_p),
                ("offset", ctypes.c_uint64), ("bytes", ctypes.c_uint64),
                ("flags", ctypes.c_uint32), ("reserved", ctypes.c_uint32)]


class ExportV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("ctx", ctypes.c_void_p),
        ("streams", ctypes.POINTER(StreamDesc)), ("n_streams", ctypes.c_uint64),
        ("graphs", ctypes.POINTER(GraphDesc)), ("n_graphs", ctypes.c_uint64),
        ("buffers", ctypes.POINTER(BufferDesc)), ("n_buffers", ctypes.c_uint64),
        ("capsule_regions", ctypes.POINTER(RegionDesc)),
        ("n_capsule_regions", ctypes.c_uint64),
        ("fingerprint", ctypes.c_uint64),
        ("identity", ctypes.c_char_p), ("manifest_json", ctypes.c_char_p),
        ("owner", ctypes.c_void_p),
        ("retain", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
        ("release", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]


def make_setup():
    """One trivial 'model': src --memcpy--> dst captured as a graph."""
    ctx = ex.Ctx()
    sid = ctx.stream(0)
    src = ctx.buffer("src", 4096)
    dst = ctx.buffer("dst", 4096)
    g = ctx.graph("copy", 1)

    def record(stream):
        ex.memcpy_async(dst.dptr(), src.dptr(), 4096, stream)

    g.capture(0, record)
    return ctx, sid, src, dst, g


def build(ctx, sid, src, dst, g, identity=None, regions=None, manifest=None):
    return build_export(
        ctx,
        streams=[StreamSpec("main", sid)],
        graphs=[GraphSpec("copy", g, 0, (0,))],
        buffers=[BufferSpec("src", src, "input"),
                 BufferSpec("dst", dst, ("input", "output"))],
        regions=(regions if regions is not None else
                 [RegionSpec("state_a", src), RegionSpec("state_b", dst)]),
        identity=identity or {"model": "trivial", "quant": "none"},
        manifest_extra=manifest,
    )


def main():
    CHECKS.clear()
    setup = make_setup()
    ctx, sid, src, dst, g = setup

    print("== struct layout (ctypes mirror vs builder input) ==")
    exp = build(*setup)
    e = ExportV1.from_address(exp.ptr)
    check("abi_version == 1", e.abi_version == 1)
    check("struct_size matches mirror", e.struct_size == ctypes.sizeof(ExportV1))
    check("ctx pointer round-trips", e.ctx == ctx.raw())
    check("n_streams/n_graphs/n_buffers/n_regions", (
        e.n_streams == 1 and e.n_graphs == 1 and e.n_buffers == 2
        and e.n_capsule_regions == 2))
    check("stream desc", (
        e.streams[0].name == b"main" and e.streams[0].stream_id == sid
        and e.streams[0].priority == 0))
    gd = e.graphs[0]
    check("graph desc", (
        gd.name == b"copy" and gd.handle == g.raw() and gd.default_key == 0
        and gd.n_keys == 1 and gd.keys[0] == 0 and gd.stream_id == sid))
    check("buffer descs", (
        e.buffers[0].name == b"src" and e.buffers[0].handle == src.raw()
        and e.buffers[0].bytes == 4096 and e.buffers[0].role == ROLE_INPUT
        and e.buffers[1].role == (ROLE_INPUT | ROLE_OUTPUT)))
    rd = e.capsule_regions[0]
    check("region desc", (
        rd.name == b"state_a" and rd.buffer == src.raw() and rd.offset == 0
        and rd.bytes == 4096 and rd.flags == REGION_DEFAULT))
    check("identity string surfaces regions", (
        b"region:0:state_a:0:4096" in ctypes.string_at(
            ctypes.cast(e.identity, ctypes.c_void_p))))
    check("fingerprint == hash(identity)", (
        e.fingerprint == rt.fingerprint(
            ctypes.string_at(ctypes.cast(e.identity, ctypes.c_void_p)))))

    print("== fingerprint semantics ==")
    exp_same = build(*setup)
    check("deterministic across builds", exp_same.fingerprint == exp.fingerprint)
    exp_id = build(*setup, identity={"model": "trivial", "quant": "fp8"})
    check("sensitive to identity pairs", exp_id.fingerprint != exp.fingerprint)
    exp_ord = build(*setup, regions=[RegionSpec("state_b", dst),
                                     RegionSpec("state_a", src)])
    check("sensitive to region ORDER", exp_ord.fingerprint != exp.fingerprint)
    exp_man = build(*setup, manifest={"note": "manifest edits are free"})
    check("insensitive to manifest", exp_man.fingerprint == exp.fingerprint)
    for x in (exp_same, exp_id, exp_ord, exp_man):
        x.release()

    print("== lifetime (retain/release vs the Python anchor) ==")
    anchor_ref = weakref.ref(exp._anchor)
    rt.export_retain(exp.ptr)          # the "consumer" adopts
    ptr = exp.ptr
    exp._anchor = None
    exp.release()                       # producer drops its reference
    gc.collect()
    check("consumer retain keeps anchor alive", anchor_ref() is not None)
    rt.export_release(ptr)              # consumer done
    gc.collect()
    check("final release frees anchor", anchor_ref() is None)

    print("== exported graph still replays ==")
    exp2 = build(*setup)
    e2 = ExportV1.from_address(exp2.ptr)
    rc = g.replay(0, e2.streams[0].stream_id)
    check("replay on exported stream_id rc==0", rc == 0)
    exp2.release()

    failed = [n for n, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))
    print("PASS — runtime export ABI, fingerprint rule, and lifetime verified")


if __name__ == "__main__":
    main()


def test_main():
    main()

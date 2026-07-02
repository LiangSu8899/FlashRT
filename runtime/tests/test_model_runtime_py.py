"""Model-runtime acceptance for the PYTHON producer — model-free, one GPU graph.

Verifies build_model_runtime end to end:
  1. struct ABI layout: a ctypes mirror of frt_model_runtime_v1 (+ port/stage
     descriptors) reads back exactly what the specs declared;
  2. identity: port schema and stage DAG are fingerprinted; a port shape
     change changes the fingerprint;
  3. verbs: Python callables are reachable THROUGH THE C FUNCTION POINTERS
     (the same entry a native consumer uses), including error translation
     and the bytes-capacity protocol of get_output;
  4. lifetime: one reference spans export + ports + verbs; the anchor dies
     on the final release.

Run from the repo root (after building exec/ and runtime/):
    PYTHONPATH=.:./exec/build:./runtime/build python runtime/tests/test_model_runtime_py.py
"""

import ctypes
import gc
import weakref

import _flashrt_exec as ex
import _flashrt_runtime as rt

from flash_rt.runtime.export import (
    BufferSpec, GraphSpec, PortSpec, RegionSpec, StageSpec, StreamSpec,
    build_model_runtime, DTYPE, MODALITY, UPDATE,
)

CHECKS = []


def check(name, ok):
    CHECKS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


# --- ctypes mirrors of the v1 ABI (must match flashrt/model_runtime.h) ---
class PortDesc(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p),
                ("modality", ctypes.c_uint32), ("dtype", ctypes.c_uint32),
                ("layout", ctypes.c_uint32), ("direction", ctypes.c_uint32),
                ("update", ctypes.c_uint32), ("required", ctypes.c_uint32),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("rank", ctypes.c_uint32),
                ("cadence_hint_hz", ctypes.c_uint32),
                ("buffer", ctypes.c_void_p),
                ("offset", ctypes.c_uint64), ("bytes", ctypes.c_uint64)]


class StageDesc(ctypes.Structure):
    _fields_ = [("graph", ctypes.c_uint32), ("n_after", ctypes.c_uint32),
                ("after", ctypes.POINTER(ctypes.c_uint32))]


class Verbs(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
                ("set_input", ctypes.c_void_p), ("get_output", ctypes.c_void_p),
                ("prepare", ctypes.c_void_p), ("step", ctypes.c_void_p),
                ("last_error", ctypes.c_void_p)]


class ModelV1(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
                ("exp", ctypes.c_void_p),
                ("ports", ctypes.POINTER(PortDesc)), ("n_ports", ctypes.c_uint64),
                ("stages", ctypes.POINTER(StageDesc)), ("n_stages", ctypes.c_uint64),
                ("self_", ctypes.c_void_p), ("verbs", Verbs),
                ("owner", ctypes.c_void_p),
                ("retain", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
                ("release", ctypes.CFUNCTYPE(None, ctypes.c_void_p))]


def make_setup():
    ctx = ex.Ctx()
    sid = ctx.stream(0)
    src = ctx.buffer("src", 4096)
    dst = ctx.buffer("dst", 4096)
    g = ctx.graph("infer", 1)

    def record(stream):
        ex.memcpy_async(dst.dptr(), src.dptr(), 4096, stream)

    g.capture(0, record)
    return ctx, sid, src, dst, g


def build(setup, img_h=224, verbs=None):
    ctx, sid, src, dst, g = setup
    verbs = verbs or {}
    return build_model_runtime(
        ctx,
        streams=[StreamSpec("main", sid)],
        graphs=[GraphSpec("infer", g, 0, (0,))],
        buffers=[BufferSpec("src", src, "input"),
                 BufferSpec("dst", dst, "output")],
        regions=[RegionSpec("boundary", dst)],
        ports=[
            PortSpec("images", "image", "bf16", "nhwc", "in", "staged",
                     required=True, shape=(1, img_h, 224, 3), cadence_hz=30),
            PortSpec("obs", "state", "bf16", "flat", "in", "swap",
                     shape=(32,), buffer=src),
            PortSpec("actions", "action", "bf16", "flat", "out", "staged",
                     shape=(4,), buffer=dst),
        ],
        stages=[StageSpec("infer")],
        identity={"model": "trivial", "quant": "none"},
        **verbs,
    )


def main():
    setup = make_setup()
    ctx, sid, src, dst, g = setup

    print("== struct layout (ctypes mirror vs specs) ==")
    calls = {"set_input": [], "step": 0}

    def py_set_input(port, payload, stream):
        calls["set_input"].append((port, bytes(payload), stream))
        return 0

    def py_get_output(port, stream):
        if port != 2:
            raise ValueError("only the actions port is decodable")
        return b"\x01\x02\x03\x04"

    def py_step():
        calls["step"] += 1
        return g.replay(0, sid)

    mr = build(setup, verbs=dict(set_input=py_set_input,
                                 get_output=py_get_output, step=py_step))
    m = ModelV1.from_address(mr.ptr)
    check("abi stamp", m.abi_version == int(rt.MODEL_ABI_VERSION)
          and m.struct_size == ctypes.sizeof(ModelV1))
    check("embedded export pointer", m.exp == mr.export_ptr)
    check("port count", m.n_ports == 3 and m.n_stages == 1)
    p0 = m.ports[0]
    check("port desc round-trips", (
        p0.name == b"images" and p0.modality == MODALITY["image"]
        and p0.dtype == DTYPE["bf16"] and p0.update == UPDATE["staged"]
        and p0.required == 1 and p0.rank == 4 and p0.shape[1] == 224
        and p0.cadence_hint_hz == 30))
    check("swap port exposes the device window", (
        m.ports[1].update == UPDATE["swap"]
        and m.ports[1].buffer == src.raw()
        and m.ports[1].bytes == 4096))
    check("stage desc", m.stages[0].graph == 0 and m.stages[0].n_after == 0)
    check("introspection matches the mirror", (
        mr.ports()[0]["name"] == "images"
        and mr.stages() == [{"graph": 0, "after": []}]))

    print("== identity / fingerprint ==")
    check("identity carries port + stage records", (
        "port:0:images:" in mr.identity and "stage:0:0:" in mr.identity))
    mr2 = build(setup, img_h=256)
    check("port shape change changes the fingerprint",
          mr2.fingerprint != mr.fingerprint)
    mr2.release()

    print("== verbs through the C function pointers ==")
    rc = rt.model_set_input(mr.ptr, 1, b"\xAA\xBB", -1)
    check("set_input reaches the Python callable",
          rc == 0 and calls["set_input"] == [(1, b"\xAA\xBB", -1)])
    rc, payload, written = rt.model_get_output(mr.ptr, 2, 16, -1)
    check("get_output returns the producer's bytes",
          rc == 0 and payload == b"\x01\x02\x03\x04" and written == 4)
    rc, _, written = rt.model_get_output(mr.ptr, 2, 2, -1)
    check("get_output reports the needed size on short buffers",
          rc == -5 and written == 4)
    rc, _, _ = rt.model_get_output(mr.ptr, 0, 16, -1)
    check("producer exceptions become status + last_error",
          rc == -1 and "actions port" in rt.model_last_error(mr.ptr))
    check("step replays through the producer",
          rt.model_step(mr.ptr) == 0 and calls["step"] == 1)

    print("== lifetime ==")
    anchor_ref = weakref.ref(mr._anchor)
    rt.model_retain(mr.ptr)          # the "consumer" adopts
    ptr = mr.ptr
    mr._anchor = None
    mr.release()                      # producer drops its reference
    gc.collect()
    check("consumer retain keeps the anchor alive", anchor_ref() is not None)
    rt.model_release(ptr)             # consumer done
    gc.collect()
    check("final release frees the anchor", anchor_ref() is None)

    failed = [n for n, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))
    print("PASS — Python-produced model runtime: layout, identity, verbs, lifetime")


if __name__ == "__main__":
    main()

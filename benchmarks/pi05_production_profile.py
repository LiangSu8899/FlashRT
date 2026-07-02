#!/usr/bin/env python3
"""Production-profile Pi0.5 benchmark: Python predict vs model-runtime tick.

This is intentionally wall-clock, not graph-only:

  cold:
    import/runtime setup, load_model, first predict (prompt/calibrate/capture),
    export, and native model-runtime adapter creation.

  steady:
    host uint8 images -> preprocessing/staging -> replay -> D2H/postprocess ->
    float32 robot actions.

The current native Pi0.5 model-runtime exposes diffusion noise as a SWAP port.
There is not yet a native C++ RNG/noise-fill verb, so this benchmark reports
the native loop with a fixed/prewritten noise window separately from the Python
public predict loop, which generates fresh noise each call.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _stats(xs: list[float]) -> str:
    xs = sorted(float(x) for x in xs)
    if not xs:
        return "n=0"

    def pct(p: float) -> float:
        return xs[int(p * (len(xs) - 1))]

    return (
        f"n={len(xs)} p50={pct(0.50):.3f} p90={pct(0.90):.3f} "
        f"p95={pct(0.95):.3f} mean={statistics.fmean(xs):.3f} "
        f"min={xs[0]:.3f} max={xs[-1]:.3f}"
    )


def _load_gate_helpers():
    gate = ROOT / "cpp/tests/gate_pi05_model_runtime_export.py"
    spec = importlib.util.spec_from_file_location("pi05_model_runtime_gate", gate)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module: {gate}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _time_s(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - t0


def _bench_ms(label: str, iters: int, fn) -> list[float]:
    times = []
    for i in range(iters):
        t0 = time.perf_counter()
        fn(i)
        times.append((time.perf_counter() - t0) * 1000.0)
    print(f"{label:<30}: {_stats(times)}", flush=True)
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-views", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--prompt", default="pick up the red block")
    ap.add_argument("--fp8", action="store_true",
                    help="use production FP8/BF16 path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--lib", default=str(
        ROOT / "cpp/build-container/libflashrt_cpp_pi05_c.so"))
    args = ap.parse_args()

    helper, import_s = _time_s(_load_gate_helpers)
    np = helper.np
    torch = helper.torch
    flash_rt = helper.flash_rt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("===== PI0.5 PRODUCTION PROFILE =====", flush=True)
    print(f"device                  : {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_capability(0)}", flush=True)
    print(f"precision               : {'fp8/bf16' if args.fp8 else 'fp16'}", flush=True)
    print(f"checkpoint              : {args.checkpoint}", flush=True)
    print(f"num_views/steps         : {args.num_views}/{args.steps}", flush=True)
    print("scope                   : host uint8 images -> robot float32 actions",
          flush=True)

    image_seq = [
        helper._make_images(args.num_views, args.seed + i)
        for i in range(max(args.iters, args.warmup) + 1)
    ]
    obs_seq = [helper._make_obs(images) for images in image_seq]
    view_seq = [helper._make_image_views(images) for images in image_seq]

    def load_model():
        return flash_rt.load_model(
            args.checkpoint,
            framework="torch",
            config="pi05",
            hardware="auto",
            num_views=args.num_views,
            num_steps=args.steps,
            cache_frames=1,
            use_fp8=bool(args.fp8),
            use_fp16=not args.fp8,
        )

    model, load_s = _time_s(load_model)
    torch.cuda.synchronize()

    _, first_predict_s = _time_s(
        lambda: model.predict(image_seq[0], prompt=args.prompt))
    torch.cuda.synchronize()

    pipe = model._pipe
    pl = pipe.pipeline
    assert getattr(pl, "_graph", None) is not None, "Pi05 graph was not captured"

    def make_model_runtime():
        export = pl.export_runtime(identity={"bench": "pi05_production_profile"})
        lib = helper._load_lib(args.lib)
        dtype_id, torch_dtype = helper._dtype_from_frontend(pipe)
        action_mean, action_stddev = helper._action_affine(pipe.norm_stats)

        cfg = helper.Pi05RuntimeConfig()
        cfg.struct_size = helper.ctypes.sizeof(helper.Pi05RuntimeConfig)
        cfg.num_views = args.num_views
        cfg.chunk = int(pl.chunk_size)
        cfg.model_action_dim = 32
        cfg.robot_action_dim = helper.LIBERO_ACTION_DIM
        cfg.action_mean = action_mean.ctypes.data_as(
            helper.ctypes.POINTER(helper.ctypes.c_float))
        cfg.n_action_mean = action_mean.size
        cfg.action_stddev = action_stddev.ctypes.data_as(
            helper.ctypes.POINTER(helper.ctypes.c_float))
        cfg.n_action_stddev = action_stddev.size
        cfg.graph_name = b"infer"
        cfg.image_buffer_name = b"observation_images_normalized"
        cfg.action_buffer_name = b"diffusion_noise"
        cfg.image_dtype = dtype_id
        cfg.action_dtype = dtype_id

        m_ptr = helper.ctypes.c_void_p()
        rc = lib.frt_pi05_model_runtime_create(
            helper.ctypes.c_void_p(export.ptr),
            helper.ctypes.byref(cfg),
            helper.ctypes.byref(m_ptr),
        )
        if rc != 0:
            export.release()
            raise RuntimeError(f"frt_pi05_model_runtime_create failed rc={rc}")
        m = helper.ctypes.cast(
            m_ptr, helper.ctypes.POINTER(helper.FrtModelRuntimeV1)).contents
        return export, m, torch_dtype

    (export, m, torch_dtype), export_adopt_s = _time_s(make_model_runtime)

    try:
        start_noise = helper._seed_noise(pipe, pl, args.seed + 1009)

        # Fixed-noise equivalence check between Python manual replay and the
        # generic native model-runtime tick.
        py_raw = helper._python_replay(pipe, pl, obs_seq[0], start_noise)
        helper._upload_bytes(pl.input_noise_buf, start_noise)
        helper._model_set_images(m, view_seq[0])
        cxx_actions = helper._model_step_get_actions(m, int(pl.chunk_size))
        cxx_raw = helper._read(pl.input_noise_buf)
        py_raw_f = helper._raw_to_float(py_raw, torch_dtype).reshape(
            int(pl.chunk_size), 32)
        py_actions = helper.unnormalize_actions(py_raw_f, pipe.norm_stats)[
            :, :helper.LIBERO_ACTION_DIM].astype(np.float32)
        raw_exact = bool(np.array_equal(py_raw, cxx_raw))
        act_max = float(np.max(np.abs(py_actions - cxx_actions)))
        act_ok = bool(np.allclose(py_actions, cxx_actions,
                                  rtol=1e-4, atol=1e-3))

        print("\n===== COLD START =====", flush=True)
        print(f"python imports            : {import_s:.3f} s", flush=True)
        print(f"load_model                : {load_s:.3f} s", flush=True)
        print(f"first predict/setup       : {first_predict_s:.3f} s", flush=True)
        print(f"export + native adopt     : {export_adopt_s * 1000.0:.3f} ms",
              flush=True)
        print(f"python ready total        : {import_s + load_s + first_predict_s:.3f} s",
              flush=True)
        print(f"hybrid ready total        : "
              f"{import_s + load_s + first_predict_s + export_adopt_s:.3f} s",
              flush=True)

        print("\n===== FIXED-NOISE EQUIVALENCE =====", flush=True)
        print(f"raw action exact          : {raw_exact}", flush=True)
        print(f"robot action allclose     : {act_ok}  max_abs={act_max:.6g}",
              flush=True)

        for i in range(args.warmup):
            idx = i % len(image_seq)
            model.predict(image_seq[idx], prompt=args.prompt)
            helper._upload_bytes(pl.input_noise_buf, start_noise)
            helper._model_set_images(m, view_seq[idx])
            helper._model_step_get_actions(m, int(pl.chunk_size))
        torch.cuda.synchronize()
        pipe.latency_records.clear()

        print("\n===== STEADY WALL-CLOCK =====", flush=True)
        _bench_ms(
            "python predict fresh-noise",
            args.iters,
            lambda i: model.predict(
                image_seq[i % len(image_seq)], prompt=args.prompt),
        )
        torch.cuda.synchronize()
        if pipe.latency_records:
            print(f"python predict internal     : {_stats(list(pipe.latency_records))}",
                  flush=True)

        def native_fixed(i: int) -> None:
            idx = i % len(view_seq)
            helper._upload_bytes(pl.input_noise_buf, start_noise)
            helper._model_set_images(m, view_seq[idx])
            helper._model_step_get_actions(m, int(pl.chunk_size))

        _bench_ms("model-runtime fixed-noise", args.iters, native_fixed)

        print("\nNOTE", flush=True)
        print("model-runtime fixed-noise includes host uint8 image staging, graph "
              "replay, D2H action readback, and C++ postprocess. It does not "
              "include native C++ RNG because the current ABI exposes noise as "
              "a SWAP window; adding a native RNG/fill stage is the next gap for "
              "a fully no-Python production loop.", flush=True)
    finally:
        m.release(m.owner)
        export.release()


if __name__ == "__main__":
    main()

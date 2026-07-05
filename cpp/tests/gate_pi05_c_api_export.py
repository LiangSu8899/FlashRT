"""Gate a real Pi0.5 export through the native C++ C API.

This is intentionally an in-process phase-1 test:
  1. Python loads the real model, captures the CUDA graph, and exports
     frt_runtime_export_v1.
  2. libflashrt_cpp_pi05_c.so adopts that export.
  3. The test compares Python frontend staging/replay against C++ runtime
     prepare_vision/replay_tick/read_actions on the same graph and buffers.

Run inside the CUDA container from the repo root:
    PYTHONPATH=.:./exec/build:./runtime/build \
    python cpp/tests/gate_pi05_c_api_export.py \
      --checkpoint "${PI05_CHECKPOINT:-/path/to/pi05_libero_pytorch}" --fp8
"""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
for rel in ("", "exec/build-container", "runtime/build-container",
            "exec/build", "runtime/build"):
    p = str(ROOT / rel) if rel else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import LIBERO_ACTION_DIM, unnormalize_actions  # noqa: E402


FRT_PI05_PIXEL_RGB8 = 0
FRT_PI05_DTYPE_BFLOAT16 = 1
FRT_PI05_DTYPE_FLOAT16 = 2
FRT_PI05_DTYPE_FLOAT32 = 3


class Pi05RuntimeConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("num_views", ctypes.c_int),
        ("chunk", ctypes.c_int),
        ("model_action_dim", ctypes.c_int),
        ("robot_action_dim", ctypes.c_int),
        ("action_mean", ctypes.POINTER(ctypes.c_float)),
        ("n_action_mean", ctypes.c_uint64),
        ("action_stddev", ctypes.POINTER(ctypes.c_float)),
        ("n_action_stddev", ctypes.c_uint64),
        ("graph_name", ctypes.c_char_p),
        ("image_buffer_name", ctypes.c_char_p),
        ("action_buffer_name", ctypes.c_char_p),
        ("image_dtype", ctypes.c_int),
        ("action_dtype", ctypes.c_int),
    ]


class Pi05VisionFrame(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("name", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("bytes", ctypes.c_uint64),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("stride_bytes", ctypes.c_int),
        ("pixel_format", ctypes.c_int),
        ("timestamp_ns", ctypes.c_uint64),
    ]


def _load_lib(path: str):
    lib = ctypes.CDLL(path)
    lib.frt_pi05_runtime_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Pi05RuntimeConfig),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.frt_pi05_runtime_create.restype = ctypes.c_int
    lib.frt_pi05_runtime_destroy.argtypes = [ctypes.c_void_p]
    lib.frt_pi05_runtime_destroy.restype = None
    lib.frt_pi05_runtime_prepare_vision.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Pi05VisionFrame),
        ctypes.c_uint64,
    ]
    lib.frt_pi05_runtime_prepare_vision.restype = ctypes.c_int
    lib.frt_pi05_runtime_replay_tick.argtypes = [ctypes.c_void_p]
    lib.frt_pi05_runtime_replay_tick.restype = ctypes.c_int
    lib.frt_pi05_runtime_read_actions.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.frt_pi05_runtime_read_actions.restype = ctypes.c_int
    lib.frt_pi05_runtime_last_error.argtypes = [ctypes.c_void_p]
    lib.frt_pi05_runtime_last_error.restype = ctypes.c_char_p
    return lib


def _check_c(rc: int, lib, rt, what: str) -> None:
    if rc == 0:
        return
    msg = b""
    if rt:
        msg = lib.frt_pi05_runtime_last_error(rt) or b""
    raise RuntimeError(f"{what} failed rc={rc}: {msg.decode(errors='replace')}")


def _read(buf) -> np.ndarray:
    return buf.download_new((buf.nbytes,), np.uint8).copy()


def _upload_bytes(buf, data: np.ndarray) -> None:
    data = np.ascontiguousarray(data, dtype=np.uint8)
    assert data.nbytes == buf.nbytes, (data.nbytes, buf.nbytes)
    buf.upload(data)


def _dtype_from_value(dtype) -> tuple[int, torch.dtype]:
    text = str(dtype).lower()
    if dtype == torch.bfloat16:
        return FRT_PI05_DTYPE_BFLOAT16, torch.bfloat16
    if dtype == torch.float16:
        return FRT_PI05_DTYPE_FLOAT16, torch.float16
    if dtype == torch.float32:
        return FRT_PI05_DTYPE_FLOAT32, torch.float32
    if text in ("bf16", "bfloat16"):
        return FRT_PI05_DTYPE_BFLOAT16, torch.bfloat16
    if text in ("f16", "float16", "fp16"):
        return FRT_PI05_DTYPE_FLOAT16, torch.float16
    if text in ("f32", "float32", "fp32"):
        return FRT_PI05_DTYPE_FLOAT32, torch.float32
    raise RuntimeError(f"unsupported Pi05 action dtype: {dtype}")


def _dtype_from_producer(pipe, pl) -> tuple[int, torch.dtype]:
    dtype = getattr(pl, "tensor_dtype", None)
    if dtype is not None:
        return _dtype_from_value(dtype)
    for attr in ("_noise_out", "_g_noise"):
        tensor = getattr(pipe, attr, None)
        if tensor is not None:
            return _dtype_from_value(tensor.dtype)
    raise RuntimeError("producer does not expose a Pi05 action dtype")


def _raw_to_float(raw: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    t = torch.frombuffer(raw.tobytes(), dtype=dtype).float()
    return t.cpu().numpy()


def _cos(a: np.ndarray, b: np.ndarray, dtype: torch.dtype) -> float:
    af = _raw_to_float(a, dtype)
    bf = _raw_to_float(b, dtype)
    na = float(np.linalg.norm(af))
    nb = float(np.linalg.norm(bf))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(af, bf) / (na * nb))


def _make_images(num_views: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        np.ascontiguousarray(
            rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8))
        for _ in range(num_views)
    ]


def _make_frames(images: list[np.ndarray]) -> tuple[ctypes.Array, list[bytes]]:
    names = [b"image", b"wrist_image", b"wrist_image_right"]
    keepalive = names[:len(images)]
    frames = (Pi05VisionFrame * len(images))()
    for i, im in enumerate(images):
        frames[i].struct_size = ctypes.sizeof(Pi05VisionFrame)
        frames[i].name = keepalive[i]
        frames[i].data = ctypes.c_void_p(im.ctypes.data)
        frames[i].bytes = im.nbytes
        frames[i].width = int(im.shape[1])
        frames[i].height = int(im.shape[0])
        frames[i].stride_bytes = int(im.strides[0])
        frames[i].pixel_format = FRT_PI05_PIXEL_RGB8
        frames[i].timestamp_ns = 0
    return frames, keepalive


def _obs_images(pipe, obs) -> list[np.ndarray]:
    nv = int(getattr(pipe, "num_views", len(obs.get("images", [])) or 1))
    if "images" in obs:
        return list(obs["images"][:nv])
    images = [obs["image"]]
    if nv >= 2:
        images.append(obs.get("wrist_image", obs["image"]))
    if nv >= 3:
        images.append(obs.get("wrist_image_right", images[-1]))
    return images[:nv]


def _normalize_images_fp16(pipe, obs) -> np.ndarray:
    frames = []
    for im in _obs_images(pipe, obs):
        if isinstance(im, torch.Tensor):
            frames.append(im.to(dtype=torch.float16).cpu().numpy())
        elif getattr(im, "dtype", None) == np.float16:
            frames.append(np.asarray(im))
        else:
            frames.append((np.asarray(im).astype(np.float32) / 127.5 - 1.0)
                          .astype(np.float16))
    return np.ascontiguousarray(np.stack(frames))


def _noise_bytes(nbytes: int, dtype: torch.dtype, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dtype == torch.float32:
        data = rng.standard_normal(nbytes // 4).astype(np.float32).tobytes()
    elif dtype == torch.bfloat16:
        values = rng.standard_normal(nbytes // 2).astype(np.float32)
        try:
            import ml_dtypes  # noqa: WPS433

            data = values.astype(ml_dtypes.bfloat16).tobytes()
        except Exception:  # noqa: BLE001
            data = (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    else:
        data = rng.standard_normal(nbytes // 2).astype(np.float16).tobytes()
    return np.frombuffer(data, dtype=np.uint8).copy()


def _action_affine(norm_stats) -> tuple[np.ndarray, np.ndarray]:
    q01 = np.asarray(norm_stats["actions"]["q01"], dtype=np.float32)
    q99 = np.asarray(norm_stats["actions"]["q99"], dtype=np.float32)
    scale = (q99 - q01 + 1e-6) / 2.0
    mean = q01 + scale
    return np.ascontiguousarray(mean), np.ascontiguousarray(scale)


def _python_stage_images(pipe, pl, obs) -> np.ndarray:
    if not hasattr(pipe, "_graph_torch_stream"):
        pl.input_images_buf.upload(_normalize_images_fp16(pipe, obs))
        return _read(pl.input_images_buf)
    with torch.cuda.stream(pipe._graph_torch_stream):
        stream_int = pipe._graph_torch_stream.cuda_stream
        pipe._fill_img_buf(obs)
        pipe._copy_tensor_to_pipeline_buf_stream(
            pipe._img_buf, pl.input_images_buf, stream_int)
    pipe._cudart.cudaStreamSynchronize(
        ctypes.c_void_p(pipe._graph_torch_stream.cuda_stream))
    return _read(pl.input_images_buf)


def _seed_noise(pipe, pl, seed: int, dtype: torch.dtype) -> np.ndarray:
    if not hasattr(pipe, "_graph_torch_stream"):
        data = _noise_bytes(pl.input_noise_buf.nbytes, dtype, seed)
        _upload_bytes(pl.input_noise_buf, data)
        return _read(pl.input_noise_buf)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.cuda.stream(pipe._graph_torch_stream):
        stream_int = pipe._graph_torch_stream.cuda_stream
        pipe._noise_buf.normal_()
        pipe._copy_tensor_to_pipeline_buf_stream(
            pipe._noise_buf, pl.input_noise_buf, stream_int)
    pipe._cudart.cudaStreamSynchronize(
        ctypes.c_void_p(pipe._graph_torch_stream.cuda_stream))
    return _read(pl.input_noise_buf)


def _python_replay(pipe, pl, obs, start_noise: np.ndarray) -> np.ndarray:
    _upload_bytes(pl.input_noise_buf, start_noise)
    if not hasattr(pipe, "_graph_torch_stream"):
        pl.input_images_buf.upload(_normalize_images_fp16(pipe, obs))
        pl.forward()
        return _read(pl.input_noise_buf)
    with torch.cuda.stream(pipe._graph_torch_stream):
        stream_int = pipe._graph_torch_stream.cuda_stream
        pipe._fill_img_buf(obs)
        pipe._copy_tensor_to_pipeline_buf_stream(
            pipe._img_buf, pl.input_images_buf, stream_int)
        pl.forward()
    pipe._cudart.cudaStreamSynchronize(
        ctypes.c_void_p(pipe._graph_torch_stream.cuda_stream))
    return _read(pl.input_noise_buf)


def _cxx_prepare(lib, rt, frames) -> None:
    _check_c(lib.frt_pi05_runtime_prepare_vision(rt, frames, len(frames)),
             lib, rt, "frt_pi05_runtime_prepare_vision")


def _cxx_replay_read(lib, rt, out_actions: np.ndarray) -> int:
    _check_c(lib.frt_pi05_runtime_replay_tick(rt), lib, rt,
             "frt_pi05_runtime_replay_tick")
    n_written = ctypes.c_uint64(0)
    _check_c(lib.frt_pi05_runtime_read_actions(
        rt,
        out_actions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_actions.size,
        ctypes.byref(n_written),
    ), lib, rt, "frt_pi05_runtime_read_actions")
    return int(n_written.value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-views", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--prompt", default="pick up the red block")
    ap.add_argument("--fp8", action="store_true", help="use production FP8/BF16 path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench-iters", type=int, default=10)
    ap.add_argument("--lib", default=str(ROOT / "cpp/build/libflashrt_cpp_pi05_c.so"))
    args = ap.parse_args()

    images = _make_images(args.num_views, args.seed)
    obs = {"images": images, "image": images[0]}
    if args.num_views >= 2:
        obs["wrist_image"] = images[1]
    if args.num_views >= 3:
        obs["wrist_image_right"] = images[2]

    print(f"precision: {'fp8/bf16' if args.fp8 else 'fp16'}")
    model = flash_rt.load_model(
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
    model.predict(images, prompt=args.prompt)
    pipe = model._pipe
    pl = pipe.pipeline

    export = pl.export_runtime(identity={"gate": "cpp_pi05_c_api"})
    lib = _load_lib(args.lib)
    dtype_id, torch_dtype = _dtype_from_producer(pipe, pl)
    action_mean, action_stddev = _action_affine(pipe.norm_stats)

    cfg = Pi05RuntimeConfig()
    cfg.struct_size = ctypes.sizeof(Pi05RuntimeConfig)
    cfg.num_views = args.num_views
    cfg.chunk = int(pl.chunk_size)
    cfg.model_action_dim = 32
    cfg.robot_action_dim = LIBERO_ACTION_DIM
    cfg.action_mean = action_mean.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    cfg.n_action_mean = action_mean.size
    cfg.action_stddev = action_stddev.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    cfg.n_action_stddev = action_stddev.size
    cfg.graph_name = b"infer"
    cfg.image_buffer_name = b"observation_images_normalized"
    cfg.action_buffer_name = b"diffusion_noise"
    cfg.image_dtype = dtype_id
    cfg.action_dtype = dtype_id

    rt = ctypes.c_void_p()
    rc = lib.frt_pi05_runtime_create(
        ctypes.c_void_p(export.ptr), ctypes.byref(cfg), ctypes.byref(rt))
    _check_c(rc, lib, rt, "frt_pi05_runtime_create")

    try:
        frames, _names = _make_frames(images)

        py_img = _python_stage_images(pipe, pl, obs)
        _cxx_prepare(lib, rt, frames)
        cxx_img = _read(pl.input_images_buf)
        img_exact = bool(np.array_equal(py_img, cxx_img))
        img_cos = _cos(py_img, cxx_img, torch_dtype)
        img_max = float(np.max(np.abs(
            _raw_to_float(py_img, torch_dtype) -
            _raw_to_float(cxx_img, torch_dtype))))

        start_noise = _seed_noise(pipe, pl, args.seed + 1009, torch_dtype)
        py_raw = _python_replay(pipe, pl, obs, start_noise)
        _upload_bytes(pl.input_noise_buf, start_noise)
        _cxx_prepare(lib, rt, frames)
        cxx_actions = np.empty((int(pl.chunk_size), LIBERO_ACTION_DIM),
                               dtype=np.float32)
        n_written = _cxx_replay_read(lib, rt, cxx_actions)
        cxx_raw = _read(pl.input_noise_buf)

        py_raw_f = _raw_to_float(py_raw, torch_dtype).reshape(
            int(pl.chunk_size), 32)
        py_actions = unnormalize_actions(py_raw_f, pipe.norm_stats)[
            :, :LIBERO_ACTION_DIM].astype(np.float32)
        cxx_actions = cxx_actions.reshape(int(pl.chunk_size), LIBERO_ACTION_DIM)

        raw_exact = bool(np.array_equal(py_raw, cxx_raw))
        raw_cos = _cos(py_raw, cxx_raw, torch_dtype)
        raw_max = float(np.max(np.abs(
            _raw_to_float(py_raw, torch_dtype) -
            _raw_to_float(cxx_raw, torch_dtype))))
        act_max = float(np.max(np.abs(py_actions - cxx_actions)))
        act_ok = bool(np.allclose(py_actions, cxx_actions, rtol=1e-4, atol=1e-3))

        print("\n===== REAL PI0.5 C++ C API EXPORT GATE =====")
        print(f"export fingerprint     : 0x{export.fingerprint:016x}")
        print(f"image buffer exact     : {img_exact}  cos={img_cos:.8f}  max_abs={img_max:.6g}")
        print(f"raw action exact       : {raw_exact}  cos={raw_cos:.8f}  max_abs={raw_max:.6g}")
        print(f"robot action allclose  : {act_ok}  max_abs={act_max:.6g}  n={n_written}")

        assert img_cos >= 0.999, f"image preprocess cosine too low: {img_cos}"
        assert raw_cos >= 0.999, f"raw replay cosine too low: {raw_cos}"
        assert act_ok, f"robot actions differ: max_abs={act_max}"

        if args.bench_iters > 0:
            out = np.empty((int(pl.chunk_size), LIBERO_ACTION_DIM), dtype=np.float32)
            for _ in range(3):
                _cxx_prepare(lib, rt, frames)
                _cxx_replay_read(lib, rt, out)
            times = []
            for _ in range(args.bench_iters):
                t0 = time.perf_counter()
                _cxx_prepare(lib, rt, frames)
                _cxx_replay_read(lib, rt, out)
                times.append((time.perf_counter() - t0) * 1000.0)
            print(
                "c++ hot tick ms        : "
                f"p50={statistics.median(times):.3f} "
                f"mean={statistics.fmean(times):.3f} "
                f"min={min(times):.3f} max={max(times):.3f} "
                f"(prepare_vision+replay_tick+read_actions, n={len(times)})"
            )

        print("\nPASS - C++ Pi05 runtime adopts a real FlashRT export")
    finally:
        if rt:
            lib.frt_pi05_runtime_destroy(rt)
        export.release()


if __name__ == "__main__":
    main()

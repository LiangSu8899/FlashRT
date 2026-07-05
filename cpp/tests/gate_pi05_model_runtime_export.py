"""Gate a real Pi0.5 export through the generic frt_model_runtime_v1 face.

Run inside the CUDA container from the repo root:

    PYTHONPATH=.:./exec/build-container:./runtime/build-container \
    python cpp/tests/gate_pi05_model_runtime_export.py \
      --checkpoint "${PI05_CHECKPOINT:-/path/to/pi05_libero_pytorch}" --fp8 \
      --lib cpp/build-container/libflashrt_cpp_pi05_c.so

The gate compares three surfaces:
  1. Python frontend staging/replay/postprocess.
  2. Native Pi05 model-runtime verbs over the producer-declared full graph.
  3. Native Pi05 model-runtime verbs over the producer-declared context/action
     split graph.
  3. Public Python predict() latency for an end-to-end Python baseline.
"""

from __future__ import annotations

import argparse
import ctypes
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
from flash_rt.subgraphs.pi05.context_action import enable as enable_context_action  # noqa: E402
from flash_rt.subgraphs.pi05.rtc_prefix import enable as enable_rtc_prefix  # noqa: E402


FRT_RT_PIXEL_RGB8 = 0

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


class FrtImageView(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("pixel_format", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("bytes", ctypes.c_uint64),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("stride_bytes", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("timestamp_ns", ctypes.c_uint64),
    ]


SetInputFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int)
GetOutputFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_int)
PrepareFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64)
StepFn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
LastErrorFn = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p)
RetainReleaseFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class FrtModelRuntimeVerbs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("set_input", SetInputFn),
        ("get_output", GetOutputFn),
        ("prepare", PrepareFn),
        ("step", StepFn),
        ("last_error", LastErrorFn),
    ]


class FrtModelRuntimeV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("exp", ctypes.c_void_p),
        ("ports", ctypes.c_void_p),
        ("n_ports", ctypes.c_uint64),
        ("stages", ctypes.c_void_p),
        ("n_stages", ctypes.c_uint64),
        ("self", ctypes.c_void_p),
        ("verbs", FrtModelRuntimeVerbs),
        ("owner", ctypes.c_void_p),
        ("retain", RetainReleaseFn),
        ("release", RetainReleaseFn),
    ]


def _load_lib(path: str):
    lib = ctypes.CDLL(path)
    lib.frt_pi05_model_runtime_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Pi05RuntimeConfig),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.frt_pi05_model_runtime_create.restype = ctypes.c_int
    lib.frt_pi05_model_runtime_create_over.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Pi05RuntimeConfig),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.frt_pi05_model_runtime_create_over.restype = ctypes.c_int
    return lib


def _model_error(m: FrtModelRuntimeV1) -> str:
    msg = m.verbs.last_error(m.self)
    return (msg or b"").decode(errors="replace")


def _check_model(rc: int, m: FrtModelRuntimeV1, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed rc={rc}: {_model_error(m)}")


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
    if text in ("bf16", "bfloat16", "3"):
        return FRT_PI05_DTYPE_BFLOAT16, torch.bfloat16
    if text in ("f16", "float16", "fp16", "2"):
        return FRT_PI05_DTYPE_FLOAT16, torch.float16
    if text in ("f32", "float32", "fp32", "1"):
        return FRT_PI05_DTYPE_FLOAT32, torch.float32
    raise RuntimeError(f"unsupported Pi05 action dtype: {dtype}")


def _dtype_from_model_runtime(mr) -> tuple[int, torch.dtype]:
    for port in mr.ports():
        if port.get("name") == "noise":
            return _dtype_from_value(port.get("dtype"))
    raise RuntimeError("model runtime does not declare a noise port dtype")


def _raw_to_float(raw: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    t = torch.frombuffer(bytearray(raw.tobytes()), dtype=dtype).float()
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


def _make_obs(images: list[np.ndarray]) -> dict:
    obs = {"images": images, "image": images[0]}
    if len(images) >= 2:
        obs["wrist_image"] = images[1]
    if len(images) >= 3:
        obs["wrist_image_right"] = images[2]
    return obs


def _make_image_views(images: list[np.ndarray]) -> ctypes.Array:
    views = (FrtImageView * len(images))()
    for i, im in enumerate(images):
        views[i].struct_size = ctypes.sizeof(FrtImageView)
        views[i].pixel_format = FRT_RT_PIXEL_RGB8
        views[i].data = ctypes.c_void_p(im.ctypes.data)
        views[i].bytes = im.nbytes
        views[i].width = int(im.shape[1])
        views[i].height = int(im.shape[0])
        views[i].stride_bytes = int(im.strides[0])
        views[i].reserved = 0
        views[i].timestamp_ns = 0
    return views


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


def _python_replay_actions(pipe, pl, obs, start_noise: np.ndarray,
                           dtype: torch.dtype) -> np.ndarray:
    raw = _python_replay(pipe, pl, obs, start_noise)
    raw_f = _raw_to_float(raw, dtype).reshape(int(pl.chunk_size), 32)
    return unnormalize_actions(raw_f, pipe.norm_stats)[
        :, :LIBERO_ACTION_DIM].astype(np.float32)


def _model_set_images(m: FrtModelRuntimeV1, views: ctypes.Array) -> None:
    ptr = ctypes.cast(views, ctypes.c_void_p)
    _check_model(m.verbs.set_input(
        m.self, 0, ptr, ctypes.sizeof(views), -1), m, "model.set_input(images)")


def _model_step_get_actions(m: FrtModelRuntimeV1, chunk: int) -> np.ndarray:
    _check_model(m.verbs.step(m.self), m, "model.step")
    out = np.empty((chunk, LIBERO_ACTION_DIM), dtype=np.float32)
    written = ctypes.c_uint64(0)
    _check_model(m.verbs.get_output(
        m.self, 2, ctypes.c_void_p(out.ctypes.data), out.nbytes,
        ctypes.byref(written), -1), m, "model.get_output(actions)")
    assert int(written.value) == out.nbytes, (written.value, out.nbytes)
    return out


def _create_over(lib, mr, cfg) -> FrtModelRuntimeV1:
    m_ptr = ctypes.c_void_p()
    rc = lib.frt_pi05_model_runtime_create_over(
        ctypes.c_void_p(mr.ptr), ctypes.byref(cfg), ctypes.byref(m_ptr))
    if rc != 0:
        raise RuntimeError(f"frt_pi05_model_runtime_create_over failed rc={rc}")
    return ctypes.cast(m_ptr, ctypes.POINTER(FrtModelRuntimeV1)).contents


def _bench(label: str, n: int, fn) -> list[float]:
    for _ in range(min(3, n)):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    print(
        f"{label:<27}: p50={statistics.median(times):.3f} "
        f"mean={statistics.fmean(times):.3f} "
        f"min={min(times):.3f} max={max(times):.3f} n={len(times)}"
    )
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
    ap.add_argument("--bench-iters", type=int, default=20)
    ap.add_argument("--rtc-prefix-len", type=int, default=2)
    ap.add_argument("--lib", default=str(
        ROOT / "cpp/build-container/libflashrt_cpp_pi05_c.so"))
    args = ap.parse_args()

    images = _make_images(args.num_views, args.seed)
    obs = _make_obs(images)
    views = _make_image_views(images)

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
    enable_context_action(model)
    enable_rtc_prefix(model, prefix_len=args.rtc_prefix_len)
    model.predict(images, prompt=args.prompt)
    pipe = model._pipe
    pl = pipe.pipeline

    mr_full = pl.export_model_runtime(
        identity={"gate": "cpp_pi05_model_runtime", "plan": "full"},
        stage_plan="full",
        io="native",
    )
    mr_split = pl.export_model_runtime(
        identity={"gate": "cpp_pi05_model_runtime", "plan": "context_action"},
        stage_plan="context_action",
        io="native",
    )
    mr_rtc = pl.export_model_runtime(
        identity={
            "gate": "cpp_pi05_model_runtime",
            "plan": "context_rtc_prefix_action",
            "rtc_prefix_len": str(args.rtc_prefix_len),
        },
        stage_plan="context_rtc_prefix_action",
        stage_plan_kwargs={"prefix_len": args.rtc_prefix_len},
        io="native",
    )
    lib = _load_lib(args.lib)
    dtype_id, torch_dtype = _dtype_from_model_runtime(mr_full)
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
    cfg.graph_name = None
    cfg.image_buffer_name = b"observation_images_normalized"
    cfg.action_buffer_name = b"diffusion_noise"
    cfg.image_dtype = dtype_id
    cfg.action_dtype = dtype_id

    m_full = _create_over(lib, mr_full, cfg)
    m_split = _create_over(lib, mr_split, cfg)
    m_rtc = _create_over(lib, mr_rtc, cfg)

    try:
        assert m_full.abi_version == 1, m_full.abi_version
        assert m_full.n_ports == 3, m_full.n_ports
        assert m_full.n_stages == 1, m_full.n_stages
        assert m_split.n_ports == 3, m_split.n_ports
        assert m_split.n_stages == 2, m_split.n_stages
        assert m_rtc.n_ports == 5, m_rtc.n_ports
        assert m_rtc.n_stages == 2, m_rtc.n_stages
        assert mr_full.fingerprint != mr_split.fingerprint
        assert mr_rtc.fingerprint not in (
            mr_full.fingerprint, mr_split.fingerprint)

        py_img = _python_stage_images(pipe, pl, obs)
        _model_set_images(m_full, views)
        cxx_img = _read(pl.input_images_buf)
        img_exact = bool(np.array_equal(py_img, cxx_img))
        img_cos = _cos(py_img, cxx_img, torch_dtype)
        img_max = float(np.max(np.abs(
            _raw_to_float(py_img, torch_dtype) -
            _raw_to_float(cxx_img, torch_dtype))))

        start_noise = _seed_noise(pipe, pl, args.seed + 1009, torch_dtype)
        py_raw = _python_replay(pipe, pl, obs, start_noise)

        _upload_bytes(pl.input_noise_buf, start_noise)
        _model_set_images(m_full, views)
        full_actions = _model_step_get_actions(m_full, int(pl.chunk_size))
        full_raw = _read(pl.input_noise_buf)

        _upload_bytes(pl.input_noise_buf, start_noise)
        _model_set_images(m_split, views)
        split_actions = _model_step_get_actions(m_split, int(pl.chunk_size))
        split_raw = _read(pl.input_noise_buf)

        rtc_prev = _seed_noise(pipe, pl, args.seed + 2003, torch_dtype)
        _upload_bytes(pl.input_noise_buf, start_noise)
        _upload_bytes(pl.input_rtc_prev_action_chunk_buf, rtc_prev)
        _model_set_images(m_rtc, views)
        rtc_actions = _model_step_get_actions(m_rtc, int(pl.chunk_size))
        rtc_raw = _read(pl.input_noise_buf)
        prefix_len = int(args.rtc_prefix_len)
        prefix_bytes = prefix_len * 32 * 2
        rtc_prefix_exact = bool(np.array_equal(
            rtc_raw[:prefix_bytes], rtc_prev[:prefix_bytes]))
        rtc_prefix_max = 0.0
        if prefix_bytes:
            rtc_prefix_max = float(np.max(np.abs(
                _raw_to_float(rtc_raw[:prefix_bytes], torch_dtype) -
                _raw_to_float(rtc_prev[:prefix_bytes], torch_dtype))))

        py_raw_f = _raw_to_float(py_raw, torch_dtype).reshape(
            int(pl.chunk_size), 32)
        py_actions = unnormalize_actions(py_raw_f, pipe.norm_stats)[
            :, :LIBERO_ACTION_DIM].astype(np.float32)

        raw_exact = bool(np.array_equal(py_raw, full_raw))
        raw_cos = _cos(py_raw, full_raw, torch_dtype)
        raw_max = float(np.max(np.abs(
            _raw_to_float(py_raw, torch_dtype) -
            _raw_to_float(full_raw, torch_dtype))))
        act_max = float(np.max(np.abs(py_actions - full_actions)))
        act_ok = bool(np.allclose(py_actions, full_actions, rtol=1e-4, atol=1e-3))
        split_raw_exact = bool(np.array_equal(full_raw, split_raw))
        split_raw_cos = _cos(full_raw, split_raw, torch_dtype)
        split_raw_max = float(np.max(np.abs(
            _raw_to_float(full_raw, torch_dtype) -
            _raw_to_float(split_raw, torch_dtype))))
        split_act_max = float(np.max(np.abs(full_actions - split_actions)))
        split_act_ok = bool(np.allclose(
            full_actions, split_actions, rtol=1e-4, atol=1e-3))

        print("\n===== REAL PI0.5 MODEL-RUNTIME EXPORT GATE =====")
        print(f"full fingerprint       : 0x{mr_full.fingerprint:016x}")
        print(f"split fingerprint      : 0x{mr_split.fingerprint:016x}")
        print(f"rtc fingerprint        : 0x{mr_rtc.fingerprint:016x}")
        print(f"full runtime           : ports={m_full.n_ports} stages={m_full.n_stages}")
        print(f"split runtime          : ports={m_split.n_ports} stages={m_split.n_stages}")
        print(f"rtc runtime            : ports={m_rtc.n_ports} stages={m_rtc.n_stages} prefix={prefix_len}")
        print(f"image buffer exact     : {img_exact}  cos={img_cos:.8f}  max_abs={img_max:.6g}")
        print(f"py vs full raw exact   : {raw_exact}  cos={raw_cos:.8f}  max_abs={raw_max:.6g}")
        print(f"py vs full action      : {act_ok}  max_abs={act_max:.6g}")
        print(f"full vs split raw exact: {split_raw_exact}  cos={split_raw_cos:.8f}  max_abs={split_raw_max:.6g}")
        print(f"full vs split action   : {split_act_ok}  max_abs={split_act_max:.6g}")
        print(f"rtc prefix exact       : {rtc_prefix_exact}  max_abs={rtc_prefix_max:.6g}")

        assert img_cos >= 0.999, f"image preprocess cosine too low: {img_cos}"
        assert raw_cos >= 0.999, f"raw replay cosine too low: {raw_cos}"
        assert act_ok, f"robot actions differ: max_abs={act_max}"
        assert split_raw_exact, (
            "split replay must be bit-exact against full replay; "
            f"cos={split_raw_cos:.8f} max_abs={split_raw_max:.6g}")
        assert split_act_ok, (
            f"split robot actions differ: max_abs={split_act_max}")
        assert rtc_prefix_exact, (
            "RTC-prefix action graph did not preserve prev_action_chunk "
            f"prefix; max_abs={rtc_prefix_max:.6g}")

        if args.bench_iters > 0:
            print("\n===== LATENCY =====")
            _bench(
                "python predict",
                args.bench_iters,
                lambda: model.predict(images, prompt=args.prompt),
            )
            _bench(
                "python fixed replay",
                args.bench_iters,
                lambda: _python_replay_actions(
                    pipe, pl, obs, start_noise, torch_dtype),
            )

            def model_runtime_tick():
                _upload_bytes(pl.input_noise_buf, start_noise)
                _model_set_images(m_full, views)
                _model_step_get_actions(m_full, int(pl.chunk_size))

            _bench(
                "model-runtime full",
                args.bench_iters,
                model_runtime_tick,
            )

            def split_runtime_tick():
                _upload_bytes(pl.input_noise_buf, start_noise)
                _model_set_images(m_split, views)
                _model_step_get_actions(m_split, int(pl.chunk_size))

            _bench(
                "model-runtime split",
                args.bench_iters,
                split_runtime_tick,
            )

            def rtc_prefix_tick():
                _upload_bytes(pl.input_noise_buf, start_noise)
                _upload_bytes(pl.input_rtc_prev_action_chunk_buf, rtc_prev)
                _model_set_images(m_rtc, views)
                _model_step_get_actions(m_rtc, int(pl.chunk_size))

            _bench(
                "model-runtime rtc-prefix",
                args.bench_iters,
                rtc_prefix_tick,
            )

        print("\nPASS - Pi05 full, context/action, and RTC-prefix model runtimes passed")
    finally:
        m_full.release(m_full.owner)
        m_split.release(m_split.owner)
        m_rtc.release(m_rtc.owner)
        mr_full.release()
        mr_split.release()
        mr_rtc.release()


if __name__ == "__main__":
    main()

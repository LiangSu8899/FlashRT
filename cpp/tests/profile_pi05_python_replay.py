"""Developer profiling tool for one Pi0.5 Python replay CUDA range.

This utility produces diagnostic traces; it is not an acceptance test.
"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
for rel in ("", "exec/build-container", "runtime/build-container",
            "exec/build", "runtime/build"):
    path = str(ROOT / rel) if rel else str(ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)

import flash_rt  # noqa: E402


def _check_cuda(rc: int, operation: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {rc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(f"Pi0.5 native profiling requires SM120, got {capability}")

    rng = np.random.default_rng(7)
    images = [
        rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        for _ in range(args.num_views)
    ]
    state = np.linspace(-0.8, 0.8, 8, dtype=np.float32)
    model = flash_rt.load_model(
        args.checkpoint,
        framework="torch",
        config="pi05",
        hardware="rtx_sm120",
        num_views=args.num_views,
        num_steps=args.steps,
        cache_frames=1,
        use_fp8=False,
        state_prompt_mode="fixed",
    )
    model.predict(images, prompt="pick up the black bowl", state=state)

    pipe = model._pipe
    pipeline = pipe.pipeline
    observation = {
        "images": images,
        "image": images[0],
        "state": state,
    }
    if len(images) >= 2:
        observation["wrist_image"] = images[1]
    if len(images) >= 3:
        observation["wrist_image_right"] = images[2]

    with torch.cuda.stream(pipe._graph_torch_stream):
        stream = pipe._graph_torch_stream.cuda_stream
        pipe._noise_buf.zero_()
        pipe._copy_tensor_to_pipeline_buf_stream(
            pipe._noise_buf, pipeline.input_noise_buf, stream)
        pipe._fill_img_buf(observation)
        pipe._copy_tensor_to_pipeline_buf_stream(
            pipe._img_buf, pipeline.input_images_buf, stream)
    _check_cuda(
        pipe._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream)),
        "cudaStreamSynchronize before profiling",
    )

    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaProfilerStart.restype = ctypes.c_int
    cudart.cudaProfilerStop.restype = ctypes.c_int
    _check_cuda(cudart.cudaProfilerStart(), "cudaProfilerStart")
    with torch.cuda.stream(pipe._graph_torch_stream):
        pipeline.forward()
    _check_cuda(
        pipe._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream)),
        "cudaStreamSynchronize after replay",
    )
    _check_cuda(cudart.cudaProfilerStop(), "cudaProfilerStop")
    print("PASS Pi0.5 Python frontend replay profiler range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

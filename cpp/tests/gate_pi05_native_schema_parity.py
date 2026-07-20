#!/usr/bin/env python3
"""Compare Python and C++ native-v2 port/stage/region declarations."""

from __future__ import annotations

import argparse
import ctypes
import difflib
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).with_name("data") / "pi05_native_v2_schema.records"
sys.path.insert(0, str(ROOT))
configured_build = os.environ.get("FLASHRT_BUILD_DIR")
if not configured_build:
    raise SystemExit("Set FLASHRT_BUILD_DIR to the Python producer CMake build")
for subdir in ("exec", "runtime"):
    sys.path.insert(0, str(Path(configured_build).resolve() / subdir))

import _flashrt_runtime as runtime_abi  # noqa: E402
import flash_rt  # noqa: E402
from cpp.tests.pi05_runtime_ctypes import (  # noqa: E402
    FRT_PI05_DTYPE_BFLOAT16,
    FRT_PI05_DTYPE_FLOAT16,
    Pi05RuntimeConfig,
    PromptLengthUpdateFn,
    load_pi05_library,
    native_overlay,
)


def canonical_records(identity: str) -> list[str]:
    prefixes = ("region:", "port:", "stage:")
    return [line for line in identity.splitlines()
            if line.startswith(prefixes)]


def assert_records(label: str, actual: list[str], expected: list[str]) -> None:
    if actual == expected:
        return
    diff = "\n".join(difflib.unified_diff(
        expected, actual, fromfile="golden", tofile=label, lineterm="",
    ))
    raise AssertionError(f"{label} schema mismatch:\n{diff}")


def make_native_v2_overlay(model, tokenizer: Path, library: Path):
    pipe = model._pipe
    pipeline = pipe.pipeline
    embedding = pipe.embedding_weight
    if embedding.dtype == torch.bfloat16:
        dtype = FRT_PI05_DTYPE_BFLOAT16
    elif embedding.dtype == torch.float16:
        dtype = FRT_PI05_DTYPE_FLOAT16
    else:
        raise RuntimeError(f"unsupported embedding dtype: {embedding.dtype}")
    if not embedding.is_cuda or not embedding.is_contiguous():
        raise RuntimeError("native-v2 gate requires a contiguous CUDA embedding")
    prompt_buffer = pipeline._lang_embeds_buf
    action_q01 = np.ascontiguousarray(
        pipe.norm_stats["actions"]["q01"], dtype=np.float32)
    action_q99 = np.ascontiguousarray(
        pipe.norm_stats["actions"]["q99"], dtype=np.float32)
    action_stddev = np.ascontiguousarray(
        (action_q99 - action_q01 + 1e-6) * 0.5, dtype=np.float32)
    action_mean = np.ascontiguousarray(
        action_q01 + action_stddev, dtype=np.float32)
    state_q01 = np.ascontiguousarray(
        pipe.norm_stats["state"]["q01"], dtype=np.float32)
    state_q99 = np.ascontiguousarray(
        pipe.norm_stats["state"]["q99"], dtype=np.float32)
    tokenizer_bytes = str(tokenizer).encode()

    @PromptLengthUpdateFn
    def update_prompt_length(_user, prompt_len):
        try:
            prompt_len = int(prompt_len)
            pipeline._current_prompt_len = prompt_len
            pipeline._set_decoder_rope_for_prompt(prompt_len)
            pipeline.attn.set_fixed_valid_len(
                pipeline.vision_seq_enc + prompt_len)
            return 0
        except Exception:  # ctypes callbacks must return a status code
            return -1

    config = Pi05RuntimeConfig()
    config.struct_size = ctypes.sizeof(Pi05RuntimeConfig)
    config.num_views = int(pipeline.num_views)
    config.chunk = int(pipeline.chunk_size)
    config.model_action_dim = 32
    config.robot_action_dim = int(action_mean.size)
    config.action_mean = action_mean.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    config.n_action_mean = action_mean.size
    config.action_stddev = action_stddev.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    config.n_action_stddev = action_stddev.size
    config.image_buffer_name = b"observation_images_normalized"
    config.action_buffer_name = b"diffusion_noise"
    config.image_dtype = dtype
    config.action_dtype = dtype
    config.prompt_tokenizer_model_path = tokenizer_bytes
    config.prompt_embedding_table_data = embedding.data_ptr()
    config.prompt_embedding_table_bytes = (
        embedding.numel() * embedding.element_size())
    config.prompt_embedding_table_dtype = dtype
    config.prompt_embedding_vocab_size = embedding.shape[0]
    config.prompt_embedding_hidden_dim = embedding.shape[1]
    config.prompt_embedding_data = prompt_buffer.ptr.value
    config.prompt_embedding_bytes = prompt_buffer.nbytes
    config.prompt_embedding_dtype = dtype
    config.max_prompt_tokens = int(pipeline.max_prompt_len)
    config.prompt_embedding_scale = math.sqrt(float(embedding.shape[1]))
    config.state_q01 = state_q01.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    config.n_state_q01 = state_q01.size
    config.state_q99 = state_q99.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    config.n_state_q99 = state_q99.size
    config.prompt_length_update = update_prompt_length
    config.prompt_embedding_on_device = 1

    lib = load_pi05_library(library)
    return native_overlay(
        lib, config,
        keepalive=(embedding, prompt_buffer, action_mean, action_stddev,
                   state_q01, state_q99, tokenizer_bytes,
                   update_prompt_length),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--native-probe", type=Path, required=True)
    parser.add_argument("--lib", type=Path)
    args = parser.parse_args()
    for name in ("checkpoint", "tokenizer", "native_probe"):
        path = getattr(args, name).resolve()
        if not path.exists():
            parser.error(f"--{name.replace('_', '-')} does not exist: {path}")
        setattr(args, name, path)
    if args.lib is None:
        args.lib = args.native_probe.parent / "libflashrt_cpp_pi05_c.so"
    args.lib = args.lib.resolve()
    if not args.lib.exists():
        parser.error(f"--lib does not exist: {args.lib}")
    golden_records = GOLDEN.read_text(encoding="utf-8").splitlines()
    expected_ports = sum(line.startswith("port:") for line in golden_records)
    expected_stages = sum(line.startswith("stage:") for line in golden_records)
    expected_regions = sum(line.startswith("region:") for line in golden_records)

    rng = np.random.default_rng(20260710)
    images = [
        np.ascontiguousarray(
            rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        )
        for _ in range(2)
    ]
    state = np.linspace(-0.25, 0.25, 8, dtype=np.float32)
    model = flash_rt.load_model(
        str(args.checkpoint), framework="torch", config="pi05",
        hardware="auto", num_views=2, num_steps=10, cache_frames=1,
        use_fp8=True, use_fp16=False, state_prompt_mode="fixed",
    )
    model.predict(images, prompt="pick up the red block", state=state)
    overlay = make_native_v2_overlay(model, args.tokenizer, args.lib)
    producer = model._pipe.pipeline.export_model_runtime(
        identity={"gate": "native_v2_schema_parity"},
        stage_plan="full", io="native_v2",
        robot_action_dim=len(model._pipe.norm_stats["actions"]["q01"]),
        state_dim=len(model._pipe.norm_stats["state"]["q01"]),
        native_overlay=overlay,
    )
    try:
        counts = dict(runtime_abi.export_counts(producer.export_ptr))
        if len(producer.ports()) != expected_ports or \
                len(producer.stages()) != expected_stages or \
                counts.get("capsule_regions") != expected_regions:
            raise RuntimeError(
                f"unexpected Python native-v2 counts: ports="
                f"{len(producer.ports())} stages={len(producer.stages())} "
                f"regions={counts.get('capsule_regions')}"
            )
        python_records = canonical_records(producer.identity)
        assert_records("python-native-v2", python_records, golden_records)
        with tempfile.TemporaryDirectory(prefix="pi05_schema_parity_") as tmp:
            native_path = Path(tmp) / "native.schema"
            env = dict(os.environ)
            env["FLASHRT_SCHEMA_OUTPUT"] = str(native_path)
            env["FLASHRT_SCHEMA_ONLY"] = "1"
            probe_lib_dir = str(args.native_probe.parent)
            inherited = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                probe_lib_dir if not inherited
                else f"{probe_lib_dir}:{inherited}")
            subprocess.run(
                [str(args.native_probe), str(args.checkpoint),
                 str(args.tokenizer)],
                check=True, env=env,
            )
            native_records = native_path.read_text().splitlines()

        assert_records("cpp-native-v2", native_records, golden_records)

        print("\n===== PI0.5 NATIVE-V2 SCHEMA PARITY =====")
        print(f"records     : {len(python_records)}")
        print(f"ports/stage : {expected_ports} / {expected_stages}")
        print(f"regions     : {expected_regions}")
        print("PASS - Python and C++ native-v2 schemas match the golden records")
        return 0
    finally:
        producer.release()


if __name__ == "__main__":
    raise SystemExit(main())

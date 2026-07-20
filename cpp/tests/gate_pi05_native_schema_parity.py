#!/usr/bin/env python3
"""Compare Python and C++ native-v2 port/stage/region declarations."""

from __future__ import annotations

import argparse
import difflib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--native-probe", type=Path, required=True)
    args = parser.parse_args()
    for name in ("checkpoint", "tokenizer", "native_probe"):
        path = getattr(args, name).resolve()
        if not path.exists():
            parser.error(f"--{name.replace('_', '-')} does not exist: {path}")
        setattr(args, name, path)
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
    producer = model._pipe.pipeline.export_model_runtime(
        identity={"gate": "native_v2_schema_parity"},
        stage_plan="full", io="native_v2",
        robot_action_dim=len(model._pipe.norm_stats["actions"]["q01"]),
        state_dim=len(model._pipe.norm_stats["state"]["q01"]),
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

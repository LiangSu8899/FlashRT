"""Compare replay-only Pi0.5 native and Python Nsight kernel traces."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path


def _read_names(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    try:
        header = next(
            index for index, line in enumerate(lines)
            if line.startswith("Start (ns),")
        )
    except StopIteration as exc:
        raise ValueError(f"{path}: cuda_gpu_trace CSV header is missing") from exc
    names = [row["Name"] for row in csv.DictReader(lines[header:])]
    if not names:
        raise ValueError(f"{path}: CUDA trace is empty")
    return names


def _classify(name: str) -> tuple[str | None, str | None]:
    # These two nodes are an implementation detail of the selected GEMM
    # algorithm. A split-K algorithm substitutes a reduction for workspace
    # initialization without changing the surrounding logical GEMM sequence.
    if name == "[CUDA memset]":
        return None, "gemm_workspace_init"
    if "cublasLt::splitKreduce_kernel" in name:
        return None, "gemm_splitk_reduce"

    patterns = (
        ("copy", "[CUDA memcpy"),
        ("attention_fill", "FillFunctor<float>"),
        ("attention_fill", "fill_negative_infinity"),
        ("gemm", "cutlass::Kernel2"),
        ("gemm", "gemmSN_NN_kernel"),
        ("attention_combine", "flash_fwd_splitkv_combine_kernel"),
        ("attention_split", "flash_fwd_splitkv_kernel"),
        ("attention", "flash_fwd_kernel"),
        ("ada_norm", "ada_rms_norm_style_kernel"),
        ("gate_residual", "gate_mul_res_kernel"),
        ("gate_silu", "gate_silu_mul_kernel"),
        ("qkv_devpos", "qkv_split_rope_devpos_kernel"),
        ("qkv_rope", "qkv_split_rope_kernel"),
        ("qkv", "qkv_split_kernel"),
        ("bias", "bias_res_kernel"),
        ("bias", "add_bias"),
        ("layer_norm", "layer_norm_kernel"),
        ("rms_norm", "rms_norm_kernel"),
        ("gelu", "gelu_kernel"),
        ("residual", "res_add_kernel"),
        ("patch", "patch_im2col_kernel"),
    )
    for logical, marker in patterns:
        if marker in name:
            return logical, None
    raise ValueError(f"kernel is not in the explicit Pi0.5 whitelist: {name}")


def _normalize(names: list[str]) -> tuple[list[str], Counter[str]]:
    result = []
    ignored: Counter[str] = Counter()
    for name in names:
        logical, ignored_kind = _classify(name)
        if ignored_kind is not None:
            ignored[ignored_kind] += 1
        else:
            result.append(logical)
    return result, ignored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()

    native_names = _read_names(args.native)
    python_names = _read_names(args.python)
    if len(native_names) != len(python_names):
        raise AssertionError(
            f"raw event count differs: native={len(native_names)} "
            f"python={len(python_names)}"
        )
    native, native_ignored = _normalize(native_names)
    python, python_ignored = _normalize(python_names)
    if sum(native_ignored.values()) != sum(python_ignored.values()):
        raise AssertionError(
            "allowlisted GEMM helper count differs: "
            f"native={dict(native_ignored)} python={dict(python_ignored)}"
        )
    if native != python:
        mismatch = next(
            (index for index, pair in enumerate(zip(native, python))
             if pair[0] != pair[1]),
            min(len(native), len(python)),
        )
        raise AssertionError(
            f"logical kernel sequence differs at {mismatch}: "
            f"native={native[mismatch:mismatch + 8]} "
            f"python={python[mismatch:mismatch + 8]}"
        )
    print({
        "ok": True,
        "raw_events": len(native_names),
        "logical_events": len(native),
        "native_gemm_helpers": dict(native_ignored),
        "python_gemm_helpers": dict(python_ignored),
    })
    print("PASS Pi0.5 native/Python logical kernel sequences are identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

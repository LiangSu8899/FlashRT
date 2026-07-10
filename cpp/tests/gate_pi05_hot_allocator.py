"""Reject CUDA allocation APIs in a replay-only Pi0.5 Nsight trace."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path


FORBIDDEN = (
    "cudaMalloc",
    "cudaFree",
    "cudaHostAlloc",
    "cudaHostRegister",
    "cudaHostUnregister",
    "cudaMemPoolCreate",
    "cudaMemPoolDestroy",
    "cuMemAlloc",
    "cuMemFree",
    "cuMemHostAlloc",
    "cuMemHostRegister",
    "cuMemHostUnregister",
    "cuMemCreate",
    "cuMemMap",
    "cuMemUnmap",
    "cuMemAddressReserve",
    "cuMemAddressFree",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    try:
        header = next(
            index for index, line in enumerate(lines)
            if line.startswith("Start (ns),")
        )
    except StopIteration as exc:
        raise ValueError(f"{path}: cuda_api_trace CSV header is missing") from exc
    rows = list(csv.DictReader(lines[header:]))
    if not rows:
        raise ValueError(f"{path}: CUDA API trace is empty")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--expected-replays", type=int, default=1000)
    args = parser.parse_args()
    if args.expected_replays <= 0:
        parser.error("--expected-replays must be positive")

    rows = _read_rows(args.trace)
    failed = [row for row in rows if int(row["Result"]) != 0]
    if failed:
        raise AssertionError(f"CUDA API calls failed: {failed[:4]}")
    names = Counter(row["Name"] for row in rows)
    graph_launches = sum(
        count for name, count in names.items()
        if name.startswith("cudaGraphLaunch") or name.startswith("cuGraphLaunch")
    )
    if graph_launches != args.expected_replays:
        raise AssertionError(
            f"graph replay count differs: actual={graph_launches} "
            f"expected={args.expected_replays}"
        )
    allocator_calls = {
        name: count for name, count in names.items()
        if any(marker in name for marker in FORBIDDEN)
    }
    if allocator_calls:
        raise AssertionError(f"hot path called CUDA allocators: {allocator_calls}")
    print({
        "ok": True,
        "graph_replays": graph_launches,
        "allocator_calls": 0,
        "cuda_api_calls": sum(names.values()),
    })
    print("PASS Pi0.5 hot replay performed no CUDA allocation calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

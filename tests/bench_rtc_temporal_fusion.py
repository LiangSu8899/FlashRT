"""Microbenchmark three-way fusion of 50 x 32 action chunks."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from flash_rt.runtime.rtc_temporal_fusion import (
    TemporalFusionBuffer,
    TemporalFusionConfig,
)


def _one_fusion(chunks: list[np.ndarray], config: TemporalFusionConfig) -> None:
    buffer = TemporalFusionBuffer(config, clock=lambda: 0.0)
    for index, actions in enumerate(chunks):
        started = index * config.period_s
        ticket = buffer.begin_prediction(started_at=started)
        buffer.complete_prediction(ticket, actions, ready_at=started + 1e-4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")

    rng = np.random.default_rng(1234)
    chunks = [rng.standard_normal((50, 32), dtype=np.float32)
              for _ in range(3)]
    config = TemporalFusionConfig(
        action_hz=50, max_chunks=3, decay=0.1, epoch_s=0)

    for _ in range(args.warmup):
        _one_fusion(chunks, config)
    samples_ms = []
    for _ in range(args.iterations):
        start = time.perf_counter_ns()
        _one_fusion(chunks, config)
        samples_ms.append((time.perf_counter_ns() - start) / 1e6)

    ordered = sorted(samples_ms)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
    print("RTC temporal fusion: 3 chunks x 50 actions x 32 dims")
    print(f"iterations={args.iterations} warmup={args.warmup}")
    print(f"median_ms={statistics.median(samples_ms):.6f}")
    print(f"p95_ms={p95:.6f} p99_ms={p99:.6f}")
    print(f"mean_ms={statistics.fmean(samples_ms):.6f}")


if __name__ == "__main__":
    main()

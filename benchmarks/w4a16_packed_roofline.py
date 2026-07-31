#!/usr/bin/env python3
"""What fraction of a part's bandwidth the packed 4-bit GEMV reaches.

A weight-only GEMV at batch one reads its whole weight and does two flops per
value, so bandwidth is the only ceiling that means anything and the fraction of
it reached is the whole result. This is the figure a projected token rate rests
on: the shapes a checkpoint issues, timed against a bandwidth measured on the
same part rather than a number from a datasheet.

The reference is a device-to-device copy, which moves each byte twice. A kernel
that only reads can exceed it, so a fraction above 100% means the reference is
the floor of the read bandwidth and not that something is wrong.

    python benchmarks/w4a16_packed_roofline.py
    python benchmarks/w4a16_packed_roofline.py --hidden 5120 --intermediate 17408
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from flash_rt import flash_rt_kernels as fvk


def measured_bandwidth(device: str, mib: int = 256) -> float:
    source = torch.empty(mib * 2**20 // 2, dtype=torch.bfloat16, device=device)
    source.normal_()
    destination = torch.empty_like(source)
    for _ in range(3):
        destination.copy_(source)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    rounds = 10
    for _ in range(rounds):
        destination.copy_(source)
    torch.cuda.synchronize(device)
    return rounds * 2 * source.numel() * 2 / (time.perf_counter() - started)


def time_call(call, warmup: int, rounds: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(rounds):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / rounds


def shapes(hidden: int, intermediate: int, layers: int, full_period: int,
           q_width: int, kv_width: int, lin_qkv: int, lin_value: int):
    """Every weight a decode step reads, and how many layers read it."""
    full = layers // full_period
    linear = layers - full
    return [
        ("mlp gate+up", 2 * intermediate, hidden, layers),
        ("mlp down", hidden, intermediate, layers),
        ("gdn in_proj", lin_qkv + lin_value, hidden, linear),
        ("gdn out", hidden, lin_value, linear),
        ("attn qkv", q_width + 2 * kv_width, hidden, full),
        ("attn o", hidden, q_width // 2, full),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--intermediate", type=int, default=9216)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--full-period", type=int, default=4)
    parser.add_argument("--q-width", type=int, default=8192)
    parser.add_argument("--kv-width", type=int, default=1024)
    parser.add_argument("--lin-qkv", type=int, default=8192)
    parser.add_argument("--lin-value", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--output")
    args = parser.parse_args()

    bandwidth = measured_bandwidth(args.device)
    name = torch.cuda.get_device_name(args.device)
    print(f"{name}: device-to-device copy at {bandwidth / 1e9:.1f} GB/s\n")
    print(f"{'weight':<14}{'N':>7}{'K':>7}{'x':>4}{'MiB':>8}{'us':>9}"
          f"{'GB/s':>9}{'% BW':>7}")

    rows = []
    total = 0.0
    total_bytes = 0
    stream = torch.cuda.current_stream(args.device).cuda_stream
    for label, n, k, count in shapes(
            args.hidden, args.intermediate, args.layers, args.full_period,
            args.q_width, args.kv_width, args.lin_qkv, args.lin_value):
        packed = torch.randint(-(2 ** 31), 2 ** 31 - 1, (n, k // 8),
                               dtype=torch.int32, device=args.device)
        scale = (torch.rand(n, k // args.group, device=args.device) * 0.02).to(
            torch.bfloat16)
        x = torch.randn(1, k, dtype=torch.bfloat16, device=args.device)
        out = torch.empty(1, n, dtype=torch.bfloat16, device=args.device)

        def call():
            rc = fvk.w4a16_packed_matvec_bf16(
                x.data_ptr(), packed.data_ptr(), scale.data_ptr(),
                out.data_ptr(), n, k, args.group, stream)
            if rc:
                raise RuntimeError(f"{label} returned {rc}")

        seconds = time_call(call, 30, args.rounds)
        weight_bytes = n * k // 2 + n * (k // args.group) * 2
        total += seconds * count
        total_bytes += weight_bytes * count
        rows.append({"weight": label, "n": n, "k": k, "layers": count,
                     "us": seconds * 1e6,
                     "gb_s": weight_bytes / seconds / 1e9,
                     "fraction": weight_bytes / seconds / bandwidth})
        print(f"{label:<14}{n:>7}{k:>7}{count:>4}{weight_bytes / 2**20:>8.1f}"
              f"{seconds * 1e6:>9.1f}{weight_bytes / seconds / 1e9:>9.0f}"
              f"{weight_bytes / seconds / bandwidth * 100:>6.0f}%")

    # The tied output projection, which for a small model is a large share.
    head_bytes = args.vocab * args.hidden * 2
    print(f"\nbackbone   {total_bytes / 2**30:6.3f} GiB in "
          f"{total * 1e3:6.2f} ms")
    print(f"lm_head    {head_bytes / 2**30:6.3f} GiB bf16 "
          f"({head_bytes / 2 / 2**30:.3f} GiB at int8)")
    for label, head in (("bf16 head", head_bytes),
                        ("int8 head", head_bytes // 2)):
        seconds = total + head / bandwidth
        print(f"  step, {label}: {(total_bytes + head) / 2**30:6.3f} GiB "
              f"-> {seconds * 1e3:6.2f} ms = {1 / seconds:5.1f} tok/s")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"device": name, "bandwidth_gb_s": bandwidth / 1e9,
                       "shapes": rows}, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()

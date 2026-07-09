#!/usr/bin/env python3
"""Micro-benchmark for SM89 FP8 block-128 GEMV kernels.

Tests all three variants (w4, w8, w16) across representative (N, K) shapes
from Qwen3-VL 8B and 2B decode. Use with ncu for roofline analysis:

    ncu --set full -k "gemv_fp8_block128" -o gemv_profile \
        python scripts/bench_gemv_sm89.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
import pathlib

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _event_ms(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        times.append(float(s.elapsed_time(e)))
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--ncu", action="store_true",
                   help="single iteration for ncu profiling")
    args = p.parse_args()
    torch.cuda.set_device(torch.device(args.device))

    try:
        from flash_rt import flash_rt_qwen3_vl_kernels as fvk
    except ImportError:
        from flash_rt import flash_rt_kernels as fvk

    shapes_8b = [
        ("8B_qkv",     4096,  5120),
        ("8B_gate_up", 4096, 28672),
        ("8B_down",   14336,  4096),
        ("8B_lm_head", 4096, 152064),
    ]
    shapes_2b = [
        ("2B_qkv",     2048,  4096),
        ("2B_gate_up", 2048, 12288),
        ("2B_down",    6144,  2048),
        ("2B_lm_head", 2048, 151936),
    ]

    all_shapes = shapes_8b + shapes_2b
    s = torch.cuda.current_stream()

    print(f"{'name':<14} {'K':>6} {'N':>7} {'variant':>7} "
          f"{'median_us':>9} {'GB/s':>7} {'%peak':>6}")
    print("-" * 68)

    peak_bw = 1008.0  # RTX 4090 GB/s

    for name, K, N in all_shapes:
        A = torch.randn(K, device=args.device).to(torch.float8_e4m3fn)
        B = torch.randn(N, K, device=args.device).to(torch.float8_e4m3fn)
        D = torch.empty(N, device=args.device, dtype=torch.bfloat16)
        K128 = K // 128
        act_scale = torch.ones(K128, device=args.device, dtype=torch.float32)
        w_scale = torch.ones(((N + 127) // 128) * K128,
                             device=args.device, dtype=torch.float32)

        variants = [
            ("w4",  fvk.ht_gemv_fp8_block128_m1_w4),
            ("w8",  fvk.ht_gemv_fp8_block128_m1_w8),
            ("w16", fvk.ht_gemv_fp8_block128_m1_w16),
        ]

        data_bytes = N * K  # weight matrix (FP8, 1 byte each)

        for vname, fn in variants:
            def run():
                fn(A.data_ptr(), B.data_ptr(), D.data_ptr(),
                   1, N, K,
                   act_scale.data_ptr(), w_scale.data_ptr(),
                   1.0, s.cuda_stream)

            if args.ncu:
                run()
            else:
                times = _event_ms(run, warmup=10, iters=args.iters)
                med_us = statistics.median(times) * 1000
                bw = data_bytes / (med_us * 1e-6) / 1e9
                pct = bw / peak_bw * 100
                print(f"{name:<14} {K:>6} {N:>7} {vname:>7} "
                      f"{med_us:>9.1f} {bw:>7.1f} {pct:>5.1f}%")
        if not args.ncu:
            print()


if __name__ == "__main__":
    main()

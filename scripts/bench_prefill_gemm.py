#!/usr/bin/env python3
"""Trusted SM89 FP8 block-128 prefill GEMM micro-bench.

Three L2 regimes, all timed with CUDA events on the default stream (correct
graph/stream handling):

  warm   — one B reused; B fully L2-resident. Upper bound for a single tile
           (L2 hits, compute-bound). Useful for relative tile ranking at the
           "kernel is fast" limit.
  layer  — N_LAYERS distinct B matrices cycled (one per virtual layer), each
           timed launch steps through them round-robin. Closest to real
           prefill: per-layer weight is distinct, so weights do NOT all fit L2
           (8B gate_up 112MB > 72MB) and there is partial eviction. This is
           the PRIMARY metric.
  cold   — 128MB L2 pollute before each timed launch. True cold-B. Lower
           bound (HBM-load dominated). Use to check HBM-bound shapes.

Peak = 165 TFLOPS dense FP8 (RTX 4090, no 2:4 sparsity). NOT 330 (that's sparse).
Report median/min over --iters timed launches, % of dense peak.

Usage:
  python bench_prefill_gemm.py --M 512 --regime layer
  python bench_prefill_gemm.py --M 512 --regime layer --tiles 64x64,64x64_s1
"""
from __future__ import annotations
import argparse, statistics, sys, pathlib
import torch
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PEAK_DENSE_TFLOPS = 165.0  # RTX 4090 FP8 dense (no 2:4 sparsity)

# (name, M-tile, N-tile, warps, stages) — mirrors csrc DEFINE list
TILES = {
    "16x64":      ("bench_fp8_block128_gemm_bs_sm89_16x64x128_w4",      16, 64),
    "32x64":      ("bench_fp8_block128_gemm_bs_sm89_32x64x128_w4",      32, 64),
    "64x64":      ("bench_fp8_block128_gemm_bs_sm89_64x64x128_w4",      64, 64),
    "64x64_s1":   ("bench_fp8_block128_gemm_bs_sm89_64x64x128_w4_s1",   64, 64),
    "32x128":     ("bench_fp8_block128_gemm_bs_sm89_32x128x128_w4",     32, 128),
    "64x128_w8":  ("bench_fp8_block128_gemm_bs_sm89_64x128x128_w8",     64, 128),
    "128x128_w8": ("bench_fp8_block128_gemm_bs_sm89_128x128x128_w8",    128, 128),
    "128x128_w8_s1": ("bench_fp8_block128_gemm_bs_sm89_128x128x128_w8_s1", 128, 128),
}

SHAPES_2B = [("2B_qkv", 4096, 2048), ("2B_o", 2048, 2048),
             ("2B_gate_up", 12288, 2048), ("2B_down", 2048, 6144)]
SHAPES_8B = [("8B_qkv", 6144, 4096), ("8B_o", 4096, 4096),
             ("8B_gate_up", 24576, 4096), ("8B_down", 4096, 12288)]


def bench(fn, args, warmup, iters, pollute=None):
    s = torch.cuda.current_stream().cuda_stream
    # All bench GEMM bindings take (A,B,D,M,N,K,asc,wsc,stream). The cold/warm
    # callers already pass `s` as the trailing positional arg, so forward as-is.
    def run():
        fn(*args)
    for _ in range(warmup):
        if pollute is not None: pollute()
        run()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if pollute is not None: pollute()
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); run(); en.record(); en.synchronize()
        times.append(float(st.elapsed_time(en)) * 1000)  # us
    return statistics.median(times), min(times), statistics.stdev(times) if len(times) > 1 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--M", type=int, nargs="+", default=[512])
    p.add_argument("--regime", choices=["warm", "layer", "cold"], default="layer")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--layers", type=int, default=36)
    p.add_argument("--tiles", default="", help="comma-separated subset of tiles")
    p.add_argument("--model", choices=["2B", "8B", "both"], default="both")
    args = p.parse_args()
    torch.cuda.set_device(torch.device(args.device))
    from flash_rt import flash_rt_qwen3_vl_kernels as fvk

    sel = list(TILES.keys()) if not args.tiles else args.tiles.split(",")
    shapes = []
    if args.model in ("2B", "both"): shapes += SHAPES_2B
    if args.model in ("8B", "both"): shapes += SHAPES_8B

    poll = torch.empty(128 * 1024 * 1024, device=args.device, dtype=torch.uint8)
    print(f"device={torch.cuda.get_device_name()} regime={args.regime} "
          f"peak={PEAK_DENSE_TFLOPS} TFLOPS dense FP8")
    print(f"{'shape':<12} {'M':>5} {'N':>6} {'K':>5} {'tile':<14} "
          f"{'med_us':>8} {'min_us':>7} {'std_us':>6} {'TFLOPS':>7} {'%peak':>6}")
    print("-" * 84)

    for M in args.M:
        for name, N, K in shapes:
            flops = 2.0 * M * N * K
            K128 = K // 128
            # per-layer distinct B (layer regime); single B otherwise
            nB = args.layers if args.regime == "layer" else 1
            Bs = [torch.randn(N, K, device=args.device).to(torch.float8_e4m3fn)
                  for _ in range(nB)]
            A = torch.randn(M, K, device=args.device).to(torch.float8_e4m3fn)
            D = torch.empty(M, N, device=args.device, dtype=torch.bfloat16)
            asc = torch.ones(M, K128, device=args.device, dtype=torch.float32)
            wsc = torch.ones((N // 128) * K128, device=args.device, dtype=torch.float32)
            s = torch.cuda.current_stream().cuda_stream
            B0 = Bs[0]

            for tname in sel:
                if tname not in TILES:
                    print(f"  {tname}: unknown tile"); continue
                bind_name, _, _ = TILES[tname]
                fn = getattr(fvk, bind_name)
                try:
                    if args.regime == "cold":
                        med, mn, sd = bench(fn, (A.data_ptr(), B0.data_ptr(), D.data_ptr(),
                                                 M, N, K, asc.data_ptr(), wsc.data_ptr(), s),
                                            args.warmup, args.iters,
                                            pollute=lambda: poll.fill_(0))
                    elif args.regime == "warm":
                        med, mn, sd = bench(fn, (A.data_ptr(), B0.data_ptr(), D.data_ptr(),
                                                 M, N, K, asc.data_ptr(), wsc.data_ptr(), s),
                                            args.warmup, args.iters)
                    else:  # layer: step through distinct B
                        idx = [0]
                        def step():
                            b = Bs[idx[0] % len(Bs)]
                            fn(A.data_ptr(), b.data_ptr(), D.data_ptr(),
                               M, N, K, asc.data_ptr(), wsc.data_ptr(), s)
                            idx[0] += 1
                        for _ in range(args.warmup): step()
                        torch.cuda.synchronize()
                        ts = []
                        for _ in range(args.iters):
                            st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
                            st.record(); step(); en.record(); en.synchronize()
                            ts.append(float(st.elapsed_time(en)) * 1000)
                        med, mn, sd = statistics.median(ts), min(ts), statistics.stdev(ts)
                    tf = flops / (med * 1e-6) / 1e12
                    print(f"{name:<12} {M:>5} {N:>6} {K:>5} {tname:<14} "
                          f"{med:>8.2f} {mn:>7.2f} {sd:>6.2f} {tf:>7.1f} {tf/PEAK_DENSE_TFLOPS*100:>5.1f}%")
                except Exception as e:
                    print(f"{name:<12} {M:>5} {N:>6} {K:>5} {tname:<14} ERROR: {e}")
            print()
        if M != args.M[-1]: print()


if __name__ == "__main__":
    main()

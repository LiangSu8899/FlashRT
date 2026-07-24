#!/usr/bin/env python3
"""Correctness + micro-bench for the SM89 FP8 residual-fold GEMM epilogue.

Verifies that bench_<tile>_resid (D = bf16(acc) + resid) is bit-identical to
the two-step baseline (plain bench_<tile> then D += resid), then micro-benches
both to measure the residual-fold win (saves 1 launch + 1 D HBM round-trip).

Usage:
    python scripts/test_resid_fold_correctness.py --M 512
"""
from __future__ import annotations
import argparse, pathlib, sys, statistics
import torch
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from flash_rt import flash_rt_qwen3_vl_kernels as vlk

# 8B down-proj: M=S, N=4096, K=12288 (mirror the prefill down GEMM).
TILES = {
    "32x64":   ("bench_fp8_block128_gemm_bs_sm89_32x64x128_w4",      "bench_fp8_block128_gemm_bs_sm89_32x64x128_w4_resid"),
    "64x64":   ("bench_fp8_block128_gemm_bs_sm89_64x64x128_w4",      "bench_fp8_block128_gemm_bs_sm89_64x64x128_w4_resid"),
    "64x64_s1":("bench_fp8_block128_gemm_bs_sm89_64x64x128_w4_s1",   "bench_fp8_block128_gemm_bs_sm89_64x64x128_w4_s1_resid"),
    "128x128_s1":("bench_fp8_block128_gemm_bs_sm89_128x128x128_w8_s1","bench_fp8_block128_gemm_bs_sm89_128x128x128_w8_s1_resid"),
}

def ev_ms(fn, iters=50, warmup=20):
    s=[torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e=[torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    for i in range(iters): s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a,b in zip(s,e)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--M", type=int, nargs="+", default=[128,512,1024])
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=12288)
    args=p.parse_args()
    dev=torch.device("cuda:0")
    torch.manual_seed(0)
    for M in args.M:
        N,K=args.N,args.K
        A=torch.randn(M,K,device=dev).to(torch.float8_e4m3fn)
        B=torch.randn(N,K,device=dev).to(torch.float8_e4m3fn)
        resid=torch.randn(M,N,device=dev,dtype=torch.bfloat16)*0.1
        asc=(torch.randn(M,K//128,device=dev)*0.1).contiguous()
        wsc=(torch.randn(N//128,K//128,device=dev)*0.1).contiguous()
        s=torch.cuda.current_stream().cuda_stream
        print(f"\n=== M={M} N={N} K={K} ===")
        print(f"{'tile':>12} {'base_us':>8} {'resid_us':>9} {'delta_us':>9} {'cos':>10} {'maxabs':>10}")
        for name,(base_fn,resid_fn) in TILES.items():
            fb=getattr(vlk,base_fn); fr=getattr(vlk,resid_fn)
            D_base=torch.empty(M,N,device=dev,dtype=torch.bfloat16)
            D_resid=torch.empty(M,N,device=dev,dtype=torch.bfloat16)
            # baseline: GEMM then D += resid
            fb(int(A.data_ptr()),int(B.data_ptr()),int(D_base.data_ptr()),M,N,K,int(asc.data_ptr()),int(wsc.data_ptr()),s)
            D_base += resid
            # residual-fold: one launch
            fr(int(A.data_ptr()),int(B.data_ptr()),int(D_resid.data_ptr()),M,N,K,int(asc.data_ptr()),int(wsc.data_ptr()),int(resid.data_ptr()),s)
            cos=torch.nn.functional.cosine_similarity(D_base.flatten().unsqueeze(0),D_resid.flatten().unsqueeze(0)).item()
            maxabs=(D_base.float()-D_resid.float()).abs().max().item()
            # bench (layer-ish: reuse A/B, cold-ish resid each iter to mimic per-step residual)
            def run_base():
                fb(int(A.data_ptr()),int(B.data_ptr()),int(D_base.data_ptr()),M,N,K,int(asc.data_ptr()),int(wsc.data_ptr()),s)
                D_base.add_(resid)
            def run_resid():
                fr(int(A.data_ptr()),int(B.data_ptr()),int(D_resid.data_ptr()),M,N,K,int(asc.data_ptr()),int(wsc.data_ptr()),int(resid.data_ptr()),s)
            b=statistics.median(ev_ms(run_base)); r=statistics.median(ev_ms(run_resid))
            flag="OK" if maxabs==0 else ("close" if cos>0.9999 else "FAIL")
            print(f"{name:>12} {b*1e3:>8.1f} {r*1e3:>9.1f} {(b-r)*1e3:>+9.1f} {cos:>10.6f} {maxabs:>10.2e} {flag}")

if __name__=="__main__":
    main()

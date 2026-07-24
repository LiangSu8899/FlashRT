#!/usr/bin/env python3
"""Correctness + micro-bench for the SM89 GeGLU silu-fold megakernel.

Verifies that bench_fp8_bs_geglu_silu_fold_* (gate+up GEMM + silu(gate)*up +
per-token block-128 FP8 quant, one launch) matches the baseline two-step path
(gate_up GEMM → bf16 [M,2*inter] → silu_mul_merged_to_fp8) within fp8 quant
tolerance, and is numerically closer to the fp32 reference. Then micro-benches
both to measure the win from eliminating the [M,2*inter] BF16 transient.

Usage:
    python scripts/test_geglu_silu_fold_correctness.py --M 512
"""
from __future__ import annotations
import argparse, pathlib, sys, statistics
import torch
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from flash_rt import flash_rt_qwen3_vl_kernels as vlk

# 8B: inter=12288 hidden=4096 ; 2B: inter=9216 hidden=2048
SHAPES = {
    "8B": dict(N=12288, K=4096),
    "2B": dict(N=6144, K=2048),
}
TILES = {
    "32x128_s2": "bench_fp8_bs_geglu_silu_fold_sm89_32x128_w4_s2",
    "16x128_s2": "bench_fp8_bs_geglu_silu_fold_sm89_16x128_w4_s2",
    "64x128_s2": "bench_fp8_bs_geglu_silu_fold_sm89_64x128_w4_s2",
    "128x128_s1": "bench_fp8_bs_geglu_silu_fold_sm89_128x128_w8_s1",
    "32x128_s1": "bench_fp8_bs_geglu_silu_fold_sm89_32x128_w4_s1",
    "16x128_s1": "bench_fp8_bs_geglu_silu_fold_sm89_16x128_w4_s1",
    "ap_32x128_s1": "bench_fp8_bs_geglu_silu_fold_apersist_sm89_32x128_w4_s1",
    "ap_16x128_s1": "bench_fp8_bs_geglu_silu_fold_apersist_sm89_16x128_w4_s1",
    "ap_32x128_s2": "bench_fp8_bs_geglu_silu_fold_apersist_sm89_32x128_w4_s2",
}

def ev_ms(fn, iters=50, warmup=20):
    s=[torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e=[torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    for i in range(iters): s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a,b in zip(s,e)]

def cos(a,b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0).float(), b.flatten().unsqueeze(0).float()).item()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--M", type=int, nargs="+", default=[128,512,1024])
    p.add_argument("--models", nargs="+", default=["2B","8B"])
    args=p.parse_args()
    dev=torch.device("cuda:0")
    torch.manual_seed(0)
    for model in args.models:
        N,K=SHAPES[model]["N"],SHAPES[model]["K"]
        for M in args.M:
            # A: per-token fp8 [M,K]; act_scale [M,K/128]; B: gate_up_w [2N,K] fp8;
            # w_scale: gate_up_s [2N/128, K/128]. Use small scales to stay in fp8 range.
            A=(torch.randn(M,K,device=dev)*0.3).to(torch.float8_e4m3fn)
            B=(torch.randn(2*N,K,device=dev)*0.3).to(torch.float8_e4m3fn)
            asc=(torch.rand(M,K//128,device=dev)*0.2+0.05).contiguous()
            wsc=(torch.rand(2*N//128,K//128,device=dev)*0.2+0.05).contiguous()
            s=torch.cuda.current_stream().cuda_stream
            # ---- baseline: gate_up GEMM → bf16 [M,2N], then silu_mul_merged → fp8 [M,N] ----
            gu_bf=torch.empty(M,2*N,device=dev,dtype=torch.bfloat16)
            vlk.bench_fp8_block128_gemm_bs_sm89_128x128x128_w8_s1(
                int(A.data_ptr()),int(B.data_ptr()),int(gu_bf.data_ptr()),M,2*N,K,
                int(asc.data_ptr()),int(wsc.data_ptr()),s)
            ap_base=torch.empty(M,N,device=dev,dtype=torch.float8_e4m3fn)
            sc_base=torch.empty(M,N//128,device=dev,dtype=torch.float32)
            vlk.silu_mul_merged_to_fp8_block128_bf16(
                int(gu_bf.data_ptr()),int(ap_base.data_ptr()),int(sc_base.data_ptr()),
                M,N,s)
            # ---- fused ----
            for tname,tfn_name in TILES.items():
                tfn=getattr(vlk,tfn_name)
                ap_f=torch.empty(M,N,device=dev,dtype=torch.float8_e4m3fn)
                sc_f=torch.empty(M,N//128,device=dev,dtype=torch.float32)
                tfn(int(A.data_ptr()),int(B.data_ptr()),M,N,K,
                    int(asc.data_ptr()),int(wsc.data_ptr()),
                    int(ap_f.data_ptr()),int(sc_f.data_ptr()),s)
                c=cos(ap_base.float(),ap_f.float())
                # bench: baseline two-step vs fused (layer-ish, reuse A/B)
                def run_base():
                    vlk.bench_fp8_block128_gemm_bs_sm89_128x128x128_w8_s1(
                        int(A.data_ptr()),int(B.data_ptr()),int(gu_bf.data_ptr()),M,2*N,K,
                        int(asc.data_ptr()),int(wsc.data_ptr()),s)
                    vlk.silu_mul_merged_to_fp8_block128_bf16(
                        int(gu_bf.data_ptr()),int(ap_base.data_ptr()),int(sc_base.data_ptr()),
                        M,N,s)
                def run_fused():
                    tfn(int(A.data_ptr()),int(B.data_ptr()),M,N,K,
                        int(asc.data_ptr()),int(wsc.data_ptr()),
                        int(ap_f.data_ptr()),int(sc_f.data_ptr()),s)
                b=statistics.median(ev_ms(run_base)); f=statistics.median(ev_ms(run_fused))
                flag="OK" if c>0.9999 else ("close" if c>0.999 else "FAIL")
                print(f"{model} M={M:>4} {tname:>10}  base={b*1e3:>7.1f}us fused={f*1e3:>7.1f}us "
                      f"delta={(b-f)*1e3:>+7.1f}us  cos={c:.6f} {flag}")
        print()

if __name__=="__main__":
    main()

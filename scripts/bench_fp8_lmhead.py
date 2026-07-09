#!/usr/bin/env python3
"""A/B benchmark: BF16 vs FP8 lm_head on Qwen3-VL 8B SM89 decode.

Loads two text frontends (one BF16 lm_head, one FP8), warms up decode
graphs, and compares per-step latency. Reports the lm_head delta.

Usage:
    python scripts/bench_fp8_lmhead.py \
        --checkpoint /path/to/Qwen3-VL-8B-Instruct-FP8
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def bench_decode(fe, label, iters=50, warmup=10, decode_pos=63):
    fe.reset_state()
    hidden = int(fe._cfg['hidden_size'])
    embed = fe._weights.anchors[0]
    ids = torch.arange(0, decode_pos + 1, device=fe.device, dtype=torch.long)
    h = embed[ids].to(torch.bfloat16).view(decode_pos + 1, hidden)
    cos = fe._rope_cos_table[:decode_pos + 1]
    sin = fe._rope_sin_table[:decode_pos + 1]
    fe.forward_hidden_prefill_fp8_blockscaled(h, cos, sin, 0)
    torch.cuda.synchronize()

    tok = 100
    for _ in range(warmup):
        fe.decode_step_with_graph(tok, decode_pos)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fe.decode_step_with_graph(tok, decode_pos)
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))

    med = statistics.median(times)
    mn = statistics.mean(times)
    lo = min(times)
    print(f"  [{label}] median={med:.3f} ms  mean={mn:.3f} ms  "
          f"min={lo:.3f} ms  ({iters} iters)")
    return times


def run_one(checkpoint, device, max_seq, fp8_lm_head, iters, decode_pos):
    import gc
    from flash_rt.frontends.torch.qwen3_vl_fp8_sm89 import (
        Qwen3VlFp8Sm89TextFrontend,
    )

    label = 'fp8_lmhead' if fp8_lm_head else 'bf16_lmhead'
    print(f"\nLoading {label} frontend...")
    fe = Qwen3VlFp8Sm89TextFrontend(
        checkpoint, device=device, max_seq=max_seq,
        use_fp8_lm_head=fp8_lm_head)
    times = bench_decode(fe, label, iters, decode_pos=decode_pos)

    logits = fe.decode_step_with_graph(100, decode_pos)
    torch.cuda.synchronize()
    top = int(logits.float().argmax())

    del fe
    gc.collect()
    torch.cuda.empty_cache()
    return times, top


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seq', type=int, default=2048)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--decode-pos', type=int, default=63)
    args = p.parse_args()

    torch.cuda.set_device(torch.device(args.device))
    print(f"GPU: {torch.cuda.get_device_name(torch.device(args.device))}")
    print(f"checkpoint: {args.checkpoint}")

    t_bf16, top_bf16 = run_one(args.checkpoint, args.device, args.max_seq,
                                False, args.iters, args.decode_pos)
    t_fp8, top_fp8 = run_one(args.checkpoint, args.device, args.max_seq,
                              True, args.iters, args.decode_pos)

    med_bf16 = statistics.median(t_bf16)
    med_fp8 = statistics.median(t_fp8)
    delta = med_bf16 - med_fp8
    print(f"\n  Summary:")
    print(f"    bf16_lmhead median={med_bf16:.3f} ms")
    print(f"    fp8_lmhead  median={med_fp8:.3f} ms")
    print(f"    delta={delta:.3f} ms  ({delta/med_bf16*100:.1f}%)")
    print(f"    top_bf16={top_bf16}  top_fp8={top_fp8}  "
          f"same_top={'YES' if top_bf16 == top_fp8 else 'NO'}")


if __name__ == '__main__':
    main()

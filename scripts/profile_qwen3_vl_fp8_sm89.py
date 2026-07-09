#!/usr/bin/env python3
"""nsys / ncu profiling harness for Qwen3-VL FP8 SM89.

Separates prefill and decode into distinct NVTX ranges so nsys can
report per-phase kernel time breakdowns.

Usage (nsys):
    nsys profile -t cuda,nvtx -o qwen3_vl_profile \
        --force-overwrite --capture-range=cudaProfilerApi \
        python scripts/profile_qwen3_vl_fp8_sm89.py \
            --checkpoint /path/to/Qwen3-VL-8B-Instruct-FP8 \
            --decode-steps 32

Usage (ncu, single decode step):
    ncu --set full -o decode_kernel \
        python scripts/profile_qwen3_vl_fp8_sm89.py \
            --checkpoint /path/to/Qwen3-VL-8B-Instruct-FP8 \
            --ncu-mode decode --ncu-kernel-skip 0 --ncu-kernel-count 999
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--image", default=str(REPO_ROOT / "FlashRT.png"))
    p.add_argument("--prompt", default="Describe this image in one sentence.")
    p.add_argument("--decode-steps", type=int, default=32,
                   help="number of decode steps to profile")
    p.add_argument("--warmup-rounds", type=int, default=2,
                   help="warmup iterations before profiling (outside cudaProfiler)")
    p.add_argument("--profile-rounds", type=int, default=1,
                   help="profiled iterations (inside cudaProfiler)")
    p.add_argument("--no-graph", dest="use_graph", action="store_false",
                   default=True,
                   help="use the eager (non-graph) path instead of CUDA graphs")
    p.add_argument("--ncu-mode", choices=["prefill", "decode", "none"],
                   default="none",
                   help="for ncu: only profile this phase (skip the other)")
    args = p.parse_args()

    torch.cuda.set_device(torch.device(args.device))
    dev = args.device

    from flash_rt.frontends.torch.qwen3_vl_fp8_sm89_multimodal import (
        Qwen3VlFp8Sm89Frontend,
    )

    print(f"GPU: {torch.cuda.get_device_name(torch.device(dev))}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"loading model...", flush=True)

    fe = Qwen3VlFp8Sm89Frontend(
        args.checkpoint, device=dev, max_seq=args.max_seq,
        max_pixels=args.max_pixels)

    img = Image.open(args.image).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": args.prompt},
        ],
    }]

    # ── Warmup (outside profiler capture) ──
    # Run all decode positions so every CUDA Graph is captured before profiling.
    print(f"warming up ({args.warmup_rounds} rounds, "
          f"{args.decode_steps} decode steps each)...", flush=True)
    for w in range(args.warmup_rounds):
        fe.set_prompt(messages)
        logits = fe.prefill_graph() if args.use_graph else fe.prefill()
        pr = fe._prompt
        base_slot = int(pr["S"])
        tok = int(logits[0].float().argmax())
        for i in range(args.decode_steps):
            if args.use_graph:
                logits = fe.decode_step_with_graph(tok, base_slot + i)
            else:
                logits = fe.decode_step(tok, base_slot + i)
            tok = int(logits[0].float().argmax())
    torch.cuda.synchronize()
    print("warmup done.", flush=True)

    # ── Profile ──
    # Reuse the warm prompt from the last warmup round (same messages,
    # same S / mrope_max / graph keys). Avoids re-capturing graphs.
    pr = fe._prompt
    assert pr is not None
    base_slot = int(pr["S"])

    torch.cuda.cudart().cudaProfilerStart()

    for r in range(args.profile_rounds):
        # -- prefill --
        if args.ncu_mode != "decode":
            fe.set_prompt(messages)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"prefill_r{r}")
            if args.use_graph:
                logits = fe.prefill_graph()
            else:
                logits = fe.prefill()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        else:
            fe.set_prompt(messages)
            logits = fe.prefill_graph() if args.use_graph else fe.prefill()
            torch.cuda.synchronize()

        tok = int(logits[0].float().argmax())

        # -- decode (graphs already warm from warmup) --
        if args.ncu_mode != "prefill":
            torch.cuda.nvtx.range_push(f"decode_r{r}")
            t0 = time.perf_counter()
            for i in range(args.decode_steps):
                torch.cuda.nvtx.range_push(f"step_{i}")
                if args.use_graph:
                    logits = fe.decode_step_with_graph(tok, base_slot + i)
                else:
                    logits = fe.decode_step(tok, base_slot + i)
                tok = int(logits[0].float().argmax())
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            torch.cuda.nvtx.range_pop()
            print(f"[r{r}] {args.decode_steps} decode steps: "
                  f"{dt:.1f} ms total, {dt/args.decode_steps:.2f} ms/tok, "
                  f"{1000*args.decode_steps/dt:.1f} tok/s")

    torch.cuda.cudart().cudaProfilerStop()
    print("profiling done.")


if __name__ == "__main__":
    main()

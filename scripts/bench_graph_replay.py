#!/usr/bin/env python3
"""Test graph replay after re-prompting.

Verifies that decode graph replay does not regress when set_prompt() is
called again with the same prompt (multi-turn scenario).

Usage:
    python scripts/bench_graph_replay.py \
        --checkpoint /path/to/Qwen3-VL-8B-Instruct-FP8
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import torch
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def bench_decode_steps(fe, n_steps, label):
    p = fe._prompt
    assert p is not None
    base_slot = int(p['S'])
    logits = fe.prefill_graph()
    tok = int(logits[0].float().argmax())

    for i in range(n_steps):
        logits = fe.decode_step_with_graph(tok, base_slot + i)
        tok = int(logits[0].float().argmax())
    torch.cuda.synchronize()

    times = []
    for _ in range(3):
        fe.set_prompt(fe._last_messages)
        logits = fe.prefill_graph()
        tok = int(logits[0].float().argmax())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n_steps):
            logits = fe.decode_step_with_graph(tok, base_slot + i)
            tok = int(logits[0].float().argmax())
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        per_step = dt / n_steps
        times.append(per_step)
        print(f"  [{label}] {n_steps} steps: {dt:.1f} ms total, "
              f"{per_step:.2f} ms/step")
    return times


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seq', type=int, default=2048)
    p.add_argument('--decode-steps', type=int, default=16)
    p.add_argument('--image', default=str(REPO_ROOT / 'FlashRT.png'))
    p.add_argument('--prompt', default='Describe this image in one sentence.')
    args = p.parse_args()

    torch.cuda.set_device(torch.device(args.device))
    print(f"GPU: {torch.cuda.get_device_name(torch.device(args.device))}")

    from flash_rt.frontends.torch.qwen3_vl_fp8_sm89_multimodal import (
        Qwen3VlFp8Sm89Frontend,
    )

    fe = Qwen3VlFp8Sm89Frontend(
        args.checkpoint, device=args.device, max_seq=args.max_seq)
    img = Image.open(args.image).convert('RGB')
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': img},
            {'type': 'text', 'text': args.prompt},
        ],
    }]
    fe._last_messages = messages

    print(f"\n--- Round 1: initial set_prompt + warmup ---")
    fe.set_prompt(messages)
    t1 = bench_decode_steps(fe, args.decode_steps, 'round1')

    print(f"\n--- Round 2: re-prompt (same messages), graphs should be warm ---")
    t2 = bench_decode_steps(fe, args.decode_steps, 'round2')

    print(f"\n--- Round 3: re-prompt again ---")
    t3 = bench_decode_steps(fe, args.decode_steps, 'round3')

    med1 = statistics.median(t1)
    med2 = statistics.median(t2)
    med3 = statistics.median(t3)
    print(f"\n  Summary (median ms/step):")
    print(f"    round1={med1:.2f}  round2={med2:.2f}  round3={med3:.2f}")
    if med1 > 0:
        ratio = med2 / med1
        print(f"    round2/round1={ratio:.2f}x "
              f"{'(regression!)' if ratio > 1.5 else '(OK)'}")


if __name__ == '__main__':
    main()

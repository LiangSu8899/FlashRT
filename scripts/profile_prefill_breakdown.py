#!/usr/bin/env python3
"""Prefill breakdown profiling: ViT vs LLM on Qwen3-VL 8B SM89.

Runs vision and LLM prefill in separate NVTX ranges for nsys analysis.

Usage:
    nsys profile -t cuda,nvtx -o prefill_breakdown \
        --force-overwrite true --capture-range=cudaProfilerApi \
        python scripts/profile_prefill_breakdown.py \
            --checkpoint /path/to/Qwen3-VL-8B-Instruct-FP8
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
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seq', type=int, default=2048)
    p.add_argument('--image', default=str(REPO_ROOT / 'FlashRT.png'))
    p.add_argument('--prompt', default='Describe this image in one sentence.')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--profile-rounds', type=int, default=1)
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

    for w in range(args.warmup):
        fe.set_prompt(messages)
        fe.prefill()
        torch.cuda.synchronize()
    print("warmup done.", flush=True)

    torch.cuda.cudart().cudaProfilerStart()

    for r in range(args.profile_rounds):
        fe.set_prompt(messages)
        pr = fe._prompt
        assert pr is not None
        llm = fe.llm
        llm.reset_state()
        hidden = int(llm._cfg['hidden_size'])
        S = int(pr['S'])
        embed = llm._weights.anchors[0]

        torch.cuda.synchronize()

        # ── Phase 1: Vision tower ──
        torch.cuda.nvtx.range_push(f'vision_r{r}')
        h = embed[pr['input_ids']].to(torch.bfloat16).view(S, hidden)
        seg_deepstacks = []
        off = 0
        for (a, b), n_patch in zip(pr['spans'], pr['seg_patches']):
            sl = slice(off, off + n_patch)
            off += n_patch
            emb, ds = fe.vision.forward(
                pr['pixel_values'][sl], pr['pos_embeds'][sl],
                pr['vcos'][sl], pr['vsin'][sl])
            h[a:b].copy_(emb.to(torch.bfloat16))
            seg_deepstacks.append(ds)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        # ── Phase 2: LLM prefill ──
        deep = {}
        for layer in range(fe._deepstack_layers):
            rows = []
            for (a, b_), ds in zip(pr['spans'], seg_deepstacks):
                rows.append((a, b_, ds[layer].to(torch.bfloat16).contiguous()))
            deep[layer] = rows

        torch.cuda.nvtx.range_push(f'llm_prefill_r{r}')
        logits = llm.forward_hidden_prefill_fp8_blockscaled(
            h, pr['mcos'], pr['msin'], 0, deepstack_by_layer=deep)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        # ── Phase 3: lm_head only (already in LLM prefill, just timing) ──
        print(f"[r{r}] S={S} top={int(logits[0].float().argmax())}")

    torch.cuda.cudart().cudaProfilerStop()
    print("profiling done.")


if __name__ == '__main__':
    main()

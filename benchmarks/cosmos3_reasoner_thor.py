#!/usr/bin/env python
"""Cosmos3-Edge Reasoner FlashRT benchmark + golden parity (Thor, batch 1, greedy).

Input preparation replicates the official
``OmniMoTModel.generate_reasoner_text`` flow exactly (same processor, same
chat-template call), then runs the FlashRT engine and compares the generated
text with the official golden outputs.

Run inside the cosmos venv with cosmos-framework importable:
    python benchmarks/cosmos3_reasoner_thor.py \
      --checkpoint /work/models/Cosmos3-Edge \
      --golden-dir /work/.tmp_cosmos_edge_outputs_reasoner
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def build_processor(checkpoint: str):
    from cosmos_framework.data.generator.processors import build_processor_lazy

    return build_processor_lazy(tokenizer_type=checkpoint)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/work/models/Cosmos3-Edge")
    ap.add_argument("--golden-dir", default="/work/.tmp_cosmos_edge_outputs_reasoner")
    ap.add_argument("--inputs-dir", default="/work/.tmp_cosmos_edge_inputs")
    ap.add_argument("--modes", default="text,image,video")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--iters", type=int, default=3, help="timed generate() repeats per mode")
    ap.add_argument("--quant", default="bf16", choices=("bf16", "fp4"))
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from PIL import Image

    from cosmos_framework.inference.inference import _decode_reasoner_video
    from flash_rt.models.cosmos3_reasoner.pipeline_thor import CosmosReasonerThor

    processor = build_processor(args.checkpoint)
    tok = getattr(processor, "tokenizer", processor)

    engine = CosmosReasonerThor(
        args.checkpoint, max_new_tokens=args.max_new_tokens,
        quant=args.quant, use_graph=not args.no_graph)

    results: dict[str, dict] = {}
    for mode in args.modes.split(","):
        sample = json.loads((Path(args.inputs_dir) / f"reasoner_{mode}_0.json").read_text())
        prompt = sample["prompt"]
        media_item = None
        if mode == "image":
            media_item = {"type": "image", "image": Image.open(sample["vision_path"]).convert("RGB")}
        elif mode == "video":
            v = _decode_reasoner_video(sample["vision_path"], sample.get("video_fps"))
            media_item = {"type": "video", "video": v["frames"], "fps": v["fps"]}

        if media_item is None:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            messages = [{"role": "user", "content": [media_item, {"type": "text", "text": prompt}]}]
        pin = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = pin["input_ids"].reshape(-1)
        kwargs: dict = {}
        if mode == "image":
            kwargs = {"pixel_values": pin["pixel_values"], "grid_thw": pin["image_grid_thw"], "is_video": False}
        elif mode == "video":
            kwargs = {"pixel_values": pin["pixel_values_videos"], "grid_thw": pin["video_grid_thw"], "is_video": True}

        stats = None
        out = None
        for _ in range(args.iters):
            out, stats = engine.generate(input_ids, max_new_tokens=args.max_new_tokens, **kwargs)
        text = tok.decode([t for t in out if t != engine.eos_token_id], skip_special_tokens=True)

        golden_path = Path(args.golden_dir) / f"reasoner_{mode}_0" / "reasoner_text.txt"
        golden = golden_path.read_text() if golden_path.exists() else None
        match = None
        first_diff = None
        if golden is not None:
            match = text.strip() == golden.strip()
            if not match:
                for i, (a, b) in enumerate(zip(text, golden)):
                    if a != b:
                        first_diff = i
                        break
                else:
                    first_diff = min(len(text), len(golden))
        results[mode] = {
            **stats,
            "golden_match": match,
            "first_char_diff": first_diff,
            "text": text,
            "token_ids": out,
        }
        print(f"[{mode}] prompt={stats['prompt_tokens']} prefill={stats['prefill_s']:.3f}s "
              f"decode={stats['decode_tok_s']:.1f} tok/s new={stats['new_tokens']} "
              f"golden_match={match} first_diff={first_diff}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
    if any(r.get("golden_match") is False for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

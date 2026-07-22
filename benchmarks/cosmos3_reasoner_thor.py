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
import statistics
import sys
from pathlib import Path

import torch


def build_processor(checkpoint: str):
    from cosmos_framework.data.generator.processors import build_processor_lazy

    return build_processor_lazy(tokenizer_type=checkpoint)


NVIDIA_PUBLIC_PROFILE = {
    "text": {
        "prompt": "Describe a modern robotics research laboratory in one sentence.",
        "prompt_tokens": 1705,
    },
    "image": {
        "prompt": "Describe what is happening in this image in one sentence.",
        "asset": "robot_153.jpg",
        "prompt_tokens": 911,
    },
    "video": {
        "prompt": "Describe what is happening in this video in one sentence.",
        "asset": "video_caption.mp4",
        "video_fps": 0.5,
        "prompt_tokens": 1263,
    },
}


def pad_prompt_tokens(input_ids: torch.Tensor, target: int, filler_id: int) -> torch.Tensor:
    """Pad a prepared chat prompt to an exact benchmark ISL.

    Padding is inserted after the first chat-template token so media sentinel
    runs and the assistant-generation suffix remain intact.
    """
    current = int(input_ids.numel())
    if current > target:
        raise ValueError(f"prepared prompt has {current} tokens, above target ISL {target}")
    if current == target:
        return input_ids
    fill = torch.full((target - current,), filler_id, dtype=input_ids.dtype)
    return torch.cat((input_ids[:1], fill, input_ids[1:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/work/models/Cosmos3-Edge")
    ap.add_argument("--golden-dir", default="/work/.tmp_cosmos_edge_outputs_reasoner")
    ap.add_argument("--inputs-dir", default="/work/.tmp_cosmos_edge_inputs")
    ap.add_argument(
        "--nvidia-assets-dir",
        default=None,
        help="Use NVIDIA Cosmos cookbook assets and published 1705/911/1263-token profile",
    )
    ap.add_argument("--modes", default="text,image,video")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--warmup-iters", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5, help="timed generate() repeats per mode")
    ap.add_argument("--quant", default="bf16", choices=("bf16", "fp4"))
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from PIL import Image

    from cosmos_framework.inference.inference import _decode_reasoner_video
    from flash_rt.models.cosmos3_reasoner.pipeline_thor import CosmosReasonerThor

    processor = build_processor(args.checkpoint)
    tok = getattr(processor, "tokenizer", processor)
    filler_ids = tok.encode(" benchmark", add_special_tokens=False)
    if not filler_ids:
        raise RuntimeError("tokenizer produced no benchmark filler token")
    filler_id = int(filler_ids[0])

    engine = CosmosReasonerThor(
        args.checkpoint, max_new_tokens=args.max_new_tokens,
        quant=args.quant, use_graph=not args.no_graph)

    results: dict[str, dict] = {}
    for mode in args.modes.split(","):
        profile = NVIDIA_PUBLIC_PROFILE[mode] if args.nvidia_assets_dir else None
        if profile:
            sample = dict(profile)
            if "asset" in profile:
                sample["vision_path"] = str(Path(args.nvidia_assets_dir) / profile["asset"])
        else:
            sample = json.loads((Path(args.inputs_dir) / f"reasoner_{mode}_0.json").read_text())
        prompt = str(sample["prompt"])
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
        prepared_prompt_tokens = int(input_ids.numel())
        if profile:
            input_ids = pad_prompt_tokens(input_ids, int(profile["prompt_tokens"]), filler_id)
        kwargs: dict = {}
        if mode == "image":
            kwargs = {"pixel_values": pin["pixel_values"], "grid_thw": pin["image_grid_thw"], "is_video": False}
        elif mode == "video":
            kwargs = {"pixel_values": pin["pixel_values_videos"], "grid_thw": pin["video_grid_thw"], "is_video": True}

        for _ in range(args.warmup_iters):
            engine.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                ignore_eos=bool(profile),
                **kwargs,
            )

        samples = []
        out = None
        for _ in range(args.iters):
            out, stats = engine.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                ignore_eos=bool(profile),
                **kwargs,
            )
            samples.append(stats)
        stats = dict(samples[-1])
        stats["decode_s_p50"] = statistics.median(s["decode_s"] for s in samples)
        stats["decode_tok_s_p50"] = statistics.median(s["decode_tok_s"] for s in samples)
        stats["prefill_s_p50"] = statistics.median(s["prefill_s"] for s in samples)
        stats["prepared_prompt_tokens"] = prepared_prompt_tokens
        stats["raw_samples"] = samples
        text = tok.decode([t for t in out if t != engine.eos_token_id], skip_special_tokens=True)

        golden_path = Path(args.golden_dir) / f"reasoner_{mode}_0" / "reasoner_text.txt"
        golden = golden_path.read_text() if golden_path.exists() else None
        if profile:
            golden = None
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
        print(f"[{mode}] prompt={stats['prompt_tokens']} prefill_p50={stats['prefill_s_p50']:.3f}s "
              f"decode_p50={stats['decode_tok_s_p50']:.1f} tok/s new={stats['new_tokens']} "
              f"golden_match={match} first_diff={first_diff}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
    if any(r.get("golden_match") is False for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

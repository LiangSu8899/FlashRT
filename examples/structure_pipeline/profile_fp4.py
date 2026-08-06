"""Kernel-bucket profile of the FP4-specialist captured graph.

Same harness as ``profile_arms.py`` but for the specialist assembly
(FA4 interface + generic seats + the native DiT chain): assemble, wire,
capture, profile 30 replays, bucket kernel time by name, and report the
top kernels. The buckets name where the remaining milliseconds against
the native pipeline actually go.

Usage: same arguments as ``groot_n17_fp4.py``.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from flash_rt.structures import capture as capture_stage

from dit_fp4_chain import apply_dit_fp4_chain, prepare_dit_fp4_chain
from full_graph import pin_action_noise
from groot_n17 import build, clone_tree, load_policy
from groot_n17_fp4 import install_fa4_interface

BUCKETS = [
    ("fp4_chain", r"fp4|e2m1|nvfp4|sfa|ada_layer_norm|no_affine"),
    ("fp8_structures", r"fp8|e4m3|quant|packed"),
    ("attention", r"fmha|attention|softmax|splitkv|sdpa|flash|fa4|cute"),
    ("gemm", r"gemm|cutlass|cublas|nvjet|s16816|tensorop|gemv|matmul"),
    ("norm_rope", r"norm|rms|rope|rotary"),
    ("triton_fused", r"triton"),
    ("memops", r"memcpy|memset|copy|cat|index|elementwise|vectorized"),
]


def bucket_of(name):
    low = name.lower()
    for bucket, pattern in BUCKETS:
        if re.search(pattern, low):
            return bucket
    return "other"


def profile_replays(stage, iters=30):
    for _ in range(5):
        stage.replay(sync=False)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            stage.replay(sync=False)
        torch.cuda.synchronize()
    per_bucket = collections.Counter()
    per_kernel = collections.Counter()
    for evt in prof.key_averages():
        us = (getattr(evt, "self_device_time_total", 0)
              or getattr(evt, "self_cuda_time_total", 0) or 0)
        if us <= 0:
            continue
        if "cuda" not in str(getattr(evt, "device_type", "")).lower():
            continue
        per_bucket[bucket_of(evt.key)] += us / iters / 1000.0
        per_kernel[evt.key] += us / iters / 1000.0
    return per_bucket, per_kernel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-assets", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    policy = load_policy(args.host, args.checkpoint, args.backbone_assets)
    model = policy.model
    fixture = torch.load(args.fixture, map_location="cpu",
                         weights_only=False)["inputs"]

    from flash_rt.structures import swap
    from flash_rt.structures.impls.cadence_static.cross_attention import (
        wire_refresh_to_producer)

    captured = {}
    original = model.get_action

    def spy(inputs, options=None):
        captured["inputs"] = clone_tree(inputs)
        return original(inputs, options)

    model.get_action = spy
    with torch.inference_mode():
        policy.get_action(fixture)
    model.get_action = original
    backbone_inputs, action_inputs = model.prepare_input(
        dict(captured["inputs"]))
    backbone_inputs = clone_tree(backbone_inputs)
    action_inputs = clone_tree(action_inputs)

    unpin = pin_action_noise()

    def hot():
        out = model.backbone(backbone_inputs)
        return model.action_head.get_action(
            out, action_inputs)["action_pred"]

    def run_once():
        with torch.inference_mode():
            hot()

    try:
        with torch.inference_mode():
            ref0 = hot().detach().clone()

        import os
        fa4_note = install_fa4_interface(model)
        if os.environ.get("FRT_NO_CHAIN"):
            chain, chain_note = None, "disabled by FRT_NO_CHAIN"
        else:
            chain = prepare_dit_fp4_chain(model)
            chain_note = (chain or {}).get("unavailable")

        asm, extras = build(model, run_once)
        handle = swap.attach(model, asm.swaps,
                             observe=extras["observed"],
                             revert=extras["revert"],
                             on_guard_fail="raise")
        if extras["cadence_statics"]:
            wire_refresh_to_producer(model, extras["cadence_statics"],
                                     run_once)
        undo_chain = None
        if chain_note is None:
            undo_chain = apply_dit_fp4_chain(model, chain, run_once)

        torch._dynamo.reset()
        stage = capture_stage(
            torch.compile(hot, mode="max-autotune-no-cudagraphs",
                          fullgraph=False),
            model=model, warmup=8, gate_cos=0, min_speedup=0)
        buckets, kernels = profile_replays(stage)
        stage.replay()
        cos = float(torch.nn.functional.cosine_similarity(
            stage.output.detach().float().cpu().flatten(),
            ref0.detach().float().cpu().flatten(), dim=0))
        report = {
            "arm": "fp4_specialist_chain",
            "fa4_interface": fa4_note or "installed",
            "dit_chain": chain_note or "native transcription",
            "parity": cos,
            "bucket_ms": {k: round(v, 3) for k, v in
                          sorted(buckets.items(), key=lambda kv: -kv[1])},
            "total_ms": round(sum(buckets.values()), 3),
            "top_kernels": [
                {"kernel": k[:110], "ms": round(v, 3)}
                for k, v in kernels.most_common(40)],
        }
        print(json.dumps(report, indent=1, default=str))
        if args.report:
            args.report.write_text(json.dumps(report, indent=1,
                                              default=str))
        if undo_chain is not None:
            undo_chain()
        handle.detach()
        stage.restore_host()
    finally:
        unpin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

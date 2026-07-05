#!/usr/bin/env python3
"""Qwen3.6 MTP/DFlash output parity probe.

Compares:
  * MTP K=1 vs MTP K=6
  * MTP K=1 vs DFlash K=15
  * MTP K=6 vs DFlash K=15

for the raw-doc and compact-chat prompts used by
``qwen36_dflash_decode_bench.py``.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    max_new: int
    chat: bool


CASES = (
    Case(
        "doc_raw",
        "Explain quantum entanglement in one short paragraph.",
        128,
        False,
    ),
    Case(
        "compact_chat",
        "Output a compact JSON plan with two robot actions.",
        32,
        True,
    ),
)


def _input_ids(fe: Any, case: Case):
    from flash_rt.frontends.torch.spec_session import as_input_ids_tensor

    if case.chat:
        encoded = fe._tokenizer.apply_chat_template(
            [{"role": "user", "content": case.prompt}],
            add_generation_prompt=True,
            enable_thinking=True,
            return_tensors="pt",
        )
        return as_input_ids_tensor(encoded, device=fe.device)
    return fe._tokenizer(
        case.prompt, return_tensors="pt").input_ids.to(fe.device)


def _frontend(args):
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    return Qwen36TorchFrontendRtx(
        args.checkpoint, quant="nvfp4", max_seq=args.max_seq)


def _cleanup():
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _run_mtp(args, case: Case, K: int):
    import torch

    fe = _frontend(args)
    ids = _input_ids(fe, case)
    out = fe.generate_own_speculative_KN_nvfp4(
        ids, max_new_tokens=case.max_new, K=K)
    torch.cuda.synchronize()
    stats = {
        "prompt_len": int(ids.shape[1]),
        "new": int(out.shape[1] - ids.shape[1]),
        "attempts": int(fe._spec_attempts),
        "accepts": int(fe._spec_accepts),
        "al": float(fe._spec_accepts) / max(1, int(fe._spec_attempts)),
    }
    tok = fe._tokenizer
    out_cpu = out.detach().cpu()
    del fe, ids, out
    _cleanup()
    return tok, out_cpu, stats


def _run_dflash(args, case: Case):
    import torch

    mtp_env = None
    if args.dflash_without_mtp:
        mtp_env = os.environ.pop("FLASHRT_QWEN36_MTP_CKPT_DIR", None)
    try:
        fe = _frontend(args)
        fe.init_dflash_drafter(args.dflash_checkpoint or None)
    finally:
        if mtp_env is not None:
            os.environ["FLASHRT_QWEN36_MTP_CKPT_DIR"] = mtp_env
    dflash_mode = {
        "pertoken": bool(getattr(fe, "_dflash_pertoken_window", False)),
        "window": getattr(fe, "_dflash_pertoken_win", None),
        "step_saves": bool(fe._dflash_step_saves_enabled()),
        "chunk_saves": bool(fe._dflash_chunk_saves_enabled()),
    }
    ids = _input_ids(fe, case)
    session = fe.make_dflash_session(
        max_new_tokens=case.max_new, K=args.dflash_k)
    out = session.generate(ids)
    torch.cuda.synchronize()
    stats = {
        "prompt_len": int(ids.shape[1]),
        "new": int(out.shape[1] - ids.shape[1]),
        "attempts": int(fe._spec_attempts),
        "accepts": int(fe._spec_accepts),
        "al": float(fe._spec_accepts) / max(1, int(fe._spec_attempts)),
        "dflash_mode": dflash_mode,
    }
    tok = fe._tokenizer
    out_cpu = out.detach().cpu()
    del fe, ids, out, session
    _cleanup()
    return tok, out_cpu, stats


def _first_diff(a, b):
    diff = (a != b).nonzero()
    if not diff.numel():
        return None
    return int(diff[0, 1].item())


def _text_window(tokenizer, tokens, idx: int | None, radius: int):
    if idx is None:
        return None
    start = max(0, idx - radius)
    end = min(int(tokens.shape[1]), idx + radius + 1)
    return {
        "token_range": [start, end],
        "token_ids": tokens[0, start:end].tolist(),
        "text": tokenizer.decode(
            tokens[0, start:end].tolist(), skip_special_tokens=False),
    }


def _compare(tokenizer, prompt_len: int, name_a: str, out_a,
             name_b: str, out_b, radius: int):
    idx = _first_diff(out_a, out_b)
    return {
        "left": name_a,
        "right": name_b,
        "equal": idx is None,
        "first_diff": None if idx is None else {
            "absolute_index": idx,
            "generated_offset": idx - prompt_len,
            "left_token": int(out_a[0, idx].item()),
            "right_token": int(out_b[0, idx].item()),
            "left_context": _text_window(tokenizer, out_a, idx, radius),
            "right_context": _text_window(tokenizer, out_b, idx, radius),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("FLASHRT_QWEN36_NVFP4_CKPT_DIR", ""),
        help="Qwen3.6 NVFP4 checkpoint directory")
    parser.add_argument("--dflash-checkpoint", default="")
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--dflash-k", type=int, default=15)
    parser.add_argument(
        "--dflash-without-mtp", action="store_true",
        help=(
            "Temporarily unset FLASHRT_QWEN36_MTP_CKPT_DIR while loading "
            "the DFlash frontend. Useful for RTX step-save experiments "
            "that do not fit with MTP and DFlash resident together."))
    parser.add_argument("--context-radius", type=int, default=12)
    args = parser.parse_args()

    if not args.checkpoint:
        raise SystemExit(
            "set --checkpoint or FLASHRT_QWEN36_NVFP4_CKPT_DIR")

    import torch

    results = []
    for case in CASES:
        tokenizer, mtp1, mtp1_stats = _run_mtp(args, case, 1)
        _, mtp6, mtp6_stats = _run_mtp(args, case, 6)
        _, dflash15, dflash_stats = _run_dflash(args, case)
        prompt_len = mtp1_stats["prompt_len"]
        if prompt_len != mtp6_stats["prompt_len"]:
            raise RuntimeError(f"{case.name}: MTP prompt length mismatch")
        if prompt_len != dflash_stats["prompt_len"]:
            raise RuntimeError(f"{case.name}: DFlash prompt length mismatch")
        results.append({
            "case": case.name,
            "prompt_len": prompt_len,
            "max_new": case.max_new,
            "chat_template": case.chat,
            "stats": {
                "mtp_k1": mtp1_stats,
                "mtp_k6": mtp6_stats,
                "dflash_k15": dflash_stats,
            },
            "comparisons": [
                _compare(
                    tokenizer, prompt_len, "mtp_k1", mtp1, "mtp_k6",
                    mtp6, args.context_radius),
                _compare(
                    tokenizer, prompt_len, "mtp_k1", mtp1, "dflash_k15",
                    dflash15, args.context_radius),
                _compare(
                    tokenizer, prompt_len, "mtp_k6", mtp6, "dflash_k15",
                    dflash15, args.context_radius),
            ],
        })

    print(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "results": results,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

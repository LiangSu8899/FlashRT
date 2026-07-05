#!/usr/bin/env python3
"""Qwen3.6 DFlash scenario benchmark.

This is the steadier scenario-style counterpart to
``qwen36_dflash_decode_bench.py``. It avoids tiny 32-token completions and
reports a 64/256-token delta decode metric for the same kind of robot/prose
prompts used in the Thor DFlash validation notes.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Scenario:
    name: str
    prompt: str
    chat: bool = True


SCENARIOS = (
    Scenario(
        "robot_json_plan",
        (
            "Output a compact JSON action list for a robot to pick up the "
            "red cube, place it on the tray, then report success. Use "
            "fields action, target, destination, and safety_check."
        ),
    ),
    Scenario(
        "robot_navigation_plan",
        (
            "Plan robot navigation from the charging dock to shelf B while "
            "avoiding a wet floor marker and a moving cart. Return concise "
            "JSON steps with waypoint, constraint, and expected_state."
        ),
    ),
    Scenario(
        "prose_explanation",
        (
            "Explain why speculative decoding can improve latency for a "
            "greedy language-model server, including one limitation."
        ),
    ),
)


def _resolve_frontend_name(name: str) -> str:
    if name == "auto":
        import torch

        return "thor" if torch.cuda.get_device_capability(0) == (11, 0) else "rtx"
    return name


def _frontend_cls(name: str):
    name = _resolve_frontend_name(name)
    if name == "thor":
        from flash_rt.frontends.torch.qwen36_thor import (
            Qwen36TorchFrontendThor,
        )

        return Qwen36TorchFrontendThor
    if name == "rtx":
        from flash_rt.frontends.torch.qwen36_rtx import (
            Qwen36TorchFrontendRtx,
        )

        return Qwen36TorchFrontendRtx
    raise ValueError(f"unknown frontend {name!r}")


def _input_ids(fe: Any, scenario: Scenario):
    from flash_rt.frontends.torch.spec_session import as_input_ids_tensor

    if scenario.chat:
        encoded = fe._tokenizer.apply_chat_template(
            [{"role": "user", "content": scenario.prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        return as_input_ids_tensor(encoded, device=fe.device)
    return fe._tokenizer(
        scenario.prompt, return_tensors="pt").input_ids.to(fe.device)


def _mean(rows, key: str) -> float:
    return float(statistics.mean(r[key] for r in rows))


def _cleanup() -> None:
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _summarize(rows):
    return {
        "runs": rows,
        "new_mean": _mean(rows, "new"),
        "prefill_ms_mean": _mean(rows, "prefill_ms"),
        "decode_ms_mean": _mean(rows, "decode_ms"),
        "decode_tok_s_mean": _mean(rows, "decode_tok_s"),
        "e2e_tok_s_mean": _mean(rows, "e2e_tok_s"),
        "al_mean": _mean(rows, "al"),
    }


def _delta(short_rows, long_rows):
    short_new = _mean(short_rows, "new")
    long_new = _mean(long_rows, "new")
    short_decode = _mean(short_rows, "decode_ms")
    long_decode = _mean(long_rows, "decode_ms")
    short_e2e = _mean(short_rows, "prefill_ms") + short_decode
    long_e2e = _mean(long_rows, "prefill_ms") + long_decode
    d_new = long_new - short_new
    d_decode = long_decode - short_decode
    d_e2e = long_e2e - short_e2e
    return {
        "new_delta": d_new,
        "decode_ms_delta": d_decode,
        "decode_tok_s_delta": (
            d_new * 1000.0 / d_decode if d_decode > 0 else None),
        "e2e_ms_delta": d_e2e,
        "e2e_tok_s_delta": (
            d_new * 1000.0 / d_e2e if d_e2e > 0 else None),
    }


def _bench_mtp(args, scenario: Scenario, max_new: int):
    import torch

    fe = _frontend_cls(args.frontend)(
        args.checkpoint, quant="nvfp4", max_seq=args.max_seq)
    ids = _input_ids(fe, scenario)
    rows = []

    fe.generate_own_speculative_KN_nvfp4(
        ids, max_new_tokens=max_new, K=args.mtp_k)
    torch.cuda.synchronize()

    for _ in range(args.repeats):
        out = fe.generate_own_speculative_KN_nvfp4(
            ids, max_new_tokens=max_new, K=args.mtp_k)
        torch.cuda.synchronize()
        new = int(out.shape[1] - ids.shape[1])
        decode_ms = float(fe._long_ctx_decode_ms)
        prefill_ms = float(fe._long_ctx_prefill_ms)
        rows.append({
            "new": new,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "decode_tok_s": (new - 1) * 1000.0 / decode_ms,
            "e2e_tok_s": new * 1000.0 / (prefill_ms + decode_ms),
            "attempts": int(fe._spec_attempts),
            "accepts": int(fe._spec_accepts),
            "al": float(fe._spec_accepts) / max(1, int(fe._spec_attempts)),
        })
    prompt_len = int(ids.shape[1])
    del fe, ids
    _cleanup()
    return prompt_len, rows


def _bench_dflash(args, scenario: Scenario, max_new: int):
    import torch

    fe = _frontend_cls(args.frontend)(
        args.checkpoint, quant="nvfp4", max_seq=args.max_seq)
    fe.init_dflash_drafter(args.dflash_checkpoint or None)
    ids = _input_ids(fe, scenario)
    rows = []

    session = fe.make_dflash_session(max_new_tokens=max_new, K=args.dflash_k)
    session.generate(ids)
    torch.cuda.synchronize()

    for _ in range(args.repeats):
        session = fe.make_dflash_session(
            max_new_tokens=max_new, K=args.dflash_k)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev2 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        session.begin(ids)
        ev1.record()
        while not session.done():
            session.step()
        ev2.record()
        torch.cuda.synchronize()
        new = len(session.generated)
        decode_ms = ev1.elapsed_time(ev2)
        prefill_ms = ev0.elapsed_time(ev1)
        rows.append({
            "new": new,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "decode_tok_s": (new - 1) * 1000.0 / decode_ms,
            "e2e_tok_s": new * 1000.0 / (prefill_ms + decode_ms),
            "attempts": int(fe._spec_attempts),
            "accepts": int(fe._spec_accepts),
            "al": float(fe._spec_accepts) / max(1, int(fe._spec_attempts)),
        })
    prompt_len = int(ids.shape[1])
    del fe, ids, session
    _cleanup()
    return prompt_len, rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("FLASHRT_QWEN36_NVFP4_CKPT_DIR", ""))
    p.add_argument("--dflash-checkpoint", default="")
    p.add_argument("--max-seq", type=int, default=32768)
    p.add_argument("--short-new", type=int, default=64)
    p.add_argument("--long-new", type=int, default=256)
    p.add_argument("--mtp-k", type=int, default=6)
    p.add_argument("--dflash-k", type=int, default=15)
    p.add_argument(
        "--frontend", choices=("auto", "thor", "rtx"), default="auto")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--path", choices=("both", "mtp", "dflash"), default="both")
    args = p.parse_args()

    if not args.checkpoint:
        raise SystemExit(
            "set --checkpoint or FLASHRT_QWEN36_NVFP4_CKPT_DIR")

    import torch

    results = []
    for scenario in SCENARIOS:
        row = {
            "scenario": scenario.name,
            "chat_template": scenario.chat,
        }
        if args.path in ("both", "mtp"):
            prompt_len, mtp_short = _bench_mtp(
                args, scenario, args.short_new)
            prompt_len2, mtp_long = _bench_mtp(
                args, scenario, args.long_new)
            if prompt_len != prompt_len2:
                raise RuntimeError("MTP prompt length mismatch")
            row["prompt_len"] = prompt_len
            row["mtp"] = {
                f"max_new_{args.short_new}": _summarize(mtp_short),
                f"max_new_{args.long_new}": _summarize(mtp_long),
                "delta": _delta(mtp_short, mtp_long),
            }
        if args.path in ("both", "dflash"):
            prompt_len, dflash_short = _bench_dflash(
                args, scenario, args.short_new)
            prompt_len2, dflash_long = _bench_dflash(
                args, scenario, args.long_new)
            if prompt_len != prompt_len2:
                raise RuntimeError("DFlash prompt length mismatch")
            row.setdefault("prompt_len", prompt_len)
            row["dflash"] = {
                f"max_new_{args.short_new}": _summarize(dflash_short),
                f"max_new_{args.long_new}": _summarize(dflash_long),
                "delta": _delta(dflash_short, dflash_long),
            }
        results.append(row)
        torch.cuda.empty_cache()

    print(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "frontend": args.frontend,
        "resolved_frontend": _resolve_frontend_name(args.frontend),
        "metric": (
            "scenario steady delta: (long_new - short_new) / "
            "(long_decode_ms - short_decode_ms); prefill reported "
            "separately and e2e delta included"),
        "short_new": args.short_new,
        "long_new": args.long_new,
        "results": results,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Qwen3.6 NVFP4 MTP-vs-DFlash decode-active benchmark.

This benchmark intentionally reports the same metric used in
``docs/qwen36_nvfp4.md``:

    decode tok/s = (new_tokens - 1) * 1000 / decode_ms

Prefill/TTFT is reported separately and never folded into decode tok/s.
MTP and DFlash are measured with separate frontend instances so their
CUDA-graph caches do not pollute each other or cause avoidable OOM.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
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


def _summary(rows):
    return {
        "runs": rows,
        "decode_tok_s_mean": statistics.mean(
            r["decode_tok_s"] for r in rows),
        "prefill_ms_mean": statistics.mean(r["prefill_ms"] for r in rows),
        "decode_ms_mean": statistics.mean(r["decode_ms"] for r in rows),
        "al_mean": statistics.mean(r["al"] for r in rows),
    }


def _resolve_frontend_name(name: str) -> str:
    if name == "auto":
        import torch

        cap = torch.cuda.get_device_capability(0)
        return "thor" if cap == (11, 0) else "rtx"
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


def _bench_mtp(args, case: Case):
    import torch

    fe = _frontend_cls(args.frontend)(
        args.checkpoint, quant="nvfp4", max_seq=args.max_seq)
    ids = _input_ids(fe, case)

    fe.generate_own_speculative_KN_nvfp4(
        ids, max_new_tokens=case.max_new, K=args.mtp_k)
    torch.cuda.synchronize()

    rows = []
    outputs = []
    for _ in range(args.repeats):
        out = fe.generate_own_speculative_KN_nvfp4(
            ids, max_new_tokens=case.max_new, K=args.mtp_k)
        torch.cuda.synchronize()
        new = int(out.shape[1] - ids.shape[1])
        decode_ms = float(fe._long_ctx_decode_ms)
        prefill_ms = float(fe._long_ctx_prefill_ms)
        rows.append({
            "new": new,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "decode_tok_s": (new - 1) * 1000.0 / decode_ms,
            "attempts": int(fe._spec_attempts),
            "accepts": int(fe._spec_accepts),
            "al": float(fe._spec_accepts) / max(1, int(fe._spec_attempts)),
        })
        outputs.append(out.detach().cpu())
    return int(ids.shape[1]), rows, outputs


def _bench_dflash(args, case: Case):
    import torch

    fe = _frontend_cls(args.frontend)(
        args.checkpoint, quant="nvfp4", max_seq=args.max_seq)
    fe.init_dflash_drafter(args.dflash_checkpoint or None)
    ids = _input_ids(fe, case)

    session = fe.make_dflash_session(
        max_new_tokens=case.max_new, K=args.dflash_k)
    session.begin(ids)
    while not session.done():
        session.step()
    torch.cuda.synchronize()

    rows = []
    outputs = []
    for _ in range(args.repeats):
        session = fe.make_dflash_session(
            max_new_tokens=case.max_new, K=args.dflash_k)
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

        out = torch.cat([ids] + session.generated, dim=1)
        new = int(out.shape[1] - ids.shape[1])
        decode_ms = ev1.elapsed_time(ev2)
        prefill_ms = ev0.elapsed_time(ev1)
        rows.append({
            "new": new,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "decode_tok_s": (new - 1) * 1000.0 / decode_ms,
            "attempts": int(fe._spec_attempts),
            "accepts": int(fe._spec_accepts),
            "al": float(fe._spec_accepts) / max(1, int(fe._spec_attempts)),
        })
        outputs.append(out.detach().cpu())
    return int(ids.shape[1]), rows, outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("FLASHRT_QWEN36_NVFP4_CKPT_DIR", ""),
        help="Qwen3.6 NVFP4 checkpoint directory")
    parser.add_argument("--dflash-checkpoint", default="")
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--mtp-k", type=int, default=6)
    parser.add_argument("--dflash-k", type=int, default=15)
    parser.add_argument(
        "--frontend", choices=("auto", "thor", "rtx"), default="auto",
        help="frontend implementation; auto selects Thor on SM110")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if not args.checkpoint:
        raise SystemExit(
            "set --checkpoint or FLASHRT_QWEN36_NVFP4_CKPT_DIR")

    import torch

    results = []
    for case in CASES:
        prompt_len, mtp_rows, mtp_outputs = _bench_mtp(args, case)
        torch.cuda.empty_cache()
        prompt_len2, dflash_rows, dflash_outputs = _bench_dflash(args, case)
        if prompt_len != prompt_len2:
            raise RuntimeError(
                f"prompt length mismatch for {case.name}: "
                f"{prompt_len} != {prompt_len2}")

        parity = [
            bool(torch.equal(a, b))
            for a, b in zip(mtp_outputs, dflash_outputs)
        ]
        first_diff = None
        if not all(parity):
            for a, b in zip(mtp_outputs, dflash_outputs):
                diff = (a != b).nonzero()
                if diff.numel():
                    first_diff = diff[0].tolist()
                    break

        results.append({
            "case": case.name,
            "prompt_len": prompt_len,
            "max_new": case.max_new,
            "chat_template": case.chat,
            "mtp": _summary(mtp_rows),
            "dflash": _summary(dflash_rows),
            "parity_all": all(parity),
            "first_diff": first_diff,
        })

    print(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "frontend": args.frontend,
        "resolved_frontend": _resolve_frontend_name(args.frontend),
        "metric": (
            "decode-active tok/s = (new_tokens - 1) * 1000 / decode_ms; "
            "prefill excluded"),
        "results": results,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

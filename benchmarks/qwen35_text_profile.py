#!/usr/bin/env python3
"""Where a decode step's time goes, against where its bytes go.

A step at batch one reads every weight once and does two flops per value, so
what it should cost is decided by how many bytes it reads and how fast the
part reads them. This prints both: the bytes each piece of the step accounts
for, and the time it actually takes. A piece whose share of the time is larger
than its share of the bytes is where the work is, and the ratio of the two
says whether the answer is a better kernel or fewer bytes.

    python benchmarks/qwen35_text_profile.py --checkpoint PATH
    python benchmarks/qwen35_text_profile.py --checkpoint PATH --int8-head
    python benchmarks/qwen35_text_profile.py --checkpoint PATH --verify

``--verify`` is the check to run after changing a kernel. It reports how far
down the distribution the true continuation of a passage sits: a working model
puts it near the top, and one that is subtly wrong still emits fluent tokens
but pushes it into the tens of thousands. That failure is silent in every
other measurement here.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from flash_rt.frontends.torch import _qwen35_text_decode as decode
from flash_rt.frontends.torch.qwen35_text import TextRuntime

PASSAGE = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "Machine learning models are trained on large datasets to predict the "
    "next token in a sequence."
)


def measured_bandwidth(device: str, mib: int = 256) -> float:
    """Device-to-device copy, which moves each byte twice.

    A kernel that only reads can exceed it, so a fraction above 100% means
    this is the floor of the read bandwidth rather than that something is
    wrong.
    """
    source = torch.empty(mib * 2 ** 20 // 2, dtype=torch.bfloat16,
                         device=device)
    source.normal_()
    destination = torch.empty_like(source)
    for _ in range(3):
        destination.copy_(source)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    rounds = 10
    for _ in range(rounds):
        destination.copy_(source)
    torch.cuda.synchronize(device)
    return rounds * 2 * source.numel() * 2 / (time.perf_counter() - started)


def packed_bytes(rows: int, columns: int, group: int) -> int:
    """A four-bit weight: its body, plus one scale per group along K."""
    return rows * columns // 2 + rows * (columns // group) * 2


def byte_budget(runtime: TextRuntime) -> dict[str, int]:
    """What one token reads, by piece."""
    dims = runtime.dims
    group = runtime.work.group_size
    linear = len(dims.linear_attention_layers)
    full = len(dims.full_attention_layers)
    head_row = 1 if runtime.weights.top.get("lm_head_scale") else 2

    return {
        "mlp block": (packed_bytes(2 * dims.intermediate, dims.hidden, group)
                      + packed_bytes(dims.hidden, dims.intermediate, group)
                      ) * dims.num_layers,
        "gdn block": (packed_bytes(dims.lin_qkv_width + dims.lin_value_width,
                                   dims.hidden, group)
                      + packed_bytes(dims.hidden, dims.lin_value_width, group)
                      + 2 * dims.lin_value_heads * dims.hidden * 2) * linear,
        "attn block": (packed_bytes(dims.q_width + 2 * dims.kv_width,
                                    dims.hidden, group)
                       + packed_bytes(dims.hidden, dims.attn_width, group)
                       ) * full,
        "residual+norm": 2 * dims.num_layers * dims.hidden * 2 * 3,
        "lm_head": dims.vocab_size * (dims.hidden * head_row + 2),
        "argmax": dims.vocab_size * 2,
    }


def time_pieces(runtime: TextRuntime, rounds: int) -> dict[str, float]:
    """Milliseconds each piece of a step accounts for, measured in place."""
    weights, work, fvk = runtime.weights, runtime.work, runtime.fvk
    dims = runtime.dims
    stream = torch.cuda.current_stream(runtime.device).cuda_stream
    runtime.reset()
    runtime.read_prompt([1, 2, 3, 4])
    torch.cuda.synchronize(runtime.device)

    def bench(call, warm: int = 20) -> float:
        for _ in range(warm):
            call()
        torch.cuda.synchronize(runtime.device)
        started = time.perf_counter()
        for _ in range(rounds):
            call()
        torch.cuda.synchronize(runtime.device)
        return (time.perf_counter() - started) / rounds * 1e3

    gdn = dims.linear_attention_layers[0]
    full = dims.full_attention_layers[0]
    x, out = work.normed.address, work.mixed.address
    linear = len(dims.linear_attention_layers)
    attention = len(dims.full_attention_layers)

    return {
        "gdn block": linear * bench(lambda: decode.linear_attention_block(
            weights.layers[gdn], work, fvk, work.state_slot[gdn], x, out, 1,
            stream)),
        "attn block": attention * bench(lambda: decode.full_attention_block(
            weights.layers[full], work, fvk, work.state_slot[full], x, out, 1,
            stream)),
        "mlp block": dims.num_layers * bench(lambda: decode.mlp_block(
            weights.layers[gdn], work, fvk, x, out, 1, stream)),
        "residual+norm": 2 * dims.num_layers * bench(
            lambda: fvk.residual_add_rms_norm(
                work.residual.address, work.mixed.address,
                weights.layers[0]["post_norm"], work.normed.address, 1,
                dims.hidden, dims.rms_norm_eps, stream)),
        "lm_head": bench(lambda: decode.project_to_vocabulary(
            weights, work, fvk, stream)),
        "argmax": bench(lambda: fvk.qwen36_argmax_bf16(
            work.logits.address, work.token.address, 1, dims.vocab_size,
            stream)),
    }


def verify(runtime: TextRuntime, token_ids: list[int]) -> dict[str, float]:
    """How far down the distribution the true continuation sits.

    A model that is subtly wrong -- a normalization off, a decode reading the
    wrong nibble -- still emits fluent tokens, and every timing here still
    looks right. This is the measurement that does not.
    """
    weights, work, fvk = runtime.weights, runtime.work, runtime.fvk
    stream = torch.cuda.current_stream(runtime.device).cuda_stream
    runtime.reset()
    ranks = []
    for step, token in enumerate(token_ids[:-1]):
        work.token.tensor[0] = token
        work.seek(step, 1)
        decode.forward(weights, work, fvk, 1, stream)
        decode.project_to_vocabulary(weights, work, fvk, stream)
        torch.cuda.synchronize(runtime.device)
        row = work.logits.tensor[0].float()
        ranks.append(int((row > row[token_ids[step + 1]]).sum()))
    ordered = sorted(ranks)
    return {
        "positions": float(len(ranks)),
        "top1": float(sum(rank == 0 for rank in ranks)) / len(ranks),
        "median_rank": float(ordered[len(ordered) // 2]),
        "mean_rank": sum(ranks) / len(ranks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--int8-head", action="store_true")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--token-ids", type=int, nargs="+",
                        help="a passage to verify against, already tokenized")
    parser.add_argument("--output")
    args = parser.parse_args()

    runtime = TextRuntime.from_checkpoint(
        args.checkpoint, device=args.device, max_seq=512, max_chunk=8,
        quantize_tied_table=args.int8_head)

    bandwidth = measured_bandwidth(args.device)
    print(f"{torch.cuda.get_device_name(args.device)}: "
          f"device-to-device copy at {bandwidth / 1e9:.1f} GB/s")
    print(f"tied table: {'int8' if args.int8_head else 'bfloat16'}\n")

    budget = byte_budget(runtime)
    times = time_pieces(runtime, args.rounds)
    total_bytes = sum(budget.values())
    total_ms = sum(times.values())

    print(f"{'piece':<16}{'MiB':>9}{'% bytes':>9}{'ms':>9}{'% time':>9}"
          f"{'GB/s':>9}{'% BW':>7}")
    for piece in sorted(budget, key=lambda name: -times[name]):
        nbytes, ms = budget[piece], times[piece]
        rate = nbytes / (ms / 1e3)
        print(f"{piece:<16}{nbytes / 2**20:>9.1f}"
              f"{100 * nbytes / total_bytes:>8.1f}%{ms:>9.3f}"
              f"{100 * ms / total_ms:>8.1f}%{rate / 1e9:>9.0f}"
              f"{100 * rate / bandwidth:>6.0f}%")
    print(f"{'step':<16}{total_bytes / 2**20:>9.1f}{'':>9}{total_ms:>9.3f}"
          f"{'':>9}{total_bytes / (total_ms / 1e3) / 1e9:>9.0f}"
          f"{100 * total_bytes / (total_ms / 1e3) / bandwidth:>6.0f}%")
    print(f"\nsum of the pieces: {total_ms:.3f} ms = "
          f"{1e3 / total_ms:.1f} tok/s. A whole step is slower than this: "
          "the pieces are timed back to back and overlap, and a step is "
          "sequential.")

    report = {"device": torch.cuda.get_device_name(args.device),
              "bandwidth_gb_s": bandwidth / 1e9,
              "int8_head": args.int8_head,
              "bytes": budget, "milliseconds": times}

    if args.verify:
        token_ids = args.token_ids
        if not token_ids:
            from transformers import AutoTokenizer

            token_ids = AutoTokenizer.from_pretrained(
                args.checkpoint)(PASSAGE)["input_ids"]
        quality = verify(runtime, token_ids)
        print(f"\nover {int(quality['positions'])} positions of ordinary "
              f"text: the true next token is the argmax "
              f"{100 * quality['top1']:.0f}% of the time, median rank "
              f"{quality['median_rank']:.0f}, mean rank "
              f"{quality['mean_rank']:.0f}.")
        print("A median rank in the thousands means the model is wrong, "
              "however fluent it sounds.")
        report["quality"] = quality

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()

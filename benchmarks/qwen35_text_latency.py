#!/usr/bin/env python3
"""What a turn costs: time to the first token, and the tokens after it.

Both halves matter for different reasons and are reported separately. A short
prompt answered often is dominated by the prompt pass and by whatever fixed
cost each token carries; a long answer is dominated by the weight read. The
per-token figure is a distribution rather than a mean, because a loop that
usually meets its deadline and occasionally does not is a loop that misses.

    python benchmarks/qwen35_text_latency.py --checkpoint PATH
    python benchmarks/qwen35_text_latency.py --checkpoint PATH --no-graph
    python benchmarks/qwen35_text_latency.py --checkpoint PATH \
        --prompt-tokens 16 64 256 --new-tokens 64
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from flash_rt.frontends.torch.qwen35_text import StepTimings, TextRuntime


def synthetic_prompt(length: int, vocab: int) -> list[int]:
    """A prompt of a given length without needing a tokenizer.

    The tokens are arbitrary: what this measures is the shape of the work,
    which depends on how many positions there are and not on which ones.
    """
    return [(index * 7919) % (vocab - 1) + 1 for index in range(length)]


def run(runtime: TextRuntime, prompt: list[int], new_tokens: int) -> dict:
    timings = StepTimings()
    runtime.generate(prompt, max_new_tokens=new_tokens, timings=timings)
    return timings.summary()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-tokens", type=int, nargs="+",
                        default=[16, 64, 256])
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--max-chunk", type=int, default=64)
    parser.add_argument("--no-graph", action="store_true",
                        help="do not capture the decode step")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()

    started = time.perf_counter()
    runtime = TextRuntime.from_checkpoint(
        args.checkpoint, device=args.device, max_seq=args.max_seq,
        max_chunk=args.max_chunk)
    load_seconds = time.perf_counter() - started

    footprint = runtime.footprint()
    print(f"{torch.cuda.get_device_name(args.device)}")
    print(f"loaded in {load_seconds:.2f}s: "
          f"{footprint['weights_gib']:.3f} GiB of weights, "
          f"{footprint['state_gib']:.3f} GiB of state, "
          f"{footprint['reserved_gib']:.3f} GiB reserved\n")

    vocab = runtime.dims.vocab_size
    if not args.no_graph:
        # Capture from a state that exists, then put it back.
        runtime.reset()
        runtime.read_prompt(synthetic_prompt(8, vocab))
        runtime.capture()
        print("decode step captured as a graph\n")

    print(f"{'prompt':>8}{'TTFT ms':>10}{'p50 ms':>9}{'p99 ms':>9}"
          f"{'max ms':>9}{'tok/s':>9}")
    results = []
    for length in args.prompt_tokens:
        if length >= args.max_seq:
            print(f"{length:>8}   skipped: longer than --max-seq")
            continue
        prompt = synthetic_prompt(length, vocab)
        best = None
        for _ in range(args.rounds):
            summary = run(runtime, prompt, args.new_tokens)
            if best is None or summary["p50_ms"] < best["p50_ms"]:
                best = summary
        results.append(best)
        print(f"{length:>8}{best['prefill_ms']:>10.2f}{best['p50_ms']:>9.3f}"
              f"{best['p99_ms']:>9.3f}{best['max_ms']:>9.3f}"
              f"{best['tokens_per_second']:>9.1f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"device": torch.cuda.get_device_name(args.device),
                       "graph": not args.no_graph,
                       "footprint": footprint,
                       "rows": results}, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()

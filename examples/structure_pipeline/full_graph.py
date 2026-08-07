"""The explicit pipeline at full speed: whole-graph CUDA capture.

``groot_n17.py`` builds the assembly and times it under ``torch.compile``.
That form still runs Python between compiled regions — every guarded
seam's admission check, every host glue call — and on a graph with two
hundred seams that Python is a real cost. The deployed form removes it:
compile once, then capture the whole hot path as a single CUDA graph and
replay it. Capture records kernels, not Python, so the guards and the
glue are paid once at capture time and never again.

Capture needs fixed shapes. The Qwen3-VL backbone computes positions,
rope tables and token routing from the request at every call; for a
fixed observation shape those are constants, so this file precomputes
them once and pins the handful of host functions that would otherwise
recompute them (and, in two places, synchronise — which capture
forbids). All constants depend on shapes and token placement, never on
image or state values; the parity check at the end is against the
untouched eager host.

Usage: same arguments as ``groot_n17.py``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import types
from pathlib import Path

import torch

from flash_rt.structures import capture as capture_stage

from groot_n17 import Assembly, build, clone_tree, load_policy


def pin_action_noise():
    """Pin the flow-matching noise to one tensor across every call.

    Seeding is not enough once a graph is captured: replays advance the
    device RNG, so the fiftieth replay denoises a different draw than
    the reference did and the outputs are not comparable. Returning one
    saved tensor for the noise shape bakes it into the captured graph
    as a constant input — every replay, and the eager reference, then
    integrate from the same starting noise. Returns an undo callable.
    """
    original = torch.randn
    box = {}

    def fixed(*size, **kwargs):
        try:
            shape = tuple(kwargs.get("size", ())) or (
                tuple(size[0])
                if len(size) == 1 and not isinstance(size[0], int)
                else tuple(size))
        except TypeError:
            shape = None
        if shape is not None and box.get("shape") == shape:
            return box["value"]
        value = original(*size, **kwargs)
        if value.is_cuda and "value" not in box:
            box["shape"] = tuple(value.shape)
            box["value"] = value.detach().clone()
        return value

    torch.randn = fixed
    return lambda: setattr(torch, "randn", original)






def replay_ms(stages, *, iters=50, rounds=9):
    timings = {name: [] for name in stages}
    names = list(stages)
    for r in range(rounds):
        for name in names[r % len(names):] + names[:r % len(names)]:
            stage = stages[name]
            for _ in range(5):
                stage.replay(sync=False)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            with torch.cuda.stream(stage.stream):
                start.record()
            for _ in range(iters):
                stage.replay(sync=False)
            with torch.cuda.stream(stage.stream):
                end.record()
            torch.cuda.synchronize()
            timings[name].append(start.elapsed_time(end) / iters)
    return {name: statistics.median(v) for name, v in timings.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("explicit", "auto"),
                        default="explicit")
    parser.add_argument("--scheme", default=None,
                        help="named precision scheme for the auto arm "
                             "(e.g. nvfp4_balance); default negotiates "
                             "FP8 chains")
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
    from flash_rt.structures.impls import unavailable_report
    from flash_rt.structures.impls.cadence_static.cross_attention import (
        wire_refresh_to_producer)

    captured = {}
    original_get_action = model.get_action

    def spy(inputs, options=None):
        captured["inputs"] = clone_tree(inputs)
        return original_get_action(inputs, options)

    model.get_action = spy
    with torch.inference_mode():
        policy.get_action(fixture)
    model.get_action = original_get_action

    backbone_inputs, action_inputs = model.prepare_input(
        dict(captured["inputs"]))
    backbone_inputs = clone_tree(backbone_inputs)
    action_inputs = clone_tree(action_inputs)

    unpin = pin_action_noise()

    def hot():
        out = model.backbone(backbone_inputs)
        return model.action_head.get_action(
            out, action_inputs)["action_pred"]

    try:
        with torch.inference_mode():
            reference = hot().detach().float().cpu()

        # the host at full speed: the registered family lowering pins
        # the shape glue, the capture door does the rest
        stock = capture_stage(
            torch.compile(hot, mode="max-autotune-no-cudagraphs",
                          fullgraph=False),
            model=model, warmup=8, gate_cos=0, min_speedup=0)

        def run_once():
            with torch.inference_mode():
                hot()

        if args.arm == "explicit":
            asm, extras = build(model, run_once)
            handle = swap.attach(model, asm.swaps, consume=False,
                                 observe=extras["observed"],
                                 revert=extras["revert"],
                                 on_guard_fail="raise")
            statics = extras["cadence_statics"]
            seats = {"seats_bound": dict(asm.families),
                     "swaps": len(asm.swaps),
                     "refused": len(asm.refused)}
        else:
            from flash_rt import structures
            from flash_rt.structures.impls.cadence_static.cross_attention \
                import (bind_cross_attention_kv, capture_cross_attention_kv,
                        discover_cross_attention_kv)
            sites = discover_cross_attention_kv(model)
            caps = capture_cross_attention_kv(sites, run_once)
            scheme_kw = ({"scheme": args.scheme}
                         if args.scheme else {})
            plan = structures.auto_swaps(model, run_once, verbose=True,
                                         **scheme_kw)
            statics = []
            if sites:
                cadence_swaps, statics = bind_cross_attention_kv(
                    sites, caps, replacements=plan.swaps)
                plan.swaps.update(cadence_swaps)
            handle = swap.attach(model, plan.swaps, consume=False,
                                 observe=plan.observed,
                                 revert=plan.revert)
            seats = {"swaps": len(plan.swaps),
                     "observed": len(plan.observed),
                     "refused": len(plan.notes.get("refused", [])),
                     "format_race": plan.notes.get("format_race")}
        if statics:
            # the refresh rides the producer's forward: the captured
            # graph writes the banks from this call's own features
            wire_refresh_to_producer(model, statics, run_once)

        torch._dynamo.reset()
        treated = capture_stage(
            torch.compile(hot, mode="max-autotune-no-cudagraphs",
                          fullgraph=False),
            model=model, warmup=8, gate_cos=0, min_speedup=0)

        medians = replay_ms({"stock_graph": stock,
                             "structures_graph": treated})
        treated.replay()
        parity = float(torch.nn.functional.cosine_similarity(
            treated.output.detach().float().cpu().flatten(),
            reference.flatten(), dim=0))
        report = {
            "arm": args.arm,
            "scheme": args.scheme,
            "device": torch.cuda.get_device_name(),
            **seats,
            "kernel_unavailable": unavailable_report(),
            "graph_lowering": treated.certification.get("graph_lowering"),
            "parity_cosine": parity,
            "stock_graph_ms": medians["stock_graph"],
            "structures_graph_ms": medians["structures_graph"],
            "speedup_vs_stock_graph": (medians["stock_graph"]
                                       / medians["structures_graph"]),
            "ledger": handle.summary(),
        }
        from flash_rt.structures.storage import WeightStore
        report["weights"] = handle.consume(
            WeightStore(checkpoint=str(args.checkpoint)))
        print(json.dumps(report, indent=2, default=str))
        if args.report:
            args.report.write_text(json.dumps(report, indent=2,
                                              default=str))
        handle.detach()
        treated.restore_host()
        stock.restore_host()
    finally:
        unpin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

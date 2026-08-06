"""LeRobot GR00T N1.7 at the captured form — explicit and automatic arms.

Same protocol as the official-host measurement, through the same
mechanism door: the registered Qwen3-VL family lowering recognizes this
host too (including its legacy position-id glue, pinned via a
capability probe), so this file is loading, arms and timing — no
host-specific capture engineering.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from transformers.feature_extraction_utils import BatchFeature

from flash_rt.structures import capture as capture_stage

from full_graph import pin_action_noise, replay_ms
from groot_n17 import build


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("explicit", "auto"),
                        required=True)
    parser.add_argument("--lerobot-src", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.lerobot_src))

    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    from flash_rt import structures
    from flash_rt.structures import swap
    from flash_rt.structures.impls import unavailable_report
    from flash_rt.structures.impls.cadence_static.cross_attention import (
        refresh_cross_attention_kv)

    model = GR00TN17.from_pretrained(args.checkpoint).to(
        device="cuda", dtype=torch.bfloat16).eval()
    payload = torch.load(args.inputs, map_location="cpu",
                         weights_only=False)
    vl_input = BatchFeature(data={
        k: (v.cuda() if torch.is_tensor(v) else v)
        for k, v in payload["backbone_inputs"].items()})
    action_input = BatchFeature(data={
        k: (v.cuda() if torch.is_tensor(v) else v)
        for k, v in payload["action_inputs"].items()})

    unpin = pin_action_noise()

    def hot():
        features = model.backbone(vl_input)
        return model.action_head.get_action(
            features, action_input)["action_pred"]

    try:
        with torch.inference_mode():
            reference = hot().detach().float().cpu()

        stock = capture_stage(
            torch.compile(hot, mode="max-autotune-no-cudagraphs",
                          fullgraph=False),
            model=model, warmup=8, gate_cos=0, min_speedup=0)

        def run_once():
            with torch.inference_mode():
                hot()

        if args.arm == "explicit":
            asm, extras = build(model, run_once)
            handle = swap.attach(model, asm.swaps,
                                 observe=extras["observed"],
                                 on_guard_fail="raise")
            statics = extras["cadence_statics"]
            seats = {"seats_bound": dict(asm.families),
                     "swaps": len(asm.swaps),
                     "refused": len(asm.refused)}
        else:
            plan = structures.auto_swaps(model, run_once, verbose=True)
            handle = swap.attach(model, plan.swaps,
                                 observe=plan.observed,
                                 revert=plan.revert)
            statics = []
            seats = {"swaps": len(plan.swaps),
                     "observed": len(plan.observed),
                     "refused": len(plan.notes.get("refused", []))}

        if statics:
            with torch.inference_mode():
                features = model.backbone(vl_input)
                processed = model.action_head.process_backbone_output(
                    features)
                refresh_cross_attention_kv(
                    statics, processed["backbone_features"])

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
            "host": "lerobot GR00TN17",
            "arm": args.arm,
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

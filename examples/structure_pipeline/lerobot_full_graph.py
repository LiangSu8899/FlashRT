"""LeRobot GR00T N1.7 at the captured form — explicit and automatic arms.

Symmetry with the official-host measurement: same checkpoint, same
prepared inputs, same fixed-shape lowering, same capture-and-replay
protocol. The LeRobot backbone adds two per-call glue helpers on top of
the shared Qwen3-VL forward; both early-exit when their result is
already present in the input, so precomputing the two tensors once
removes the capture-illegal work without touching host code.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from transformers.feature_extraction_utils import BatchFeature

from full_graph import (capture, lower_backbone_to_fixed_shapes,
                        pin_action_noise, replay_ms)
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
    vl_data = {k: (v.cuda() if torch.is_tensor(v) else v)
               for k, v in payload["backbone_inputs"].items()}
    action_input = BatchFeature(data={
        k: (v.cuda() if torch.is_tensor(v) else v)
        for k, v in payload["action_inputs"].items()})

    # Precompute the two per-call glue tensors once. Both helpers
    # early-exit when their key is already present, so the per-call
    # path becomes pure tensor compute with fixed shapes.
    glue = {k: vl_data[k] for k in
            ("input_ids", "attention_mask", "pixel_values",
             "image_grid_thw")}
    model.backbone._ensure_mm_token_type_ids(glue)
    model.backbone._ensure_legacy_qwen3_position_ids(glue)
    if "mm_token_type_ids" in glue:
        vl_data["mm_token_type_ids"] = glue["mm_token_type_ids"]
    vl_input = BatchFeature(data=vl_data)

    # position_ids is not on the backbone's input whitelist, so passing
    # it through the request cannot reach the helper's early exit; pin
    # the helper itself to the precomputed tensor. Same class of act as
    # the rest of the lowering: a shape-derived constant of one fixed
    # request, never a value-dependent quantity.
    fixed_position_ids = glue.get("position_ids")
    if fixed_position_ids is not None:
        def pinned_position_ids(model_input,
                                _pids=fixed_position_ids):
            model_input["position_ids"] = _pids
        model.backbone._ensure_legacy_qwen3_position_ids =             pinned_position_ids

    unpin = pin_action_noise()

    def hot():
        features = model.backbone(vl_input)
        return model.action_head.get_action(
            features, action_input)["action_pred"]

    with torch.inference_mode():
        reference = hot().detach().float().cpu()

    undo = lower_backbone_to_fixed_shapes(model, vl_data)
    try:
        with torch.inference_mode():
            lowered = hot().detach().float().cpu()
        lowering_cos = float(torch.nn.functional.cosine_similarity(
            lowered.flatten(), reference.flatten(), dim=0))

        pool = torch.cuda.graph_pool_handle()
        stock_graph, _, _ = capture(hot, pool)

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

        treated_graph, treated_out, _ = capture(hot, pool)
        medians = replay_ms({"stock_graph": stock_graph,
                             "structures_graph": treated_graph})
        treated_graph.replay()
        torch.cuda.synchronize()
        parity = float(torch.nn.functional.cosine_similarity(
            treated_out.detach().float().cpu().flatten(),
            reference.flatten(), dim=0))

        report = {
            "host": "lerobot GR00TN17",
            "arm": args.arm,
            "device": torch.cuda.get_device_name(),
            **seats,
            "kernel_unavailable": unavailable_report(),
            "lowering_cosine": lowering_cos,
            "parity_cosine": parity,
            "stock_graph_ms": medians["stock_graph"],
            "structures_graph_ms": medians["structures_graph"],
            "speedup_vs_stock_graph": (medians["stock_graph"]
                                       / medians["structures_graph"]),
            "ledger": handle.summary(),
        }
        print(json.dumps(report, indent=2, default=str))
        if args.report:
            args.report.write_text(
                json.dumps(report, indent=2, default=str))
        handle.detach()
    finally:
        undo()
        unpin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

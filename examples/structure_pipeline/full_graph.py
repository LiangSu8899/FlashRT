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


def lower_backbone_to_fixed_shapes(model, backbone_inputs):
    """Pin the shape-derived constants of one fixed request.

    Returns an ``undo`` callable restoring every patched attribute.
    """
    qwen = model.backbone.model
    base = qwen.model
    visual = base.visual
    input_ids = backbone_inputs["input_ids"]
    attention_mask = backbone_inputs["attention_mask"]
    grid_thw = backbone_inputs["image_grid_thw"]
    if not bool(torch.all(attention_mask == 1).item()):
        raise RuntimeError("fixed-shape capture requires an all-one mask")

    try:
        position_ids, rope_deltas = base.get_rope_index(
            input_ids, grid_thw, None, attention_mask=attention_mask)
        have_rope_index = True
    except (TypeError, IndexError):
        # newer transformers changed this helper's signature; a host
        # that injects position_ids into the request never calls it,
        # so there is nothing to pin — leave it untouched
        have_rope_index = False
    pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
    rotary = visual.rot_pos_emb(grid_thw)
    rotary_pair = torch.cat((rotary, rotary), dim=-1)
    position_embeddings = (rotary_pair.cos(), rotary_pair.sin())
    fixed_cu_seqlens = torch.nn.functional.pad(
        torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32), (1, 0), value=0)
    vision_lengths = tuple(int(v) for v in
                           (fixed_cu_seqlens[1:] - fixed_cu_seqlens[:-1]).cpu().tolist())
    split_sizes = tuple(int(v) for v in (
        grid_thw.prod(-1) // visual.spatial_merge_size ** 2).cpu().tolist())
    image_mask = (input_ids == qwen.config.image_token_id).unsqueeze(-1)
    video_mask = (input_ids == qwen.config.video_token_id).unsqueeze(-1)
    visual_indices = (input_ids.reshape(-1)
                      == qwen.config.image_token_id).nonzero().flatten()

    # Two vision-contract generations exist in the host's transformers:
    # the older one returns ``(merged, deepstack_list)`` from the visual
    # tower, the newer one wraps the same three tensors in an output
    # class and splits the pooled embeddings inside get_image_features
    # (with a .tolist() the capture cannot record). Probe by the class,
    # not the version string.
    import importlib
    modeling = importlib.import_module(type(visual).__module__)
    output_cls = getattr(modeling, "BaseModelOutputWithDeepstackFeatures",
                         None)

    saved = {
        "visual_forward": visual.forward,
        "attn_forwards": [b.attn.forward for b in visual.blocks],
        "get_image_features": base.get_image_features,
        "get_placeholder_mask": base.get_placeholder_mask,
        "get_rope_index": base.get_rope_index,
        "deepstack": base.language_model._deepstack_process,
        "lm_head": qwen.lm_head,
        "use_cache": qwen.config.text_config.use_cache,
    }
    from transformers import masking_utils
    saved["ignore_mask"] = masking_utils._ignore_causal_mask_sdpa

    def fixed_visual_forward(self, hidden_states, grid_thw=None, **kwargs):
        del grid_thw
        kwargs.pop("return_dict", None)
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + pos_embeds
        seq, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq, -1)
        deepstack_features = []
        for index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens=fixed_cu_seqlens,
                                  position_embeddings=position_embeddings,
                                  **kwargs)
            if index in self.deepstack_visual_indexes:
                merger = self.deepstack_visual_indexes.index(index)
                deepstack_features.append(
                    self.deepstack_merger_list[merger](hidden_states))
        merged = self.merger(hidden_states)
        if output_cls is not None:
            return output_cls(last_hidden_state=hidden_states,
                              pooler_output=merged,
                              deepstack_features=deepstack_features)
        return merged, deepstack_features

    def fixed_vision_attention(self, hidden_states, cu_seqlens=None,
                               rotary_pos_emb=None,
                               position_embeddings=None, **kwargs):
        del rotary_pos_emb
        from transformers.models.qwen3_vl import modeling_qwen3_vl as m

        seq = hidden_states.shape[0]
        query, key, value = (self.qkv(hidden_states)
                             .reshape(seq, 3, self.num_heads, -1)
                             .permute(1, 0, 2, 3).unbind(0))
        cos_t, sin_t = position_embeddings
        query, key = m.apply_rotary_pos_emb_vision(query, key, cos_t, sin_t)
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        interface = m.eager_attention_forward
        if self.config._attn_implementation != "eager":
            interface = m.ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation]
        chunks = (torch.split(t, vision_lengths, dim=2)
                  for t in (query, key, value))
        outputs = [interface(self, q, k, v, attention_mask=None,
                             scaling=self.scaling, dropout=0.0,
                             is_causal=False, **kwargs)[0]
                   for q, k, v in zip(*chunks)]
        return self.proj(torch.cat(outputs, dim=1)
                         .reshape(seq, -1).contiguous())

    def fixed_get_image_features(self, pixel_values, image_grid_thw=None,
                                 **kwargs):
        del kwargs
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw)
        if output_cls is not None:
            vision_output.pooler_output = torch.split(
                vision_output.pooler_output, split_sizes)
            return vision_output
        embeds, deepstack = vision_output
        return torch.split(embeds, split_sizes), deepstack

    def fixed_get_placeholder_mask(self, input_ids_arg, inputs_embeds,
                                   image_features=None, video_features=None):
        del self, input_ids_arg, image_features, video_features
        return (image_mask.expand_as(inputs_embeds),
                video_mask.expand_as(inputs_embeds))

    def fixed_deepstack(self, hidden_states, visual_pos_masks,
                        visual_embeds):
        del self, visual_pos_masks
        b, s, c = hidden_states.shape
        flat = hidden_states.reshape(b * s, c)
        updated = flat.index_select(0, visual_indices) + visual_embeds
        return flat.index_copy(0, visual_indices, updated).reshape(b, s, c)

    class EmptyLMHead(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states.new_empty(*hidden_states.shape[:-1], 0)

    visual.forward = types.MethodType(fixed_visual_forward, visual)
    for block in visual.blocks:
        block.attn.forward = types.MethodType(fixed_vision_attention,
                                              block.attn)
    base.get_image_features = types.MethodType(fixed_get_image_features,
                                               base)
    base.get_placeholder_mask = types.MethodType(fixed_get_placeholder_mask,
                                                 base)
    if have_rope_index:
        base.get_rope_index = types.MethodType(
            lambda self, *a, **k: (position_ids, rope_deltas), base)
    base.language_model._deepstack_process = types.MethodType(
        fixed_deepstack, base.language_model)
    qwen.lm_head = EmptyLMHead()
    qwen.config.text_config.use_cache = False
    masking_utils._ignore_causal_mask_sdpa = lambda *a, **k: True

    def undo():
        visual.forward = saved["visual_forward"]
        for block, fwd in zip(visual.blocks, saved["attn_forwards"]):
            block.attn.forward = fwd
        base.get_image_features = saved["get_image_features"]
        base.get_placeholder_mask = saved["get_placeholder_mask"]
        if have_rope_index:
            base.get_rope_index = saved["get_rope_index"]
        base.language_model._deepstack_process = saved["deepstack"]
        qwen.lm_head = saved["lm_head"]
        qwen.config.text_config.use_cache = saved["use_cache"]
        masking_utils._ignore_causal_mask_sdpa = saved["ignore_mask"]

    return undo


def capture(fn, pool):
    """Compile, warm every lazy autotune, then record one CUDA graph."""
    torch._dynamo.reset()
    compiled = torch.compile(fn, mode="max-autotune-no-cudagraphs",
                             fullgraph=False)
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        for _ in range(8):
            output = compiled()
            torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph, stream=stream,
                                                  pool=pool):
        output = compiled()
    torch.cuda.synchronize()
    return graph, output, compiled


def replay_ms(graphs, *, iters=50, rounds=9):
    timings = {name: [] for name in graphs}
    names = list(graphs)
    for r in range(rounds):
        for name in names[r % len(names):] + names[:r % len(names)]:
            graph = graphs[name]
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(iters):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            timings[name].append(start.elapsed_time(end) / iters)
    return {name: statistics.median(v) for name, v in timings.items()}


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
    from flash_rt.structures.impls import unavailable_report
    from flash_rt.structures.impls.cadence_static.cross_attention import (
        refresh_cross_attention_kv)

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

    with torch.inference_mode():
        reference = hot().detach().float().cpu()

    undo = lower_backbone_to_fixed_shapes(model, backbone_inputs)
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

        asm, extras = build(model, run_once)
        handle = swap.attach(model, asm.swaps,
                             observe=extras["observed"],
                             on_guard_fail="raise")

        # cross-attention K/V follow the observation, not the denoise
        # step: refresh the static buffers once here, outside anything
        # captured. A deployment does the same once per observation;
        # the replayed graph reads whatever the buffers hold.
        if extras["cadence_statics"]:
            with torch.inference_mode():
                out = model.backbone(backbone_inputs)
                processed = model.action_head.process_backbone_output(out)
                refresh_cross_attention_kv(
                    extras["cadence_statics"],
                    processed["backbone_features"])

        treated_graph, treated_out, _ = capture(hot, pool)
        medians = replay_ms({"stock_graph": stock_graph,
                             "structures_graph": treated_graph})
        treated_graph.replay()
        torch.cuda.synchronize()
        parity = float(torch.nn.functional.cosine_similarity(
            treated_out.detach().float().cpu().flatten(),
            reference.flatten(), dim=0))
        ledger = handle.summary()

        report = {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "seats_bound": dict(asm.families),
            "swaps": len(asm.swaps),
            "refused": len(asm.refused),
            "kernel_unavailable": unavailable_report(),
            "lowering_cosine": lowering_cos,
            "parity_cosine": parity,
            "stock_graph_ms": medians["stock_graph"],
            "structures_graph_ms": medians["structures_graph"],
            "speedup_vs_stock_graph": (medians["stock_graph"]
                                       / medians["structures_graph"]),
            "ledger": ledger,
        }
        print(json.dumps(report, indent=2, default=str))
        if args.report:
            args.report.write_text(json.dumps(report, indent=2,
                                              default=str))
        handle.detach()
    finally:
        undo()
        unpin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 1: capture per-stage activations from the official PyTorch reference
on a deterministic input.

Output: ``tests/fixtures/gr00t_n17_ref_e2e_<tag>_<views>v_traj<id>_step<s>.pt``.

The fixture contains:
  * ``meta``           — embodiment_tag, num_views, ckpt path/sha, dtypes
  * ``inputs``         — raw obs dict + post-processor tensors (pixel_values,
                         input_ids, attention_mask, image_grid_thw, image_mask)
  * ``activations``    — per-block hidden states for every stage:
        vit_block_{i}                     (i in 0..23)
        deepstack_merged_{j}              (j in 0..2)         (post merger MLP)
        llm_input_embeds                  (text + visual tokens)
        llm_layer_{i}                     (i in 0..15)
        backbone_features                 (final hidden_states[-1])
        vlln_out                          (post LayerNorm 2048)
        vl_self_attn_block_{i}            (i in 0..3)
        sa_input                          (state+action features pre-DiT)
        dit_block_{i}                     (i in 0..31, last denoise step only)
        action_pred_velocity_step_{s}     (s in 0..3)
        actions_unnormalized              (final post-decode_action)
  * ``noise_seed``     — int, deterministic init for the diffusion noise

Usage:
  python tools/groot_n17/gen_reference.py --tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --traj 1 --step 0
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
                   help="EmbodimentTag enum name (case-insensitive)")
    p.add_argument("--traj", type=int, default=1, help="trajectory id within the dataset")
    p.add_argument("--step", type=int, default=0, help="step index within the trajectory")
    p.add_argument("--seed", type=int, default=0, help="deterministic noise seed")
    p.add_argument("--ckpt", default=None,
                   help="local N1.7 ckpt dir (auto-detected from HF cache if None)")
    p.add_argument("--dataset", default="/gr00t/Isaac-GR00T/demo_data/droid_sample",
                   help="LeRobot demo dataset path (must match --tag)")
    p.add_argument("--out-dir", default="/work/tests/fixtures",
                   help="where to write the .pt fixture")
    return p.parse_args()


def autodetect_ckpt() -> str:
    root = "/root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/"
    sha = sorted(os.listdir(root))[0]
    return root + sha


def register_block_hooks(model, store: dict[str, torch.Tensor]) -> list:
    """Hook every named block of interest. Returns hook handles for later cleanup."""

    def make_hook(key: str):
        def _hook(_module, _args, output):
            t = output[0] if isinstance(output, tuple) else output
            if isinstance(t, torch.Tensor):
                store[key] = t.detach().to(torch.float32).cpu().clone()
        return _hook

    handles = []

    # Backbone: Qwen3-VL ViT (24 blocks) + LLM (16 layers truncated)
    visual = model.backbone.model.model.visual
    for i, blk in enumerate(visual.blocks):
        handles.append(blk.register_forward_hook(make_hook(f"vit_block_{i}")))
    # DeepStack mergers
    for j, m in enumerate(visual.deepstack_merger_list):
        handles.append(m.register_forward_hook(make_hook(f"deepstack_merger_{j}")))

    lm = model.backbone.model.model.language_model
    for i, layer in enumerate(lm.layers):
        handles.append(layer.register_forward_hook(make_hook(f"llm_layer_{i}")))

    # Action head
    ah = model.action_head
    if hasattr(ah, "vlln") and ah.vlln is not None:
        handles.append(ah.vlln.register_forward_hook(make_hook("vlln_out")))
    if hasattr(ah, "vl_self_attention") and ah.vl_self_attention is not None:
        # SelfAttentionTransformer.transformer_blocks
        if hasattr(ah.vl_self_attention, "transformer_blocks"):
            for i, blk in enumerate(ah.vl_self_attention.transformer_blocks):
                handles.append(blk.register_forward_hook(make_hook(f"vlsa_block_{i}")))

    # DiT: 32 transformer blocks
    if hasattr(ah, "model") and hasattr(ah.model, "transformer_blocks"):
        for i, blk in enumerate(ah.model.transformer_blocks):
            handles.append(blk.register_forward_hook(make_hook(f"dit_block_{i}")))

    return handles


def main():
    args = parse_args()
    ckpt = args.ckpt or autodetect_ckpt()
    print(f"ckpt:    {ckpt}")
    print(f"dataset: {args.dataset}")
    print(f"tag:     {args.tag}  traj={args.traj} step={args.step} seed={args.seed}")

    # Deterministic
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(False)  # GR00T's flash-attn isn't deterministic

    import gr00t.model  # registers Gr00tN1d7 model_class
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.eval.open_loop_eval import parse_observation_gr00t

    print("loading Gr00tPolicy ...", flush=True)
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.tag),
        model_path=ckpt,
        device="cuda:0",
    )
    model = policy.model.eval()
    print("loaded:", type(model).__name__, "params=", sum(p.numel() for p in model.parameters()) / 1e9, "B")

    activations: dict[str, torch.Tensor] = {}
    handles = register_block_hooks(model, activations)
    print(f"hooks:   {len(handles)} forward hooks installed")

    # Build observation from the LeRobot dataset
    print("loading dataset ...", flush=True)
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset,
        modality_configs=policy.modality_configs,
        # Demo data is AV1-encoded; opencv-python's FFmpeg lacks AV1 sw decoder
        # on aarch64. ffmpeg CLI backend goes through the system /usr/bin/ffmpeg
        # which has libdav1d. (torchcodec wheel ships LFS-pointer-only on aarch64.)
        video_backend="ffmpeg",
    )
    traj = loader[args.traj]  # __getitem__ returns the trajectory pd.DataFrame
    data_point = extract_step_data(traj, args.step, policy.modality_configs, policy.embodiment_tag)

    obs: dict[str, Any] = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for lang_key in loader.modality_configs["language"].modality_keys:
        obs[lang_key] = data_point.text
    parsed = parse_observation_gr00t(obs, loader.modality_configs)
    print("input keys:", list(parsed.keys()))

    print("running inference ...", flush=True)
    with torch.inference_mode():
        action_dict, info = policy.get_action(parsed)
    print("got actions:", {k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in action_dict.items()})

    # Cleanup hooks
    for h in handles:
        h.remove()

    # Sanity: every expected activation is present
    expected_keys = (
        [f"vit_block_{i}" for i in range(24)]
        + [f"deepstack_merger_{j}" for j in range(3)]
        + [f"llm_layer_{i}" for i in range(16)]
        + ["vlln_out"]
        + [f"vlsa_block_{i}" for i in range(4)]
        + [f"dit_block_{i}" for i in range(32)]
    )
    missing = [k for k in expected_keys if k not in activations]
    print(f"activations captured: {len(activations)}/{len(expected_keys)} expected; missing={missing[:5]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nv = len([k for k in obs if k.startswith("video.")])
    fname = f"gr00t_n17_ref_{args.tag.lower()}_{nv}v_traj{args.traj}_step{args.step}_seed{args.seed}.pt"
    out_path = out_dir / fname

    torch.save(
        {
            "meta": {
                "tag": args.tag,
                "ckpt": ckpt,
                "traj": args.traj,
                "step": args.step,
                "seed": args.seed,
                "num_views": nv,
                "torch_version": torch.__version__,
            },
            "inputs": parsed,  # nested dict of np arrays
            "actions": {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in action_dict.items()},
            "activations": activations,
        },
        out_path,
    )
    sz_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path} ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()

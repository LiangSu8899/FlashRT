"""Capture auxiliary tensors needed by Phase 3b.2 LLM cosine tests.

The base fixture (``gen_reference.py``) only saves per-block hidden state
activations. For the LLM forward we additionally need:

  * ``input_ids``           — (1, S) int64 — used to derive visual_pos_masks
  * ``visual_pos_masks``    — (1, S) bool  — DeepStack injection mask
  * ``rope_cos``, ``rope_sin`` — (1, S, HD) fp32 — M-RoPE tables (HF outputs
    a tuple of (cos, sin) shape ``(3, 1, S, HD)``; we collapse axis 0 which is
    only used inside the apply_rotary_pos_emb_vision helper for split sections)

Outputs alongside the existing fixture as ``..._llm_aux.pt``.
"""
from __future__ import annotations

import argparse
import functools
import glob
import os
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture",
                    default="/work/tests/fixtures/gr00t_n17_ref_oxe_droid_relative_eef_relative_joint_2v_traj1_step0_seed0.pt")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
    args = ap.parse_args()

    ckpt = args.ckpt or sorted(glob.glob(
        "/root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/*"))[0]

    fx = torch.load(args.fixture, weights_only=False, map_location="cpu")

    import gr00t.model  # noqa: F401  (registers Gr00tN1d7)
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag

    print(f"loading policy from {ckpt} ...", flush=True)
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.tag),
        model_path=ckpt, device="cuda:0",
    )

    captured = {}

    # Hook the language_model.forward to grab inputs_embeds, position_ids, etc.
    lm = policy.model.backbone.model.model.language_model
    orig_lm_forward = lm.forward

    def lm_hook(self, *fargs, **fkwargs):
        # The language_model is called positional-style with **kw — capture both
        # the inputs_embeds and the kwargs of interest.
        if "inputs_embeds" in fkwargs:
            captured["inputs_embeds_shape"] = tuple(fkwargs["inputs_embeds"].shape)
            # Capture the actual fused text+image embed tensor — this is the
            # input to the truncated LLM (pre-DeepStack-injection).
            captured["llm_input_embeds"] = (
                fkwargs["inputs_embeds"].detach().to(torch.float32).cpu())
        if "input_ids" in fkwargs and fkwargs["input_ids"] is not None:
            captured["input_ids"] = fkwargs["input_ids"].detach().cpu()
        if "visual_pos_masks" in fkwargs:
            vpm = fkwargs["visual_pos_masks"]
            captured["visual_pos_masks"] = vpm.detach().cpu() if vpm is not None else None
        if "deepstack_visual_embeds" in fkwargs:
            ds = fkwargs["deepstack_visual_embeds"]
            if ds is not None:
                captured["deepstack_visual_embeds_shapes"] = [
                    tuple(t.shape) for t in ds]
        if "position_ids" in fkwargs:
            pid = fkwargs["position_ids"]
            captured["position_ids"] = (
                pid.detach().cpu() if pid is not None else None)
        return orig_lm_forward(*fargs, **fkwargs)

    lm.forward = functools.partial(lm_hook, lm)

    # Capture input to ViT block 0 (= post-patch_embed + pos_embed pixel features)
    # Capture initial noise: monkey-patch action_head.get_action_with_features
    # so we record the very first ``actions = torch.randn(...)`` it produces.
    ah = policy.model.action_head
    orig_gawf = ah.get_action_with_features

    def gawf_hook(self, *args, **kwargs):
        # Patch torch.randn for the duration of this call to capture the noise.
        import torch as _t
        orig_randn = _t.randn

        def patched_randn(*ra, **rkw):
            t = orig_randn(*ra, **rkw)
            if "initial_noise" not in captured:
                captured["initial_noise"] = t.detach().to(_t.float32).cpu().clone()
            return t
        _t.randn = patched_randn
        try:
            return orig_gawf(*args, **kwargs)
        finally:
            _t.randn = orig_randn

    ah.get_action_with_features = functools.partial(gawf_hook, ah)

    # Also capture the final post-decode_action tensor (pre-denorm).
    ad = ah.action_decoder
    orig_ad_forward = ad.forward

    def ad_hook(self, *args, **kwargs):
        out = orig_ad_forward(*args, **kwargs)
        captured["last_action_decoder_out"] = out.detach().to(torch.float32).cpu()
        return out
    ad.forward = functools.partial(ad_hook, ad)

    # Capture per-step DiT input + output across the 4 inference timesteps.
    # ah.model is the AlternateVLDiT; hook its forward.
    dit = ah.model
    orig_dit_forward = dit.forward
    captured["dit_step_input"] = []
    captured["dit_step_output"] = []
    captured["dit_step_temb"] = []

    def dit_hook(self, hidden_states, *fa, **fk):
        captured["dit_step_input"].append(
            hidden_states.detach().to(torch.float32).cpu())
        if "timestep" in fk:
            captured["dit_step_temb"].append(
                fk["timestep"].detach().cpu().clone())
        out = orig_dit_forward(hidden_states, *fa, **fk)
        if isinstance(out, tuple):
            t0 = out[0]
        else:
            t0 = out
        captured["dit_step_output"].append(t0.detach().to(torch.float32).cpu())
        return out
    dit.forward = functools.partial(dit_hook, dit)

    visual = policy.model.backbone.model.model.visual
    block0 = visual.blocks[0]

    def block0_pre_hook(module, args, kwargs):
        # First positional arg is hidden_states (S, D)
        h = args[0] if args else kwargs.get("hidden_states")
        captured["pixel_features"] = h.detach().to(torch.float32).cpu()
        # Also grab grid_thw for ViT rope construction
        return None
    block0.register_forward_pre_hook(block0_pre_hook, with_kwargs=True)

    # Also hook the visual.forward to capture grid_thw
    orig_visual_forward = visual.forward

    def visual_hook(self, hidden_states, grid_thw, **kw):
        captured["grid_thw"] = grid_thw.detach().cpu()
        return orig_visual_forward(hidden_states, grid_thw, **kw)
    visual.forward = functools.partial(visual_hook, visual)

    # Also hook rotary_emb to capture cos/sin
    rot = lm.rotary_emb
    orig_rot_forward = rot.forward

    def rot_hook(self, x, position_ids):
        cos, sin = orig_rot_forward(x, position_ids)
        captured["rope_cos"] = cos.detach().cpu()
        captured["rope_sin"] = sin.detach().cpu()
        return cos, sin

    rot.forward = functools.partial(rot_hook, rot)

    parsed = fx["inputs"]
    # Seed RNG identically to gen_reference.py (which uses --seed=0 by
    # default and calls torch.manual_seed/np.random.seed before
    # policy.get_action). Without this the noise sampled inside
    # policy.get_action diverges from the noise used to produce
    # fx["actions"], so any e2e cos comparing our infer(aux noise) vs
    # fx["actions"] is comparing two different diffusion trajectories.
    seed = int(fx["meta"].get("seed", 0)) if isinstance(fx.get("meta"), dict) else 0
    print(f"seeding torch / numpy with seed={seed}", flush=True)
    torch.manual_seed(seed)
    import numpy as _np
    _np.random.seed(seed)
    print("running inference ...", flush=True)
    with torch.inference_mode():
        policy.get_action(parsed)

    out = {
        k: captured[k] for k in (
            "input_ids", "visual_pos_masks", "position_ids",
            "rope_cos", "rope_sin", "inputs_embeds_shape",
            "deepstack_visual_embeds_shapes",
            "llm_input_embeds", "pixel_features", "grid_thw",
            "initial_noise", "last_action_decoder_out",
            "dit_step_input", "dit_step_output", "dit_step_temb",
        ) if k in captured
    }
    print("captured keys:")
    for k, v in out.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {v}")

    out_path = Path(args.fixture).with_name(
        Path(args.fixture).stem + "_llm_aux.pt")
    torch.save(out, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()

"""Step 1 of layer-bisect: run HF native bf16 DiT with backbone == ours,
capture per-layer outputs into a side-file next to this script.

Run our frontend's set_prompt to get backbone first (cos≈0.9996 vs HF), so the
encoder_hidden_states fed to HF DiT are byte-identical to what we feed to our
DiT in step 2 of the bisect — that isolates 'DiT block implementation' as the
only difference.

Path overrides (all optional):
    GROOT_N17_CKPT  — directory containing the N1.7 snapshot (one level
                      above ``model-*.safetensors``). Defaults to the
                      first match under the local HF cache.
    GR00T_SRC       — extra sys.path entry for the gr00t package; only
                      used if ``import gr00t`` fails out of the box.
"""
from __future__ import annotations

import functools
import glob
import os
import pathlib
import sys

import torch

_REPO = pathlib.Path(__file__).resolve().parents[3]
_HERE = pathlib.Path(__file__).resolve().parent


def _default_ckpt_dir() -> str:
    """Resolve to the ``snapshots`` parent (the one that contains
    ``<hash>/model-*.safetensors`` subdirs); main() does the per-snapshot
    glob from there."""
    env = os.environ.get("GROOT_N17_CKPT")
    if env:
        return env
    return os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots")


CKPT_DIR = _default_ckpt_dir()
FX = str(_REPO / "tests" / "fixtures"
         / "gr00t_n17_ref_oxe_droid_relative_eef_relative_joint_2v_traj1_step0_seed0.pt")
AUX = FX.replace(".pt", "_llm_aux.pt")
OUT = str(_HERE / "_hf_dit_layer_ref.pt")


def main() -> None:
    snap = sorted(glob.glob(os.path.join(CKPT_DIR, "*")))[0]
    aux = torch.load(AUX, weights_only=False)

    # ── Compute "our" backbone first (a one-shot set_prompt) and stash. ──
    from flash_vla.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor
    fe = GrootN17TorchFrontendThor(
        snap, num_views=2,
        embodiment_tag="oxe_droid_relative_eef_relative_joint")
    fe.set_prompt(aux=aux, prompt="x")
    encoder_hs = fe._backbone_features.detach().to(torch.bfloat16).cpu()
    image_mask = aux["visual_pos_masks"].cpu()

    # Free our model — HF load needs the GPU.
    del fe
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ── Load HF model in bf16 native and run DiT once. ──
    # Try a plain import first; fall back to GR00T_SRC env var (e.g. a
    # local Isaac-GR00T checkout that isn't on the PYTHONPATH yet).
    try:
        import gr00t.model.gr00t_n1d7.gr00t_n1d7  # noqa: F401
    except ImportError:
        gr00t_src = os.environ.get("GR00T_SRC")
        if gr00t_src:
            sys.path.insert(0, gr00t_src)
        import gr00t.model.gr00t_n1d7.gr00t_n1d7  # noqa: F401  side-effect: register
    from transformers import AutoModel
    print("Loading HF model (bf16)...", flush=True)
    m = AutoModel.from_pretrained(
        snap, dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()
    dit = m.action_head.model

    layer_outs: list[torch.Tensor] = []

    def hook(_idx, _mod, _args, _kwargs, out):
        t = out[0] if isinstance(out, tuple) else out
        layer_outs.append(t.detach().to(torch.float32).cpu())

    handles = [
        b.register_forward_hook(functools.partial(hook, i), with_kwargs=True)
        for i, b in enumerate(dit.transformer_blocks)
    ]

    hidden = aux["dit_step_input"][0].to(torch.bfloat16).cuda()
    enc_cuda = encoder_hs.to(torch.bfloat16).cuda()
    img_mask = image_mask.cuda()
    bb_mask = torch.ones_like(img_mask)
    timestep = torch.tensor([int(aux["dit_step_temb"][0].item())],
                             dtype=torch.long).cuda()

    with torch.no_grad():
        final = dit(hidden, enc_cuda, timestep=timestep,
                     image_mask=img_mask, backbone_attention_mask=bb_mask)
    final = final.detach().to(torch.float32).cpu()

    for h in handles:
        h.remove()

    payload = {
        "encoder_hs": encoder_hs,        # (1, S, 2048) bf16, the SAME we'll
                                          # feed our DiT in step 2
        "hidden_input": hidden.cpu(),
        "image_mask": image_mask,
        "timestep": int(timestep.item()),
        "layer_outs": layer_outs,
        "final": final,
    }
    torch.save(payload, OUT)
    print(f"saved → {OUT}", flush=True)
    print(f"HF DiT final cos vs aux dit_step_output[0] = "
          f"{(lambda a,b:float((a.flatten().double()@b.flatten().double())/(a.flatten().double().norm()*b.flatten().double().norm())))(final.squeeze(0), aux['dit_step_output'][0].squeeze(0)):.6f}",
          flush=True)


if __name__ == "__main__":
    main()

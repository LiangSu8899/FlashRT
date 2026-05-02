"""Layer-by-layer numerical-equivalence test of our DiT vs HF reference.

Pre-req: run debug_dit_capture_hf.py first to dump per-layer HF outputs.

Pass A: re-inject HF output_{li-1} as our layer-li input, run our single
layer, compare to HF output_li. Pure single-layer cos — accumulation
contribution removed.
"""
from __future__ import annotations

import glob
import os
import pathlib
import sys

import torch

_REPO = pathlib.Path(__file__).resolve().parents[3]
_HERE = pathlib.Path(__file__).resolve().parent


def _default_ckpt_dir() -> str:
    env = os.environ.get("GROOT_N17_CKPT")
    if env:
        return env
    return os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots")


CKPT_DIR = _default_ckpt_dir()
FX = str(_REPO / "tests" / "fixtures"
         / "gr00t_n17_ref_oxe_droid_relative_eef_relative_joint_2v_traj1_step0_seed0.pt")
AUX = FX.replace(".pt", "_llm_aux.pt")
REF = str(_HERE / "_hf_dit_layer_ref.pt")


def _co(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    return float(a @ b / (a.norm() * b.norm()))


def main() -> None:
    snap = sorted(glob.glob(os.path.join(CKPT_DIR, "*")))[0]
    fx = torch.load(FX, weights_only=False)
    aux = torch.load(AUX, weights_only=False)
    ref = torch.load(REF, weights_only=False)

    from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor
    fe = GrootN17TorchFrontendThor(
        snap, num_views=2,
        embodiment_tag="oxe_droid_relative_eef_relative_joint")
    fe.set_prompt(aux=aux, prompt="x")  # internal warmup populates everything

    # Sanity: backbone delta vs HF reference (≈ bf16 round, expect ~0.1).
    delta = (fe._backbone_features.cpu().to(torch.float32) -
             ref["encoder_hs"].to(torch.float32)).abs().max().item()
    print(f"  | backbone delta vs reference (bf16 round): {delta:g}", flush=True)

    Sa = 41
    bufs = fe._infer_bufs

    # set_prompt → _warmup_infer already ran; cuBLAS hot. Don't call
    # fe._precompute_dit_cross_kv() — it would reallocate _dit_cross_K/V
    # and dangle the backend's K_layers/V_layers slot pointers.

    # Inject HF input + use HF temb to compute MOD modulators identically
    # to a real infer call.
    t_disc = int(ref["timestep"])
    temb = fe._compute_timestep_emb(t_disc)
    shifts, scales = fe._compute_dit_adaln_modulators(temb)

    # Build dit_forward dims/bufs/weights dicts.
    weights = {
        "scale_msa": [t.data_ptr() for t in scales],
        "shift_msa": [t.data_ptr() for t in shifts],
        "q_w": [w.data_ptr() for w in fe._dit_q_w],
        "q_b": [b.data_ptr() for b in fe._dit_q_b],
        "k_w": [w.data_ptr() for w in fe._dit_k_w],
        "k_b": [b.data_ptr() for b in fe._dit_k_b],
        "v_w": [w.data_ptr() for w in fe._dit_v_w],
        "v_b": [b.data_ptr() for b in fe._dit_v_b],
        "o_w": [w.data_ptr() for w in fe._dit_o_w],
        "o_b": [b.data_ptr() for b in fe._dit_o_b],
        "ff_proj_w": [w.data_ptr() for w in fe._dit_ff_proj_w],
        "ff_proj_b": [b.data_ptr() for b in fe._dit_ff_proj_b],
        "ff_down_w": [w.data_ptr() for w in fe._dit_ff_down_w],
        "ff_down_b": [b.data_ptr() for b in fe._dit_ff_down_b],
    }
    Skv_text = int(fe._dit_cross_K[0].shape[0])
    Skv_image = int(fe._dit_cross_K[1].shape[0])
    dims = {"Sa": Sa, "D": 1536, "FF": 6144,
            "Skv_text": Skv_text, "Skv_image": Skv_image}
    bufs_ptrs = {
        "h": bufs["dit_h"].data_ptr(),
        "xn": bufs["dit_xn"].data_ptr(),
        "o_proj_out": bufs["dit_o_proj_out"].data_ptr(),
        "ff_proj_out": bufs["dit_ff_proj_out"].data_ptr(),
    }

    from flash_rt.models.groot_n17 import pipeline_thor

    print("\n  Pass A: per-layer equivalence (re-inject HF output_{li-1})")
    print("  layer  type    cos(ours, HF)")
    print("  ----- ------- ----------------")
    for li in range(32):
        prev_in = (ref["hidden_input"] if li == 0 else ref["layer_outs"][li - 1])
        bufs["dit_h"][:Sa].copy_(prev_in.to(torch.bfloat16).cuda().squeeze(0))
        pipeline_thor.dit_forward(
            gemm=fe._gemm, fvk=fe._fvk,
            bufs=bufs_ptrs, weights=weights, dims=dims,
            attn=fe._dit_attn, layers_subset=[li],
        )
        torch.cuda.synchronize()
        ours = bufs["dit_h"][:Sa].clone().to(torch.float32).cpu()
        hf = ref["layer_outs"][li].squeeze(0)
        c = _co(ours, hf)
        kind = ("self" if (li % 2 == 1)
                else ("cross-T" if (li % 4 == 0) else "cross-I"))
        print(f"  {li:5d}  {kind:7s}  {c:.6f}", flush=True)


if __name__ == "__main__":
    main()

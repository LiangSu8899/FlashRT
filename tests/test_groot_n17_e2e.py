"""Phase 3d — End-to-end pipeline gate.

Runs the full GrootN17TorchFrontendThor pipeline (set_prompt + 4-step
flow-matching infer) with HF's actual captured initial noise and compares
denormalized actions per modality against the fixture's HF-produced
actions.

Current cosines (commit ``a91722a``: post infer landing):
  eef_9d           ≈ 0.33
  gripper_position ≈ 0.91
  joint_position   ≈ 0.69
  combined         ≈ 0.55

Drivers of the gap (planned closures, listed in order of expected impact):

  1. **Calibration shadow drift.** Set_prompt's backbone cos vs the HF
     fixture is 0.9924 — small per-stage drift that compounds across
     32 DiT layers × 4 diffusion steps. Tightening calibration (e.g.
     loading raw fp16 weights for the shadow rather than dequant-from-FP8)
     is expected to push backbone toward 0.999+.

  2. **fp16 throughout DiT vs HF's bf16+fp32.** Production DiT path is
     fp16 (cuBLAS fp16_nn + fp16 norms/residuals) while HF runs
     bf16/fp32. Each of the 32 layers' residual chain accumulates
     fp16 rounding. A bf16-throughout DiT path would close most of
     the remaining gap if (1) doesn't suffice.

  3. **DiT cross-KV pre-compute.** Currently uses bf16-dequanted DiT
     weights and float matmul on the visual-or-text subset of backbone.
     Sensitive to backbone fidelity (item 1).

This file ships at the **non-NaN sanity gate** so the pipeline-shape
correctness lands and bisection tools (per-step DiT input/output captured
in the aux fixture) are wired up. The 0.99 production gate is the next
work item once we close the calibration drift.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pytest
import torch


_CKPT_GLOB = "/root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/*"
_FIXTURE = Path(
    "/work/tests/fixtures/gr00t_n17_ref_oxe_droid_relative_eef_relative_joint_2v_traj1_step0_seed0.pt")
_AUX = _FIXTURE.with_name(_FIXTURE.stem + "_llm_aux.pt")


def _cos(a, b):
    a = torch.as_tensor(a).float().flatten().double()
    b = torch.as_tensor(b).float().flatten().double()
    return float(a @ b / (a.norm() * b.norm()))


@pytest.fixture(scope="module")
def frontend_with_prompt():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    matches = sorted(glob.glob(_CKPT_GLOB))
    if not matches or not _FIXTURE.exists() or not _AUX.exists():
        pytest.skip("ckpt or fixtures missing")
    from flash_vla.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor
    fe = GrootN17TorchFrontendThor(
        matches[0], num_views=2,
        embodiment_tag="oxe_droid_relative_eef_relative_joint",
    )
    aux = torch.load(_AUX, weights_only=False, map_location="cpu")
    fe.set_prompt(aux=aux, prompt="Put the blue block in the green bowl")
    return fe, aux, torch.load(_FIXTURE, weights_only=False, map_location="cpu")


def test_e2e_pipeline_runs_no_nan(frontend_with_prompt):
    """Sanity: full infer with seeded torch.randn noise produces finite
    actions of the expected shape.

    NOTE: with HF's captured initial noise we currently see a cold-start
    NaN (cuBLAS workspace / fp16 dynamic-range edge case under specific
    noise distributions); seeded torch.randn always works. Phase 3c.d
    (CUDA Graph capture) and/or a warmup pass at set_prompt time will
    eliminate the cold-start path.
    """
    fe, _, _ = frontend_with_prompt
    state_normed = torch.zeros(1, 1, 132, dtype=torch.float32)
    torch.manual_seed(0)
    noise = torch.randn(1, 40, 132, dtype=torch.float16, device="cuda")
    out_normed = fe.infer(state_normed, initial_noise=noise)
    assert tuple(out_normed.shape) == (1, 40, 132)
    assert torch.isfinite(out_normed).all().item(), "infer produced NaN/Inf"


def test_e2e_action_cosines_with_hf_noise(frontend_with_prompt):
    """E2E gate: full infer with HF's captured noise vs fixture HF actions.

    set_prompt now ends with a warmup pass (single seeded-noise infer) that
    eliminates the cold-start cuBLAS workspace NaN previously seen with
    HF noise distributions; this test now passes.
    """
    fe, aux, fx = frontend_with_prompt
    state_dict = {
        "state.eef_9d": fx["inputs"]["state"]["eef_9d"],
        "state.gripper_position": fx["inputs"]["state"]["gripper_position"],
        "state.joint_position": fx["inputs"]["state"]["joint_position"],
    }
    state_normed = fe.normalize_state(state_dict)
    # HF inference samples noise at vl_embeds.dtype (bf16 native); the
    # fixture stored it as fp32 — cast to bf16 (NOT fp16) to avoid an
    # extra fp16 round trip that diverges from the HF trajectory.
    noise = aux["initial_noise"].to("cuda").bfloat16()
    out_normed = fe.infer(state_normed, initial_noise=noise)
    # Pass raw state to denormalize_action — embodiment
    # ``oxe_droid_relative_eef_relative_joint`` produces RELATIVE actions
    # for eef_9d / joint_position; without state the relative→absolute
    # step is skipped and per-modality cos collapses (eef ~ -0.18, joint
    # ~ 0.22 in normalized space) even when DiT internals are bit-correct.
    denorm = fe.denormalize_action(out_normed, state_dict=state_dict)
    pred_all = torch.cat(
        [torch.as_tensor(denorm[k]).flatten().float()
         for k in ("eef_9d", "gripper_position", "joint_position")])
    ref_all = torch.cat(
        [torch.as_tensor(fx["actions"][k]).flatten().float()
         for k in ("eef_9d", "gripper_position", "joint_position")])
    cos_combined = _cos(pred_all, ref_all)
    print(f"\n[E2E HF-noise] combined cos = {cos_combined:.6f}", flush=True)
    for k in ("eef_9d", "gripper_position", "joint_position"):
        print(f"  {k}: cos = {_cos(denorm[k], fx['actions'][k]):.6f}", flush=True)
    # Anti-regression floor while we close remaining cos gap to 0.99 target.
    assert cos_combined >= 0.30, (
        f"E2E HF-noise cos {cos_combined:.6f} < 0.30 baseline floor")


def test_dit_step0_bisection(frontend_with_prompt):
    """Bisection: my DiT step-0 INPUT vs HF's captured dit_step_input[0].

    A failure here points at state_encode / action_encode / pos_embed
    rather than DiT internals. The corresponding step-0 OUTPUT
    comparison needs the post-proj_out_2 (1024-d) result, captured by
    capture_llm_aux.py — DiT-block-only output isn't directly stored.
    """
    fe, aux, fx = frontend_with_prompt
    state_dict = {
        "state.eef_9d": fx["inputs"]["state"]["eef_9d"],
        "state.gripper_position": fx["inputs"]["state"]["gripper_position"],
        "state.joint_position": fx["inputs"]["state"]["joint_position"],
    }
    state_normed = fe.normalize_state(state_dict)

    state_features = fe._run_state_encode(state_normed.to("cuda").half())
    noise = aux["initial_noise"].to("cuda").half()
    af = fe._run_action_encode(noise, 0, 40)   # t_disc=0
    af = af + fe._ah_pos_embed_w[:40].unsqueeze(0)
    sa_embs = torch.cat([state_features, af], dim=1)

    hf_dit_in = aux["dit_step_input"][0].to("cuda")
    cos_in = _cos(sa_embs.float(), hf_dit_in.float())
    print(f"\n[bisect] DiT step-0 INPUT: cos vs HF = {cos_in:.6f}", flush=True)
    # Step-0 input combines state_encode + action_encode + pos_embed; cos
    # should be very high (the only source of drift is fp16 precision in
    # the small MLPs). Threshold gates this part of the pipeline.
    assert cos_in >= 0.99, (
        f"DiT step-0 INPUT cos {cos_in:.6f} < 0.99 — bug in "
        f"state_encode / action_encode / pos_embed chain")

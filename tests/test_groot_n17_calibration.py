"""Phase 3c.b1 — smoke test for the calibration module.

Verifies that ``flash_rt.models.groot_n17.calibration`` runs end-to-end
on a real ckpt + Phase 1 fixture inputs without crashing, returns
sensible amax values, and the per-stage hidden-state shapes line up.

Cosine fidelity is NOT asserted here — the per-stage cos depends on
exact upstream chains (post-patch-embed pixel features for ViT layer 0,
post-embed LLM input, etc.) which are produced by ``set_prompt`` in
Phase 3c.b2. This test just validates the calibration math/shapes.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
import torch


_CKPT_GLOB = "/root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/*"
_FIXTURE = Path(
    "/work/tests/fixtures/gr00t_n17_ref_oxe_droid_relative_eef_relative_joint_2v_traj1_step0_seed0.pt")
_AUX = _FIXTURE.with_name(_FIXTURE.stem + "_llm_aux.pt")


@pytest.fixture(scope="module")
def frontend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    matches = sorted(glob.glob(_CKPT_GLOB))
    if not matches:
        pytest.skip("N1.7 ckpt not in HF cache")
    from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor
    return GrootN17TorchFrontendThor(
        matches[0], num_views=2,
        embodiment_tag="oxe_droid_relative_eef_relative_joint",
    )


@pytest.fixture(scope="module")
def fixtures():
    if not _FIXTURE.exists() or not _AUX.exists():
        pytest.skip(f"fixtures missing ({_FIXTURE} / {_AUX})")
    fx = torch.load(_FIXTURE, weights_only=False, map_location="cpu")
    aux = torch.load(_AUX, weights_only=False, map_location="cpu")
    return fx, aux


def _all_finite_positive(values, name):
    assert all(v > 0 and v < 1e6 for v in values), \
        f"{name}: non-positive or absurd amax: min={min(values)} max={max(values)}"


def test_calibrate_vit(frontend, fixtures):
    from flash_rt.models.groot_n17 import calibration as cal
    import sys; sys.path.insert(0, str(_FIXTURE.parent.parent))
    from _groot_n17_runner import build_vit_rope_tables

    fx, _ = fixtures
    cos_v, sin_v = build_vit_rope_tables([(1, 16, 16)] * 4, head_dim=64)
    vit_in = fx["activations"]["vit_block_0"].to("cuda").float()
    out = cal.calibrate_vit(
        frontend, vit_in, cos_v.float(), sin_v.float(), num_views=4)

    assert len(out["vit_act_qkv"]) == 24
    assert len(out["vit_act_o"]) == 24
    assert len(out["vit_act_fc1"]) == 24
    assert len(out["vit_act_fc2"]) == 24
    _all_finite_positive(out["vit_act_qkv"], "vit_act_qkv")
    _all_finite_positive(out["vit_act_fc1"], "vit_act_fc1")
    assert set(out["deepstack_taps"].keys()) == {5, 11, 17}
    assert tuple(out["vit_final"].shape) == (1024, 1024)


def test_calibrate_deepstack(frontend, fixtures):
    from flash_rt.models.groot_n17 import calibration as cal
    fx, _ = fixtures
    taps = {tap: fx["activations"][f"vit_block_{tap}"].to("cuda").float()
            for tap in (5, 11, 17)}
    out = cal.calibrate_deepstack(frontend, taps)
    assert len(out["deepstack_act_fc1"]) == 3
    assert len(out["deepstack_act_fc2"]) == 3
    _all_finite_positive(out["deepstack_act_fc1"], "deepstack_act_fc1")
    assert len(out["features"]) == 3
    for f in out["features"]:
        assert tuple(f.shape) == (256, 2048)


def test_calibrate_llm(frontend, fixtures):
    from flash_rt.models.groot_n17 import calibration as cal
    fx, aux = fixtures
    llm_in = fx["activations"]["llm_layer_0"].to("cuda").float()
    if llm_in.dim() == 2:
        llm_in = llm_in.unsqueeze(0)
    rope_cos = aux["rope_cos"][0].to("cuda").float()
    rope_sin = aux["rope_sin"][0].to("cuda").float()
    mask = aux["visual_pos_masks"][0].to("cuda")
    ds_feats = [fx["activations"][f"deepstack_merger_{j}"].to("cuda").float()
                for j in range(3)]
    out = cal.calibrate_llm(frontend, llm_in, rope_cos, rope_sin, mask, ds_feats)
    assert len(out["llm_act_qkv"]) == 16
    assert len(out["llm_act_o"]) == 16
    assert len(out["llm_act_gateup"]) == 16
    assert len(out["llm_act_down"]) == 16
    _all_finite_positive(out["llm_act_qkv"], "llm_act_qkv")
    assert tuple(out["llm_final"].shape) == (1, 277, 2048)


def test_calibrate_vlsa(frontend, fixtures):
    from flash_rt.models.groot_n17 import calibration as cal
    fx, _ = fixtures
    # Use llm_layer_15 directly (post-layer 15, pre-vlln) as a stand-in
    # llm_final to exercise the vlsa shadow.
    llm_final = fx["activations"]["llm_layer_15"].to("cuda").float()
    if llm_final.dim() == 2:
        llm_final = llm_final.unsqueeze(0)
    out = cal.calibrate_vlsa(frontend, llm_final)
    assert len(out["vlsa_act_qkv"]) == 4
    _all_finite_positive(out["vlsa_act_qkv"], "vlsa_act_qkv")
    assert tuple(out["backbone_features"].shape) == (1, 277, 2048)


def test_frontend_set_prompt(frontend, fixtures):
    """End-to-end set_prompt: shadow chain + alpha bake + cache save.

    Validates:
      * set_prompt completes (no crashes)
      * Per-stage alpha + d_act_scale lists are the right length & sensible
      * backbone_features shape matches expected (1, S, 2048)
      * Backbone-vs-fixture cosine indicates the shadow chain reproduces the
        HF reference within the user's E2E gate (≥0.99)
      * Calibration cache JSON is written
    """
    import os
    fx, aux = fixtures
    frontend.set_prompt(aux=aux, prompt="Put the blue block in the green bowl")

    assert frontend.Se == 277
    assert tuple(frontend._backbone_features.shape) == (1, 277, 2048)
    assert frontend._backbone_features.dtype == torch.float16

    # alpha lists
    assert len(frontend._vit_alpha_q) == 24
    assert len(frontend._llm_alpha_qkv) == 16
    assert len(frontend._vlsa_alpha_q) == 4
    assert len(frontend._dsm_alpha_fc1) == 3

    # d_act_scale device tensor lists
    assert len(frontend._vit_act_qkv_dev) == 24
    assert all(t.dtype == torch.float32 and t.numel() == 1
               for t in frontend._vit_act_qkv_dev)

    # DeepStack inject buffers (256 visual rows non-zero)
    for j in range(3):
        nz = (frontend._deepstack_inject[j].abs().sum(-1) > 0).sum().item()
        assert nz == 256, f"deepstack_inject[{j}] has {nz} non-zero rows, expected 256"

    # Backbone vs fixture vlsa_block_3 — cos≥0.99 is the production E2E gate.
    def co(a, b):
        a, b = a.flatten().double(), b.flatten().double()
        return float(a @ b / (a.norm() * b.norm()))
    ref = fx["activations"]["vlsa_block_3"]
    pred = frontend._backbone_features.float().cpu()
    cos = co(pred, ref)
    assert cos >= 0.99, f"backbone shadow vs fixture cos {cos:.6f} < 0.99"

    # Cache written
    assert hasattr(frontend, "_calibration_cache_path")
    assert os.path.exists(frontend._calibration_cache_path)
    assert os.path.getsize(frontend._calibration_cache_path) > 1000


def test_amax_helpers():
    from flash_rt.models.groot_n17 import calibration as cal
    s = cal.amax_to_dev_scale(44.8)
    assert s.dtype == torch.float32
    assert s.numel() == 1
    assert abs(s.item() - 0.1) < 1e-6
    a = cal.alpha(amax_act=44.8, weight_scale=0.001)
    assert abs(a - 0.0001) < 1e-9

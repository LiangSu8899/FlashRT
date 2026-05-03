"""Integration tests for GROOT N1.6 multi-sample calibration on RTX.

Runs on RTX 5090 / 4090. Skipped if GPU or the GR00T-N1.6-3B checkpoint
are unavailable. Set ``GROOT_N16_CHECKPOINT`` to override the default
checkpoint directory.

Run::

    python -m pytest tests/test_groot_calibrate_rtx.py -v
"""

import os

import numpy as np
import pytest
import torch

CKPT_GROOT = os.environ.get(
    "GROOT_N16_CHECKPOINT", "<ckpts>/GR00T-N1.6-3B")

_GPU_AVAILABLE = torch.cuda.is_available()
_CKPT_AVAILABLE = os.path.isdir(CKPT_GROOT)


def _bootstrap_pipeline(rt):
    """Build the GROOT sub-pipelines and FP8 calibration state, keeping
    ``_sd`` alive so the shadow-forward helpers remain callable."""
    dummy_obs = {
        "image": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "state": np.zeros(32, dtype=np.float32),
    }
    rt._build_pipeline_and_capture(dummy_obs, release_sd=False)
    return dummy_obs


def _make_synthetic_ie(qwen3, seed):
    """Build a deterministic synthetic ``ie_fp16`` tensor of the right shape."""
    from flash_rt.models.groot.pipeline_rtx import QWEN3_D
    torch.manual_seed(seed)
    return torch.randn(qwen3.Se, QWEN3_D, device="cuda", dtype=torch.float16)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"GR00T-N1.6-3B ckpt missing at {CKPT_GROOT}")
def test_qwen3_per_sample_n1_matches_legacy_impl():
    """``_calibrate_qwen3_per_sample([ie])`` must match ``_calibrate_qwen3_impl(ie)``.

    Guards the refactor that splits raw-amax collection from FP8 scale
    conversion: the single-sample legacy path must remain bit-identical.
    """
    from flash_rt.frontends.torch.groot_rtx import (
        GrootTorchFrontendRtx,
        _cal_scale,
        _calibrate_qwen3_impl,
        _calibrate_qwen3_per_sample,
    )

    rt = GrootTorchFrontendRtx(CKPT_GROOT, num_views=2,
                               embodiment_tag="gr1")
    rt.set_prompt("pick up the red block")
    _bootstrap_pipeline(rt)
    ie = _make_synthetic_ie(rt._qwen3, seed=0)

    legacy = _calibrate_qwen3_impl(
        rt._sd, rt._gemm, rt._fvk, rt._qwen3, ie)
    matrix = _calibrate_qwen3_per_sample(
        rt._sd, rt._gemm, rt._fvk, rt._qwen3, [ie])

    assert matrix.shape[0] == 1
    persample = [_cal_scale(a) for a in matrix[0].tolist()]
    np.testing.assert_array_equal(
        np.asarray(legacy, dtype=np.float64),
        np.asarray(persample, dtype=np.float64))


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"GR00T-N1.6-3B ckpt missing at {CKPT_GROOT}")
def test_qwen3_per_sample_shape_and_finiteness():
    """N>=2 must yield ``[N, 3*L]`` finite non-negative amax values."""
    from flash_rt.frontends.torch.groot_rtx import (
        GrootTorchFrontendRtx,
        _calibrate_qwen3_per_sample,
    )
    from flash_rt.models.groot.pipeline_rtx import QWEN3_L

    rt = GrootTorchFrontendRtx(CKPT_GROOT, num_views=2,
                               embodiment_tag="gr1")
    rt.set_prompt("pick up the red block")
    _bootstrap_pipeline(rt)
    ies = [_make_synthetic_ie(rt._qwen3, seed=s) for s in range(3)]

    matrix = _calibrate_qwen3_per_sample(
        rt._sd, rt._gemm, rt._fvk, rt._qwen3, ies)

    assert matrix.shape == (3, 3 * QWEN3_L), \
        f"expected (3, {3 * QWEN3_L}), got {matrix.shape}"
    assert np.isfinite(matrix).all()
    assert (matrix >= 0).all()


def test_qwen3_per_sample_empty_raises():
    """Empty input list raises ``ValueError`` without touching CUDA."""
    from flash_rt.frontends.torch.groot_rtx import _calibrate_qwen3_per_sample

    with pytest.raises(ValueError, match="non-empty"):
        _calibrate_qwen3_per_sample(None, None, None, None, [])


def test_dit_per_sample_empty_raises():
    """Empty cal_tensors list raises ``ValueError`` without touching CUDA."""
    from flash_rt.frontends.torch.groot_rtx import _calibrate_dit_per_sample

    with pytest.raises(ValueError, match="non-empty"):
        _calibrate_dit_per_sample(None, None, None, None, [], [], [])


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"GR00T-N1.6-3B ckpt missing at {CKPT_GROOT}")
def test_calibrate_multi_frame_end_to_end():
    """``calibrate([obs1, obs2, obs3])`` must populate scales + precision_spec
    and leave the model in a runnable state."""
    from flash_rt.frontends.torch.groot_rtx import GrootTorchFrontendRtx
    from flash_rt.models.groot.pipeline_rtx import DIT_L, QWEN3_L

    rt = GrootTorchFrontendRtx(CKPT_GROOT, num_views=2, embodiment_tag="gr1")
    rt.set_prompt("pick up the red block")

    rng = np.random.RandomState(0)
    obs_list = [{
        "image": rng.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "wrist_image": rng.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "state": rng.randn(32).astype(np.float32),
    } for _ in range(3)]

    rt.calibrate(obs_list, percentile=99.9)

    spec = rt.precision_spec
    assert spec is not None
    assert spec.source == "calibration"
    assert len(spec.encoder_layer_specs) == QWEN3_L * 3
    assert len(spec.decoder_layer_specs) == DIT_L * 3

    first_qwen3 = next(iter(spec.encoder_layer_specs.values()))
    assert first_qwen3.calibration_method == "percentile"
    assert first_qwen3.calibration_samples == 3
    assert first_qwen3.calibration_percentile == 99.9
    assert np.isfinite(first_qwen3.scale[0]) and first_qwen3.scale[0] > 0

    # All scales finite and positive (DiT cross placeholders == 1/_FP8_MAX)
    for entry in list(spec.encoder_layer_specs.values()) + list(
            spec.decoder_layer_specs.values()):
        assert np.isfinite(entry.scale[0]) and entry.scale[0] > 0

    # Inference still runs after multi-frame calibration
    out = rt.infer(obs_list[0])
    assert "actions" in out
    assert np.isfinite(out["actions"]).all()


def test_dit_per_sample_length_mismatch_raises():
    """Mismatched per-sample list lengths raise ``ValueError``."""
    from flash_rt.frontends.torch.groot_rtx import _calibrate_dit_per_sample

    with pytest.raises(ValueError, match="length mismatch"):
        _calibrate_dit_per_sample(
            None, None, None, None,
            cal_tensors_list=[{}, {}],
            state_feat_list=[None],
            actions_fp32_list=[None, None])

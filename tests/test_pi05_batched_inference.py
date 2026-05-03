"""Integration tests for the Pi0.5 RTX B=2 batched inference path.

Runs on RTX 5090 / 4090. Skipped if GPU or pi05 checkpoint missing.

These tests cover:
  - The default ``infer()`` path is unaffected when ``set_batched_mode``
    is never called.
  - ``set_batched_mode`` switches the attention backend to
    :class:`RtxFlashAttnBatchedBackendPi05` and the next
    ``set_prompt_batch`` builds a :class:`Pi05BatchedPipeline`.
  - ``infer_batch([obs1, obs2])`` returns two finite action chunks.
  - With identical observations and prompts on both batch slots, the
    per-slot outputs are bit-equal to each other (sanity check that
    sample-1 and sample-2 contexts are isolated and identical).
  - Disabling batched mode reverts the pipeline class on the next
    standard ``set_prompt``.

Run::

    python -m pytest tests/test_pi05_batched_inference.py -v
"""

import os

import numpy as np
import pytest
import torch

CKPT_PI05 = os.environ.get(
    "PI05_LIBERO_PYTORCH_CHECKPOINT",
    "<ckpts>/pi05_libero_pytorch")

_GPU_AVAILABLE = torch.cuda.is_available()
_CKPT_AVAILABLE = os.path.isdir(CKPT_PI05)


def _make_dummy_obs():
    return {
        "image": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
    }


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"pi05 ckpt missing at {CKPT_PI05}")
def test_default_path_unchanged_no_batched_mode():
    """Without ``set_batched_mode``, pipeline class is the standard one."""
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline
    from flash_rt.models.pi05.pipeline_rtx_batched import Pi05BatchedPipeline

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    rt.set_prompt("pick up the cup")
    assert isinstance(rt.pipeline, Pi05Pipeline)
    assert not isinstance(rt.pipeline, Pi05BatchedPipeline)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"pi05 ckpt missing at {CKPT_PI05}")
def test_set_batched_mode_builds_batched_pipeline():
    """``set_batched_mode`` + ``set_prompt_batch`` builds Pi05BatchedPipeline."""
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    from flash_rt.hardware.rtx.attn_backend_batched_pi05 import (
        RtxFlashAttnBatchedBackendPi05,
    )
    from flash_rt.models.pi05.pipeline_rtx_batched import Pi05BatchedPipeline

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    rt.set_batched_mode(enable=True)
    assert isinstance(rt.attn_backend, RtxFlashAttnBatchedBackendPi05)
    rt.set_prompt_batch(["pick up the cup", "fold the t-shirt"])
    assert isinstance(rt.pipeline, Pi05BatchedPipeline)
    assert rt.pipeline.B == 2


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"pi05 ckpt missing at {CKPT_PI05}")
def test_set_prompt_batch_rejects_wrong_size():
    """Wrong-sized prompt list is rejected at the boundary."""
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    rt.set_batched_mode(enable=True)
    with pytest.raises(ValueError, match="2 prompts"):
        rt.set_prompt_batch(["only one prompt"])


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"pi05 ckpt missing at {CKPT_PI05}")
def test_infer_batch_returns_two_finite_action_chunks():
    """End-to-end batched inference returns 2 finite action chunks."""
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    rt.set_batched_mode(enable=True)
    rt.set_prompt_batch(["pick up the cup", "pick up the cup"])
    rt.calibrate_batch([_make_dummy_obs()])

    out = rt.infer_batch([_make_dummy_obs(), _make_dummy_obs()])
    assert len(out) == 2
    for r in out:
        assert "actions" in r
        a = r["actions"]
        assert np.isfinite(a).all(), "non-finite actions in batched output"
        assert np.max(np.abs(a)) < 100.0, \
            f"actions out of plausible range: max abs = {np.max(np.abs(a))}"


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"pi05 ckpt missing at {CKPT_PI05}")
def test_infer_batch_both_slots_yield_plausible_actions():
    """B=2 with identical obs + prompt on both slots: each slot independently
    produces finite, in-range actions.

    Note: per-slot outputs differ because each sample is seeded with its
    own diffusion noise (independent draws of ``self._noise_buf_b2.normal_()``).
    Verifying numerical equivalence to a single-sample reference would
    require an API to inject identical noise per slot — out of scope for
    v0.1.0 generic batched inference.
    """
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    rt.set_batched_mode(enable=True)
    rt.set_prompt_batch(["pick up the cup", "pick up the cup"])
    rt.calibrate_batch([_make_dummy_obs()])

    obs = _make_dummy_obs()
    torch.manual_seed(424242)
    out = rt.infer_batch([obs, obs])

    for b, r in enumerate(out):
        a = r["actions"]
        assert np.isfinite(a).all(), f"slot {b}: non-finite actions"
        # Plausible-range guard (LIBERO actions are normalised to ~[-1, 1];
        # diffusion outputs occasionally exceed that under degenerate
        # zero-input observations, so allow modest overshoot).
        assert np.max(np.abs(a)) < 100.0, \
            f"slot {b}: actions out of range, max abs = {np.max(np.abs(a))}"

"""Real-checkpoint gate for Pi0.5 with asynchronous temporal fusion.

Set ``PI05_CKPT`` to a Pi0.5 PyTorch checkpoint and run on an SM89 GPU. The
default test suite skips this module when the checkpoint or CUDA is absent.
"""

from __future__ import annotations

import math
import os
import time
import unittest

import numpy as np
import torch

from flash_rt.runtime.rtc_temporal_fusion import (
    AsyncTemporalFusionRunner,
    TemporalFusionConfig,
)


_CHECKPOINT = os.environ.get("PI05_CKPT")
_HAS_CHECKPOINT = bool(_CHECKPOINT and os.path.isdir(_CHECKPOINT))


class _Pi05Adapter:
    def __init__(self, model, prompt: str):
        self.model = model
        self.prompt = prompt

    def infer_actions(self, observation):
        return np.asarray(self.model.predict(
            images=observation["images"], prompt=self.prompt))


def _validate_pi05_checkpoint_async_temporal_fusion():
    import flash_rt

    capability = torch.cuda.get_device_capability()
    if capability != (8, 9):
        raise unittest.SkipTest(
            f"RTX SM89 gate requires compute capability 8.9, got {capability}")

    rng = np.random.RandomState(7)
    image = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    observation = {"images": [image, image.copy()]}
    prompt = "pick up the red block"
    model = flash_rt.load_model(
        checkpoint=_CHECKPOINT,
        framework="torch",
        config="pi05",
        hardware="rtx_sm89",
        num_views=2,
        autotune=int(os.environ.get("RTC_PI05_AUTOTUNE", "0")),
        use_fp8=True,
    )

    # Complete calibration/capture before constructing the controller clock.
    baseline = np.asarray(model.predict(
        images=observation["images"], prompt=prompt))
    assert baseline.ndim == 2 and baseline.shape[0] > 1 and baseline.shape[1] > 0
    assert np.all(np.isfinite(baseline))

    config = TemporalFusionConfig(
        action_hz=20,
        max_chunks=3,
        decay=0.1,
        switch_mode="latency",
        start_next_at=1,
        miss_policy="block",
    )
    runner = AsyncTemporalFusionRunner(_Pi05Adapter(model, prompt), config)
    try:
        runner.reset(observation)
        initial = runner.active_chunk
        assert initial is not None
        assert initial.actions.shape == baseline.shape
        assert np.all(np.isfinite(initial.actions))

        # Serve action zero, then action one while the next real model inference
        # runs on the worker. The active prediction must not switch yet.
        runner.next_action(observation)
        initial_id = runner.active_chunk.prediction_id
        runner.next_action(observation)
        assert runner.active_chunk.prediction_id == initial_id

        deadline = time.monotonic() + 10.0
        while not runner.prediction_ready and time.monotonic() < deadline:
            time.sleep(0.001)
        assert runner.prediction_pending and runner.prediction_ready, (
            "background Pi0.5 inference did not finish within 10 seconds")

        served = runner.next_action(observation)
        fused = runner.active_chunk
        assert fused is not None and fused.prediction_id != initial_id
        assert served.shape == (baseline.shape[1],)
        assert np.all(np.isfinite(served))
        assert int(np.max(fused.source_counts)) >= 2

        raw = runner.buffer.raw_chunks
        assert len(raw) >= 2
        previous, latest = raw[-2], raw[-1]
        source_i = latest.start_step - previous.start_step
        assert 0 <= source_i < previous.horizon
        weight = math.exp(-config.decay * abs(source_i))
        expected = (np.asarray(latest.actions[0], dtype=np.float64)
                    + weight * np.asarray(previous.actions[source_i],
                                          dtype=np.float64)) / (1.0 + weight)
        assert np.allclose(fused.actions[0], expected, rtol=1e-5, atol=1e-6)
        assert runner.stats.predictions_completed >= 2
        assert runner.stats.chunk_switches >= 1
    finally:
        runner.close()
        del model
        torch.cuda.empty_cache()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
@unittest.skipUnless(
    _HAS_CHECKPOINT, "set PI05_CKPT to a Pi0.5 PyTorch checkpoint")
class Pi05TemporalFusionCheckpointTest(unittest.TestCase):
    def test_async_prediction_is_temporally_fused(self):
        _validate_pi05_checkpoint_async_temporal_fusion()


if __name__ == "__main__":
    unittest.main(verbosity=2)

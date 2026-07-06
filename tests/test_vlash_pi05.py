"""Real-checkpoint gate for Pi0.5 with asynchronous VLASh.

Set ``PI05_CKPT`` to a Pi0.5 PyTorch checkpoint and run on an SM89 GPU. The
normal test suite skips this module when the checkpoint or CUDA is absent.
"""

from __future__ import annotations

import os
import threading
import time
import unittest

import numpy as np
import torch

from flash_rt.runtime.vlash import AsyncVLAShRunner, VLAShConfig


_CHECKPOINT = (os.environ.get("PI05_CKPT")
               or os.environ.get("PI05_LIBERO_PYTORCH_CHECKPOINT"))
_HAS_CHECKPOINT = bool(_CHECKPOINT and os.path.isdir(_CHECKPOINT))


class _Pi05VLAShAdapter:
    def __init__(self, model, prompt: str):
        self.model = model
        self.prompt = prompt
        self.states: list[np.ndarray] = []
        self._lock = threading.Lock()

    def infer_actions(self, observation):
        state = np.asarray(observation["state"], dtype=np.float32)
        with self._lock:
            self.states.append(state.copy())
        return np.asarray(self.model.predict(
            images=observation["images"], prompt=self.prompt, state=state))

    def recorded_states(self) -> list[np.ndarray]:
        with self._lock:
            return [s.copy() for s in self.states]


class Pi05VLAShCheckpointTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    @unittest.skipUnless(
        _HAS_CHECKPOINT, "set PI05_CKPT to a Pi0.5 PyTorch checkpoint")
    def test_async_prediction_uses_projected_state(self):
        import flash_rt

        capability = torch.cuda.get_device_capability()
        if capability != (8, 9):
            raise unittest.SkipTest(
                "RTX SM89 gate requires compute capability 8.9, "
                f"got {capability}")

        rng = np.random.RandomState(11)
        image = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        observation = {"images": [image, image.copy()]}
        prompt = "pick up the red block"
        state_dim = 7
        state0 = np.zeros(state_dim, dtype=np.float32)
        measured = np.linspace(-0.2, 0.2, state_dim).astype(np.float32)

        model = flash_rt.load_model(
            checkpoint=_CHECKPOINT,
            framework="torch",
            config="pi05",
            hardware="rtx_sm89",
            num_views=2,
            autotune=int(os.environ.get("VLASH_PI05_AUTOTUNE", "0")),
            use_fp8=True,
            state_prompt_mode="fixed",
        )
        adapter = _Pi05VLAShAdapter(model, prompt)
        runner = AsyncVLAShRunner(
            adapter,
            VLAShConfig(
                action_hz=20,
                lookahead_steps=1,
                start_next_at=1,
                miss_policy="block",
                state_action_indices=tuple(range(state_dim)),
            ),
        )
        try:
            runner.reset(observation, state=state0)
            initial = runner.current_chunk
            self.assertIsNotNone(initial)
            assert initial is not None
            self.assertEqual(initial.actions.ndim, 2)
            self.assertGreater(initial.actions.shape[0], 1)
            self.assertGreaterEqual(initial.actions.shape[1], state_dim)
            self.assertTrue(np.all(np.isfinite(initial.actions)))

            first = runner.next_action(observation, state=state0)
            self.assertTrue(np.all(np.isfinite(first)))

            expected_projected = measured + initial.actions[1, :state_dim]
            second = runner.next_action(observation, state=measured)
            self.assertTrue(np.all(np.isfinite(second)))

            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                states = adapter.recorded_states()
                pending = getattr(runner, "_pending", None)
                if len(states) >= 2 and pending is not None and pending.done():
                    break
                time.sleep(0.001)
            states = adapter.recorded_states()
            self.assertGreaterEqual(
                len(states), 2,
                "background Pi0.5 inference did not start within 10 seconds")
            np.testing.assert_allclose(states[0], state0, rtol=0, atol=0)
            np.testing.assert_allclose(
                states[1], expected_projected.astype(np.float32),
                rtol=1e-5, atol=1e-5)

            served = runner.next_action(observation, state=expected_projected)
            current = runner.current_chunk
            self.assertIsNotNone(current)
            assert current is not None
            self.assertTrue(np.all(np.isfinite(served)))
            self.assertEqual(current.projected_action_count, 1)
            np.testing.assert_allclose(
                current.start_state, measured, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(
                current.projected_state, expected_projected,
                rtol=1e-5, atol=1e-5)
            self.assertGreaterEqual(runner.stats.chunks_started, 1)
            self.assertGreaterEqual(runner.stats.chunks_completed, 1)
            self.assertGreaterEqual(runner.stats.swaps, 1)
        finally:
            runner.close()
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main(verbosity=2)

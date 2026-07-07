import math
import threading
import time

import numpy as np
import pytest

from flash_rt.runtime.rtc_temporal_fusion import (
    AsyncTemporalFusionRunner,
    TemporalFusionBuffer,
    TemporalFusionConfig,
)


def test_fusion_aligns_actions_by_controller_step():
    cfg = TemporalFusionConfig(action_hz=10, decay=0.5, epoch_s=0)
    buf = TemporalFusionBuffer(cfg, clock=lambda: 0.0)
    one = buf.begin_prediction(started_at=0.0)
    buf.complete_prediction(one, [[0.0], [1.0], [2.0], [3.0]], ready_at=0.01)
    two = buf.begin_prediction(started_at=0.2)
    fused = buf.complete_prediction(two, [[10.0], [11.0], [12.0], [13.0]], ready_at=0.21)

    old_weight = math.exp(-1.0)  # old index 2 vs latest index 0
    assert np.isclose(fused.actions[0, 0], (10 + old_weight * 2) / (1 + old_weight))
    assert np.isclose(fused.actions[1, 0], (11 + old_weight * 3) / (1 + old_weight))
    assert np.array_equal(fused.source_counts, [2, 2, 1, 1])
    assert np.allclose(fused.expected_times_s, [0.2, 0.3, 0.4, 0.5])


def test_fusion_uses_up_to_three_raw_chunks():
    cfg = TemporalFusionConfig(action_hz=10, max_chunks=3,
                               decay=0.2, epoch_s=0)
    buf = TemporalFusionBuffer(cfg, clock=lambda: 0.0)
    for start, value in ((0.0, 0.0), (0.1, 10.0)):
        ticket = buf.begin_prediction(started_at=start)
        buf.complete_prediction(ticket, np.full((5, 1), value),
                                ready_at=start + 0.01)
    ticket = buf.begin_prediction(started_at=0.2)
    fused = buf.complete_prediction(ticket, np.full((5, 1), 20.0),
                                    ready_at=0.21)

    w_oldest = math.exp(-0.4)
    w_previous = math.exp(-0.2)
    expected = (20 + w_previous * 10) / (1 + w_previous + w_oldest)
    assert np.isclose(fused.actions[0, 0], expected)
    assert fused.source_counts[0] == 3


def test_raw_chunks_are_immutable_and_expire_after_last_step():
    cfg = TemporalFusionConfig(action_hz=10, epoch_s=0)
    buf = TemporalFusionBuffer(cfg, clock=lambda: 0.0)
    source = np.array([[1.0], [2.0], [3.0]])
    ticket = buf.begin_prediction(started_at=0)
    buf.complete_prediction(ticket, source, ready_at=0.01)
    source[:] = 99

    assert np.array_equal(buf.raw_chunks[0].actions[:, 0], [1, 2, 3])
    assert buf.prune_expired(now=0.2) == 0
    assert buf.prune_expired(now=0.3) == 1
    assert buf.total_pruned == 1


def test_completion_counts_chunks_pruned_before_fusion():
    cfg = TemporalFusionConfig(action_hz=10, epoch_s=0)
    buf = TemporalFusionBuffer(cfg, clock=lambda: 0.0)
    first = buf.begin_prediction(started_at=0)
    buf.complete_prediction(first, np.ones((2, 1)), ready_at=0.01)

    second = buf.begin_prediction(started_at=0.3)
    buf.complete_prediction(second, np.ones((2, 1)), ready_at=0.31)

    assert buf.total_pruned == 1


def test_latency_switch_uses_elapsed_step_count():
    cfg = TemporalFusionConfig(action_hz=10, epoch_s=0,
                               switch_mode="latency")
    buf = TemporalFusionBuffer(cfg, clock=lambda: 0.0)
    ticket = buf.begin_prediction(started_at=0)
    fused = buf.complete_prediction(ticket, np.arange(5)[:, None],
                                    ready_at=0.25)
    assert fused.switch_index == 2


def test_latency_switch_can_use_later_foreground_poll_time():
    cfg = TemporalFusionConfig(action_hz=10, epoch_s=0,
                               switch_mode="latency")
    buf = TemporalFusionBuffer(cfg, clock=lambda: 0.0)
    ticket = buf.begin_prediction(started_at=0)
    fused = buf.complete_prediction(ticket, np.arange(5)[:, None],
                                    ready_at=0.15, switch_at=0.32)
    assert fused.switch_index == 3


def test_absolute_state_switch_chooses_nearest_action():
    cfg = TemporalFusionConfig(action_hz=10, epoch_s=0,
                               switch_mode="state")
    buf = TemporalFusionBuffer(cfg, clock=lambda: 0.0)
    ticket = buf.begin_prediction(state=[0.0], started_at=0)
    fused = buf.complete_prediction(ticket, [[1.0], [5.0], [9.0]],
                                    ready_at=0.01, current_state=[5.2])
    assert fused.switch_index == 1


def test_delta_state_switch_integrates_from_prediction_start_state():
    cfg = TemporalFusionConfig(action_hz=10, epoch_s=0,
                               switch_mode="state",
                               action_representation="delta",
                               delta_mode="cumulative")
    buf = TemporalFusionBuffer(cfg, clock=lambda: 0.0)
    ticket = buf.begin_prediction(state=[10.0], started_at=0)
    fused = buf.complete_prediction(ticket, [[1.0], [2.0], [-1.0]],
                                    ready_at=0.01, current_state=[12.9])
    # Estimated post-action states are [11, 13, 12].
    assert fused.switch_index == 1


def test_async_runner_consumes_old_chunk_during_prediction():
    class Adapter:
        def infer_actions(self, observation):
            if observation["chunk"]:
                time.sleep(0.01)
            base = observation["chunk"] * 10
            return np.arange(base, base + 4, dtype=np.float32)[:, None]

    runner = AsyncTemporalFusionRunner(
        Adapter(),
        TemporalFusionConfig(action_hz=1000, start_next_at=2,
                             decay=100),
    )
    try:
        runner.reset({"chunk": 0})
        assert runner.next_action({"chunk": 1}).item() == 0
        assert runner.next_action({"chunk": 1}).item() == 1
        assert runner.next_action({"chunk": 1}).item() == 2
        time.sleep(0.02)
        assert runner.next_action({"chunk": 2}).item() >= 10
        assert runner.stats.chunk_switches >= 1
    finally:
        runner.close()


def test_async_runner_snapshots_mutable_observation():
    started = threading.Event()
    release = threading.Event()
    seen = []

    class Adapter:
        def __init__(self):
            self.calls = 0

        def infer_actions(self, observation):
            self.calls += 1
            if self.calls > 1:
                started.set()
                assert release.wait(timeout=1)
                seen.append((observation["step"], observation["image"].copy()))
            return np.arange(4, dtype=np.float32)[:, None]

    runner = AsyncTemporalFusionRunner(
        Adapter(), TemporalFusionConfig(action_hz=10, start_next_at=0))
    observation = {"step": 1, "image": np.array([1.0, 2.0])}
    try:
        runner.reset(observation)
        runner.next_action(observation)
        assert started.wait(timeout=1)
        observation["step"] = 2
        observation["image"][:] = 9
        release.set()
        deadline = time.monotonic() + 1
        while not runner.prediction_ready and time.monotonic() < deadline:
            time.sleep(0.001)
        assert runner.prediction_pending
        assert runner.prediction_ready
        runner.next_action(observation)
        assert seen[0][0] == 1
        assert np.array_equal(seen[0][1], [1.0, 2.0])
    finally:
        release.set()
        runner.close()


def test_async_runner_close_waits_for_running_worker_by_default():
    class Adapter:
        def __init__(self):
            self.calls = 0
            self.started = threading.Event()
            self.release = threading.Event()

        def infer_actions(self, observation):
            call = self.calls
            self.calls += 1
            if call == 0:
                return np.arange(2, dtype=np.float32)[:, None]
            self.started.set()
            self.release.wait(timeout=2.0)
            return np.arange(10, 12, dtype=np.float32)[:, None]

    adapter = Adapter()
    runner = AsyncTemporalFusionRunner(
        adapter, TemporalFusionConfig(action_hz=10, start_next_at=0))
    try:
        runner.reset({"chunk": 0})
        assert runner.next_action({"chunk": 1}).item() == 0
        assert adapter.started.wait(timeout=1.0)

        close_done = threading.Event()

        def close_runner():
            runner.close()
            close_done.set()

        close_thread = threading.Thread(target=close_runner)
        close_thread.start()
        assert not close_done.wait(timeout=0.05)
        adapter.release.set()
        close_thread.join(timeout=1.0)
        assert close_done.is_set()
    finally:
        adapter.release.set()
        runner.close(wait=False)


def test_async_runner_clears_failed_pending_future_and_ticket():
    class Adapter:
        def __init__(self):
            self.calls = 0

        def infer_actions(self, observation):
            call = self.calls
            self.calls += 1
            if call == 0:
                return np.arange(2, dtype=np.float32)[:, None]
            if call == 1:
                raise RuntimeError("worker failed")
            return np.arange(10, 12, dtype=np.float32)[:, None]

    runner = AsyncTemporalFusionRunner(
        Adapter(), TemporalFusionConfig(action_hz=10, start_next_at=0))
    try:
        runner.reset({"chunk": 0})
        assert runner.next_action({"chunk": 1}).item() == 0
        assert runner.prediction_pending
        deadline = time.monotonic() + 1.0
        while not runner.prediction_ready and time.monotonic() < deadline:
            time.sleep(0.001)
        assert runner.prediction_ready

        with pytest.raises(RuntimeError, match="worker failed"):
            runner.next_action({"chunk": 2})
        assert not runner.prediction_pending
        assert not runner.buffer._pending

        assert runner.next_action({"chunk": 3}).item() == 1
        assert runner.prediction_pending
    finally:
        runner.close()

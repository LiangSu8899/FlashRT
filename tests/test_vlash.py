import time

import numpy as np
import pytest

from flash_rt.runtime.vlash import AsyncVLAShRunner, VLAShConfig
from flash_rt.subgraphs.pi05.stage_plans import vlash


class RecordingAdapter:
    def __init__(self):
        self.states = []
        self.calls = 0

    def infer_actions(self, observation):
        self.states.append(np.asarray(observation["state"]).copy())
        call = self.calls
        self.calls += 1
        if call:
            time.sleep(0.01)
            return np.array([[100.0], [101.0], [102.0], [103.0]])
        return np.array([[1.0], [2.0], [3.0], [4.0]])


def test_vlash_projects_state_with_next_t_actions():
    runner = AsyncVLAShRunner(
        RecordingAdapter(), VLAShConfig(action_hz=20, lookahead_steps=2))
    try:
        projected, count = runner.project_state(
            np.array([10.0]),
            np.array([[1.0], [2.0], [3.0], [4.0]]),
            start_index=1,
        )
        assert count == 2
        assert np.array_equal(projected, [15.0])  # 10 + 2 + 3
    finally:
        runner.close()


def test_vlash_does_not_truncate_float_deltas_for_integer_state():
    runner = AsyncVLAShRunner(
        RecordingAdapter(), VLAShConfig(action_hz=20, lookahead_steps=1))
    try:
        projected, _ = runner.project_state(
            np.array([10], dtype=np.int32), np.array([[0.5]]), start_index=0)
        assert projected.dtype == np.float64
        assert np.array_equal(projected, [10.5])
    finally:
        runner.close()


def test_vlash_projection_supports_state_action_dimension_mapping():
    runner = AsyncVLAShRunner(
        RecordingAdapter(),
        VLAShConfig(action_hz=20, lookahead_steps=2,
                    state_action_indices=(0, 2)),
    )
    try:
        projected, count = runner.project_state(
            np.array([10.0, 20.0]),
            np.array([[1.0, 99.0, 2.0], [3.0, 99.0, 4.0]]),
            start_index=0,
        )
        assert count == 2
        assert np.array_equal(projected, [14.0, 26.0])
    finally:
        runner.close()


def test_vlash_uses_available_actions_when_fewer_than_t_remain():
    runner = AsyncVLAShRunner(
        RecordingAdapter(), VLAShConfig(action_hz=20, lookahead_steps=5))
    try:
        projected, count = runner.project_state(
            np.array([10.0]), np.array([[1.0], [2.0], [3.0]]),
            start_index=2)
        assert count == 1
        assert np.array_equal(projected, [13.0])
    finally:
        runner.close()


def test_vlash_action_quantization_is_disabled_by_default():
    runner = AsyncVLAShRunner(
        RecordingAdapter(), VLAShConfig(action_hz=20, lookahead_steps=1))
    actions = np.arange(10, dtype=np.float32).reshape(5, 2)
    try:
        quantized = runner.quantize_actions(actions)
        assert np.array_equal(quantized, actions)
        assert quantized is not actions
    finally:
        runner.close()


def test_vlash_action_quantization_sums_groups_and_keeps_tail():
    runner = AsyncVLAShRunner(
        RecordingAdapter(),
        VLAShConfig(
            action_hz=20,
            lookahead_steps=1,
            action_quantization_enabled=True,
            action_quantization_granularity=2,
        ),
    )
    actions = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
    ])
    try:
        quantized = runner.quantize_actions(actions)
        assert np.array_equal(quantized, [
            [3.0, 30.0],
            [7.0, 70.0],
            [5.0, 50.0],
        ])
    finally:
        runner.close()


@pytest.mark.parametrize("granularity", [0, -1])
def test_vlash_rejects_non_positive_quantization_granularity(granularity):
    with pytest.raises(ValueError, match="granularity must be positive"):
        VLAShConfig(
            action_hz=20,
            lookahead_steps=1,
            action_quantization_granularity=granularity,
        )


def test_async_vlash_stores_and_serves_quantized_actions():
    adapter = RecordingAdapter()
    runner = AsyncVLAShRunner(
        adapter,
        VLAShConfig(
            action_hz=20,
            lookahead_steps=1,
            action_quantization_enabled=True,
            action_quantization_granularity=2,
        ),
    )
    try:
        runner.reset({"state": np.array([0.0])}, state=[0.0])
        assert np.array_equal(runner.current_chunk.actions, [[3.0], [7.0]])
        assert runner.current_chunk.metadata["original_action_count"] == 4
        assert runner.current_chunk.metadata["stored_action_count"] == 2
        assert runner.next_action({"state": np.array([0.0])}, state=[0.0]).item() == 3
        assert runner.next_action({"state": np.array([0.0])}, state=[0.0]).item() == 7
    finally:
        runner.close()


def test_async_vlash_conditions_next_prediction_and_switches_at_zero():
    adapter = RecordingAdapter()
    original_observation = {"state": np.array([-1.0]), "frame": "frame"}
    runner = AsyncVLAShRunner(
        adapter,
        VLAShConfig(action_hz=20, lookahead_steps=2, start_next_at=1),
    )
    try:
        runner.reset(original_observation, state=np.array([0.0]))
        assert runner.next_action(original_observation, state=[0.0]).item() == 1

        # Submission happens before action index 1 is served.  The projected
        # state is current state 10 + old actions[1:3] = 10 + 2 + 3 = 15.
        assert runner.next_action(original_observation, state=[10.0]).item() == 2
        time.sleep(0.02)

        # VLASh starts the new trajectory at index zero; it does not skip by
        # measured inference latency.
        assert runner.next_action(original_observation, state=[15.0]).item() == 100
        assert np.array_equal(adapter.states[1], [15.0])
        assert np.array_equal(original_observation["state"], [-1.0])
        assert runner.current_chunk.projected_action_count == 2
    finally:
        runner.close()


def test_vlash_stage_plan_reuses_context_and_plain_decoder():
    plan = vlash(lookahead_steps=3)
    assert [stage.graph_name() for stage in plan.stages] == [
        "context", "decode_only"]
    assert plan.stages[1].after == ("context",)
    assert plan.metadata["lookahead_steps"] == 3

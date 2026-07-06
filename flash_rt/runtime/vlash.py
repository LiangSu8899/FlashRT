"""Lightweight asynchronous VLASh execution for action-chunk policies.

While the foreground consumes one action chunk, one background worker predicts
the next.  At prediction start, the runner estimates the robot state after the
next ``lookahead_steps`` actions from the active chunk::

    projected_state = start_state + sum(active_actions[i:i+t])

The projected state is inserted into a copy of the observation passed to the
model.  Because the new trajectory is conditioned on the estimated state at
prediction completion, a completed chunk is activated from index zero.  There
is deliberately no temporal fusion and no latency/state index search here.

Actions supplied by the adapter must be state deltas in the same units as the
robot state.  Use ``state_action_indices`` when action vectors contain extra
dimensions such as a gripper command.

Optional action quantization reduces the temporal horizon by summing every
``k`` consecutive delta actions.  Once enabled, horizon/index configuration is
expressed in quantized actions.  Summing is appropriate only for additive
delta actions; absolute targets and discrete commands need a task-specific
aggregation rule instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable

import numpy as np

from .rtc import ActionChunkAdapter


StateSetter = Callable[[Any, np.ndarray], Any]
ObservationSnapshotter = Callable[[Any], Any]


def _snapshot_observation(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _snapshot_observation(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_observation(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_observation(item) for item in value)
    return value


@dataclass(frozen=True)
class VLAShConfig:
    """Configuration for asynchronous projected-state chunk generation."""

    action_hz: float
    lookahead_steps: int
    action_horizon: int | None = None
    start_next_at: int | None = None
    miss_policy: str = "hold_last"
    state_key: str = "state"
    state_action_indices: tuple[int, ...] | None = None
    action_quantization_enabled: bool = False
    action_quantization_granularity: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.action_hz) or self.action_hz <= 0:
            raise ValueError("action_hz must be positive and finite")
        if self.lookahead_steps < 0:
            raise ValueError("lookahead_steps must be non-negative")
        if self.action_horizon is not None and self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.start_next_at is not None and self.start_next_at < 0:
            raise ValueError("start_next_at must be non-negative")
        if self.miss_policy not in {"hold_last", "block"}:
            raise ValueError("miss_policy must be 'hold_last' or 'block'")
        if not self.state_key:
            raise ValueError("state_key must be non-empty")
        if not isinstance(self.action_quantization_enabled, (bool, np.bool_)):
            raise TypeError("action_quantization_enabled must be a boolean")
        if (isinstance(self.action_quantization_granularity, (bool, np.bool_))
                or not isinstance(self.action_quantization_granularity,
                                  (int, np.integer))):
            raise TypeError(
                "action_quantization_granularity must be an integer")
        if self.action_quantization_granularity <= 0:
            raise ValueError(
                "action_quantization_granularity must be positive")
        object.__setattr__(self, "action_quantization_granularity",
                           int(self.action_quantization_granularity))
        if self.state_action_indices is not None:
            indices = tuple(int(i) for i in self.state_action_indices)
            if any(i < 0 for i in indices) or len(set(indices)) != len(indices):
                raise ValueError(
                    "state_action_indices must be unique and non-negative")
            object.__setattr__(self, "state_action_indices", indices)

    @property
    def period_s(self) -> float:
        return 1.0 / self.action_hz


@dataclass(frozen=True)
class VLAShChunkResult:
    actions: np.ndarray = field(repr=False)
    latency_s: float
    observation_time_s: float
    ready_time_s: float
    start_state: np.ndarray = field(repr=False)
    projected_state: np.ndarray = field(repr=False)
    projected_action_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VLAShStats:
    chunks_started: int = 0
    chunks_completed: int = 0
    actions_served: int = 0
    state_projections: int = 0
    deadline_misses: int = 0
    held_actions: int = 0
    swaps: int = 0
    last_latency_s: float = 0.0
    max_latency_s: float = 0.0


class AsyncVLAShRunner:
    """Consume actions while one worker predicts from a projected state.

    Call :meth:`next_action` once per robot tick with the latest observation
    and measured robot state.  The first call blocks for the initial chunk.
    Later calls launch prediction at ``start_next_at`` and continue serving the
    active chunk.  When the prediction completes, its action zero is served
    next.
    """

    def __init__(self, adapter: ActionChunkAdapter, config: VLAShConfig, *,
                 state_setter: StateSetter | None = None,
                 observation_snapshotter: ObservationSnapshotter | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.adapter = adapter
        self.config = config
        self.stats = VLAShStats()
        self._state_setter = state_setter or self._set_state_on_mapping
        self._observation_snapshotter = (
            observation_snapshotter or _snapshot_observation)
        self._clock = clock
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.RLock()
        self._current: VLAShChunkResult | None = None
        self._pending: Future[VLAShChunkResult] | None = None
        self._idx = 0
        self._last_action: np.ndarray | None = None
        self._closed = False

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> "AsyncVLAShRunner":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def current_chunk(self) -> VLAShChunkResult | None:
        with self._lock:
            return self._current

    def reset(self, observation: Any, *, state: Any) -> None:
        """Synchronously generate the initial chunk from the measured state."""
        self._raise_if_closed()
        measured = self._validate_state(state)
        prepared = self._state_setter(observation, measured)
        result = self._run_inference(
            prepared, start_state=measured, projected_state=measured,
            projected_count=0)
        with self._lock:
            self._current = result
            self._pending = None
            self._idx = 0
            self._last_action = None

    def next_action(self, observation: Any, *, state: Any,
                    block_if_empty: bool = True) -> np.ndarray:
        """Return one action while preparing the next VLASh chunk."""
        self._raise_if_closed()
        measured = self._validate_state(state)
        if self._current is None:
            if not block_if_empty:
                raise RuntimeError("VLASh runner has no current chunk")
            self.reset(observation, state=measured)

        with self._lock:
            self._promote_ready_locked()
            current = self._current
            if current is None:
                raise RuntimeError("VLASh runner failed to initialize")
            horizon = self._horizon(current.actions)
            if self._idx >= self._start_next_at(horizon):
                self._submit_locked(observation, measured)
            if self._idx >= horizon:
                self._handle_exhausted_locked(observation, measured)
                current = self._current
                if current is None:
                    raise RuntimeError("VLASh runner has no chunk after exhaustion")
            action = np.asarray(current.actions[self._idx]).copy()
            self._idx += 1
            self._last_action = action
            self.stats.actions_served += 1
            return action

    def _submit_locked(self, observation: Any, state: np.ndarray) -> None:
        if self._pending is not None:
            return
        current = self._current
        if current is None:
            raise RuntimeError("cannot project state without an active chunk")
        projected, count = self.project_state(
            state, current.actions, start_index=self._idx)
        snapshot = self._observation_snapshotter(observation)
        prepared = self._state_setter(snapshot, projected)
        self.stats.chunks_started += 1
        self.stats.state_projections += 1
        self._pending = self._executor.submit(
            self._run_inference, prepared,
            start_state=state.copy(), projected_state=projected,
            projected_count=count)

    def project_state(self, state: Any, actions: Any, *,
                      start_index: int) -> tuple[np.ndarray, int]:
        """Integrate the next ``t`` stored action deltas into measured state.

        With action quantization enabled, each stored action is already the sum
        of up to ``action_quantization_granularity`` model actions, and ``t``
        therefore counts quantized actions.
        """
        measured = self._validate_state(state)
        chunk = np.asarray(actions)
        if chunk.ndim != 2:
            raise ValueError("actions must be [horizon, action_dim]")
        begin = max(0, int(start_index))
        end = min(chunk.shape[0], begin + self.config.lookahead_steps)
        selected = chunk[begin:end]
        deltas = self._state_deltas(selected, measured.size)
        projected = measured + np.sum(deltas, axis=0, dtype=np.float64)
        return projected.astype(measured.dtype, copy=False), end - begin

    def _run_inference(self, observation: Any, *, start_state: np.ndarray,
                       projected_state: np.ndarray,
                       projected_count: int) -> VLAShChunkResult:
        t0 = self._clock()
        actions = np.asarray(self.adapter.infer_actions(observation))
        t1 = self._clock()
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or not actions.shape[0] or not actions.shape[1]:
            raise ValueError(
                f"expected action chunk [horizon, action_dim], got {actions.shape}")
        original_action_count = int(actions.shape[0])
        actions = self.quantize_actions(actions)
        return VLAShChunkResult(
            actions=actions,
            latency_s=t1 - t0,
            observation_time_s=t0,
            ready_time_s=t1,
            start_state=start_state.copy(),
            projected_state=projected_state.copy(),
            projected_action_count=projected_count,
            metadata={
                "action_quantization_enabled":
                    bool(self.config.action_quantization_enabled),
                "action_quantization_granularity":
                    self.config.action_quantization_granularity,
                "original_action_count": original_action_count,
                "stored_action_count": int(actions.shape[0]),
            },
        )

    def quantize_actions(self, actions: Any) -> np.ndarray:
        """Return a chunk with every consecutive ``k`` actions summed.

        The final group is retained even when it contains fewer than ``k``
        actions.  Quantization is applied along the horizon axis only, so the
        action dimension is unchanged.  A copy is returned when quantization
        is disabled, keeping model-owned output buffers isolated from runtime
        mutation.
        """
        chunk = np.asarray(actions)
        if chunk.ndim != 2 or not chunk.shape[0] or not chunk.shape[1]:
            raise ValueError(
                f"expected action chunk [horizon, action_dim], got {chunk.shape}")
        if not self.config.action_quantization_enabled:
            return np.array(chunk, copy=True)

        granularity = self.config.action_quantization_granularity
        groups = [
            np.sum(chunk[start:start + granularity], axis=0)
            for start in range(0, chunk.shape[0], granularity)
        ]
        return np.stack(groups, axis=0)

    def _promote_ready_locked(self) -> bool:
        if self._pending is None or not self._pending.done():
            return False
        self._activate_pending_result_locked()
        return True

    def _handle_exhausted_locked(self, observation: Any,
                                 state: np.ndarray) -> None:
        if self._promote_ready_locked():
            return
        self.stats.deadline_misses += 1
        if self.config.miss_policy == "block":
            self._submit_locked(observation, state)
            if self._pending is None:
                raise RuntimeError("failed to submit recovery chunk")
            self._activate_pending_result_locked()
            return
        if self._last_action is None:
            raise RuntimeError("cannot hold before any action was served")
        self.stats.held_actions += 1
        now = self._clock()
        self._current = VLAShChunkResult(
            actions=self._last_action[None, :].copy(), latency_s=0.0,
            observation_time_s=now, ready_time_s=now,
            start_state=state.copy(), projected_state=state.copy(),
            projected_action_count=0, metadata={"held": True})
        self._idx = 0

    def _activate_pending_result_locked(self) -> None:
        result = self._consume_pending_result_locked()
        self._current = result
        self._idx = 0  # VLASh trajectory starts at prediction completion.
        self.stats.chunks_completed += 1
        self.stats.swaps += 1
        self.stats.last_latency_s = result.latency_s
        self.stats.max_latency_s = max(
            self.stats.max_latency_s, result.latency_s)

    def _consume_pending_result_locked(self) -> VLAShChunkResult:
        if self._pending is None:
            raise RuntimeError("VLASh runner has no pending chunk")
        try:
            result = self._pending.result()
        except BaseException:
            self._pending = None
            raise
        self._pending = None
        return result

    def _state_deltas(self, actions: np.ndarray,
                      state_dim: int) -> np.ndarray:
        indices = self.config.state_action_indices
        if indices is None:
            if actions.shape[1] < state_dim:
                raise ValueError(
                    f"action dimension {actions.shape[1]} is smaller than "
                    f"state dimension {state_dim}")
            return np.asarray(actions[:, :state_dim], dtype=np.float64)
        if len(indices) != state_dim:
            raise ValueError(
                "state_action_indices length must equal state dimension")
        if indices and max(indices) >= actions.shape[1]:
            raise ValueError("state_action_indices exceed action dimension")
        return np.asarray(actions[:, indices], dtype=np.float64)

    def _set_state_on_mapping(self, observation: Any,
                              state: np.ndarray) -> Any:
        if not isinstance(observation, Mapping):
            raise TypeError(
                "default VLASh state injection requires a mapping "
                "observation; pass state_setter for custom types")
        copied = dict(observation)
        copied[self.config.state_key] = state.copy()
        return copied

    @staticmethod
    def _validate_state(state: Any) -> np.ndarray:
        result = np.asarray(state)
        if result.ndim != 1 or not result.size:
            raise ValueError("state must be a non-empty 1D vector")
        if not np.issubdtype(result.dtype, np.number):
            raise TypeError("state must have a numeric dtype")
        if not np.all(np.isfinite(result)):
            raise ValueError("state must contain only finite values")
        dtype = result.dtype if np.issubdtype(result.dtype, np.floating) else np.float64
        return np.array(result, dtype=dtype, copy=True)

    def _horizon(self, actions: np.ndarray) -> int:
        horizon = int(actions.shape[0])
        if self.config.action_horizon is not None:
            horizon = min(horizon, self.config.action_horizon)
        return horizon

    def _start_next_at(self, horizon: int) -> int:
        if self.config.start_next_at is not None:
            return min(self.config.start_next_at, horizon)
        return max(1, horizon // 2)

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("VLASh runner is closed")


__all__ = [
    "AsyncVLAShRunner",
    "ObservationSnapshotter",
    "StateSetter",
    "VLAShChunkResult",
    "VLAShConfig",
    "VLAShStats",
]

"""Temporal fusion and asynchronous execution for action-chunk policies.

Actions are aligned on a controller-step grid.  If a chunk starts at global
step ``s``, action ``i`` addresses step ``s + i``.  Overlapping raw predictions
are fused with ``exp(-decay * abs(old_index - latest_index))`` weights.  Fused
values never replace raw predictions; raw chunks live until their final step
expires.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Callable, Literal, Mapping

import numpy as np

from .rtc import ActionChunkAdapter


SwitchMode = Literal["latency", "state"]
ObservationSnapshotter = Callable[[Any], Any]


@dataclass(frozen=True)
class TemporalFusionConfig:
    action_hz: float
    max_chunks: int = 3
    decay: float = 0.1
    switch_mode: SwitchMode = "latency"
    action_representation: Literal["absolute", "delta"] = "absolute"
    delta_mode: Literal["cumulative", "from_start"] = "cumulative"
    distance_metric: Literal["l1", "l2"] = "l1"
    state_action_indices: tuple[int, ...] | None = None
    start_next_at: int | None = None
    action_horizon: int | None = None
    miss_policy: Literal["hold_last", "block"] = "hold_last"
    epoch_s: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.action_hz) or self.action_hz <= 0:
            raise ValueError("action_hz must be positive and finite")
        if self.max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        if not math.isfinite(self.decay) or self.decay < 0:
            raise ValueError("decay must be non-negative and finite")
        if self.switch_mode not in {"latency", "state"}:
            raise ValueError("switch_mode must be 'latency' or 'state'")
        if self.action_representation not in {"absolute", "delta"}:
            raise ValueError("action_representation must be 'absolute' or 'delta'")
        if self.delta_mode not in {"cumulative", "from_start"}:
            raise ValueError("delta_mode must be 'cumulative' or 'from_start'")
        if self.distance_metric not in {"l1", "l2"}:
            raise ValueError("distance_metric must be 'l1' or 'l2'")
        if self.start_next_at is not None and self.start_next_at < 0:
            raise ValueError("start_next_at must be non-negative")
        if self.action_horizon is not None and self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.miss_policy not in {"hold_last", "block"}:
            raise ValueError("miss_policy must be 'hold_last' or 'block'")
        if self.epoch_s is not None and not math.isfinite(self.epoch_s):
            raise ValueError("epoch_s must be finite")
        if self.state_action_indices is not None:
            indices = tuple(int(i) for i in self.state_action_indices)
            if any(i < 0 for i in indices) or len(set(indices)) != len(indices):
                raise ValueError("state_action_indices must be unique and non-negative")
            object.__setattr__(self, "state_action_indices", indices)

    @property
    def period_s(self) -> float:
        return 1.0 / self.action_hz


@dataclass(frozen=True)
class PredictionTicket:
    prediction_id: int
    started_time_s: float
    start_step: int
    start_state: np.ndarray | None = field(repr=False)


@dataclass(frozen=True)
class TimedActionChunk:
    prediction_id: int
    actions: np.ndarray = field(repr=False)
    started_time_s: float
    ready_time_s: float
    start_step: int
    start_state: np.ndarray | None = field(repr=False)

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def end_step_exclusive(self) -> int:
        return self.start_step + self.horizon

    def covers(self, step: int) -> bool:
        return self.start_step <= step < self.end_step_exclusive


@dataclass(frozen=True)
class FusedChunk:
    prediction_id: int
    actions: np.ndarray = field(repr=False)
    start_step: int
    expected_times_s: np.ndarray = field(repr=False)
    source_counts: np.ndarray = field(repr=False)
    switch_index: int
    started_time_s: float
    ready_time_s: float

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])


@dataclass
class TemporalFusionStats:
    predictions_started: int = 0
    predictions_completed: int = 0
    actions_served: int = 0
    chunk_switches: int = 0
    deadline_misses: int = 0
    held_actions: int = 0
    raw_chunks_pruned: int = 0
    last_latency_s: float = 0.0
    max_latency_s: float = 0.0


def _frozen_array(value: Any, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


class TemporalFusionBuffer:
    """Record raw predictions and create step-aligned fused chunks."""

    def __init__(self, config: TemporalFusionConfig, *,
                 clock: Callable[[], float] = time.monotonic):
        self.config = config
        self._clock = clock
        self._epoch_s = float(config.epoch_s) if config.epoch_s is not None else clock()
        self._next_id = 0
        self._pending: dict[int, PredictionTicket] = {}
        self._chunks: list[TimedActionChunk] = []
        self._latest: FusedChunk | None = None
        self._total_pruned = 0
        self._lock = threading.RLock()

    @property
    def epoch_s(self) -> float:
        return self._epoch_s

    @property
    def raw_chunks(self) -> tuple[TimedActionChunk, ...]:
        with self._lock:
            return tuple(self._chunks)

    @property
    def latest(self) -> FusedChunk | None:
        with self._lock:
            return self._latest

    @property
    def total_pruned(self) -> int:
        with self._lock:
            return self._total_pruned

    def step_at(self, timestamp_s: float) -> int:
        value = float(timestamp_s)
        if not math.isfinite(value):
            raise ValueError("timestamp must be finite")
        return math.floor((value - self._epoch_s) / self.config.period_s + 1e-9)

    def time_at(self, step: int) -> float:
        return self._epoch_s + int(step) * self.config.period_s

    def begin_prediction(self, *, state: Any | None = None,
                         started_at: float | None = None,
                         start_step: int | None = None) -> PredictionTicket:
        started = self._clock() if started_at is None else float(started_at)
        if not math.isfinite(started):
            raise ValueError("started_at must be finite")
        step = self.step_at(started) if start_step is None else int(start_step)
        start_state = None if state is None else _frozen_array(state, 1, "state")
        with self._lock:
            ticket = PredictionTicket(self._next_id, started, step, start_state)
            self._next_id += 1
            self._pending[ticket.prediction_id] = ticket
            return ticket

    def cancel_prediction(self, ticket: PredictionTicket) -> None:
        with self._lock:
            self._pending.pop(ticket.prediction_id, None)

    def complete_prediction(self, ticket: PredictionTicket, actions: Any, *,
                            ready_at: float | None = None,
                            switch_at: float | None = None,
                            current_state: Any | None = None,
                            switch_mode: SwitchMode | None = None) -> FusedChunk:
        ready = self._clock() if ready_at is None else float(ready_at)
        if not math.isfinite(ready) or ready < ticket.started_time_s:
            raise ValueError("ready_at must be finite and not precede started_at")
        switch_time = ready if switch_at is None else float(switch_at)
        if not math.isfinite(switch_time) or switch_time < ticket.started_time_s:
            raise ValueError("switch_at must be finite and not precede started_at")
        raw = _frozen_array(actions, 2, "actions")
        if not raw.shape[0] or not raw.shape[1]:
            raise ValueError("actions must have non-zero horizon and dimension")
        if self.config.action_horizon is not None:
            raw = _frozen_array(raw[:self.config.action_horizon], 2, "actions")

        with self._lock:
            registered = self._pending.pop(ticket.prediction_id, None)
            if registered is not ticket:
                raise ValueError("unknown, cancelled, or completed prediction ticket")
            self._prune_locked(ready)
            if self._chunks and self._chunks[-1].action_dim != raw.shape[1]:
                raise ValueError("action dimension changed between predictions")
            chunk = TimedActionChunk(ticket.prediction_id, raw,
                                     ticket.started_time_s, ready,
                                     ticket.start_step, ticket.start_state)
            self._chunks.append(chunk)
            fused, counts = self._fuse_locked(chunk)
            index = self._switch_index(chunk, fused, switch_time,
                                       current_state,
                                       switch_mode or self.config.switch_mode)
            times = np.array([self.time_at(chunk.start_step + i)
                              for i in range(chunk.horizon)], dtype=np.float64)
            for array in (fused, counts, times):
                array.setflags(write=False)
            result = FusedChunk(chunk.prediction_id, fused, chunk.start_step,
                                times, counts, index,
                                chunk.started_time_s, chunk.ready_time_s)
            self._latest = result
            return result

    def prune_expired(self, *, now: float | None = None) -> int:
        with self._lock:
            return self._prune_locked(self._clock() if now is None else float(now))

    def _prune_locked(self, now: float) -> int:
        current = self.step_at(now)
        before = len(self._chunks)
        self._chunks[:] = [c for c in self._chunks if c.end_step_exclusive > current]
        pruned = before - len(self._chunks)
        self._total_pruned += pruned
        return pruned

    def _fuse_locked(self, latest: TimedActionChunk) -> tuple[np.ndarray, np.ndarray]:
        fused = np.empty(latest.actions.shape,
                         dtype=np.result_type(latest.actions.dtype, np.float32))
        counts = np.empty(latest.horizon, dtype=np.int32)
        newest_first = list(reversed(self._chunks))
        for latest_i in range(latest.horizon):
            step = latest.start_step + latest_i
            candidates: list[tuple[TimedActionChunk, int]] = []
            for chunk in newest_first:
                if chunk.covers(step):
                    candidates.append((chunk, step - chunk.start_step))
                    if len(candidates) == self.config.max_chunks:
                        break
            weighted = np.zeros(latest.action_dim, dtype=np.float64)
            weight_sum = 0.0
            for chunk, source_i in candidates:
                weight = math.exp(-self.config.decay * abs(source_i - latest_i))
                weighted += weight * np.asarray(chunk.actions[source_i], dtype=np.float64)
                weight_sum += weight
            fused[latest_i] = weighted / weight_sum
            counts[latest_i] = len(candidates)
        return fused, counts

    def _switch_index(self, latest: TimedActionChunk, fused: np.ndarray,
                      switch_time: float, current_state: Any | None,
                      mode: SwitchMode) -> int:
        if mode == "latency":
            return int(np.clip(self.step_at(switch_time) - latest.start_step,
                               0, latest.horizon - 1))
        if mode != "state":
            raise ValueError(f"unknown switch mode {mode!r}")
        if current_state is None:
            raise ValueError("current_state is required for state switching")
        state = np.asarray(current_state, dtype=np.float64)
        if state.ndim != 1 or not state.size or not np.all(np.isfinite(state)):
            raise ValueError("current_state must be a finite, non-empty 1D vector")
        action_state = self._action_state(fused, state.size)
        if self.config.action_representation == "absolute":
            estimates = action_state
        else:
            if latest.start_state is None:
                raise ValueError("prediction start state is required for delta actions")
            base = np.asarray(latest.start_state, dtype=np.float64)
            if base.shape != state.shape:
                raise ValueError("prediction start state and current state shapes differ")
            changes = (np.cumsum(action_state, axis=0)
                       if self.config.delta_mode == "cumulative" else action_state)
            estimates = base[None, :] + changes
        delta = estimates - state[None, :]
        distances = (np.sum(np.abs(delta), axis=1)
                     if self.config.distance_metric == "l1"
                     else np.linalg.norm(delta, axis=1))
        return int(np.argmin(distances))

    def _action_state(self, actions: np.ndarray, state_size: int) -> np.ndarray:
        indices = self.config.state_action_indices
        if indices is None:
            if actions.shape[1] < state_size:
                raise ValueError("action dimension is smaller than state dimension")
            return np.asarray(actions[:, :state_size], dtype=np.float64)
        if len(indices) != state_size or (indices and max(indices) >= actions.shape[1]):
            raise ValueError("state_action_indices do not match state/action dimensions")
        return np.asarray(actions[:, indices], dtype=np.float64)


@dataclass(frozen=True)
class _PredictionOutput:
    ticket: PredictionTicket
    actions: np.ndarray
    ready_time_s: float


class AsyncTemporalFusionRunner:
    """Consume an active fused chunk while one worker predicts the next.

    The default observation snapshotter recursively copies mappings, sequences,
    and NumPy arrays before handing them to the worker. This prevents a camera
    or state buffer updated by the foreground loop from changing an in-flight
    prediction. Supply ``observation_snapshotter`` for custom observation types
    such as device-buffer wrappers.
    """

    def __init__(self, adapter: ActionChunkAdapter, config: TemporalFusionConfig,
                 *, clock: Callable[[], float] = time.monotonic,
                 observation_snapshotter: ObservationSnapshotter | None = None):
        self.adapter = adapter
        self.config = config
        self.buffer = TemporalFusionBuffer(config, clock=clock)
        self.stats = TemporalFusionStats()
        self._clock = clock
        self._observation_snapshotter = (
            observation_snapshotter or _snapshot_observation)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: Future[_PredictionOutput] | None = None
        self._active: FusedChunk | None = None
        self._index = 0
        self._last_action: np.ndarray | None = None
        self._closed = False
        self._lock = threading.RLock()

    @property
    def active_chunk(self) -> FusedChunk | None:
        with self._lock:
            return self._active

    @property
    def prediction_pending(self) -> bool:
        """Whether a background prediction has been submitted."""
        with self._lock:
            return self._pending is not None

    @property
    def prediction_ready(self) -> bool:
        """Whether the submitted prediction can be promoted without blocking."""
        with self._lock:
            return self._pending is not None and self._pending.done()

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
            if self._pending is not None and not self._pending.done():
                ticket = getattr(self._pending, "ticket", None)
                if ticket is not None:
                    self.buffer.cancel_prediction(ticket)
                self._pending.cancel()
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> "AsyncTemporalFusionRunner":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def reset(self, observation: Any, *, state: Any | None = None) -> None:
        self._check_open()
        ticket = self.buffer.begin_prediction(state=state)
        self.stats.predictions_started += 1
        try:
            actions = self.adapter.infer_actions(observation)
            ready = self._clock()
            fused = self.buffer.complete_prediction(
                ticket, actions, ready_at=ready, current_state=state)
        except BaseException:
            self.buffer.cancel_prediction(ticket)
            raise
        with self._lock:
            self._activate(fused)
            self._last_action = None
            self._record_completion(ticket, ready, switched=False)

    def next_action(self, observation: Any, *, state: Any | None = None,
                    block_if_empty: bool = True) -> np.ndarray:
        self._check_open()
        if self._active is None:
            if not block_if_empty:
                raise RuntimeError("temporal fusion runner has no active chunk")
            self.reset(observation, state=state)
        with self._lock:
            self._promote_ready(state)
            active = self._active
            assert active is not None
            trigger = (min(self.config.start_next_at, active.horizon)
                       if self.config.start_next_at is not None
                       else max(1, active.horizon // 2))
            if self._index >= trigger:
                self._submit(observation, state)
            if self._index >= active.horizon:
                self._handle_exhausted(observation, state)
                active = self._active
                assert active is not None
            action = np.asarray(active.actions[self._index]).copy()
            self._index += 1
            self._last_action = action
            self.stats.actions_served += 1
            return action

    def _run(self, ticket: PredictionTicket, observation: Any) -> _PredictionOutput:
        return _PredictionOutput(ticket,
                                 np.asarray(self.adapter.infer_actions(observation)),
                                 self._clock())

    def _submit(self, observation: Any, state: Any | None) -> None:
        if self._pending is not None:
            return
        ticket = self.buffer.begin_prediction(state=state)
        self.stats.predictions_started += 1
        snapshot = self._observation_snapshotter(observation)
        self._pending = self._executor.submit(self._run, ticket, snapshot)
        self._pending.ticket = ticket

    def _promote_ready(self, state: Any | None) -> bool:
        if self._pending is None or not self._pending.done():
            return False
        output = self._consume_pending_result_locked()
        fused = self.buffer.complete_prediction(
            output.ticket, output.actions,
            ready_at=output.ready_time_s, switch_at=self._clock(),
            current_state=state)
        self._activate(fused)
        self._record_completion(output.ticket, output.ready_time_s, switched=True)
        return True

    def _handle_exhausted(self, observation: Any, state: Any | None) -> None:
        if self._promote_ready(state):
            return
        self.stats.deadline_misses += 1
        if self.config.miss_policy == "block":
            self._submit(observation, state)
            assert self._pending is not None
            output = self._consume_pending_result_locked()
            fused = self.buffer.complete_prediction(
                output.ticket, output.actions,
                ready_at=output.ready_time_s, switch_at=self._clock(),
                current_state=state)
            self._activate(fused)
            self._record_completion(output.ticket, output.ready_time_s, switched=True)
            return
        if self._last_action is None:
            raise RuntimeError("cannot hold before any action was served")
        self.stats.held_actions += 1
        now = self._clock()
        held = np.asarray(self._last_action)[None, :].copy()
        held.setflags(write=False)
        self._active = FusedChunk(-1, held, self.buffer.step_at(now),
                                  np.asarray([now]), np.asarray([0]), 0, now, now)
        self._index = 0

    def _consume_pending_result_locked(self) -> _PredictionOutput:
        if self._pending is None:
            raise RuntimeError("temporal fusion runner has no pending prediction")
        try:
            output = self._pending.result()
        except BaseException:
            ticket = getattr(self._pending, "ticket", None)
            if ticket is not None:
                self.buffer.cancel_prediction(ticket)
            self._pending = None
            raise
        self._pending = None
        return output

    def _activate(self, fused: FusedChunk) -> None:
        self._active = fused
        self._index = fused.switch_index

    def _record_completion(self, ticket: PredictionTicket, ready: float,
                           *, switched: bool) -> None:
        latency = ready - ticket.started_time_s
        self.stats.predictions_completed += 1
        self.stats.chunk_switches += int(switched)
        self.stats.last_latency_s = latency
        self.stats.max_latency_s = max(self.stats.max_latency_s, latency)
        self.buffer.prune_expired(now=ready)
        self.stats.raw_chunks_pruned = self.buffer.total_pruned

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("temporal fusion runner is closed")


def _snapshot_observation(value: Any) -> Any:
    """Copy common mutable observation containers for worker ownership."""
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, Mapping):
        return {key: _snapshot_observation(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_observation(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_observation(item) for item in value)
    return value


__all__ = [
    "ActionChunkAdapter", "AsyncTemporalFusionRunner", "FusedChunk",
    "ObservationSnapshotter",
    "PredictionTicket", "TemporalFusionBuffer", "TemporalFusionConfig",
    "TemporalFusionStats", "TimedActionChunk",
]

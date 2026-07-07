# RTC temporal fusion

`flash_rt.runtime.rtc_temporal_fusion` provides an optional host-side runtime
for action-chunk policies. It predicts the next chunk on one background worker
while the foreground continues serving the active chunk, then fuses overlapping
raw predictions on a controller-step timeline.

This module is policy/runtime code. It does not change model kernels, CUDA graph
capture, or the Pi0.5 decoder.

## Timeline and fusion

For an action rate `f`, the controller period is `dt = 1 / f`. A prediction
started at time `t` receives a global start step

```text
start_step = floor((t - epoch) / dt)
```

Action `i` in that chunk addresses `start_step + i`. For a step covered by
several raw chunks, at most `max_chunks` newest chunks are used. If `i_new` is
the position in the newest chunk and `i_old` is the position in another chunk,
its weight is

```text
w = exp(-decay * abs(i_old - i_new))
```

The fused action is the normalized weighted mean. Fusion never overwrites raw
predictions. A raw chunk remains available until its final controller step has
expired, allowing more than two predictions to contribute.

## Basic use

The adapter boundary deliberately matches `flash_rt.runtime.rtc`: it receives
an observation and returns one numeric array shaped `[horizon, action_dim]`.

```python
import numpy as np

from flash_rt.runtime import (
    AsyncTemporalFusionRunner,
    TemporalFusionConfig,
)


class Pi05Adapter:
    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt

    def infer_actions(self, observation):
        return np.asarray(self.model.predict(
            images=observation["images"],
            prompt=self.prompt,
            state=observation.get("state"),
        ))


runner = AsyncTemporalFusionRunner(
    Pi05Adapter(model, "pick up the red block"),
    TemporalFusionConfig(
        action_hz=20,
        max_chunks=3,
        decay=0.1,
        start_next_at=25,
        switch_mode="latency",
    ),
)

observation = {"images": images}
try:
    while control_loop_running:
        action = runner.next_action(observation)
        robot.send_action(action)
finally:
    runner.close()
```

The first call is synchronous because no action exists yet. Later predictions
run on a single background thread. The adapter/model must therefore permit
inference from that worker thread. Only one model inference is in flight. Before
submission, the default snapshotter recursively copies mappings, lists, tuples,
and NumPy arrays so foreground camera/state updates cannot alter the in-flight
observation. Pass `observation_snapshotter=` for custom device-buffer objects.

`prediction_pending` and `prediction_ready` expose read-only worker state for
monitoring without reaching into executor internals.

## Chunk switching

`switch_mode="latency"` selects the position corresponding to the foreground
time at which the completed future is promoted. It accounts for both inference
latency and a delayed foreground poll.

`switch_mode="state"` selects the fused action whose estimated state is closest
to the current robot state:

- `action_representation="absolute"`: selected action dimensions are compared
  directly with state.
- `action_representation="delta"`: state is estimated from the state recorded
  when prediction started. `delta_mode="cumulative"` cumulatively integrates
  deltas; `"from_start"` treats each action as a displacement from that state.
- `state_action_indices` maps state dimensions when action vectors contain
  extra dimensions.
- `distance_metric` selects L1 or L2 distance.

Pass the latest state on every controller call when state switching is enabled:

```python
action = runner.next_action(observation, state=current_robot_state)
```

## Deadline policy

The next prediction starts at `start_next_at`; the default is half of the
active horizon. If the active chunk expires first:

- `miss_policy="hold_last"` repeats the last served action;
- `miss_policy="block"` waits for the pending prediction.

`runner.stats` reports prediction counts, switches, misses, held actions,
latency, and expired raw chunks.

## Configuration summary

| Option | Default | Meaning |
|---|---:|---|
| `action_hz` | required | Controller/action frequency |
| `max_chunks` | `3` | Maximum overlapping raw chunks per fused action |
| `decay` | `0.1` | Exponential position-difference decay |
| `switch_mode` | `"latency"` | Latency- or state-based chunk switch |
| `start_next_at` | half horizon | Active-chunk index that starts prediction |
| `action_horizon` | model horizon | Optional maximum raw chunk length |
| `miss_policy` | `"hold_last"` | Behavior when prediction misses the horizon |

## Validation

CPU-only mechanism tests:

```bash
PYTHONPATH=. python -m pytest tests/test_rtc_temporal_fusion.py -q
```

Reproduce the `3 x 50 x 32` host-side fusion microbenchmark with:

```bash
PYTHONPATH=. python tests/bench_rtc_temporal_fusion.py \
  --warmup 100 --iterations 1000
```

The real Pi0.5 gate loads a checkpoint, executes the RTX SM89 model, predicts a
second chunk on the background worker, and checks that two real model outputs
are step-aligned and fused according to the exponential formula:

```bash
export CUDA_VISIBLE_DEVICES=7
export PI05_CKPT=/path/to/pi05_libero_pytorch
PYTHONPATH=. python -m pytest \
  tests/test_rtc_temporal_fusion_pi05.py -q -s
```

For a clean local SM89 validation environment, keep the current checkout first
on `PYTHONPATH` and point `PI05_CKPT` at a local Pi0.5 PyTorch checkpoint:

```bash
cd "$FLASHRT_CHECKOUT"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PI05_CKPT=/path/to/pi05_libero_pytorch
export CUDA_HOME=/path/to/cuda-12.x
export CUDA_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$CONDA_PREFIX/lib"
export FLASHRT_CUDART_PATH="$CUDA_HOME/lib64/libcudart.so.12"
PYTHONPATH=$PWD \
  python -m pytest tests/test_rtc_temporal_fusion_pi05.py -q -s
```

This command imports FlashRT from the current checkout only. Build or install
the required CUDA extensions in the active Conda environment before running the
gate.

The test skips when CUDA or `PI05_CKPT` is unavailable. `RTC_PI05_AUTOTUNE`
controls Pi0.5 autotune trials and defaults to `0` for a fast gate.
It can also run without pytest, which is useful in minimal deployment Conda
environments:

```bash
PYTHONPATH=. python tests/test_rtc_temporal_fusion_pi05.py
```

This gate validates model/runtime integration, not closed-loop task success.
Before robot deployment, validate action cadence, camera/state freshness,
deadline policy, and control quality in the target simulator or robot stack.

# VLASh runtime

`flash_rt.runtime.vlash` provides an optional host-side runtime for action-chunk
policies that condition the next prediction on a projected future robot state.
It does not change model kernels, CUDA graph capture, or decoder numerics.

While the foreground consumes the active chunk, one background worker predicts
the next chunk. Before starting that worker, VLASh estimates the robot state
after the next `lookahead_steps` actions from the active chunk:

```text
projected_state = measured_state + sum(active_actions[i : i + lookahead_steps])
```

The projected state is inserted into a copied observation and passed to the
adapter. For Pi0.5, the adapter normally forwards it to
`model.predict(..., state=projected_state)`, so the state is rendered into the
prompt before the context graph runs. When the background chunk is ready, VLASh
activates it from action index zero. There is no temporal fusion and no
latency-based index skip.

Background submissions snapshot mapping, list, tuple, and NumPy-array
observation values before state injection. Deployments that own immutable or
device-resident buffers can pass a custom `observation_snapshotter` to
`AsyncVLAShRunner`.

## Basic use

The adapter boundary matches the existing action-chunk runtime: it receives an
observation and returns a numeric array shaped `[horizon, action_dim]`.

```python
import numpy as np

from flash_rt.runtime import AsyncVLAShRunner, VLAShConfig


class Pi05Adapter:
    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt

    def infer_actions(self, observation):
        return np.asarray(self.model.predict(
            images=observation["images"],
            prompt=self.prompt,
            state=observation["state"],
        ))


runner = AsyncVLAShRunner(
    Pi05Adapter(model, "pick up the red block"),
    VLAShConfig(
        action_hz=20,
        lookahead_steps=2,
        start_next_at=25,
        state_action_indices=tuple(range(7)),
    ),
)

observation = {"images": images}
try:
    runner.reset(observation, state=current_robot_state)
    while control_loop_running:
        action = runner.next_action(observation, state=current_robot_state)
        robot.send_action(action)
finally:
    runner.close()
```

Actions used for projection must be deltas in the same units as the measured
robot state. If an action vector has extra dimensions such as gripper commands,
set `state_action_indices` to choose the action dimensions that correspond to
state dimensions.

## Action quantization

Set `action_quantization_enabled=True` to store every consecutive group of
`action_quantization_granularity` actions as a summed delta. This reduces the
stored horizon and makes `lookahead_steps`, `start_next_at`, and
`action_horizon` count quantized actions. This is appropriate only for additive
delta actions; absolute targets and discrete commands need a task-specific
aggregation rule.

## Pi0.5 subgraph plan

The Pi0.5 `vlash` stage plan reuses the existing `context -> decode_only` graph
split. The upper runtime performs the state projection before context replay;
the subgraph helper only enables the required context/action graph capture.
For live state prompts, use a state-prompt mode whose shapes are graph-safe,
such as `state_prompt_mode="fixed"` or prewarmed exact buckets.

## Validation

CPU-only mechanism tests:

```bash
PYTHONPATH=. python -m pytest tests/test_vlash.py -q
```

Real Pi0.5 checkpoint gate on an SM89 GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
export PI05_CKPT=/path/to/pi05_libero_pytorch
PYTHONPATH=. python -m pytest tests/test_vlash_pi05.py -q -s
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
  python -m pytest tests/test_vlash_pi05.py -q -s
```

The checkpoint gate skips when CUDA or `PI05_CKPT` is unavailable. It validates
model/runtime integration, projected-state injection, asynchronous worker
completion, and index-zero chunk activation. It does not measure closed-loop
task success; validate cadence, state freshness, and control quality in the
target simulator or robot stack before deployment.

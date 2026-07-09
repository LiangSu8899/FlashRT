# Pi0.5 Native Model Runtime IO Contract

This document is the deployment-facing IO contract for the Pi0.5 native
model-runtime face. It is intentionally limited to the public runtime/model
runtime ABI:

- `frt_runtime_export_v1` in `runtime/include/flashrt/runtime.h`
- `frt_model_runtime_v1` in `runtime/include/flashrt/model_runtime.h`
- Pi0.5 producer declarations in `flash_rt/models/pi05/runtime_export.py`
- Pi0.5 native verb overlay in `cpp/models/pi05/`

It does not freeze the C++ implementation classes under `cpp/`. Those classes
may evolve as long as the exported ports, stages, regions, identity, and hot
contract remain valid.

## Current Native Face

The current Pi0.5 `io="native"` export declares three host-visible ports.
This is the contract implemented by `frt_pi05_model_runtime_create_over`.

| port | modality/update | direction | dtype/layout/shape | backing |
|---|---|---|---|---|
| `images` | `IMAGE/STAGED` | input | device tensor dtype, `NHWC`, `(num_views, 224, 224, 3)` | `observation_images_normalized` |
| `noise` | `TENSOR/SWAP` | input | device tensor dtype, `FLAT`, `(chunk_length, 32)` | `diffusion_noise` |
| `actions` | `ACTION/STAGED` | output | host-visible robot action chunk, `FLAT`, `(chunk_length, robot_action_dim)` | `diffusion_noise` |

Current source of truth:

- Export declaration: `flash_rt/models/pi05/runtime_export.py`,
  `export_model_runtime(..., io="native")`
- Native verb implementation: `cpp/models/pi05/src/model_runtime.cpp`
- C++ modality binding: `cpp/models/pi05/src/runtime.cpp`,
  `cpp/models/pi05/src/io.cpp`, `cpp/models/pi05/src/spec.cpp`

There is deliberately no `prompt` port on the adopted-export path today. The
prompt embedding is prepared by the producer before graph capture/export. A
producer must not declare a `TEXT/STAGED` or `STATE/STAGED` port until the
native verb can really update that input on the hot path.

## Native V2 Face

The `io="native_v2"` export adds prompt/state staging and a raw action alias.
Adding these ports changes the model-runtime identity and therefore the
fingerprint. Existing capsules from the old face must refuse restore into this
face.

| port | modality/update | direction | dtype/layout/shape | backing |
|---|---|---|---|---|
| `prompt` | `TEXT/STAGED` | input | UTF-8 bytes, `FLAT`, variable length | staged by C++ runtime |
| `state` | `STATE/STAGED` | input | host `f32`, `FLAT`, `(state_dim,)` | staged by C++ runtime |
| `images` | `IMAGE/STAGED` | input | device tensor dtype, `NHWC`, `(num_views, 224, 224, 3)` | `observation_images_normalized` |
| `noise` | `TENSOR/SWAP` | input | device tensor dtype, `FLAT`, `(chunk_length, 32)` | `diffusion_noise` |
| `actions` | `ACTION/STAGED` | output | host-visible robot action chunk, `FLAT`, `(chunk_length, robot_action_dim)` | `diffusion_noise` |
| `actions_raw` | `TENSOR/SWAP` | output | device tensor dtype, `FLAT`, `(chunk_length, 32)` | `diffusion_noise` |

For Pi0.5, proprioceptive state is not an independent model tensor. It is
normalized, discretized into OpenPI-compatible 256-bin state tokens, rendered
into the prompt text, tokenized, embedded, and written into the language rows
of `encoder_x`. Therefore prompt and state updates are one producer-owned text
staging path.

Internal model buffers such as `encoder_x`, KV/cache windows, residual
streams, and `diffusion_noise` are not `STATE` ports. They are `TENSOR` ports
when exposed as hot IO, or runtime buffers/capsule regions when they are part
of a restorable boundary.

## STAGED Payloads

The payload conventions are inherited from `frt_model_runtime_v1`.

| modality | `set_input` data | bytes |
|---|---|---|
| `IMAGE` / `DEPTH` | `frt_image_view[]` | `n_frames * sizeof(frt_image_view)` |
| `TEXT` | UTF-8 bytes | byte length |
| `TENSOR` / `STATE` / `ACTION` / `AUDIO` | raw bytes per the port dtype and shape | byte length |

For `IMAGE`, frames are matched positionally to the producer-declared camera
view order. The Pi0.5 view order is:

1. `image`
2. `wrist_image`
3. `wrist_image_right`

Deployments with fewer views export a shorter `num_views` and use the prefix
of that view order.

## Image Input

The current native image input accepts host `frt_image_view[]` and stages the
data into the device `observation_images_normalized` buffer before replay.

Producer-owned preprocessing:

- view count is checked against the exported `images` port shape;
- frame payloads are host `u8` pixels in `RGB8`/`HWC` layout;
- target tensor is `(num_views, 224, 224, 3)`;
- output layout is `NHWC`;
- output dtype is the exported tensor dtype, normally BF16 for the FP8 path;
- normalization is `x / 127.5 - 1.0`;
- resizing to 224x224 is producer-owned.

The Pi0.5 native face rejects unsupported input shape, dtype, layout, pixel
format, or view count with a shape/status error. BGR, grayscale, RGBA, CHW, and
non-`u8` frames are not silently converted at the Pi0.5 contract boundary. If a
deployment supports more pixel formats, the supported set must be documented by
the producer and tested against the CPU reference path.

## Noise Input

`noise` is a `TENSOR/SWAP` port. The host writes its raw bytes directly into
the `diffusion_noise` window, usually by `cap_swap` after Nexus adoption or by
the equivalent runtime/backend copy mechanism. Calling `set_input` on this
port is unsupported by design: SWAP means the device window is the interface.

Shape is `(chunk_length, 32)`. `chunk_length` is declared by the producer and
must be read from the port shape; host code must not assume `(10, 32)`.

## Action Output

`actions` is the host-visible robot action chunk after producer-owned
postprocess.

The logical output shape is:

```
(chunk_length, robot_action_dim)
```

For LIBERO-style Pi0.5 deployments, `robot_action_dim` is typically 7. Other
deployments may export a different fixed robot action dimension. Consumers and
schedulers must read the declared port shape instead of hard-coding `(10, 7)`.

The internal model output remains `(chunk_length, 32)` in `diffusion_noise`.
The native `actions` STAGED output slices the robot dimensions and applies the
deployment action normalization statistics. With q01/q99 stats, the affine
parameters are:

```
mean   = (q01 + q99) / 2
stddev = (q99 - q01) / 2
```

The C++ postprocess path clamps normalized action values to the configured
domain before applying the affine transform. Any raw `(chunk_length, 32)` face
must be exported as a separate `TENSOR/SWAP` output. The Pi0.5 `native_v2`
face declares this as `actions_raw`; RTC stage plans also use the same port
name. Nexus must treat it as a declared raw byte window, not model internals.

## Lifecycle Mapping

Mindon-style lifecycle names map to the existing ABI. Do not add a parallel
API family for the same phases.

| requested name | existing contract | phase |
|---|---|---|
| `Prepare` | `prepare(graph, key)` | warm only |
| `Warmup` | host policy: call `prepare` for needed variants, then run warm ticks | warm |
| `Infer` | `step()` sugar or host-scheduled stage replay | hot |
| `Sync` | host/backend stream synchronization | hot or drain |
| `GetOutput` | `get_output(port, out, capacity, &written, stream)` | hot |

`prepare` is the only place a shape-bucket miss may capture or materialize a
variant. A hot tick must not recapture, allocate, or rebind graph pointers.

## Identity and Capsule Regions

The following changes are deployment identity changes:

- adding/removing/reordering ports;
- changing a port modality, dtype, layout, direction, update class, required
  flag, shape, bound buffer index, offset, or byte window;
- changing graph names or default stream placement;
- changing the stage DAG;
- adding/removing/reordering capsule regions;
- changing a region name, buffer, offset, byte length, or flags.

The following are not deployment identity changes:

- editing `manifest_json`;
- changing `cadence_hint_hz`.

Prompt/state staging should normally add a restorable prompt context region
only after the bytes that define that context are explicit. A valid region
could include the language rows of `encoder_x` plus the fixed-prompt valid
length scalar. Region layout and order are fingerprinted; old capsules should
not restore into the new layout.

## Current Integration Lanes

There are three supported integration lanes:

- Lane A, current: Python setup/capture/export stays resident in the process;
  the hot loop adopts `frt_model_runtime_v1` and runs through C++/Nexus.
- Lane B, after prompt/state staging: same as Lane A, plus hot
  `prompt`/`state` STAGED ports.
- Lane C, future native producer: a C++ shared object implements
  `frt_model_runtime_open_v1(config_json, &out)` and produces the same public
  struct without Python setup.

The Pi0.5 C++ shared object now exports `frt_model_runtime_open_v1` as a
native-v2 configuration gate. The gate requires `io="native_v2"`,
`checkpoint_path`, `tokenizer_model_path`, `state_prompt_mode="fixed"`,
`max_prompt_tokens >= 200`, and a positive `state_dim`. It also parses
`checkpoint_path/model.safetensors` enough to verify the Pi0.5 prompt
embedding table metadata, and verifies action/state q01/q99 dimensions from
either openpi `norm_stats.json` or LeRobot policy normalizer/unnormalizer
safetensors. Valid configuration returns unsupported until native asset
materialization and graph capture are complete.

CUDA graph execs are process-local objects. They are not serialized as a
portable artifact. Removing Python from setup requires a native producer that
loads assets and captures graphs in the replay process.

## Validation

The minimum regression set for this contract is:

```
PYTHONPATH=.:./exec/build:./runtime/build python runtime/tests/test_runtime_export.py
PYTHONPATH=.:./exec/build:./runtime/build python runtime/tests/test_model_runtime_py.py
./runtime/build/test_model_runtime
ctest --test-dir cpp/build --output-on-failure
```

Real-checkpoint gates:

```
python cpp/tests/gate_pi05_model_runtime_export.py ...
python cpp/tests/gate_pi05_c_api_export.py ...
```

For prompt/state staging, add token-exact, formatter string-exact, embedding
bit-exact, fixed-vs-exact E2E cosine, and hot-contract tests before declaring
the new STAGED ports.

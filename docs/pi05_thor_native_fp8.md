# Pi0.5 Native C++ FP8 On Thor

This guide covers the Python-free Pi0.5 producer on NVIDIA Thor SM110. It
loads a safetensors checkpoint, calibrates FP8 activation scales, captures one
fixed-shape CUDA Graph, and returns `frt_model_runtime_v1` from
`frt_model_runtime_open_v1`.

The implementation is model-specific by design. The stable runtime ABI remains
model- and backend-neutral; Pi0.5 checkpoint names, dimensions, prompt/state
semantics, calibration sites, and camera names stay under `cpp/models/pi05/`.
Dataset discovery, decoding, synchronization, and sampling policy remain host
responsibilities.

## Support Matrix

| Producer | Hardware | Precision | Calibration | Native IO dtype |
|---|---|---|---|---|
| Native C++ | SM110 | FP8 E4M3 | Required | F16 |
| Native C++ | SM120 | BF16 | Not used | BF16 |
| Python | Backend-specific | Existing FP8/BF16/NVFP4 routes | Existing Python contract | Producer-declared |

`precision="auto"` selects FP8 E4M3 on SM110 and BF16 on SM120. Production
configuration should normally specify the intended precision explicitly.
Native C++ NVFP4 is not implemented by this producer; `precision="nvfp4"` is
rejected. This does not change the independent Python NVFP4 path.

## Build

The SM110 backend requires CUDA kernels, CUDA staging, exec, SentencePiece,
and a compatible CUTLASS checkout. It has been validated with CUDA 13.0.

```bash
export BUILD_DIR="$PWD/cpp/build-thor"
export CUTLASS_DIR="$PWD/third_party/cutlass"

cmake -S cpp -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=110 \
  -DFLASHRT_CPP_WITH_EXEC=ON \
  -DFLASHRT_CPP_WITH_CUDA_STAGING=ON \
  -DFLASHRT_CPP_WITH_CUDA_KERNELS=ON \
  -DFLASHRT_CPP_WITH_SENTENCEPIECE=ON \
  -DFLASHRT_CPP_WITH_FA2=OFF \
  -DFLASHRT_CPP_WITH_THOR_FP8=ON \
  -DFLASHRT_CPP_CUTLASS_DIR="$CUTLASS_DIR"
cmake --build "$BUILD_DIR" -j
```

Configure fails if the Thor backend is enabled without CUDA kernels, CUDA
staging, exec, or CUTLASS. Build `Release` for deployment. A Debug build is
useful for the same test matrix but is not a latency reference.

## Native Configuration

Calibration and runtime open consume the same model configuration. Calibration
does not need `calibration_path`; runtime open requires it on SM110.

```json
{
  "io": "native_v2",
  "checkpoint_path": "/path/to/pi05-checkpoint",
  "tokenizer_model_path": "/path/to/paligemma_tokenizer.model",
  "state_prompt_mode": "fixed",
  "precision": "fp8_e4m3fn",
  "max_prompt_tokens": 200,
  "state_dim": 8,
  "num_views": 2,
  "chunk": 10,
  "num_steps": 10,
  "vision_pool_factor": 1,
  "max_frame_width": 1280,
  "max_frame_height": 720
}
```

The checkpoint supplies state/action q01 and q99 normalization metadata. All
shape fields are fixed setup values. Increasing a capacity or changing view,
chunk, schedule, tokenizer, checkpoint, precision, or calibration artifact
produces a different deployment identity where applicable.

## Calibration API

The model-specific API is declared in
`flashrt/cpp/models/pi05/c_api.h`:

```c
frt_pi05_calibration_session* session = NULL;
int rc = frt_pi05_calibration_create_v1(
    calibration_config_json, 99.9, &session);

frt_pi05_vision_frame frames[2] = {0};
frames[0].struct_size = sizeof(frames[0]);
frames[0].name = "image";
frames[0].data = base_rgb;
frames[0].bytes = base_bytes;
frames[0].width = base_width;
frames[0].height = base_height;
frames[0].stride_bytes = base_stride;
frames[0].pixel_format = FRT_PI05_PIXEL_RGB8;

frames[1].struct_size = sizeof(frames[1]);
frames[1].name = "wrist_image";
frames[1].data = wrist_rgb;
frames[1].bytes = wrist_bytes;
frames[1].width = wrist_width;
frames[1].height = wrist_height;
frames[1].stride_bytes = wrist_stride;
frames[1].pixel_format = FRT_PI05_PIXEL_RGB8;

frt_pi05_calibration_sample_v1 sample = {0};
sample.struct_size = sizeof(sample);
sample.prompt = task_prompt;
sample.state = state_f32;
sample.n_state = state_dim;
sample.frames = frames;
sample.n_frames = 2;
sample.noise = noise_f32;
sample.n_noise = chunk * 32;

rc = frt_pi05_calibration_observe_v1(session, &sample);
rc = frt_pi05_calibration_finalize_v1(
    session, "/path/to/pi05-sm110-fp8.safetensors");
frt_pi05_calibration_destroy_v1(session);
```

Check every return code. Creation errors are available through
`frt_pi05_calibration_create_last_error_v1`; session errors are available
through `frt_pi05_calibration_last_error_v1`.

Calls on one session are serialized; the handle does not provide internal
concurrent-observe semantics. Hosts that calibrate independent streams in
parallel use independent sessions and artifacts.

### Camera Sets

One `observe` call is one complete observation. Supported camera names are the
prefix of this model-specific order:

1. `image`
2. `wrist_image`
3. `wrist_image_right`

A one-view session supplies only `image`. A two-view session supplies `image`
and `wrist_image`. The input array may be in any order; calibration canonicalizes
it by name. Missing, duplicate, or unknown names reject the entire observation
without increasing `sample_count`.

Calibration image payloads use host `u8/RGB8/HWC` frames. Each frame must fit
the setup-time width/height capacity. `timestamp_ns` is carried by the frame
descriptor but FlashRT does not synchronize cameras; the host must submit a
coherent observation.

The native model-runtime `images` STAGED payload uses `frt_image_view[]`, which
has no camera-name field and follows the producer-declared positional order.
Do not infer one payload convention from the other.

### Single And Dataset Calibration

Call `observe` once for a single-observation artifact. Call it repeatedly for
multi-timestamp or dataset calibration, then call `finalize` once. FlashRT
reduces each activation site independently with NumPy-compatible linear
percentile semantics. `99.9` is the validated dataset setting; `100.0` selects
the observed maximum.

Dataset iteration is intentionally not part of the runtime. The host chooses
episodes, tasks, timestamps, image decoding, and synchronization. Samples
should represent the deployed prompt, state, camera, and action distribution.
Broadening calibration data can improve coverage while reducing resolution for
common activations, so every artifact still needs an end-to-end action gate.

`noise` is optional F32 `[chunk, 32]`. Supply fixed noise when comparing with a
reference producer. If it is omitted with `n_noise=0`, FlashRT generates
deterministic normal F16 noise from `noise_seed + successful_sample_index`.
Malformed or non-finite state/noise payloads are rejected before the sample is
committed.

Calibration materializes the model and runs uncaptured reference forwards. It
is an offline/setup operation, not a control-loop operation. Reuse the produced
artifact until an identity input or calibration policy changes.

## Artifact Contract

The calibration file is an atomically published safetensors artifact with two
F32 tensors:

- `encoder_scales`: 72 values;
- `decoder_scales`: `num_steps * 18 * 4` values.

Metadata binds the artifact to:

- schema, model, precision, tensor dtype, and reducer version;
- observed SM architecture;
- full checkpoint and tokenizer SHA-256 digests;
- view count, prompt capacity, state dimension, chunk, denoise steps, and
  vision pooling;
- successful sample count and percentile.

Runtime open rejects incompatible metadata, non-positive/non-finite scales,
shape mismatches, checkpoint/tokenizer digest mismatches, and hardware
mismatches. The runtime identity also includes the calibration file SHA-256,
so changing scale bytes intentionally changes the fingerprint and prevents an
old capsule from restoring into different math.

Do not edit or merge artifacts manually. Re-run calibration from representative
observations.

## Runtime Open

Add the artifact path to the configuration and load the producer through the
standard model-runtime symbol:

```json
{
  "io": "native_v2",
  "checkpoint_path": "/path/to/pi05-checkpoint",
  "tokenizer_model_path": "/path/to/paligemma_tokenizer.model",
  "state_prompt_mode": "fixed",
  "precision": "fp8_e4m3fn",
  "calibration_path": "/path/to/pi05-sm110-fp8.safetensors",
  "max_prompt_tokens": 200,
  "state_dim": 8,
  "num_views": 2,
  "chunk": 10,
  "num_steps": 10,
  "vision_pool_factor": 1
}
```

Resolve `FRT_MODEL_RUNTIME_OPEN_V1_SYMBOL` as
`frt_model_runtime_open_v1_fn`, or link the producer library and call
`frt_model_runtime_open_v1` directly. The returned runtime publishes one
`infer` stage and these ordered ports:

| Port | Update | SM110 dtype | Payload |
|---|---|---|---|
| `prompt` | STAGED | U8 | UTF-8 task text |
| `state` | STAGED | F32 | raw proprioception |
| `images` | STAGED | F16 | host `frt_image_view[]` transformed into the captured window |
| `noise` | SWAP | F16 | device `[chunk, 32]` |
| `actions` | STAGED | F32 | host `[chunk, robot_action_dim]` |
| `actions_raw` | SWAP | F16 | device `[chunk, 32]` alias |

Prompt formatting, state normalization/discretization, tokenization, embedding,
vision preprocessing, and action postprocessing remain producer-owned. Nexus
or another consumer moves declared payloads and schedules declared stages; it
does not interpret Pi0.5 semantics.

## Loading Path

Native setup mmaps `model.safetensors` read-only and accepts F32, BF16, or F16
source tensors. Independent CPU transforms run in bounded parallel tasks while
typed final F16/FP8/scales are uploaded into context-owned buffers. Full-file
checkpoint hashing runs concurrently. Valid unaligned safetensors payloads use
alignment-safe reads.

Materialization uses up to eight worker threads by default, bounded by the
layer count. `FLASHRT_NATIVE_WEIGHT_WORKERS=<n>` is a setup-only diagnostic and
tuning override (`1..64`, still bounded by layer count); invalid values keep the
default. It changes scheduling only, not tensor order, bytes, identity, or
runtime math. Validate any deployment override with the same byte/action gates.

There is no implicit second weight-cache format for safetensors. This avoids
another invalidation and serialization boundary while keeping the mathematical
source path identical to the shipped producer. The OS page cache may improve
repeated file reads but is not part of the FlashRT contract.

On a representative SM110 run with a 14.47 GB F32 checkpoint, native setup was
7.90 seconds and the complete open/infer/output/teardown lifecycle was 8.62
seconds. Of setup time, 7.61 seconds was weight transform/quantization/upload,
94 ms was workspace/style setup, 163 ms was warmup, and 26 ms was graph capture.
These are reference measurements, not latency ABI guarantees.

## Validation

Run CPU/CUDA-off tests in addition to SM110 Release and Debug builds:

```bash
ctest --test-dir "$BUILD_DIR" --output-on-failure
```

The real-checkpoint gate compares native calibration and inference with the
shipped Torch producer using fixed observations and noise:

```bash
python cpp/tests/gate_pi05_native_thor_fp8.py \
  --probe "$BUILD_DIR/pi05_native_thor_fp8_probe" \
  --checkpoint "$CHECKPOINT_DIR" \
  --tokenizer "$TOKENIZER_MODEL" \
  --artifact "$CALIBRATION_FILE" \
  --samples 1 --views 1

python cpp/tests/gate_pi05_native_thor_fp8.py \
  --probe "$BUILD_DIR/pi05_native_thor_fp8_probe" \
  --checkpoint "$CHECKPOINT_DIR" \
  --tokenizer "$TOKENIZER_MODEL" \
  --artifact "$CALIBRATION_FILE" \
  --samples 3 --views 2
```

The two-view probe deliberately submits reversed camera order and checks
duplicate/incomplete/unknown names, non-RGB input rejection, malformed noise,
deterministic generated noise, artifact loading, runtime identity, one graph
variant, finite logical actions, and teardown. The reference gates require all
72 encoder scales, all 720 decoder scales for ten steps, and all 320 raw action
values to be bit-exact.

For the complete service loop, profile the CUDA profiler range around 1,000
iterations of prompt/state/image/noise update, replay, and action output:

```bash
FLASHRT_PROFILE_REPLAYS=1000 FLASHRT_PROFILE_SERVICE_LOOP=1 \
nsys profile --trace=cuda \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o "$HOT_REPORT" \
  "$BUILD_DIR/pi05_native_open_probe" \
  "$CHECKPOINT_DIR" "$TOKENIZER_MODEL" "$CALIBRATION_FILE"
```

The validated trace contained exactly 1,000 graph launches and no CUDA device
allocation/free, CUDA host allocation/registration, mempool, virtual-memory,
graph instantiation, or capture API in the measured range. A separate
1,000-update prompt/state gate measured 266 microseconds p99 against a 1 ms
limit, with one graph variant throughout.

# PI0.5 Native C++ Frontend

The PI0.5 native frontend loads a supported safetensors checkpoint, performs
model-owned preprocessing and postprocessing, captures the model graphs, and
publishes the standard `frt_model_runtime_v1` interface. Python and native C++
are producer choices; consumers use the same model-runtime ABI.

This page covers the PI0.5 producer-specific API. The generic ABI is documented
in [`model_runtime_api.md`](model_runtime_api.md).

## Ownership boundaries

- `runtime/` defines ports, stages, regions, lifetime and replay verbs. It does
  not know PI0.5, calibration datasets or hardware policy.
- `cpp/models/pi05/` owns the semantic pipeline, checkpoint schema, prompt and
  state processing, image contract, action postprocessing and calibration
  orchestration.
- SM110 and SM120 targets bind the semantic operations to existing `csrc`
  primitives and own precision-specific packing, scratch and observers. They do
  not own another model forward.
- The host owns dataset decoding, sample selection, episode policy and artifact
  placement. Nexus may transport snapshot regions, but it does not interpret
  model state or own calibration.

Calibration and graph capture are setup work. The hot path remains
`set_input` / `step` / `get_output` with no allocation, recapture or pointer
rebinding.

## Build requirements

The complete native path is opt-in. With
`FLASHRT_ENABLE_NATIVE_CPP=OFF` (the default), the PI0.5 model libraries,
native operation support, loader, tokenizer, producer and their tests are not
part of the build. Existing Python modules and their kernel linkage remain
unchanged.

Build one deployment target per CMake tree:

- `FLASHRT_ENABLE_NATIVE_CPP=ON` enables the native C++ product boundary.
- `FLASHRT_CPP_WITH_PI05_SM120_TARGET=ON` for SM120 BF16 or static FP8.
- `FLASHRT_CPP_WITH_PI05_SM110_TARGET=ON` for SM110 static FP8.
- `FLASHRT_CPP_WITH_SENTENCEPIECE=ON` is required for native prompt formatting
  and embedding.

SM120 also consumes the opt-in FA2 native C library documented in
[`fa2_native_c_api.md`](fa2_native_c_api.md). Build it first, then pass its
path to the frontend build:

```bash
cmake -S . -B <operations-build> \
  -DGPU_ARCH=120 \
  -DENABLE_FA2=ON \
  -DFLASHRT_ENABLE_NATIVE_CPP=ON \
  -DFLASHRT_BUILD_FA2_PYTHON_ADAPTER=OFF
cmake --build <operations-build> --target flashrt_fa2_raw -j

cmake -S cpp -B <frontend-build> \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLASHRT_ENABLE_NATIVE_CPP=ON \
  -DFLASHRT_CPP_WITH_SENTENCEPIECE=ON \
  -DFLASHRT_CPP_WITH_PI05_SM120_TARGET=ON \
  -DFLASHRT_CPP_FA2_LIBRARY=<operations-build>/libflashrt_fa2_raw.so
cmake --build <frontend-build> -j
```

SM110 does not depend on FA2. It consumes the existing CUTLASS FMHA path:

```bash
cmake -S cpp -B <frontend-build> \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLASHRT_ENABLE_NATIVE_CPP=ON \
  -DFLASHRT_CPP_WITH_SENTENCEPIECE=ON \
  -DFLASHRT_CPP_WITH_PI05_SM110_TARGET=ON \
  -DFLASHRT_CPP_CUTLASS_DIR=<cutlass-source>
cmake --build <frontend-build> -j
```

The shared producer exports only the documented `frt_*` C API. Internal CUDA,
C++, SentencePiece and target symbols are hidden and checked at link time.

The frontend uses existing common `csrc` operations without changing their
symbols or behavior. Native-only operation gaps live under `csrc/native_cpp/`,
are linked only by an enabled native target, and are not added to the Python
kernel modules. They are model-independent operation implementations; PI0.5
model order, dimensions, state rules and checkpoint mapping remain under
`cpp/models/pi05/`.

The SM110 target therefore reuses the shipped FP16 LayerNorm and dynamic FP8
quantizer directly. The opt-in gap library is limited to operations absent from
the common surface, including host-free setup helpers, persisted FP8 weight
packing, and the explicit padded-stride seqused attention entry point. The
strict target gate compares calibration scales and raw actions with the
unchanged Torch producer bit for bit.

| Native C++ | PI0.5 target | Result |
|---|---|---|
| OFF | OFF | Existing default build; no native PI0.5 product |
| OFF | ON | Configuration error |
| ON | OFF | Native support/model static libraries only; no deployable producer |
| ON | exactly one target | `libflashrt_cpp_pi05_c` and selected target |
| ON | both targets | Configuration error; use separate build trees |

## Configuration

Calibration and native open consume the same JSON shape. A representative VLA
calibration configuration is:

```json
{
  "io": "native_v2",
  "checkpoint_path": "CHECKPOINT_DIR",
  "tokenizer_model_path": "TOKENIZER_MODEL",
  "state_prompt_mode": "fixed",
  "precision": "fp8_e4m3fn",
  "stage_plan": "full",
  "max_prompt_tokens": 200,
  "state_dim": 8,
  "num_views": 3,
  "chunk": 10,
  "num_steps": 10,
  "vision_pool_factor": 1,
  "max_frame_width": 1280,
  "max_frame_height": 720
}
```

The checkpoint must contain `model.safetensors` and matching state/action
normalization statistics. `num_views`, `chunk`, `num_steps`, prompt capacity and
pool factor are setup dimensions. They participate in calibration identity and
must match at open. The frontend supports one to three configured views; frame
names and count must match the model's canonical view order.

`precision="auto"` chooses a supported precision for the current target. Do
not set `calibration_path` while producing the first artifact. For FP8 runtime
open, add the finalized artifact path to the same configuration. The artifact
must match the checkpoint, tokenizer, hardware and setup dimensions; identity
mismatch is a hard error.

## FP8 calibration

For the complete single-view, multi-view, and dataset workflow, including a
reusable C++ loop, input lifetime rules, artifact reuse, and troubleshooting,
see [`pi05_native_calibration.md`](pi05_native_calibration.md).

The key distinction is that one observation contains the exact configured set
of synchronized camera views, while dataset calibration calls `observe`
multiple times. A three-view runtime always requires three frames per
observation, even when calibrating from only one observation.

Include `flashrt/cpp/models/pi05/c_api.h`. Create one session, call `observe`
once per selected dataset sample, then finalize the reduced artifact:

```c
frt_pi05_calibration_session* calibration = NULL;
int rc = frt_pi05_calibration_create_v1(
    calibration_config_json, 99.9, &calibration);

for (size_t i = 0; rc == 0 && i < sample_count; ++i) {
    frt_pi05_calibration_sample_v1 sample = {0};
    sample.struct_size = sizeof(sample);
    sample.prompt = samples[i].prompt;
    sample.state = samples[i].state;
    sample.n_state = state_dim;
    sample.frames = samples[i].frames;
    sample.n_frames = num_views;
    sample.noise = samples[i].noise;       /* optional */
    sample.n_noise = chunk * 32;           /* zero when noise is omitted */
    sample.noise_seed = samples[i].seed;
    rc = frt_pi05_calibration_observe_v1(calibration, &sample);
}

if (rc == 0) {
    rc = frt_pi05_calibration_finalize_v1(calibration,
                                           calibration_artifact_path);
}
frt_pi05_calibration_destroy_v1(calibration);
```

Each frame is host `U8/HWC/RGB8` with explicit name, dimensions, stride and
capacity. Omitted noise is generated deterministically from `noise_seed` and
the committed sample index. A failed observation does not increment
`frt_pi05_calibration_sample_count_v1`.

The host deliberately owns the dataset loop. The calibration session owns only
the model traversal, per-operation observation and deterministic reduction, so
single-frame, multi-view and dataset calibration do not create separate model
pipelines.

Native FP8 open never performs an implicit first-inference calibration. The
host must produce or select an artifact before open. BF16 does not use this
artifact path.

The artifact records checkpoint and tokenizer digests, hardware, activation
dtype, view count, prompt/state/chunk/step dimensions, pool factor, percentile
and sample count. It is not portable across a mismatched identity.

## Open and execute

Load the producer through the generic symbol:

```c
frt_model_runtime_v1* model = NULL;
int rc = frt_model_runtime_open_v1(runtime_config_json, &model);
```

For FP8, `runtime_config_json` includes the finalized `calibration_path`. BF16
does not use a calibration artifact. After open:

1. Inspect `ports[]`, `stages[]` and the export identity.
2. Call `prepare` only during setup when a declared graph variant needs it.
3. Send the declared `TEXT`, `STATE` and `IMAGE` inputs with `set_input`.
4. Update optional SWAP inputs such as noise through their declared buffer
   windows.
5. Replay `step`, or schedule the declared stages, then read `actions` with
   `get_output`.
6. Release the model-runtime handle when no in-flight work remains.

`stage_plan="full"` publishes one `infer` stage. `context_action` publishes the
same semantic pipeline as `context -> decode_only`; the split is a scheduling
view, not a second forward. Hosts must use the declared action shape rather than
assuming a particular chunk or robot action width.

## Compatibility

The existing Python producer and `io="native"` verb-overlay path remain valid.
`io="native_v2"` is the fully native checkpoint producer. Both converge on
`frt_model_runtime_v1`; choosing one does not change Nexus or host lifecycle
semantics.

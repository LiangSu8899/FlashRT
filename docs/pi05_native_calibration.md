# PI0.5 Native C++ FP8 Calibration

This guide covers the explicit native C++ calibration API for the PI0.5
producer. It applies to SM120 FP8 and SM110 FP8. BF16 does not use a
calibration artifact.

Native C++ calibration is intentionally explicit:

1. create a calibration session from the deployment configuration;
2. submit one or more representative observations;
3. finalize one immutable artifact;
4. open the FP8 runtime with that artifact.

The runtime does not silently calibrate on the first inference. The host owns
artifact placement and reuse, while FlashRT validates the artifact identity at
open.

## Views, observations, and datasets

These are separate dimensions:

- A **view** is one camera image in an observation.
- An **observation** is one synchronized prompt, robot state, and exact set of
  configured views.
- A **dataset calibration** submits multiple observations by calling
  `frt_pi05_calibration_observe_v1` repeatedly on one session.

| Deployment | `num_views` | Frames per `observe` | Calls to `observe` |
|---|---:|---:|---:|
| One camera, one sample | 1 | 1 | 1 |
| Three cameras, one synchronized sample | 3 | 3 | 1 |
| Three cameras, 16 selected dataset samples | 3 | 3 | 16 |

"Single-frame calibration" is valid for a runtime configured with one view.
It is not a shortcut for calibrating a three-view runtime: a three-view
artifact requires all three views in every observation.

The canonical view names are fixed by the model contract:

| `num_views` | Required names in every observation |
|---:|---|
| 1 | `image` |
| 2 | `image`, `wrist_image` |
| 3 | `image`, `wrist_image`, `wrist_image_right` |

The input array may use any order, but names must be unique and the exact set
must be present. Multiple temporal frames are multiple observations, not extra
entries in one observation.

## Build and link

Enable one PI0.5 target and SentencePiece in the FlashRT build. When FlashRT is
included with CMake, link the C face:

```cmake
target_link_libraries(my_runner PRIVATE flashrt_cpp_pi05_c)
```

Include:

```cpp
#include "flashrt/cpp/models/pi05/c_api.h"
```

The required symbols are:

```text
frt_pi05_calibration_create_v1
frt_pi05_calibration_observe_v1
frt_pi05_calibration_finalize_v1
frt_pi05_calibration_sample_count_v1
frt_pi05_calibration_last_error_v1
frt_pi05_calibration_destroy_v1
```

## Calibration configuration

Calibration and runtime open use the same shape and asset fields. Use explicit
FP8 precision and omit `calibration_path` while generating the artifact:

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

Use the real deployment dimensions. In particular, do not use a diagnostic
`chunk=1` configuration to produce a deployment artifact. The standard PI0.5
VLA route uses `chunk=10`.

The checkpoint directory must contain `model.safetensors` and compatible
state/action normalization statistics. The state passed to `observe` is the
raw physical state; FlashRT applies the checkpoint's state normalization and
PI0.5 prompt formatting. Pass the raw task text as `prompt`; do not preformat
the `Task:` prefix or discretize the state yourself.

## Input contract

Each `frt_pi05_calibration_sample_v1` must satisfy all of the following:

- `struct_size` is `sizeof(frt_pi05_calibration_sample_v1)`;
- `prompt` is non-null and fits `max_prompt_tokens` after model formatting;
- `state` contains exactly `state_dim` finite `float` values;
- `n_frames` equals `num_views` and uses the canonical names above;
- every image is host `U8/HWC/RGB8` with positive dimensions and stride;
- each image fits `max_frame_width` and `max_frame_height`;
- `bytes` covers the complete strided image;
- explicit noise is finite F32 with exactly `chunk * 32` values, or noise is
  omitted with `noise=NULL` and `n_noise=0`.

Input pointers only need to remain valid until `observe` returns. One session
must be driven serially; do not call `observe` or `finalize` concurrently on
the same handle.

When noise is omitted, FlashRT generates deterministic Gaussian noise from
`noise_seed` and the committed sample index. For an exact comparison with
another producer, pass the same explicit noise to both producers. For normal
dataset calibration, omitting noise and fixing `noise_seed` is sufficient and
reproducible.

## Reusable C++ loop

The following helper is directly reusable after the application maps its image
decoder or dataset reader into `CalibrationObservation`:

```cpp
#include "flashrt/cpp/models/pi05/c_api.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

struct RgbView {
    std::string name;
    std::vector<std::uint8_t> pixels;
    int width = 0;
    int height = 0;
    int stride_bytes = 0;
    std::uint64_t timestamp_ns = 0;
};

struct CalibrationObservation {
    std::string prompt;
    std::vector<float> state;
    std::vector<RgbView> views;
    std::vector<float> noise;  // Empty means deterministic generated noise.
    std::uint64_t noise_seed = 0;
};

int calibrate_pi05(const std::string& config_json,
                   double percentile,
                   const std::vector<CalibrationObservation>& observations,
                   const std::string& artifact_path) {
    if (observations.empty()) {
        std::cerr << "calibration requires at least one observation\n";
        return -1;
    }

    frt_pi05_calibration_session* session = nullptr;
    int rc = frt_pi05_calibration_create_v1(
        config_json.c_str(), percentile, &session);
    if (rc != 0) {
        std::cerr << "calibration create failed: "
                  << frt_pi05_calibration_create_last_error_v1() << '\n';
        return rc;
    }

    for (const CalibrationObservation& observation : observations) {
        std::vector<frt_pi05_vision_frame> frames(observation.views.size());
        for (std::size_t i = 0; i < observation.views.size(); ++i) {
            const RgbView& view = observation.views[i];
            frt_pi05_vision_frame& frame = frames[i];
            frame = {};
            frame.struct_size = sizeof(frame);
            frame.name = view.name.c_str();
            frame.data = view.pixels.data();
            frame.bytes = view.pixels.size();
            frame.width = view.width;
            frame.height = view.height;
            frame.stride_bytes = view.stride_bytes;
            frame.pixel_format = FRT_PI05_PIXEL_RGB8;
            frame.timestamp_ns = view.timestamp_ns;
        }

        frt_pi05_calibration_sample_v1 sample{};
        sample.struct_size = sizeof(sample);
        sample.prompt = observation.prompt.c_str();
        sample.state = observation.state.data();
        sample.n_state = observation.state.size();
        sample.frames = frames.data();
        sample.n_frames = frames.size();
        sample.noise = observation.noise.empty()
                           ? nullptr
                           : observation.noise.data();
        sample.n_noise = observation.noise.size();
        sample.noise_seed = observation.noise_seed;

        rc = frt_pi05_calibration_observe_v1(session, &sample);
        if (rc != 0) {
            std::cerr << "calibration observe failed: "
                      << frt_pi05_calibration_last_error_v1(session) << '\n';
            frt_pi05_calibration_destroy_v1(session);
            return rc;
        }
    }

    if (frt_pi05_calibration_sample_count_v1(session) !=
        observations.size()) {
        std::cerr << "calibration sample count mismatch\n";
        frt_pi05_calibration_destroy_v1(session);
        return -1;
    }

    rc = frt_pi05_calibration_finalize_v1(session, artifact_path.c_str());
    if (rc != 0) {
        std::cerr << "calibration finalize failed: "
                  << frt_pi05_calibration_last_error_v1(session) << '\n';
    }
    frt_pi05_calibration_destroy_v1(session);
    return rc;
}
```

The artifact parent directory must already exist. Finalization writes a
safetensors artifact through a temporary file, flushes it, and atomically
renames it to the requested path.

## Single-sample calibration

For a one-camera deployment, set `num_views=1`, load one observation containing
the `image` view, and call the helper with one element:

```cpp
CalibrationObservation observation = load_one_camera_observation();
int rc = calibrate_pi05(config_json, 99.9, {observation},
                        "pi05_fp8_calibration.safetensors");
```

For a three-camera deployment, a single sample still contains three
synchronized images:

```cpp
CalibrationObservation observation = load_three_camera_observation();
// observation.views contains image, wrist_image, wrist_image_right.
int rc = calibrate_pi05(config_json, 99.9, {observation},
                        "pi05_fp8_calibration.safetensors");
```

With one observation, the percentile has no cross-sample effect. This mode is
useful for bring-up and a narrow, stable deployment distribution. A
representative multi-sample calibration is preferred when lighting, scenes,
tasks, cameras, or robot states vary.

## Dataset calibration

Select representative observations first, then submit all of them to one
session and finalize once:

```cpp
std::vector<CalibrationObservation> selected =
    load_stratified_calibration_observations(dataset);

int rc = calibrate_pi05(config_json, 99.9, selected,
                        "pi05_fp8_calibration.safetensors");
```

As a practical starting point, use 8 to 32 observations spread across
episodes, task prompts, frame positions, lighting, object poses, and robot
states. This is an operational recommendation, not an ABI requirement. Avoid
using only adjacent near-identical frames. Calibrate and validate on different
samples, then increase coverage only when held-out shadow results show a need.

Every observation must use the same configured view set and shape. Do not mix
checkpoints, robot normalization domains, or deployment view layouts in one
artifact. The session keeps one scale vector per submitted sample until
finalize, so select a representative subset instead of blindly iterating a
large training corpus.

At each observation site, finalization sorts the values across samples and
uses linear interpolation at the requested percentile. `99.9` is the standard
starting point; `100.0` is a strict maximum reducer. Do not lower the
percentile merely to improve one sample: require held-out accuracy and shadow
evidence.

A failed `observe` call is atomic: it does not increment the committed sample
count. Stop and fix the input instead of finalizing a partially unexpected
dataset.

## Open the calibrated runtime

After successful finalization, add `calibration_path` to an otherwise matching
runtime configuration:

```json
{
  "io": "native_v2",
  "checkpoint_path": "CHECKPOINT_DIR",
  "tokenizer_model_path": "TOKENIZER_MODEL",
  "state_prompt_mode": "fixed",
  "precision": "fp8_e4m3fn",
  "calibration_path": "pi05_fp8_calibration.safetensors",
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

Then resolve and call the normal model-runtime factory symbol. This keeps the
host on the generic producer ABI instead of adding a PI0.5-specific open ABI:

```cpp
#include "flashrt/model_runtime.h"

#include <dlfcn.h>

void* producer = dlopen("libflashrt_cpp_pi05_c.so", RTLD_NOW | RTLD_LOCAL);
auto open_model = reinterpret_cast<frt_model_runtime_open_v1_fn>(
    producer ? dlsym(producer, FRT_MODEL_RUNTIME_OPEN_V1_SYMBOL) : nullptr);

frt_model_runtime_v1* model = nullptr;
int rc = open_model ? open_model(runtime_config_json.c_str(), &model) : -1;
```

Keep the producer library loaded until the returned model and all retained
exports have been released. See [`model_runtime_api.md`](model_runtime_api.md)
for retain/release and verb sequencing. The calibration-specific error
accessors apply only to calibration create/observe/finalize.

## Artifact identity and regeneration

The artifact records and validates:

- checkpoint and tokenizer content digests;
- hardware target and activation dtype;
- `num_views`, `max_prompt_tokens`, `state_dim`, `chunk`, `num_steps`, and
  `vision_pool_factor`;
- reducer percentile and committed sample count.

Generate a new artifact after any checkpoint/tokenizer/hardware or listed
shape change. Also recalibrate after a material domain shift, camera change,
or normalization change even when the mechanical identity still matches. Do
not copy an SM120 artifact to SM110 or edit artifact metadata to bypass a
runtime mismatch.

Artifact paths and retention are host policy. A deployment may cache artifacts
by its own model/config key, but FP8 runtime open always verifies the content
identity before accepting one.

## Troubleshooting

| Error | Check |
|---|---|
| calibration target unsupported | Build SentencePiece and exactly one supported PI0.5 target |
| calibration sample is invalid | `struct_size`, prompt, exact view count/names, RGB8 metadata, state and noise pointers |
| state shape/non-finite error | Pass exactly `state_dim` raw finite values; do not pre-normalize |
| noise shape error | Pass exactly `chunk * 32` F32 values, or `NULL/0` |
| vision sample incomplete | Every observation must contain the exact configured view set |
| artifact creation/publication error | Ensure the parent directory exists and is writable |
| calibration identity mismatch at open | Compare checkpoint, tokenizer, hardware, precision and all shape fields |
| poor held-out cosine/action agreement | Recheck RGB order, prompt/state pairing, sample coverage, explicit noise parity and percentile |

Before deployment, compare calibrated C++ output against the reference
producer on held-out observations using fixed inputs and noise. Calibration is
accepted only when numerical and latency gates pass; successful artifact
loading alone is not an accuracy result.

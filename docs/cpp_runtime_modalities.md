# FlashRT C++ Native Runtime

The `cpp/` tree is the native model-runtime side above
`frt_runtime_export_v1`. It owns model IO semantics: modality preprocess,
prompt/state binding, replay inputs, and action postprocess. It is deliberately
inside FlashRT, not Nexus.

Nexus consumes only the exported runtime surface:

```
FlashRT C++ runtime
  camera/state/text/action semantics
  preprocess + postprocess
  graph/buffer ownership
      |
      v
frt_runtime_export_v1
      |
      v
Nexus adopt + capsule + schedule
```

## Boundary

Stable ABI:

- `runtime/include/flashrt/runtime.h`
- `frt_runtime_export_v1`
- `FRT_RUNTIME_OPEN_V1_SYMBOL`

Non-frozen C++ path:

- `cpp/runtime/`: native runtime manager interfaces.
- `cpp/modalities/`: reusable modality primitives.
- `cpp/families/`: model-family contracts such as VLA.
- `cpp/models/`: concrete model adapters such as Pi0.5.

The C++ API is allowed to evolve until multiple real model runtimes have forced
the common shape. The export ABI remains the stable hand-off.

## Modality Split

Common primitives live in `cpp/modalities/`:

- `types.h`: tensor view, dtype, layout, memory place, status.
- `vision.h`: view-order guarded resize/normalize/layout pack.
- `action.h`: slice, unnormalize, clamp, action schema.

Model-family contracts live in `cpp/families/<family>/`:

- define the common flow shared by a class of models;
- keep Pi0.5-specific rules out of the runtime manager;
- give the second VLA model a place to land without copying Pi0.5.

Model adapters live in `cpp/models/<model>/`:

- declare required views, target shape, dtype, normalization, output buffers;
- declare action chunk/model dim/robot dim/schema/stats;
- bind those semantics to the model's exported buffers.

Pi0.5 is the first adapter:

- vision: `image`, `wrist_image`, `wrist_image_right` -> NHWC 224x224 in the
  producer-declared dtype, normalized to `[-1, 1]`;
- action: `(chunk, 32)` model output -> first 7 robot dims, unnormalized by
  deployment stats.
- `flashrt::models::pi05::RuntimeIo` binds those specs to concrete tensor
  views and exposes `prepare_vision()` / `read_actions()`.
- `flashrt::models::pi05::Runtime` is the full C++ runtime shell for the
  adopted-export path: it retains `frt_runtime_export_v1`, binds Pi0.5 IO,
  calls replay, and exposes the VLA family interface.

Current Pi0.5 status:

- complete C++ hot-path shell: prepare vision, replay graph, read action;
- complete lifecycle for adopted Python/native exports: retain/release;
- complete build target: `flashrt_cpp_pi05`;
- C host ABI target: `flashrt_cpp_pi05_c`, exporting
  `frt_pi05_runtime_create`, `frt_pi05_runtime_prepare_vision`,
  `frt_pi05_runtime_replay_tick`, and `frt_pi05_runtime_read_actions`;
- CUDA vision path: host camera frames -> H2D raw frame -> CUDA
  resize/normalize/cast directly into export device buffers;
- conservative action staging path: device action buffer -> D2H -> CPU
  reference postprocess;
- native SM120 BF16 and SM110 FP8 checkpoint loading, tokenizer/prompt staging,
  weight materialization, calibration where required, and graph capture are
  implemented by the optional `frt_model_runtime_open_v1` producer. They
  present the same model-runtime ABI and remain FlashRT responsibilities, not
  Nexus features.

## Reference First

The portable semantic references are:

- `preprocess_vision_cpu`
- `postprocess_action_cpu`

They give every CUDA/DMA/zero-copy fast path a golden contract. The vision
device path uses CUDA resize/normalize/cast and is tested against the CPU
reference at the declared output dtype. The action device path remains
conservative D2H staging because postprocess is small; it can move to CUDA
without changing model adapters. Native SM110 calibration and inference add
separate producer-parity gates on top of these modality references.

## Hot Path Rules

Production model runtimes should make these true after setup:

1. no allocation in steady-state `prepare_tick` / replay / `read_actions`;
2. camera view order is explicit and validated;
3. tensor shape/dtype/layout mismatches fail before replay;
4. action schema and normalization stats are fingerprinted or otherwise bound
   into the deployment identity;
5. Nexus never learns model-specific modality rules.

## Tests

`cpp/tests/test_modalities.cpp` validates the first contracts:

- Pi0.5 vision spec shape/order/dtype;
- RGB/BGR -> RGB normalize -> BF16 NHWC packing;
- missing/wrong view count rejection;
- BF16 model action -> unnormalized robot action.

`cpp/tests/test_pi05_runtime.cpp` validates the model runtime shell:

- export retain/release;
- Pi0.5 manifest exposure;
- `prepare_vision -> replay_tick -> read_actions`;
- replay dispatch to the export graph/key/stream.

`cpp/tests/test_device_staging.cpp` validates the device bridge when a CUDA
device is present:

- vision preprocess into a device tensor via CUDA kernel, compared against the
  CPU reference;
- action postprocess from a device tensor via D2H staging.

`cpp/tests/test_pi05_c_api.cpp` validates the C host ABI against real exec
buffers:

- creates a `frt_runtime_export_v1` with Pi0.5 buffer names;
- creates the Pi0.5 C runtime from that export;
- prepares a host RGB frame into the export image device buffer;
- reads actions from the export action device buffer;
- verifies export retain/release ownership.

Build:

```
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure
```

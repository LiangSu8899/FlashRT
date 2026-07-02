# FlashRT C++ Runtime Modalities

This layer is the native model-runtime side above `frt_runtime_export_v1`.
It owns model IO semantics: modality preprocess, prompt/state binding, replay
inputs, and action postprocess. It is deliberately inside FlashRT, not Nexus.

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

Non-frozen C++ runtime API:

- `runtime/include/flashrt/runtime_cpp.h`
- `runtime_modalities/include/flashrt/modalities/*.h`
- `runtime_models/<model>/`

The C++ API is allowed to evolve until multiple real model runtimes have forced
the common shape. The export ABI remains the stable hand-off.

## Modality Split

Common primitives live in `runtime_modalities/`:

- `types.h`: tensor view, dtype, layout, memory place, status.
- `vision.h`: view-order guarded resize/normalize/layout pack.
- `action.h`: slice, unnormalize, clamp, action schema.

Model adapters live in `runtime_models/<model>/`:

- declare required views, target shape, dtype, normalization, output buffers;
- declare action chunk/model dim/robot dim/schema/stats;
- bind those semantics to the model's exported buffers.

Pi0.5 is the first adapter:

- vision: `image`, `wrist_image`, `wrist_image_right` -> NHWC BF16 224x224,
  normalized to `[-1, 1]`;
- action: `(chunk, 32)` model output -> first 7 robot dims, unnormalized by
  deployment stats.
- `flashrt::models::pi05::RuntimeIo` binds those specs to concrete tensor
  views and exposes `prepare_vision()` / `read_actions()`.

## CPU Reference First

The current implementation is a CPU reference path:

- `preprocess_vision_cpu`
- `postprocess_action_cpu`

This is intentional. It gives every later CUDA/DMA/zero-copy fast path a golden
contract. A GPU implementation must match the CPU reference within the declared
tolerance and preserve the same view-order, shape, dtype, and schema guards.

## Hot Path Rules

Production model runtimes should make these true after setup:

1. no allocation in steady-state `prepare_tick` / replay / `read_actions`;
2. camera view order is explicit and validated;
3. tensor shape/dtype/layout mismatches fail before replay;
4. action schema and normalization stats are fingerprinted or otherwise bound
   into the deployment identity;
5. Nexus never learns model-specific modality rules.

## Tests

`runtime/tests/test_modalities.cpp` validates the first contracts:

- Pi0.5 vision spec shape/order/dtype;
- RGB/BGR -> RGB normalize -> BF16 NHWC packing;
- missing/wrong view count rejection;
- BF16 model action -> unnormalized robot action.

Build:

```
cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/build -j
ctest --test-dir runtime/build --output-on-failure
```

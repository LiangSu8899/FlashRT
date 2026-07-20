# FlashRT Native C++ Runtime — Design

The native runtime path exists for one reason: physical-AI production ticks
need white-box, hard-real-time discipline — bounded tail latency,
high-frequency state updates, long-run stability — that a Python hot loop
cannot promise. This document is the structure map; the interface reference is
[`model_runtime_api.md`](model_runtime_api.md) and the layer norms are
[`runtime_contract.md`](runtime_contract.md).

## One struct, two producers

Everything converges on `frt_model_runtime_v1` (the standard face of one
deployed, tickable model). Either the Python setup bridge or a native
model-runtime `.so` (`frt_model_runtime_open_v1`) produces the same struct.
Consumers — FlashRT-Nexus, robot loops, FFI hosts — never change when the
producer does.

The clean hybrid path is **verb override**: the setup producer exports the
authoritative ports, stage DAG, graph streams, identity and fingerprint; a
native C++ runtime retains that declaration and replaces only
`set_input`/`get_output`/`prepare`/`step`. This keeps model-specific capture
decisions out of C++ hot-path code while still removing Python from the tick.

## Tree layout

```
runtime/                     the ONLY frozen surface (pure C ABI)
  include/flashrt/runtime.h        frt_runtime_export_v1  (execution/state kernel)
  include/flashrt/model_runtime.h  frt_model_runtime_v1   (ports · stages · verbs)
  src/                             builder + lifetime (no CUDA, no exec link)
  bindings/                        _flashrt_runtime (setup/dev bridge)

cpp/                         native implementation layers (NOT frozen)
  runtime/                   C++ manager interfaces (internal, may evolve)
  modalities/                reusable primitives: tensor views, vision
                             preprocess (CPU + CUDA), action postprocess,
                             the persistent VisionStaging pool
  families/<family>/         model-family contracts (e.g. VLA manifest)
  models/<model>/            semantic pipeline, model specs and adapters
                             binding family + modality primitives to concrete
                             buffers, normalization, action schemas and the
                             generic model-runtime face

csrc/native_cpp/             opt-in, Python-free operation implementations
                             used by native frontends; model-independent and
                             hidden from the public shared-library surface

flash_rt/runtime/export.py   the Python producer (same face, GIL-safe verbs)
```

Rule of altitude: `modalities/` knows pixels and tensors, never models;
`families/` knows a model class's IO shape, never buffer names; `models/` owns
one semantic pipeline and binds model names/constants without re-implementing
kernel primitives. Nothing under `cpp/` is ABI — the struct in `runtime/` is
the deployment surface.

## Model and hardware binding

The model boundary and the hardware boundary are intentionally different.

The **model** is selected by the native overlay/factory that the host loads:
`cpp/models/pi05/` exports `frt_pi05_model_runtime_create_over`, a future
GROOT runtime would export its own model factory, and so on. That code owns the
model's hot-path transforms: image normalization, state packing, action
postprocess, and the names/shapes of public ports it supports.

The **hardware** is selected by the setup producer. In the hybrid path, Python
chooses and captures the hardware pipeline before the C++ overlay inherits its
declarations with `frt_model_runtime_override_verbs`. In the fully native path,
the model factory queries capabilities, binds the model's one semantic pipeline
to target operations, loads/calibrates during setup, captures graphs and
publishes the same declarations. Hardware targets own kernel binding, packing
and private scratch; they do not own model topology, stage policy or a second
calibration forward.

The hybrid setup shape is:

1. The hardware-specific pipeline builds a ready model instance.
2. `flash_rt/models/<model>/runtime_export.py` exports that instance as the
   model family's standard `frt_model_runtime_v1` face.
3. `cpp/models/<model>/` overlays native hot verbs on that exact declaration.
4. Nexus or a robot loop consumes only the resulting model-runtime handle.

The fully native setup shape is:

1. `frt_model_runtime_open_v1` parses and validates model configuration.
2. The model factory resolves the device target and constructs the shared
   semantic pipeline.
3. The producer captures and publishes ports, stages, regions and identity.
4. Nexus or a robot loop consumes the same model-runtime handle.

PI0.5 native checkpoint loading and FP8 calibration usage are documented in
[`pi05_native_cpp.md`](pi05_native_cpp.md).

If two hardware pipelines expose the same logical ports and stage DAG, they can
share one native C++ overlay. If their visible contract differs, the difference
must be represented in the producer identity and handled with a distinct plan,
model key, or overlay; it should not leak into Nexus as ad hoc hardware logic.

## The production tick

Ports declare the update class; the class decides the lane:

- **SWAP** — the port is a device-buffer window; the host writes raw bytes
  directly (its own copy verb / `cap_swap`). Microsecond lane, zero model
  code in the loop.
- **STAGED** — the runtime's `set_input` transforms host data. The CUDA
  vision path runs on a fixed-capacity `VisionStaging` pool created with the
  runtime: memcpy to a pinned slot, async H2D, kernel. No `cudaMalloc` /
  `cudaFree` per frame; a frame over capacity is a hard error, never a
  fallback allocation.
- **SETUP** — legal only outside the tick.

Hot contract for both hot lanes (pinned by tests, not just prose): never
recapture, never allocate, never rebind graph pointers — only buffer contents
change, and replay output tracks them.

## Stage plans

Graph cuts are producer-owned. The model-runtime ABI stores only graph indices
and dependency indices; it does not know customer plan names or model structure.
Optional cuts are managed outside the C++ runtime under `flash_rt/subgraphs/`.
See [`subgraph_stage_plans.md`](subgraph_stage_plans.md) for the customer
registration and capture-hook workflow.

The C++ runtime does not parse manifests or hardcode split names. For Pi0.5,
`frt_pi05_model_runtime_create_over` inherits the producer's declarations and
maps only the public ports it implements (`images`, optional `noise`,
`actions`). `step` is convenience only: same-stream stage chains may replay
sequentially; cross-stream dependencies require a host scheduler.

Pi0.5's default producer plan is:

- `stage_plan="full"`: one `infer` graph.

The optional `flash_rt.subgraphs.pi05.context_action` module can be enabled
before graph capture to add `stage_plan="context_action"`: `context` (prompt
copy + vision + encoder) followed by `decode_only` (action decoder). The
correctness gate checks full replay and split replay produce equivalent
actions for the same inputs.

It also exposes two IO faces over the same captured graphs:

- `io="python"`: Python frontend hot loop; normalized tensors are SWAP ports.
- `io="native"`: native C++ hot loop; raw images/actions are STAGED and noise
  remains a SWAP port. Export requires an immediate
  `frt_pi05_model_runtime_create_over` callback, so a declaration with
  placeholder verbs can never escape to a consumer.

The native `actions` port declares the logical output chunk delivered by
`get_output`, not necessarily the raw model buffer layout. A Pi0.5 producer may
store `(chunk, 32)` diffusion state internally while exposing `(chunk, 7)`
robot actions. GROOT-like or other VLA producers can expose `(50, 7)` through
the same descriptor; the chunk length is data on the port, not a runtime
constant.

## Graph-variant cache

Each `frt_graph` is a ShapeKey→exec table, optionally bounded by
`max_variants` (LRU). The exec layer provides the cache **mechanism** —
`frt_graph_evict`, `frt_graph_evict_lru`, `frt_graph_variant_count` — and the
model runtime provides the warm-phase capture door (`prepare`). Eviction and
budget **policy** live in the host (e.g. a Nexus graph store). Discipline:
fixed-shape or bucket-keyed graphs in production; hot-path misses fail loudly
(`FRT_ERR_NO_VARIANT`); evict only at a safe point, never while a variant may
be in flight.

## Freeze and evolution

`runtime/include/flashrt/*.h` is additive-only after v1: append fields, append
enum values, never reorder or remove. `frt_runtime_export_v1` follows its
version + `struct_size` contract; `frt_model_runtime_v1` consumers require
only `FRT_MODEL_RUNTIME_V1_BASE_SIZE` and probe future tails separately while
the model-runtime ABI version remains 1. The embedded verbs table is frozen.
Everything under `cpp/` may be refactored freely as long as the produced
struct — and the identity it fingerprints — is preserved.

## Native build isolation

Native C++ deployment support is an explicit product boundary controlled by
`FLASHRT_ENABLE_NATIVE_CPP`. When it is disabled, native model producers and
their operation-only dependencies do not enter the default Python build or
change its dynamic dependencies.

An existing common `csrc` symbol is a shared behavioral contract. A native
model adaptation must not change its signature, workspace requirement,
numerical behavior, launch behavior or default build ownership. If the native
path needs stricter math or a different workspace contract, add a distinctly
named operation under the opt-in native boundary. A genuinely common fix is a
separate infrastructure change with a complete caller inventory, old-path
regression evidence and calibration-cache impact analysis.

`csrc/native_cpp/` may contain reusable operation implementations missing from
the Python-oriented build, but it must not contain model topology, checkpoint
keys, model dimensions, prompt rules or stage policy. Its entry points are
internal linkage contracts, not another deployment ABI. The model producer's
shared library exports only its documented C API.

The native directory is a gap layer, not an alternate kernel library. Target
bindings must call an existing common operation when its public contract is
already sufficient. A distinct native operation must document the missing
contract dimension and remain independently named. Producer parity is measured
against an unchanged base build so a private implementation cannot validate
itself through a branch-local reference that contains the same numerical
change.

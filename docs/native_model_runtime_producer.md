# Native model-runtime producer guide

This guide defines how a native producer joins the stable
`frt_model_runtime_v1` boundary. A model implementation is an example of the
contract, not a reason to specialize the contract.

## Ownership

`runtime/` owns opaque handles, descriptors, identity construction, verbs, and
lifetime. `exec/` owns Buffer/Graph/Plan/Event/ShapeKey mechanisms. A producer
under `cpp/models/<model>/` owns checkpoint names, model dimensions, tokenizer
and formatter behavior, preprocessing, graph capture, workspace, and output
postprocessing. Nexus and other consumers interpret none of those semantics.

Do not add `model_kind`, `backend_kind`, model dimensions, checkpoint fields,
or a State object to the frozen ABI. Express the public face with ports,
stages, regions, verbs, and producer identity.

## Native frontend structure

A native model is a frontend over shared execution and kernel mechanisms. It
has one semantic pipeline and may lower that pipeline into different physical
plans for hardware and precision. A physical plan is not a second model.

- `model/` owns semantic stage order, graph cuts, model state, and public
  artifacts.
- `plans/<target>/` binds logical operations, weight packing, calibration
  observers, and plan-private scratch for one physical execution route.
- `csrc/` owns CUDA kernel implementations. Model frontends call its launcher
  APIs and must not define or compile private copies of those kernels.

The model owns one logical workspace schema: logical names, shapes, aliases,
and state meaning are stable across plans. A lowered plan supplies the physical
activation dtype and appends private buffers or aliases required by its
operations. It must not redefine model buffers or state semantics. New hardware
therefore adds operation bindings and private requirements without cloning the
semantic pipeline.

## Construction

Use one existing construction path:

1. `frt_runtime_builder_finish_model`: one producer builds export and model
   declarations under one fingerprint.
2. `frt_model_runtime_wrap`: an adapter adds a model face to an existing
   export whose identity already covers that face.
3. `frt_model_runtime_override_verbs`: an internal handoff retains an existing
   declaration while replacing hot verbs.

All paths reject STAGED inputs without `set_input` and STAGED outputs without
`get_output`. A factory may use an unpublished intermediate while assembling
an override, but only the final object with real verbs may leave the factory.

## Identity

The builder is the only fingerprint implementation. Include actual weights,
tokenizer/configuration, graph/stream placement, port schema and windows,
stage DAG, and ordered restore regions. Query the executing device for hardware
identity; do not copy the requested CMake architecture or a model default.

Manifest fields are discovery metadata, not a substitute for identity. A
schema or restore change intentionally produces a new fingerprint and rejects
old capsules.

## Calibration artifacts

Calibration is producer setup, not a new runtime mechanism. Keep model sites,
tensor dimensions, camera names, state/prompt semantics, and artifact format in
`cpp/models/<model>/`. Generic loaders may parse a standard container, but
`runtime/`, `exec/`, and consumers must not interpret model calibration data.

The host owns dataset traversal, decoding, synchronization, and sampling
policy. A model session consumes one complete observation per call and may
reduce repeated observations according to a documented policy. Named inputs
must be canonicalized before model math and reject missing, duplicate, or
unknown names atomically.

An artifact must bind every fact that changes scale meaning: observed hardware,
model and tokenizer content digests, precision, fixed shapes, schema/reducer
version, and successful sample count. When artifact bytes change inference
math, include the artifact digest in producer identity. Loading incompatible
metadata is a hard setup error, never a warning or fallback recalibration in the
hot process.

## Multiple producers and backends

Python, native CUDA, CPU, llama.cpp, and future producers expose the same
structural boundary but may have different graph counts, internal buffers,
workspace, identities, and synchronization implementations. Validate each
producer's invariants independently. Compare only a deliberately shared
semantic face through checked-in canonical records.

FlashRT supplies one exec implementation per process. Heterogeneous backend
instances enter above it through capsule backend vtables; do not add a runtime
backend registry or backend-kind ABI field.

## Hot path

Setup allocates storage, resolves names, loads weights, captures or adopts
graphs, and prepares variants. A control tick may update SWAP windows, execute
STAGED verbs, fire stages, and read output. It must not allocate device memory,
recapture, rebind graph pointers, or grow capacity. Oversized payloads fail.

Measure CUDA allocator APIs over the complete service iteration. Host
allocation claims require a host allocation counter scoped to the component;
CUDA traces cannot prove host allocator behavior.

## Schema workflow

For a face implemented by more than one producer:

1. Check in canonical `region:`, `port:`, and `stage:` records.
2. Generate records from every producer independently.
3. Diff each producer against the golden records.
4. Derive expected counts from the records instead of repeating constants.
5. Treat a golden update as a public contract review with fingerprint and
   restore-compatibility analysis.

Do not require implementation-private graph, buffer, manifest, or identity
records to match across backends.

## Pull request evidence

- C++ runtime tests in CUDA-off and affected hardware builds.
- Python runtime contract tests when the Python producer is supported.
- Producer-local lifecycle, schema, negative-input, and hot-loop gates.
- Numerical evidence appropriate to the boundary: bit-exact for identical
  graph/input bytes; a documented fixed tolerance for genuinely different
  backend math.
- Consumer adoption tests when descriptor or enum mapping changes.
- Migration notes for payload, fingerprint, packaging, or compatibility
  changes.

Use placeholders such as `<build-dir>` and `<checkpoint>` in public commands.
Do not publish local paths, host/container names, credentials, environment
dumps, internal URLs, or proprietary dataset/checkpoint identifiers.

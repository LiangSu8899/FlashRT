# FlashRT Runtime Export (`runtime/`)

The hand-off surface between a FlashRT **model runtime** (producer) and a
**host/serving layer** (consumer). One captured, replay-ready model is packaged
as one POD struct — `frt_runtime_export_v1` — and adopted by the consumer.

The exec contract (`docs/exec_contract.md`) fixes *how to replay*. The runtime
export fixes *what a deployed model IS*: which streams, graphs, buffers, and
restorable state regions exist, and the identity that stored state is bound to.
Both layers are mechanism only. Plans are deliberately **not** exported — DAG
orchestration is the consumer's job.

## Structure

```
producer (owns model + capture)              consumer (owns loop + state policy)
─────────────────────────────────            ──────────────────────────────────
Python setup/capture                          e.g. FlashRT-Nexus capsule host,
    flash_rt/runtime/export.py
      └─ _flashrt_runtime.Builder ──┐
native setup (same struct):          │        adopt(export*)
  native model runtime .so           ├──►  frt_runtime_export_v1  ◄──┘
    frt_model_runtime_open_v1(...)───┘        │ ctx, streams[], graphs[],
                                              │ buffers[], capsule_regions[],
                                              │ fingerprint/identity/manifest,
                                              │ owner + retain/release
                                              ▼
                                        replay / snapshot / restore
                                        via exec.h handles only
```

```
runtime/
  include/flashrt/runtime.h   the ABI (structs + builder). Consumers need ONLY
                              this header + exec.h — the struct is plain data.
  src/runtime_export.cpp      builder + export lifetime (no CUDA, no exec link)
  bindings/runtime_pybind.cpp `_flashrt_runtime` (setup/dev bridge)
  tests/                      model-free acceptance
flash_rt/runtime/export.py    Python producer: RuntimeExport / build_export()
```

## The contract, in five rules

1. **One struct, two producers.** Python fills it in-process; a native model
   runtime `.so` exports `frt_model_runtime_open_v1` (symbol name is in the
   model-runtime header) and fills the *same* struct. Consumers never change.
2. **Consumers see handles, never internals.** No Python, torch, model code, or
   kernel headers cross this boundary — only `frt_*` handles, POD descriptors,
   and strings owned by the export.
3. **Identity is split from discovery.** `identity` is the canonical string
   (weights digest, quant, kernel version, arch — supplied by the producer —
   plus graph names and the full capsule-region layout, appended by the
   builder). `fingerprint` = FNV-1a 64 of `identity`, computed **only** by the
   builder: one implementation, one hashing rule. `manifest_json` is free-form
   discovery data; editing it never invalidates stored state.
4. **Region order is contractual.** Restorable state regions are matched by
   position on restore, so their order/name/offset/bytes are all fingerprinted.
5. **Lifetime is explicit.** The consumer calls `retain(owner)` on adopt and
   `release(owner)` when done — from any thread. The phase-1 Python producer
   handles GIL acquisition inside `release`. While a reference is held, every
   handle in the struct (including `native_handle` stream pointers) stays
   valid; the Python process stays resident as the setup host, because CUDA
   graph execs are process-local by construction.

## Producing an export (phase 1, Python)

```python
export = pipeline.export_runtime(identity={"weights_sha256": digest})
# hand export.ptr (an frt_runtime_export_v1*) to the native consumer
```

`Pi05Pipeline.export_runtime()` is the reference producer: streams = the
capture stream, graphs = `infer` / `decode_only`, buffers = the pipeline IO
surface, default capsule region = the rollout boundary (`diffusion_noise`, the
region set validated by `serving/robot_recap/verify_capsule.py`).

## The model runtime ABI (`flashrt/model_runtime.h`)

The export describes a captured model's static execution assets; it does not
say how dynamic inputs enter the model each tick. That is the model runtime
ABI — `frt_model_runtime_v1` — the standard face of one deployed, tickable
model:

```
                 ┌────────────────────────────────────────────┐
   host tick ──► │ frt_model_runtime_v1                       │
                 │   ports[]   modality/dtype/shape/update    │
                 │   stages[]  subgraph DAG (export graphs)   │
                 │   verbs     set_input · get_output ·       │
                 │             prepare(warm) · step(sugar)    │
                 │   exp ────► frt_runtime_export_v1          │
                 │             (state/replay kernel, frozen)  │
                 └────────────────────────────────────────────┘
```

The contract is data first, verbs as sugar. Ports carry the load-bearing
**update class** — the two-speed hot path:

- `SWAP` — the port is a device-buffer window; the host writes raw bytes
  directly (its own copy verb / `cap_swap`). Microsecond lane, zero model
  code in the loop. (observation tensors, noise seeds, numeric state)
- `STAGED` — the runtime's `set_input` transforms host data (tokenize /
  resize / normalize / embed) into bound buffers. (prompt text, camera frames)
- `SETUP` — legal only outside the tick.

Production contract for both hot classes: never recapture, never allocate,
never rebind graph pointers — only buffer contents change. Replay graphs are
fixed-shape or bucket-keyed; a bucket miss is handled by `prepare` in the
warm phase, never inside a tick. `step` fires the declared stage order for
simple hosts; scheduling hosts fire stages themselves (that is what the stage
DAG is for).

Full structure map: [`cpp_runtime_design.md`](cpp_runtime_design.md).
Field-by-field interface reference: [`model_runtime_api.md`](model_runtime_api.md).

Two construction paths: the export builder assembles export + ports + stages
under ONE identity (`frt_runtime_builder_finish_model` — a port-schema change
changes the fingerprint), or `frt_model_runtime_wrap` wraps an existing
export with ports/verbs (the native-adapter path; identity inherited).
Consumers retain/release only the model runtime; it holds the export
reference internally.

## C++ model runtime layer

The runtime export is still only the hand-off surface. Model IO semantics live
one layer above it in FlashRT's native C++ path:

- `cpp/runtime/` defines the non-frozen native runtime manager interfaces.
- `cpp/modalities/` contains reusable modality primitives: tensor views,
  vision preprocess, and action postprocess.
- `cpp/families/` contains model-family contracts such as VLA.
- `cpp/models/<model>/` contains thin model adapters that bind family +
  modality primitives to concrete buffer names, shapes, normalization, action
  schemas, and state regions.

Nexus should not implement or own these rules. It adopts `frt_runtime_export_v1`
and drives snapshot/restore/replay; FlashRT model runtimes prepare inputs and
decode outputs.

Pi0.5 is the reference C++ model runtime under `cpp/models/pi05/`. It supports
both producer forms. The adopted-export path accepts Python- or native-produced
graphs and overlays native vision/action/prompt/state verbs. The Python-free
path loads safetensors and SentencePiece assets, selects an explicitly built
hardware backend, captures one native graph, and publishes the complete
`frt_model_runtime_v1` through `frt_model_runtime_open_v1`. SM120 currently
uses BF16 plus native FA2; SM110 uses FP8 E4M3 plus an identity-bound native
calibration artifact. Both expose the same backend-neutral contract, so Nexus
and serving hosts do not change.

The `flashrt_cpp_pi05_c` target also exposes the model-specific host/calibration
C API. These functions own Pi0.5 semantic transforms; they do not extend the
frozen runtime or exec structs. See [`pi05_io_contract.md`](pi05_io_contract.md)
and [`pi05_thor_native_fp8.md`](pi05_thor_native_fp8.md).

## Extending the ABI

Additive only after v1: append struct fields (bump `FRT_RUNTIME_ABI_VERSION` +
`struct_size`), append enum values, never reorder or remove. Consumers gate on
`abi_version`/`struct_size` before reading anything else.

## Validation

```
PYTHONPATH=.:./exec/build:./runtime/build python runtime/tests/test_runtime_export.py
./runtime/build/test_model_runtime
```

The export test covers: ctypes-mirror layout check of every field, fingerprint
determinism / identity sensitivity / region-order sensitivity / manifest
insensitivity, retain-release lifetime against the Python anchor, and replay
through exported handles. The model-runtime test covers: port/stage
declaration and validation, port schema in the identity fingerprint, verb
dispatch, and lifetime on both construction paths. The consumer side is
validated in the FlashRT-Nexus repo (adopt + snapshot/restore through the
capsule core).

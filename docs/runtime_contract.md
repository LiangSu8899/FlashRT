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
today:                                        e.g. FlashRT-Nexus capsule host,
  Python setup/capture                        a robot loop, a server shell
    flash_rt/runtime/export.py
      └─ _flashrt_runtime.Builder ──┐
later (same struct, host unchanged): │        adopt(export*)
  native model runtime .so           ├──►  frt_runtime_export_v1  ◄──┘
    frt_runtime_open_v1(config,&out)─┘        │ ctx, streams[], graphs[],
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

1. **One struct, two producers.** Today Python fills it in-process; later a
   native model runtime `.so` exports `frt_runtime_open_v1` (symbol name is in
   the header) and fills the *same* struct. Consumers never change.
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

![the model-runtime face](figures/model_runtime_face.png)

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

![two-speed tick](figures/two_speed_tick.png)

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

Pi0.5 is the reference C++ model runtime under `cpp/models/pi05/`. The current
implementation is the adopted-export path: setup/capture can still be produced
by Python, while the native runtime owns vision prepare, graph replay dispatch,
action decode, and export lifetime. It includes a CUDA vision
resize/normalize/cast path for device buffers, with CPU reference tests, and a
conservative D2H action staging path. The `flashrt_cpp_pi05_c` target exposes a
small C host ABI around this path so C/Python/serving shells can drive the
adopted export without including C++ classes. A future pure C++ checkpoint
loader/tokenizer/capture path must produce the same `frt_runtime_export_v1`, so
Nexus and serving hosts do not change.

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

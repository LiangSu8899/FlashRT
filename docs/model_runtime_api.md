# Model Runtime ABI — Interface Reference

Authoritative header: [`runtime/include/flashrt/model_runtime.h`](../runtime/include/flashrt/model_runtime.h)
(v1, additive-only). Design rationale: [`cpp_runtime_design.md`](cpp_runtime_design.md);
layer norms: [`runtime_contract.md`](runtime_contract.md).

## Enums (values are ABI-frozen)

| enum | values |
|---|---|
| modality | `TENSOR 0` · `IMAGE 1` · `TEXT 2` · `STATE 3` · `ACTION 4` · `AUDIO 5` · `DEPTH 6` · `FORCE 7` |
| dtype | `U8 0` · `F32 1` · `F16 2` · `BF16 3` · `I32 4` · `I64 5` |
| layout | `FLAT 0` · `HWC 1` · `NHWC 2` · `CHW 3` · `NCHW 4` |
| direction | `IN 0` · `OUT 1` |
| update | `SWAP 0` · `STAGED 1` · `SETUP 2` |

`STATE` is reserved for real proprioception; internal embedding/residual
windows are `TENSOR`. A `STAGED` declaration is a promise the port accepts hot
updates — a producer that cannot deliver that declares `SETUP` or omits the
port, never advertise-and-refuse.

## Payload conventions (STAGED `set_input`)

| modality | `data` points at | `bytes` |
|---|---|---|
| `IMAGE` / `DEPTH` | `frt_image_view[]`, matched to camera views **positionally** in declared order | `n_frames * sizeof(frt_image_view)` |
| `TEXT` | UTF-8 (no NUL required) | byte length |
| `TENSOR` / `STATE` / `ACTION` / `AUDIO` | raw bytes per the port's dtype/shape | byte length |

## Descriptors

`frt_runtime_port_desc` — one dynamic input/output:
`name`, `modality`, `dtype` (device-side tensor), `layout`, `direction`,
`update`, `required`, `shape[rank]` (−1 = bucket-variable),
`cadence_hint_hz` (advisory only), and the SWAP window `buffer`/`offset`/
`bytes` (null buffer = staged-only). Strings/arrays are owned by the runtime
object and stay valid while a reference is held.

`frt_runtime_stage_desc` — one schedulable stage: `graph` (index into the
export's graphs) plus `after[n_after]` (earlier stage indices). Declared array
order is the sequential order `step` uses.

## The object

```c
frt_model_runtime_v1 {
  abi_version / struct_size          gate before reading anything else
  exp                                the embedded frt_runtime_export_v1
  ports / n_ports                    dynamic-IO declarations
  stages / n_stages                  subgraph DAG
  self + verbs                       producer verbs (below)
  owner / retain / release           lifetime (see below)
}
```

**Verbs** (`frt_model_runtime_verbs`; every entry is always callable — absent
producer verbs are filled with unsupported stubs returning `-3`):

| verb | phase | semantics |
|---|---|---|
| `set_input(self, port, data, bytes, stream)` | HOT | write one IN port per the payload convention; `stream` = an export stream id or −1 for the port default |
| `get_output(self, port, out, capacity, written, stream)` | HOT | read one OUT port through the producer's postprocess; `capacity`/`written` are **bytes**; short buffers return `-5` with `written` = needed size |
| `prepare(self, graph, key)` | WARM only | ensure a shape-bucket variant exists (capture-on-miss); never call inside a tick |
| `step(self)` | HOT (sugar) | fire all stages in declared order; scheduling hosts fire stages themselves |
| `last_error(self)` | — | message for the most recent failure |

Status codes follow the pi05 C face: `0` ok, `-1` invalid, `-2` not found,
`-3` unsupported, `-4` shape mismatch, `-5` insufficient storage, `-6` backend.

**Hot contract** (SWAP writes and both hot verbs): never recapture, never
allocate, never rebind graph pointers — only buffer contents change.

**Lifetime**: the consumer retains/releases only the model runtime; the owner
holds one export reference internally. `retain`/`release` are thread-safe;
the Python producer acquires the GIL inside `release`, so native consumers may
drop references from any thread.

## Construction paths

**Integrated (preferred)** — the export builder assembles export + ports +
stages under one identity:

```c
frt_runtime_builder_add_port (b, name, modality, dtype, layout, direction,
                              update, required, shape, rank, cadence_hint_hz,
                              buffer, offset, bytes);
frt_runtime_builder_add_stage(b, graph_index, after, n_after);
frt_model_runtime_v1* m = frt_runtime_builder_finish_model(
    b, &verbs, verbs_self, owner, retain_owner, release_owner);
```

Identity covers each port's schema **and its bound window** (buffer index
into the declared buffers array, offset, bytes) plus the stage DAG; only
`cadence_hint_hz` stays out. A port-schema or window change therefore changes
the fingerprint, and stored state is refused. Canonical record formats:

```
port:<i>:<name>:<modality>:<dtype>:<layout>:<dir>:<update>:<req>:<d0,d1,..>:<buf_idx>:<off>:<bytes>
stage:<i>:<graph>:<after0,after1,..>
```

**Adapter** — wrap an existing export with ports/verbs (the native path over
a Python-built export; identity inherited, ports not re-fingerprinted):

```c
frt_model_runtime_v1* m = frt_model_runtime_wrap(
    exp, ports, n_ports, stages, n_stages,
    &verbs, verbs_self, wrapper_owner, wrapper_release);
```

**Native factory (symbol convention)** — a model-runtime `.so` exports
`FRT_MODEL_RUNTIME_OPEN_V1_SYMBOL`:
`int frt_model_runtime_open_v1(const char* config_json, frt_model_runtime_v1** out)`.

**Reference producers**: `frt_pi05_model_runtime_create`
(`cpp/models/pi05/`) and `Pi05Pipeline.export_model_runtime()`
(`flash_rt/models/pi05/runtime_export.py`, via
`flash_rt.runtime.export.build_model_runtime`).

## Graph-cache verbs (exec layer)

For host eviction/budget policy — mechanism only, and only at safe points
(never while a variant may be in flight):

```c
int    frt_graph_evict(frt_graph, frt_shape_key);   /* FRT_ERR_NO_VARIANT if absent */
int    frt_graph_evict_lru(frt_graph);
size_t frt_graph_variant_count(frt_graph);
```

## Validation

```
./runtime/build/test_model_runtime                     # ABI, identity, lifetime, stubs
ctest --test-dir cpp/build                             # modalities, staging pool, pi05 faces
PYTHONPATH=.:./exec/build:./runtime/build \
  python runtime/tests/test_model_runtime_py.py        # Python producer through C fn pointers
```

The consumer side (adoption, hot-input contract, real-model tick) is
validated in the FlashRT-Nexus repository.

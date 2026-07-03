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
order is the sequential order `step` uses. Stage streams are not a separate
field: replay stream comes from the referenced graph descriptor.

Graph stream placement and the stage DAG are deployment identity. A cut from
`full` to `context_action`, a stream move, or a dependency change changes the
fingerprint; this is intentional state-safety, not a cache miss.

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
graph:<name>:<stream_id>
stage:<i>:<graph>:<after0,after1,..>
```

**Adapter** — wrap an existing export with ports/verbs; identity inherited,
ports not re-fingerprinted:

```c
frt_model_runtime_v1* m = frt_model_runtime_wrap(
    exp, ports, n_ports, stages, n_stages,
    &verbs, verbs_self, wrapper_owner, wrapper_release);
```

**Verb override** — inherit an existing model-runtime declaration and replace
only verbs. This is the standard hand-off when a setup producer owns capture,
ports, stage DAG and fingerprint, while a native C++ runtime owns hot-path
transforms:

```c
frt_model_runtime_v1* m = frt_model_runtime_override_verbs(
    producer_model, &verbs, verbs_self,
    native_owner, retain_native_owner, release_native_owner);
```

The override retains `producer_model`, so inherited port/stage pointers remain
valid even if the original producer reference is released first. Deployment
identity is unchanged.

**Native factory (symbol convention)** — a model-runtime `.so` exports
`FRT_MODEL_RUNTIME_OPEN_V1_SYMBOL`:
`int frt_model_runtime_open_v1(const char* config_json, frt_model_runtime_v1** out)`.

**Reference producers**: `Pi05Pipeline.export_model_runtime()`
(`flash_rt/models/pi05/runtime_export.py`, via
`flash_rt.runtime.export.build_model_runtime`) and the native Pi0.5 verb
overlay `frt_pi05_model_runtime_create_over` (`cpp/models/pi05/`).

## Producer layout: model contract vs hardware pipeline

The clean default is one export module per logical model family contract:

```
flash_rt/models/<model>/runtime_export.py     ports, stages, identity, export helpers
flash_rt/models/<model>/pipeline_rtx.py       RTX graph capture and live buffers
flash_rt/models/<model>/pipeline_thor.py      Thor graph capture and live buffers
cpp/models/<model>/                           native hot-path verb overlay
```

`runtime_export.py` is not the hardware implementation. It is the shared
contract that lowers a ready pipeline into `frt_model_runtime_v1`: declared
ports, graph/stage descriptors, stream placement, identity fields, and optional
stage-plan selection. Each hardware pipeline owns its graph capture, buffers,
kernel choices, calibration, and setup policy, then delegates
`export_model_runtime(...)` to that model export module.

The C++ runtime consumes the resulting declaration. It distinguishes the model
by the native factory/overlay it loads (`cpp/models/<model>/...`) and by the
declared port schema it implements; it should not branch on hardware unless a
real input/output transform differs. Hardware identity comes from the producer:
the graph handles, stream table, buffer descriptors, pipeline class, precision,
architecture, and other setup fields included in the canonical identity and
fingerprint.

This means an RTX Pi0.5 producer and a Thor Pi0.5 producer may both feed the
same native Pi0.5 C++ runtime if they expose the same logical model-runtime
contract. If a hardware path needs incompatible graph names, stage cuts, port
shapes, or hot-input semantics, it is a different deployment identity and must
either use distinct stage-plan names/model keys or a separate native overlay.

Pi0.5 export knobs:

- `stage_plan="full"` exports one `infer` stage.
- `stage_plan="context_action"` exports `context -> decode_only` only after
  enabling `flash_rt.subgraphs.pi05.context_action` before graph capture.
- `stage_plan="context_rtc_prefix_action"` exports
  `context -> decode_rtc_prefix` only after enabling
  `flash_rt.subgraphs.pi05.rtc_prefix` before graph capture. This mode adds
  raw SWAP tensor ports `prev_action_chunk` and `actions_raw` with shape
  `(chunk_length, 32)`; `prefix_len` is fixed at capture/export time through
  `stage_plan_kwargs`.
- `stage_plan="context_rtc_vjp_guided_action"` exports
  `context -> decode_rtc_vjp_guided` only when a producer-supplied
  `DenoiserVjpProvider` captured or adopted that graph. This mode adds raw
  SWAP tensor ports `prev_action_chunk`, `actions_raw`, `prefix_weights`
  `(chunk_length,) f32`, and `guidance_weight` `(1,) f32`.
- `io="python"` exports the Python frontend SWAP-tensor face.
- `io="native"` exports the C++ runtime face (`images/actions` STAGED,
  `noise` SWAP), intended for `frt_pi05_model_runtime_create_over`.

For VLA outputs, the `actions` port shape is the logical host-visible action
chunk after postprocess: `(chunk_length, robot_action_dim)` for flat robot
actions. It is not required to match the internal diffusion/action buffer
shape. Pi0.5, for example, keeps `diffusion_noise` as `(chunk_length, 32)` but
declares native `actions` as `(chunk_length, 7)` for LIBERO-style robot output.
Other deployments may export `(50, 7)` or another fixed action shape through
the same port contract; schedulers must read the declared port shape instead
of assuming `(10, 7)`.

Stage-plan names are resolved by `flash_rt.subgraphs.stage_plan`:

```python
from flash_rt.subgraphs.stage_plan import Stage, StagePlan, register_stage_plan

register_stage_plan(
    "prefill_decode",
    StagePlan((
        Stage("prefill", graph="prefill"),
        Stage("decode", graph="decode", after=("prefill",)),
    ), name="prefill_decode"),
    model="my_llm",
)
```

Subgraph packages should keep their built-in plans in `stage_plans.py` and
import that module before export. Customer plans use the same registration API.
Registered factories may receive export-time `stage_plan_kwargs` for choices
such as diffusion chunk size. The ABI still receives only graph indices and
dependency indices; it never receives the plan name as executable policy.
The full capture-hook workflow is in
[`subgraph_stage_plans.md`](subgraph_stage_plans.md).

Export-time selection:

```python
model = pipeline.export_model_runtime(
    stage_plan="prefill_decode",
    io="native",
)

chunked = pipeline.export_model_runtime(
    stage_plan="denoise_chunks",
    stage_plan_kwargs={"chunk_size": 5, "total_steps": 10},
    io="native",
)
```

Every graph named by the resolved plan must already exist in the producer's
export. Validation happens during export: unknown graph, unknown stream, stream
mismatch, duplicate stage name, or a dependency on a later stage is rejected.
For structural cuts, the required gate is bit-exact split replay versus full
replay under the same input/noise. Approximate thresholds are not sufficient
for a boundary-only split.

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

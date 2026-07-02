# Subgraph Stage Plans

Subgraph cuts are producer/customer setup policy. They are not part of
`runtime/`, not part of the C++ hot-path runtime, and not part of Nexus.
The frozen model-runtime ABI only receives the result: graph indices and
dependency indices.

The default model pipeline should stay clean: capture its normal full graph
and any built-in graph it already needs. Optional deployment cuts are enabled
through `flash_rt/subgraphs/` modules before graph capture.

## Ownership

- **Pipeline** exposes stable stage bodies or a generic capture-hook call site.
  It should not know customer plan names.
- **Subgraph module** captures/adopts extra graph handles and registers them
  for export.
- **StagePlan registry** names the DAG over those exported graph names.
- **Nexus** sees only `cap_model_runtime.stages[]`; it never parses plan names.

## The two things a cut must provide

A plan name alone is not enough. Every cut has two halves:

1. **Capture side**: create replayable graph handles named `vision`, `encoder`,
   `decoder`, `denoise_0_4`, etc.
2. **Declaration side**: register a `StagePlan` that orders those graph names.

If the declaration references a graph that was not captured/exported, export
fails during setup.

## Customer module layout

Built-in and customer cuts live under `flash_rt/subgraphs/`:

```
flash_rt/subgraphs/
  stage_plan.py              Stage / StagePlan / register_stage_plan
  capture.py                 capture hook + export graph registration helpers
  pi05/
    context_action.py        optional Pi0.5 capture hook
    rtc_prefix.py            optional Pi0.5 RTC-prefix action hook
    rtc_vjp_guided.py        Pi0.5 VJP-guided action provider contract
    stage_plans.py           Pi0.5 plan declarations
```

External deployments can keep the same shape in their own package and import
it before capture/export.

## Namespacing rules

- Register plans with a model key: `register_stage_plan(name, plan,
  model="pi05")`. Different models use different keys.
- Do not use `replace=True` except for the package that owns that model key or
  for an explicit deployment override. The default rejects accidental
  duplicate registration.
- Capture hooks are attached to one frontend or one pipeline instance. Enabling
  a cut for one deployed model does not mutate other model instances.
- A plan name is not enough to run: export still validates that every graph
  named by the resolved plan exists in the current producer. This catches using
  a plan with the wrong model, hardware path, or pipeline variant.
- If two variants of the same model intentionally need incompatible plan
  meanings, use distinct model keys or distinct plan names, e.g. `pi05_rtx`
  vs `pi05_thor`, or `context_action_rtx` vs `context_action_thor`.

## Pi0.5 example: enable `context -> action`

Default Pi0.5 capture stays unchanged. To add the optional split:

```python
from flash_rt.subgraphs.pi05.context_action import enable

model = flash_rt.load_model(...)
enable(model)                             # before model.predict / graph capture
model.predict(images, prompt=prompt)      # captures full + decode + context
pipeline = model._pipe.pipeline

runtime = pipeline.export_model_runtime(
    stage_plan="context_action",
    io="native",
)
```

`enable(...)` accepts the public `VLAModel`, the Pi0.5 frontend, or an
already-built pipeline. For lazy frontends it records a pending hook and applies
it when the pipeline is created, before graph capture.

The optional module captures a `context` graph and registers it for export.
`flash_rt.subgraphs.pi05.stage_plans` declares:

```python
from flash_rt.subgraphs.stage_plan import Stage, StagePlan, register_stage_plan

register_stage_plan(
    "context_action",
    StagePlan((
        Stage("context", graph="context"),
        Stage("action", graph="decode_only", after=("context",)),
    ), name="context_action"),
    model="pi05",
)
```

## Pi0.5 RTC-prefix action graph

`context_action` only changes graph boundaries. It does not change model math.
For RTC-style guided action chunks, the action graph must explicitly read the
previous raw action chunk inside the denoise loop. Enable the separate
RTC-prefix hook before graph capture:

```python
from flash_rt.subgraphs.pi05.rtc_prefix import enable

model = flash_rt.load_model(...)
enable(model, prefix_len=2)              # before model.predict / graph capture
model.predict(images, prompt=prompt)     # captures full + decode + context + rtc action
pipeline = model._pipe.pipeline

runtime = pipeline.export_model_runtime(
    stage_plan="context_rtc_prefix_action",
    stage_plan_kwargs={"prefix_len": 2},
    io="native",
)
```

This exports the same `context` stage plus an `action` stage over graph
`decode_rtc_prefix`. The plan adds two raw tensor ports only for this explicit
mode:

- `prev_action_chunk` — input SWAP, shape `(chunk_length, 32)`, raw model
  action space, read by the captured action graph.
- `actions_raw` — output SWAP, shape `(chunk_length, 32)`, aliases the raw
  diffusion/action buffer so a host can feed the next previous-chunk input.

`prefix_len` is a capture-time value. The current implementation captures a
fixed-size copy inside each denoise step, so different prefix lengths are
different deployment identities. A per-tick dynamic prefix length would require
a producer-owned mask kernel or separate graph variants; it is not a Nexus
scheduler parameter.

## Pi0.5 full RTC VJP-guided action contract

The complete Kinetix-style RTC path is a different producer mode:

```python
from flash_rt.subgraphs.pi05.rtc_vjp_guided import enable

enable(model, provider=my_vjp_provider)   # before graph capture
model.predict(images, prompt=prompt)

runtime = pipeline.export_model_runtime(
    stage_plan="context_rtc_vjp_guided_action",
    io="native",
)
```

This mode is intentionally provider-based. Pi0.5's stock FlashRT pipeline is an
inference graph built from custom kernels and GEMM launches; it does not expose
autograd. A real provider must capture or adopt a graph named
`decode_rtc_vjp_guided` and register it for export. The graph implements the
denoise-step correction:

```text
x1, vjp_fun, v_t = vjp(denoiser, x_t, has_aux=True)
error = (prev_action_chunk - x1) * prefix_weights[:, None]
correction = vjp_fun(error)[0]
v_t = v_t + guidance_weight * correction
x_t = x_t + dt * v_t
```

The exact guidance-weight formula and prefix schedule belong to the provider,
because they are model math. Nexus only sees the resulting stage and ports.

When this plan is selected, export adds these raw SWAP ports:

- `prev_action_chunk` — input, raw model action space, shape
  `(chunk_length, 32)`.
- `actions_raw` — output, raw model action space, shape `(chunk_length, 32)`.
- `prefix_weights` — input float32, shape `(chunk_length,)`.
- `guidance_weight` — input float32 scalar, shape `(1,)`.

If no provider registered `decode_rtc_vjp_guided`, export fails during setup.
That failure is deliberate: `context_rtc_prefix_action` demonstrates async RTC
and prefix locking; `context_rtc_vjp_guided_action` must not silently degrade to
the prefix-only graph.

## Customer example: `vision -> encoder -> decoder`

This cut should not be hardcoded into the main pipeline. Put it in a subgraph
module and enable it explicitly.

Capture hook:

```python
from flash_rt.subgraphs.capture import (
    capture_graph,
    register_captured_graph,
    register_capture_hook,
)


def enable(pipeline):
    register_capture_hook(pipeline, _capture)


def _run_vision(pl, stream):
    pl.vision_encoder(stream)


def _run_encoder(pl, stream):
    pl._copy_lang_embeds_to_encoder_x(stream=stream)
    pl.transformer_encoder(stream)


def _run_decoder(pl, stream):
    pl.transformer_decoder(stream)


def _capture(pl, stream_handle, stream_int):
    vision = capture_graph(
        pl, stream_handle, lambda: _run_vision(pl, stream_int))
    encoder = capture_graph(
        pl, stream_handle, lambda: _run_encoder(pl, stream_int))
    decoder = capture_graph(
        pl, stream_handle, lambda: _run_decoder(pl, stream_int))

    register_captured_graph(pl, "vision", vision, exec_name="my_vla_vision")
    register_captured_graph(pl, "encoder", encoder, exec_name="my_vla_encoder")
    register_captured_graph(pl, "decoder", decoder, exec_name="my_vla_decoder")
```

Plan declaration:

```python
from flash_rt.subgraphs.stage_plan import Stage, StagePlan, register_stage_plan

register_stage_plan(
    "vision_encoder_decoder",
    StagePlan((
        Stage("vision", graph="vision"),
        Stage("encoder", graph="encoder", after=("vision",)),
        Stage("decoder", graph="decoder", after=("encoder",)),
    ), name="vision_encoder_decoder"),
    model="my_vla",
)
```

Export:

```python
pipeline.export_model_runtime(
    stage_plan="vision_encoder_decoder",
    io="native",
)
```

## Parametric cuts

For diffusion or WAM-style loops, register a factory:

```python
from flash_rt.subgraphs.stage_plan import StagePlan, register_stage_plan


def denoise_chunks(*, chunk_size: int, total_steps: int = 10) -> StagePlan:
    graphs = tuple(
        f"denoise_{i}_{min(i + chunk_size, total_steps) - 1}"
        for i in range(0, total_steps, chunk_size)
    )
    return StagePlan.chain(
        "denoise_chunks",
        graphs,
        metadata={"chunk_size": chunk_size, "total_steps": total_steps},
    )


register_stage_plan("denoise_chunks", denoise_chunks, model="my_vla")
```

Call:

```python
pipeline.export_model_runtime(
    stage_plan="denoise_chunks",
    stage_plan_kwargs={"chunk_size": 5, "total_steps": 10},
    io="native",
)
```

The capture hook must create and register the matching graph names
(`denoise_0_4`, `denoise_5_9`, etc.) before export.
`register_captured_graph` is intentionally one captured CUDA graph per variant;
if a producer owns a pre-built multi-variant exec graph, register it with
`register_export_graph` instead.

## Validation

Export validates:

- stage plan is non-empty;
- stage names are unique;
- every referenced graph name exists;
- every referenced stream exists;
- stage stream matches the graph stream;
- dependencies point only to earlier stages.

These are setup-time failures. The hot path never discovers new graph cuts.

Correctness validation is stricter than cosine similarity: for a graph cut that
only reorders capture boundaries, `split stage replay == full graph replay`
must be bit-exact for the same inputs and fixed RNG/noise. If a split changes
math, precision, kernel choice, or dynamic control flow, it is a new producer
mode and must carry its own identity and acceptance gate.

The stage plan is deployment identity. Changing from `full` to
`context_action`, moving a graph to another stream, or changing dependency
edges changes the canonical identity/fingerprint; stored capsules from one
plan are deliberately refused by another plan.

Shared hand-off buffers are producer-owned. Dependencies order stages within
one iteration, but they do not make cross-iteration overlap safe. If a later
stage reads a buffer that an earlier stage writes, do not fire the next
iteration's writer while the previous reader can still be in flight. True
overlap requires the producer to capture/export double-buffered stages.

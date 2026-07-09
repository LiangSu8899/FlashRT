# Mindon Pi0.5 Integration Guide

This guide describes how a Mindon C++ host should integrate Pi0.5 through the
existing FlashRT runtime/model-runtime contracts. It is a deployment guide, not
a new ABI.

## Layer Ownership

FlashRT owns:

- checkpoint loading and graph capture;
- ports, stages, streams, buffers, capsule regions, identity, and fingerprint;
- model-specific IO semantics: tokenizer, prompt formatting, image
  preprocess, state normalization/discretization, and action postprocess;
- `set_input`, `get_output`, `prepare`, and `step` producer verbs.

Nexus owns:

- adoption of `frt_runtime_export_v1` / `frt_model_runtime_v1`;
- capsule snapshot/restore/fork/move over declared regions;
- stage scheduling and interaction modes;
- embedded and transport adapters that map external payloads to declared
  ports.

Mindon owns:

- the application/control loop;
- camera/state/prompt transport into the adopted ports;
- action publication and deadline policy.

Nexus should not learn Pi0.5 tokenizer, tensor layout, normalization, or
action schema rules. If a host needs richer state, the producer must export a
richer model-runtime face.

## Recommended Lanes

### Lane A: Available Now

Use a resident Python setup producer, then run the hot loop in C++.

Flow:

1. Start a process that embeds CPython or calls a Python setup function.
2. Load the Pi0.5 checkpoint through the FlashRT Python frontend.
3. Capture graphs and call `pipeline.export_model_runtime(io="native", ...)`.
4. Pass the returned `frt_model_runtime_v1*` to the C++ host.
5. Adopt it into Nexus with `flashrt_adopt_model_runtime`.
6. Warm any declared graph variants.
7. Drive `images`, `noise`, and `actions` through the C++ hot loop.

In Lane A, prompt is setup-time. The current adopted-export face does not
export hot `prompt` or `state` ports. A request may repeat the setup prompt for
bookkeeping, but it cannot change the model prompt dynamically.

### Lane B: After Prompt/State Staging

Use the same setup/adopt path as Lane A, but the producer additionally exports
real hot ports:

- `prompt: TEXT/STAGED`
- `state: STATE/STAGED`

The C++ host updates these ports with `cap_model_set_input` or the embedded
session equivalent. The producer formats, tokenizes, embeds, and writes the
fixed prompt window. Nexus remains unchanged.

### Lane C: Future Native Producer

Load a native FlashRT shared object and call:

```c
int frt_model_runtime_open_v1(const char* config_json,
                              frt_model_runtime_v1** out);
```

The returned struct must expose the same public model-runtime contract as the
Python setup producer. The host and Nexus adoption code must not change when
switching between Lane A and Lane C.

The current C++ shared object exports this symbol as a native-v2 configuration
gate. It validates `io`, checkpoint path, tokenizer model path, fixed prompt
mode, prompt capacity, state dimension, and the Pi0.5 embedding metadata in
`model.safetensors`. It also verifies action/state q01/q99 metadata from
openpi `norm_stats.json` or LeRobot policy normalizer/unnormalizer
safetensors, then returns unsupported until the native checkpoint
materialization and CUDA graph capture path are implemented. Hosts may use this
to wire dynamic loading and error handling, but must keep using Lane A or B for
execution.

## No-HTTP C++ Host Shape

For same-process control loops, prefer Nexus embedded/session APIs over HTTP.
The high-level loop is:

```
producer setup -> frt_model_runtime_v1
adopt -> cap_model_runtime
open embedded session
for each control tick:
  update declared input ports
  tick or fire stages
  read declared output ports
optional:
  snapshot / restore named capsules
```

The C++ loop should discover ports by name and then rely on the declared port
shape, dtype, direction, and update class. It should not hard-code `(10, 7)`,
graph names, or internal buffer names.

## Port Update Rules

For `SWAP` ports:

- write the declared buffer window directly through the capsule/backend copy
  mechanism;
- do not call `set_input`;
- verify byte count against `port.bytes`.

For `STAGED` ports:

- call the producer verb through `cap_model_set_input` or
  `nexus_embedded_set_input`;
- pass bytes in the payload convention declared by `frt_model_runtime_v1`;
- expect shape/status errors for invalid input.

For `SETUP` ports:

- never update them inside a control tick.

## Mapping Existing Mindon Calls

| Mindon call | Integration point |
|---|---|
| `Prepare` | warm phase, producer `prepare(graph, key)` |
| `Warmup` | host policy: `prepare` plus warm ticks |
| `Infer` | `cap_model_tick`, `nexus_embedded_tick`, or explicit stage firing |
| `Sync` | host/backend stream sync or embedded session synchronization |
| `GetOutput` | `cap_model_get_output` / `nexus_embedded_get_output` |

Do not introduce a second runtime API beside `frt_model_runtime_v1`. The
existing verbs already carry these phases.

## Prompt and State

Pi0.5 state is rendered into the language prompt. It is not an independent
model tensor. The producer path is:

```
raw proprioception -> normalize -> 256-bin discretize -> prompt string
-> token ids -> embedding gather -> encoder_x prompt window
```

Until Lane B lands, prompt and state changes require a setup-time producer
refresh. After Lane B, Mindon should send task text to the `prompt` port and
raw proprioception to the `state` port. The producer owns all formatting and
normalization details.

## Image Input

Mindon should pass camera frames as `frt_image_view[]` to the `images`
`IMAGE/STAGED` port, or through the matching Nexus embedded input. Frames are
matched by declared position, not by runtime graph names.

The current Pi0.5 native producer stages host pixels into the
`observation_images_normalized` device tensor and normalizes to `[-1, 1]`.
Pass `u8` `RGB8` frames in HWC layout. BGR/RGBA/GRAY, CHW, and non-`u8` inputs
are rejected instead of silently reinterpreted. Use the producer documentation
in `docs/pi05_io_contract.md` for accepted formats and shape rules.

## Action Output

Read the `actions` port shape to determine chunk length and action dimension.
The output is the host-visible robot action chunk after producer postprocess.
For raw model action state, use `actions_raw` when the producer exports it. In
the Pi0.5 `native_v2` face this is a raw `TENSOR/SWAP` alias of
`diffusion_noise` with shape `(chunk, 32)`.

## Capsule Boundaries

Capsules snapshot exactly the producer-declared regions, in declared order.
Mindon should treat capsule contents as opaque bytes. A fingerprint mismatch
on restore is a deployment mismatch and must fail loudly.

If prompt context becomes a region in a future producer face, old capsules
will not restore into that new face. That is expected and protects state
safety.

## Configuration Sketch

Lane A setup in a Python producer plugin should export:

```python
model = pipeline.export_model_runtime(
    identity={"deployment": "mindon-pi05"},
    stage_plan="full",
    io="native",
)
```

A split or RTC deployment may choose another producer-registered stage plan,
but the C++ host still sees only the adopted stage array.

## Acceptance Checklist

- The host discovers ports and shapes from `cap_model_runtime`.
- `images` updates use `STAGED`; `noise` updates use `SWAP`.
- `actions` capacity is computed from the declared output shape and dtype.
- The warm phase finishes before the first control tick.
- The hot loop performs no graph capture, allocation, or graph rebinding.
- Snapshot/restore is tested within one live capture.
- Nexus core code remains unchanged for model-specific semantics.

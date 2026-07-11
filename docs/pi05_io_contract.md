# Pi0.5 Native Model Runtime IO Contract

This document is the deployment-facing IO contract for the Pi0.5 native
model-runtime face. It is intentionally limited to the public runtime/model
runtime ABI:

- `frt_runtime_export_v1` in `runtime/include/flashrt/runtime.h`
- `frt_model_runtime_v1` in `runtime/include/flashrt/model_runtime.h`
- Pi0.5 producer declarations in `flash_rt/models/pi05/runtime_export.py`
- Pi0.5 native verb overlay in `cpp/models/pi05/`

It does not freeze the C++ implementation classes under `cpp/`. Those classes
may evolve as long as the exported ports, stages, regions, identity, and hot
contract remain valid.

## Current Native Face

The current Pi0.5 `io="native"` export declares three host-visible ports.
This is the contract implemented by `frt_pi05_model_runtime_create_over`.

| port | modality/update | direction | dtype/layout/shape | backing |
|---|---|---|---|---|
| `images` | `IMAGE/STAGED` | input | device tensor dtype, `NHWC`, `(num_views, 224, 224, 3)` | `observation_images_normalized` |
| `noise` | `TENSOR/SWAP` | input | device tensor dtype, `FLAT`, `(chunk_length, 32)` | `diffusion_noise` |
| `actions` | `ACTION/STAGED` | output | host `f32`, `FLAT`, `(chunk_length, robot_action_dim)` | staged-only; no raw window |

Current source of truth:

- Export declaration: `flash_rt/models/pi05/runtime_export.py`,
  `export_model_runtime(..., io="native")`
- Native verb implementation: `cpp/models/pi05/src/model_runtime.cpp`
- C++ modality binding: `cpp/models/pi05/src/runtime.cpp`,
  `cpp/models/pi05/src/io.cpp`, `cpp/models/pi05/src/spec.cpp`

`io="native"` and `io="native_v2"` are declaration-only handoffs. Their
discovery manifest carries `declaration_only: true`; callers must pass them to
`frt_pi05_model_runtime_create_over` before adoption. Generic Python-produced
model runtimes reject STAGED ports without matching input/output verbs.

There is deliberately no `prompt` port on the adopted-export path today. The
prompt embedding is prepared by the producer before graph capture/export. A
producer must not declare a `TEXT/STAGED` or `STATE/STAGED` port until the
native verb can really update that input on the hot path.

## Native V2 Face

The `io="native_v2"` export adds prompt/state staging and a raw action alias.
Adding these ports changes the model-runtime identity and therefore the
fingerprint. Existing capsules from the old face must refuse restore into this
face.

| port | modality/update | direction | dtype/layout/shape | backing |
|---|---|---|---|---|
| `prompt` | `TEXT/STAGED` | input | UTF-8 bytes, `FLAT`, variable length | staged by C++ runtime |
| `state` | `STATE/STAGED` | input | host `f32`, `FLAT`, `(state_dim,)` | staged by C++ runtime |
| `images` | `IMAGE/STAGED` | input | device tensor dtype, `NHWC`, `(num_views, 224, 224, 3)` | `observation_images_normalized` |
| `noise` | `TENSOR/SWAP` | input | device tensor dtype, `FLAT`, `(chunk_length, 32)` | `diffusion_noise` |
| `actions` | `ACTION/STAGED` | output | host `f32`, `FLAT`, `(chunk_length, robot_action_dim)` | staged-only; no raw window |
| `actions_raw` | `TENSOR/SWAP` | output | device tensor dtype, `FLAT`, `(chunk_length, 32)` | `diffusion_noise` |

For Pi0.5, proprioceptive state is not an independent model tensor. It is
normalized, discretized into OpenPI-compatible 256-bin state tokens, rendered
into the prompt text, tokenized, embedded, and written into the language rows
of `encoder_x`. Therefore prompt and state updates are one producer-owned text
staging path.

Internal model buffers such as `encoder_x`, KV/cache windows, residual
streams, and `diffusion_noise` are not `STATE` ports. They are `TENSOR` ports
when exposed as hot IO, or runtime buffers/capsule regions when they are part
of a restorable boundary.

## STAGED Payloads

The payload conventions are inherited from `frt_model_runtime_v1`.

| modality | `set_input` data | bytes |
|---|---|---|
| `IMAGE` / `DEPTH` | `frt_image_view[]` | `n_frames * sizeof(frt_image_view)` |
| `TEXT` | UTF-8 bytes | byte length |
| `TENSOR` / `STATE` / `ACTION` / `AUDIO` | raw bytes per the port dtype and shape | byte length |

For `IMAGE`, frames are matched positionally to the producer-declared camera
view order. The Pi0.5 view order is:

1. `image`
2. `wrist_image`
3. `wrist_image_right`

Deployments with fewer views export a shorter `num_views` and use the prefix
of that view order.

## Image Input

The current native image input accepts host `frt_image_view[]` and stages the
data into the device `observation_images_normalized` buffer before replay.

Producer-owned preprocessing:

- view count is checked against the exported `images` port shape;
- frame payloads are host `u8` pixels in `RGB8`/`HWC` layout;
- target tensor is `(num_views, 224, 224, 3)`;
- output layout is `NHWC`;
- output dtype is the exported tensor dtype, normally BF16 for the FP8 path;
- normalization is `x / 127.5 - 1.0`;
- resizing to 224x224 is producer-owned.

`stride_bytes=0` means tightly packed RGB. A positive stride may include row
padding but must be at least `width * 3`; negative and short strides are shape
errors. The native CUDA and CPU reference paths are bit-exact in BF16 over the
validation matrix (`1x1`, odd dimensions, non-4:3 inputs, wide/tall inputs,
224x224, and padded rows).

The Pi0.5 native face rejects unsupported input shape, dtype, layout, pixel
format, or view count with a shape/status error. BGR, grayscale, RGBA, CHW, and
non-`u8` frames are not silently converted at the Pi0.5 contract boundary. If a
deployment supports more pixel formats, the supported set must be documented by
the producer and tested against the CPU reference path.

Compatibility note: the legacy `frt_pi05_runtime_prepare_vision` C API keeps
accepting its explicit `BGR8`, `RGBA8`, `BGRA8`, and `GRAY8` enum values and
converts them through the shared modality implementation. This compatibility
does not expand the `frt_model_runtime_v1` native face: its `IMAGE/STAGED`
deployment contract remains strict `RGB8/u8/HWC`.

## Noise Input

`noise` is a `TENSOR/SWAP` port. The host writes its raw bytes directly into
the `diffusion_noise` window, usually by `cap_swap` after Nexus adoption or by
the equivalent runtime/backend copy mechanism. Calling `set_input` on this
port is unsupported by design: SWAP means the device window is the interface.

Shape is `(chunk_length, 32)`. `chunk_length` is declared by the producer and
must be read from the port shape; host code must not assume `(10, 32)`.

## Action Output

`actions` is the host-visible robot action chunk after producer-owned
postprocess. Its declared dtype is F32 because that is the payload returned by
`get_output`; it does not expose the BF16 diffusion window as backing.

The logical output shape is:

```
(chunk_length, robot_action_dim)
```

For LIBERO-style Pi0.5 deployments, `robot_action_dim` is typically 7. Other
deployments may export a different fixed robot action dimension. Consumers and
schedulers must read the declared port shape instead of hard-coding `(10, 7)`.

The internal model output remains `(chunk_length, 32)` in `diffusion_noise`.
The native `actions` STAGED output slices the robot dimensions and applies the
deployment action normalization statistics. With q01/q99 stats, the affine
parameters are:

```
mean   = (q01 + q99) / 2
stddev = (q99 - q01) / 2
```

The C++ postprocess path clamps normalized action values to the configured
domain before applying the affine transform. Any raw `(chunk_length, 32)` face
must be exported as a separate `TENSOR/SWAP` output. The Pi0.5 `native_v2`
face declares this as `actions_raw`; RTC stage plans also use the same port
name. Nexus must treat it as a declared raw byte window, not model internals.

## Lifecycle Mapping

Mindon-style lifecycle names map to the existing ABI. Do not add a parallel
API family for the same phases.

| requested name | existing contract | phase |
|---|---|---|
| `Prepare` | `prepare(graph, key)` | warm only |
| `Warmup` | host policy: call `prepare` for needed variants, then run warm ticks | warm |
| `Infer` | `step()` sugar or host-scheduled stage replay | hot |
| `Sync` | host/backend stream synchronization | hot or drain |
| `GetOutput` | `get_output(port, out, capacity, &written, stream)` | hot |

`prepare` is the only place a shape-bucket miss may capture or materialize a
variant. A hot tick must not recapture, allocate, or rebind graph pointers.
The Pi0.5 C++ face fixes its vision frames, action D2H staging, task/formatted
prompt strings, tokenizer ids, and normalized-state storage during setup.
Payloads that would grow those workspaces return a shape/capacity error; there
is no larger-allocation fallback in the hot path. These workspace changes do
not alter the port schema or deployment fingerprint.

## Identity and Capsule Regions

The following changes are deployment identity changes:

- adding/removing/reordering ports;
- changing a port modality, dtype, layout, direction, update class, required
  flag, shape, bound buffer index, offset, or byte window;
- changing graph names or default stream placement;
- changing the stage DAG;
- adding/removing/reordering capsule regions;
- changing a region name, buffer, offset, byte length, or flags.

The following are not deployment identity changes:

- editing `manifest_json`;
- changing `cadence_hint_hz`.

Prompt/state staging does not by itself make prompt context a capsule region.
A restorable prompt context would have to include the embedding, attention
lengths, decoder position/RoPE, and the CPU semantic cache used by later
independent prompt/state updates. The current face can rebuild those values
from its declared inputs, so it does not advertise partial prompt restoration.
Region layout and order are fingerprinted; adding a complete prompt region in
a later face will intentionally invalidate old capsules.

## Current Integration Lanes

There are three supported integration lanes:

- Lane A, current: Python setup/capture/export stays resident in the process;
  the hot loop adopts `frt_model_runtime_v1` and runs through C++/Nexus.
- Lane B: an adopted setup producer exposes real hot `prompt`/`state` STAGED
  ports and the C++ overlay owns their transforms.
- Lane C, current on RTX SM120: a C++ shared object implements
  `frt_model_runtime_open_v1(config_json, &out)` and produces the same public
  struct without Python setup.

The Pi0.5 C++ shared object exports `frt_model_runtime_open_v1` as a complete
native-v2 producer when built with CUDA kernels, native FA2, and SentencePiece.
Execution currently requires RTX SM120. The factory requires `io="native_v2"`,
`checkpoint_path`, `tokenizer_model_path`, `state_prompt_mode="fixed"`,
`max_prompt_tokens >= 200`, and a positive `state_dim`; `num_views`, `chunk`,
`num_steps`, and `vision_pool_factor` are optional fixed setup values. It parses
`checkpoint_path/model.safetensors` through the native read-only mmap loader to
verify the complete 812-tensor Pi0.5 inventory: all 27 vision layers, all 18
language encoder layers, all 18 action-expert layers, embeddings/final norms,
projectors, action projections, and time MLP. It also verifies action/state q01/q99
dimensions from either openpi `norm_stats.json` or LeRobot policy
normalizer/unnormalizer safetensors. Safetensors tensor byte ranges must match
dtype/shape, and normalization quantiles must be finite ordered pairs. Builds
without native FA2 or SentencePiece validate the config and return unsupported;
they do not advertise a runtime they cannot execute. The mmap and parsed tensor
views are setup-side assets; they never enter the model-runtime ABI or hot path.

The native setup layer also carries CPU reference transforms matching the
existing PyTorch producer: source BF16 rounding for vision/decoder weights,
`OIHW -> HWIO` patch permutation, Q/K head interleave, QKV and gate/up fusion,
and FP32 encoder RMSNorm fold before the final BF16 rounding. Real-checkpoint
gates compare the resulting BF16 bytes against PyTorch for both bare OpenPI
keys and LeRobot `model.`-prefixed keys.

Materialized device weights use `frt_buffer` allocations owned by the native
producer's `frt_ctx`. They are internal setup assets, not model ports and not
capsule regions. Upload is complete before capture; duplicate logical names or
typed shape/payload mismatches fail setup. The same store carries BF16, FP8
E4M3, INT8, and FP32 scale buffers without introducing a model-level state
object. Destroying the context releases the device weights after graphs and
plans, preserving the exec ownership order.

The composed materializer covers language-encoder, action-expert, and vision
weights. Encoder layers emit the five pipeline groups (`attn_qkv`, `attn_o`,
`ffn_gate`, `ffn_up`, and `ffn_down`) with FP32 RMS folds. Decoder layers emit
those groups plus the four AdaRMS modulation tensors and the optional merged
gate/up buffer used by the FP16 path. Vision setup emits patch/position/final
norm and multimodal-projector globals plus the twelve per-layer attention,
FFN, and normalization buffers. Decoder globals include final AdaRMS
modulation, time MLP, generated time embeddings, and action projections. The
action output projection is pre-scaled by `-1/num_steps` after source BF16
rounding; 5-step and 10-step schedules are byte-exact with the PyTorch
producer. The prompt embedding table is materialized separately to keep its
approximately 1 GiB allocation explicit. These paths have been exercised
against the two supported real checkpoint layouts. The checkpoint inventory
also validates the language final norm and expert LM head even though the
current Pi0.5 pipeline does not consume them. The native producer materializes
the full BF16 store before capture and keeps it under the graph context lifetime.

Native setup quantization reproduces the PyTorch producer's per-tensor FP8
E4M3 weights in either `kn` or `nk` layout and per-output-channel INT8 weights
in `[N,K]` layout. FP8 scalar descales and INT8 channel scales are FP32 device
buffers. Real-checkpoint gates compare both quantized bytes and scale bytes;
the precision choice remains producer setup policy and does not alter ports,
regions, or the exec mechanism.

The setup packer derives low-precision buffers from the already uploaded BF16
fallback, so both paths share exactly the same transformed source bytes. It
stores packed weights under `fp8.*` or `int8.*` names and their typed FP32
scales under the matching `.scale` names in the same context-owned store.

Full BF16 assembly has one ordered setup path: vision globals and 27 layers,
18 language-encoder layers, 18 action-expert layers, decoder globals, then the
prompt embedding table. With merged decoder gate/up buffers enabled this owns
613 logical device buffers. Assembly options make `num_steps`, merged gate/up,
and the large embedding allocation explicit; they are producer configuration,
not ABI fields.

Full FP8 packing follows the producer's exact site inventory: four GEMM
weights for each vision layer plus the projector, four for each encoder layer,
and four for each decoder layer. Encoder gate/up columns are merged during
setup. INT8 packing remains independently selectable for vision, encoder, and
decoder and preserves their existing four/five/five weights-per-layer policy.

The native kernel layer is CPython-independent and links the existing
`GemmRunner` implementation directly. Setup warms required BF16 GEMM shapes,
captures the complete `infer` graph through `frt_graph_capture`, and exports
exactly one shape-key variant (`0`).

The native core workspace maps every vision, encoder, decoder, style, action,
RTC, and reusable scratch allocation to a context-owned `frt_buffer`. There is
no model-level State object. With vision pooling disabled, `vision_x_pooled`
is an explicit alias of `vision_x` (34 logical names, 33 allocations); pooled
deployments allocate it separately. Buffer shapes are fixed from `num_views`,
`max_prompt_tokens`, `chunk_size`, `num_steps`, and `vision_pool_factor` before
capture, and BF16 RMS-one constants, attention backend storage, and generated
decoder style contents are initialized during setup.

Native RoPE setup uses the same float64 frequency/phase computation and BF16
interleaved `[cos, sin]` layout as the Python producer. Encoder and
prompt-relative decoder slices are byte-exact against NumPy/ml_dtypes for
pooled and unpooled configurations. Decoder slice updates reuse one stable
buffer across prompt lengths; vision position embeddings are expanded per view
with setup-side D2D copies from the typed weight store.

Decoder time/style precompute is also native setup work. It consumes the
generated time embeddings, time-MLP weights, 18 layers of AdaRMS modulation,
and final modulation from the typed store; it reuses existing workspace
buffers as scratch and writes the four persistent style buffers without a
temporary device allocation. The GEMM, explicit BF16 bias round-trip, and
float-SiLU sequence is BF16 bit-exact with the PyTorch producer on both
supported checkpoint layouts.

The native kernel driver also owns the BF16 forward primitives used around
GEMM and attention: RMS/Layer/AdaRMS normalization, residual and gated
residual updates, GELU/gated GELU, QKV split with fixed or device-position
RoPE, patch im2col, and vision pooling. These are direct typed calls to the
existing CUDA implementations, with CPU-reference and captured-replay gates;
they do not route through pybind or introduce a second kernel implementation.

The native vision graph composes patch im2col/embedding, all 27 SigLIP layers,
per-view FA2, optional fixed-factor spatial pooling, final LayerNorm, and the
1152-to-2048 multimodal projector. Position embedding expansion remains setup
work. With inputs restored before each of 100 replays, the graph keeps one
variant; final SigLIP and projected encoder tokens reach cosine 0.9999 or
better against the layer-by-layer PyTorch reference on both supported
checkpoint layouts.

The first composed BF16 forward segment is the encoder QKV path:
RMSNorm, the folded QKV projection, RoPE split, and writes into the selected
layer of the shared K/V cache. Layer 17 is also the complete final encoder
layer behavior because the producer intentionally stops after populating its
cache. Its outputs are bit-exact (`cos=1`, `max=0`) against the PyTorch
checkpoint path for both OpenPI and LeRobot layouts, and the segment captures
and replays with one graph variant.

Encoder layers 0-16 extend that segment through fixed-shape FA2, output
projection, residual/RMS normalization, the separate gate/up projections,
gated GELU, down projection, and the final residual update. A captured layer 0
replayed 100 times remains a single variant and reaches cosine 0.999992 versus
the original PyTorch path on both checkpoint layouts. Layer 17 keeps the
intentional cache-only early exit described above.

The native encoder composes all 18 layers into one captured graph while
preserving that final cache-only behavior. Restoring the input before each of
100 replays produces one graph variant. On both OpenPI and LeRobot checkpoint
layouts, the final encoder state and layer-17 Q/K/V each reach cosine 0.9999 or
better against the layer-by-layer PyTorch reference. This composition owns no
state object: activations and K/V remain context-backed buffers.

The native decoder composes one BF16 AdaRMS/cross-attention/FFN layer, one
flow-matching update, and the complete 10-step diffusion graph. Decoder K/V is
appended at the device-side fixed-prompt position in the encoder cache; style
and noise remain context-backed buffers. Full 10-step captures replay 100 times
with one variant on both checkpoint layouts. Independent first and final
schedule steps reach cosine 0.9999 or better against PyTorch; the accumulated
endpoint gate remains part of the real-episode end-to-end validation because
synthetic random K/V amplifies SDPA-versus-FA2 rounding across steps.

The native graph owner now assembles the completed segments into one `infer`
capture: prompt copy, vision, encoder, then diffusion. Prompt embeddings live
in a separate persistent buffer because `encoder_x` is an in-place residual
stream; each replay captures a D2D copy into its language window. Both
checkpoint layouts complete 100 full replays with one variant, bit-identical
outputs for restored inputs, and a constant workspace allocation count. The
persistent prompt source, not the overwritten encoder rows, is the primary
prompt-context capsule candidate.

RTX attention owns a separate context-backed buffer set rather than borrowing
Torch tensors: SigLIP Q/K/V, encoder Q and 18-layer shared K/V cache, decoder
Q, fixed-shape `seqused/devpos` int32 values, FA2 outputs/LSE, and split-KV
accumulators. Layer K/V pointers are stable offsets into one cache allocation.
Updating a fixed prompt length writes the same three scalar buffers without
allocation or rebinding. The Python-free attention driver calls the vendored
FA2 raw C entries directly for SigLIP, fixed-shape encoder `seqused`, and
decoder `seqused` split-KV. Its graph gate changes the prompt length after
capture, replays 100 times with one variant, and verifies the new device-side
valid length is observed. `flash_rt_fa2` remains a thin Python adapter over the
same `libflashrt_fa2_raw` kernel owner.

The native builder publishes one `infer` graph/stage and the ordered ports
`prompt`, `state`, `images`, `noise`, `actions`, and `actions_raw`. Identity
includes SM120, model/tokenizer SHA-256 values, prompt mode, fixed shapes, and
schedule parameters. The only capsule region is `rollout_boundary` over the
diffusion/action buffer. Prompt embeddings, encoder/decoder caches, attention
lengths, and RoPE remain context-owned `frt_buffer` workspace that each infer
rebuilds; they are not falsely advertised as independently restorable state.

The returned verb override retains the builder-produced base model, which
retains the export and graph owner. Releasing the final public model releases
the overlay, export, captured graph, buffers, stream, and context in ownership
order without a second lifecycle owner.

CUDA graph execs are process-local objects. They are not serialized as a
portable artifact. The native producer therefore loads assets and captures the
graph in the replay process.

## Validation

The minimum regression set for this contract is:

```
PYTHONPATH=.:./exec/build:./runtime/build python runtime/tests/test_runtime_export.py
PYTHONPATH=.:./exec/build:./runtime/build python runtime/tests/test_model_runtime_py.py
./runtime/build/test_model_runtime
ctest --test-dir cpp/build --output-on-failure
```

Real-checkpoint gates:

```
python cpp/tests/gate_pi05_native_weight_ops.py \
  --checkpoint <pi05 checkpoint> \
  --probe cpp/build/pi05_native_weight_probe
```

```
FLASHRT_BUILD_DIR=<build-dir> \
  python cpp/tests/gate_pi05_model_runtime_export.py \
  --lib <build-dir>/libflashrt_cpp_pi05_c.so ...
python cpp/tests/gate_pi05_c_api_export.py ...
```

The Python-produced overlay gate uses the FA2-enabled, SentencePiece-off SM120
build because Python already owns prompt/tokenizer setup. Native-v2 factory
gates use the SentencePiece-enabled SM120 build in a Python-free producer
process. In either lane, pybind modules and the producer library must come from
the same build directory; graph and buffer handles cannot cross exec builds.

Prompt/state STAGED ports require token-exact, formatter string-exact,
embedding bit-exact, fixed-vs-exact E2E cosine, and hot-contract coverage; a
producer must not retain the declarations if any required verb is unavailable.

The native factory lifecycle gate is:

```
<build-dir>/pi05_native_open_probe \
  <checkpoint> <tokenizer.model>
```

Run it against both OpenPI and LeRobot checkpoint layouts. It validates the
public schema, one captured variant, prompt/state/image staging, direct SWAP
noise input, finite action output, and retain/release teardown.

Python and C++ native-v2 producers must also publish identical canonical
port/stage/region records (their producer identity and fingerprints remain
different):

```
FLASHRT_BUILD_DIR=<build-dir> \
  python cpp/tests/gate_pi05_native_schema_parity.py \
  --checkpoint <openpi-pytorch-checkpoint> \
  --tokenizer <tokenizer.model> \
  --native-probe <build-dir>/pi05_native_open_probe
```

The native formatter and tokenizer must also remain token-exact over real
prompt/state traffic:

```
python cpp/tests/gate_pi05_tokenizer_corpus.py \
  --dataset <libero-lerobot-root> \
  --checkpoint <openpi-pytorch-checkpoint> \
  --tokenizer <tokenizer.model> \
  --probe <build-dir>/pi05_tokenizer_corpus_probe \
  --count 10000
```

This gate normalizes every recorded state with the checkpoint q01/q99 values,
renders the full state prompt through the native formatter, and compares every
valid token ID with OpenPI `PaligemmaTokenizer`. The 10,000-record reference
run used a lightweight oracle whose tokenizer and formatter logic is kept
line-equivalent with upstream OpenPI; it covered 20 token lengths from 43
through 62 with zero mismatches. This source-level oracle is distinct from the
official-environment end-to-end gate below.

The real-episode numerical gate compares against the official OpenPI PyTorch
`PI0Pytorch.sample_actions` path, not another native intermediate:

```
python cpp/tests/gate_pi05_native_e2e.py \
  --checkpoint <openpi-pytorch-checkpoint> \
  --tokenizer <tokenizer.model> \
  --dataset <libero-lerobot-root> \
  --probe <build-dir>/pi05_native_e2e_probe \
  --episode 0 --frame 0
```

The gate rounds the shared initial noise to the exported BF16 contract before
both runs. Raw and robot action outputs must each reach cosine 0.9999 against
the official FP32 residual path. Separately, the STAGED `actions` bytes must
match q01/q99 postprocess recomputed from the native BF16 `actions_raw` window
at `rtol=atol=1e-6`; this keeps numerical precision and IO semantics as two
independent acceptance checks. Set `OPENPI_BASELINE_SITE_PACKAGES` when the
official OpenPI Transformers replacement is installed in a separate prefix.

Collect replay-only native and Python BF16 Nsight traces with the same fixed
shape. Setup, graph capture, prompt/image/noise staging, and output copies must
remain outside the CUDA profiler range:

```
FLASHRT_PROFILE_RANGE=1 nsys profile --trace=cuda \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o <native-report> \
  <build-dir>/pi05_native_open_probe \
  <checkpoint> <tokenizer.model>

nsys profile --trace=cuda --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o <python-report> \
  python cpp/tests/profile_pi05_python_replay.py \
  --checkpoint <checkpoint> --num-views 2 --steps 10

nsys stats --report cuda_gpu_trace --format csv \
  <native-report>.nsys-rep > <native-trace>.csv
nsys stats --report cuda_gpu_trace --format csv \
  <python-report>.nsys-rep > <python-trace>.csv
python cpp/tests/gate_pi05_kernel_sequence.py \
  --native <native-trace>.csv --python <python-trace>.csv
```

The comparator rejects unknown kernels and requires equal raw event counts.
Its explicit equivalence list covers only selected GEMM kernel variants,
GEMM workspace-init versus split-K reduction helpers, `add_bias` versus the
equivalent `bias_res` form, and the two negative-infinity fill symbols. On the
reference RTX 5090 SM120 run both traces contained 3,576 raw events and their
3,172 logical-kernel sequences were exactly equal.

The separate hot allocator gate profiles 1,000 graph replays without tracing
individual graph nodes:

```
FLASHRT_PROFILE_REPLAYS=1000 nsys profile --trace=cuda \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o <hot-report> \
  <build-dir>/pi05_native_open_probe \
  <checkpoint> <tokenizer.model>
nsys stats --report cuda_api_trace --format csv \
  <hot-report>.nsys-rep > <hot-api-trace>.csv
python cpp/tests/gate_pi05_hot_allocator.py \
  --trace <hot-api-trace>.csv --expected-replays 1000
```

The gate requires exactly 1,000 `cudaGraphLaunch` calls and rejects CUDA/driver
device allocation, host registration, mempool creation, virtual-memory map,
and corresponding release APIs. The probe independently requires one graph
variant after the final replay. The reference SM120 run observed zero allocator
calls across 2,001 CUDA API calls.

The same native probe can gate the complete hot state staging chain
(normalization, formatting, tokenization, embedding gather, and prompt-length
device update):

```
FLASHRT_HOT_STATE_UPDATES=1000 FLASHRT_HOT_STATE_P99_US=1000 \
  <build-dir>/pi05_native_open_probe \
  <checkpoint> <tokenizer.model>
```

The probe varies all eight state dimensions, warms 20 updates, measures 1,000
updates, and requires the graph variant count to remain one. The reference
RTX 5090 SM120 run measured p50/p99/max of 39.31/41.70/43.44 microseconds,
well below the explicit one-millisecond p99 contract.

The unload probe has no static dependency on the Pi0.5 producer. It resolves
the factory from the shared object, exercises an extra retain/release pair,
releases the final model reference, and only then unloads the producer:

```
<build-dir>/pi05_native_dlopen_probe \
  <build-dir>/libflashrt_cpp_pi05_c.so \
  <checkpoint> <tokenizer.model> 1
```

For the ASAN build, instrument the producer, runtime, exec library, and probe,
not only the executable. CUDA needs `protect_shadow_gap=0` in this environment
to avoid an address-space collision with ASAN's default shadow gap:

```
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:protect_shadow_gap=0 \
  <asan-build-dir>/pi05_native_dlopen_probe \
  <asan-build-dir>/libflashrt_cpp_pi05_c.so \
  <checkpoint> <tokenizer.model> 1
```

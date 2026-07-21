# Cosmos3-Edge Thor Baseline

`config="cosmos3_edge"` is the Thor baseline runner for NVIDIA's official
Cosmos Framework inference path. It is intentionally upstream-first: run this
baseline, capture latency/outputs, then port the FlashRT optimized denoise/action
path behind the same config.

Current status:

- Official Cosmos Framework baseline is runnable from FlashRT.
- P0 denoise dump is captured for `av_inverse_0`.
- P1 scheduler/latent replay scaffold is wired as `backend="replay"`.
- P1 Torch reference validates the native transformer math path through all 28
  layers for step 0 action velocity, and through the full 30-step UniPC denoise
  loop as `backend="torch_ref"`.
- `backend="flashrt"` is wired to the current optimized eager denoise engine:
  static AV buffers, static und K/V cache, BF16 FlashRT GEMMs/RMSNorm/relu2,
  fused qk-norm+RoPE, cached timestep embeddings, native action input tail copy,
  native flat velocity fill, native action bias+timestep add, native action
  bias+tail clear, native action row scatter/gather, and FA4 native gen
  attention.
- `backend="official_action_only"` runs the official no-dump Cosmos Framework
  path, but skips inverse-dynamics vision decode/save and rewrites
  `sample_outputs.json` to action-only.
- `official_action_only` also has an opt-in `live_dump_out` capture mode that
  records a denoise safetensors dump from the live official run without editing
  the external Cosmos Framework checkout.
- `official_action_only` has an opt-in live FlashRT handoff prototype: step 0
  runs official `_get_velocity` to capture the live boundary, then later
  denoise steps short-circuit `_get_velocity` through FlashRT velocity.
- `--live-prelayer-bootstrap` is a stronger handoff mode: it captures the live
  boundary before official decoder layers, aborts the official transformer, and
  returns FlashRT velocity for all 30 denoise steps.
- `--live-warm-request` is the measured request path for the live handoff: it
  enables pre-layer bootstrap and one official warmup request so cold FA4/engine
  setup is paid before the timed batch.
- In live handoff mode, the Cosmos UniPC sampler is patched in-process to use
  FlashRT's native `cosmos3_edge_unipc_step_f32_bf16` path for the fixed
  single-sample Edge AV schedule when available. Set
  `FLASHRT_COSMOS3_EDGE_LIVE_NATIVE_UNIPC=0` to fall back to the official
  Python scheduler.

## Quantized whole-graph denoise engine (`models/cosmos3_edge/pipeline_thor.py`)

`CosmosEdgeThor` is the optimized denoise engine: pre-quantized gen-tower
weights, static buffers, per-layer joint und+gen K/V with the static und prefix
pre-installed, FA4 attention, native UniPC, and the entire 30-step denoise
captured in **one CUDA graph**. It reads no environment variables; precision is
selected by the caller.

- `quant="fp8"` (default): per-tensor FP8 E4M3 weights for all six gen-tower
  projections, calibrated static activation scales (one dynamic-scale denoise
  pass records per-site amax ceilings via `fp8_accumulate_scale_max`, margin
  1.25), fused `residual_add_rms_norm_fp8` / `quantize_fp8_static` /
  `cosmos3_edge_relu2_to_fp8_static_bf16` quant chain.
- `ffn_fp4=True` (opt-in): NVFP4 W4A4 up/down FFN GEMMs via `flash_rt_fp4`
  (`cutlass_fp4_sq_fp16` + fused `cosmos3_edge_res_rms_fp4_sfa_bf16` and
  `cosmos3_edge_relu2_fp4_sfa_fp16` quantizers, dynamic per-16-block scales).
- The gen-stream fused qk-norm+RoPE kernel is the warp-per-head register
  variant (one warp per `(row, head)` vector, `__shfl_xor` rope-partner
  exchange, no shared memory or barriers).
- The last layer runs slim at M=60: only the action rows feed the head, so
  final-layer Q/attention/o-proj/FFN shrink to the 60 action tokens while K/V
  still cover the full sequence (math-identical; `--no-slim-last` to disable).

Measured on Thor (15 back-to-back iterations, hot regime, `av_inverse_0`,
official eager denoise baseline 33.14s; gate cos >= 0.999, rel_l2 < 3%,
both engines re-validated on the second-seed dump):

| engine | denoise P50 | speedup | final action cos | rel_l2 |
|---|---|---|---|---|
| `--engine bf16-eager` (per-op path) | 11.998s | 2.76x | 0.9999959 | 0.29% |
| `--engine fp8` | 5.792s | 5.72x | 0.9999834 | 0.59% |
| `--engine fp8 --ffn-fp4` | 5.456s | 6.07x | 0.9997067 | 2.42% |

```bash
python benchmarks/cosmos3_edge_thor_denoise.py --engine fp8 --iters 15 \
  --warmup-steps 30 --enforce-gates            # FP8 default
python benchmarks/cosmos3_edge_thor_denoise.py --engine fp8 --ffn-fp4 \
  --iters 15 --warmup-steps 30 --enforce-gates # max-quantization variant
```

FP8 is the recommended default (large precision margin); `--ffn-fp4` trades
~6% latency for a still-passing but thinner gate margin. Measured negatives,
kept default-off or not landed: a fused QKV wide GEMM (cuBLASLt N=4096 tactic
regression, behind `qkv_fused=`) and SageAttention2 int8 gen attention (the
SM80-era int8 mma core is both numerically wrong and slower than FA4 on
SM110). The largest remaining denoise levers are a relu2 epilogue fused into
the up GEMM and a Thor-native int8/fp8 attention kernel.

## Quickstart

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_flashrt \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework
```

The example must run inside the same container/venv used for the official
baseline, with `cosmos-framework` importable. `--vae-path` is optional only when
the container can download `Wan2.2_VAE.pth` itself.

To validate the no-dump action-only official path without decoded vision output:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_action_only_official \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only
```

To capture a live official denoise dump while keeping the action-only output:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_live_dump \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-dump-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_denoise.safetensors
```

This capture mode is for boundary validation, not baseline timing: enabling the
official velocity postprocess hook makes Cosmos run an extra uncond velocity
branch even when guidance is 1.0.

To exercise the live FlashRT handoff prototype:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_live_handoff \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-flashrt-handoff \
  --live-boundary-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_handoff_boundary.safetensors
```

This is a correctness/architecture prototype. It still pays official step 0,
FlashRT engine initialization, and cold FA4 setup inside
`generate_samples_from_batch`; the current archived run reports 26.10s
`generate_batch`. `--live-boundary-out` is only for debug artifact archival; the
runtime can construct the boundary in memory, which measured 26.20s and the same
action metrics.

To skip the official step-0 transformer as well:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_live_prelayer \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-prelayer-bootstrap \
  --live-handoff-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_trace.json
```

The archived pre-layer run proves the official decoder was aborted
(`prelayer_aborted=true`) and returns action within the gate, but one-shot
latency is still dominated by FA4 first-use compilation: first FlashRT velocity
8.50s, steady FlashRT velocity 0.398s/step.

To archive the smaller native-denoise/VLM input contract at the pre-layer
boundary:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_live_prelayer_boundary \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-prelayer-bootstrap \
  --live-boundary-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_boundary.safetensors \
  --live-handoff-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_boundary_handoff_trace.json \
  --upstream-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_boundary_upstream_trace.json
```

Archived `official_action_only_live_prelayer_boundary.safetensors` metrics:

- 60.36 MiB file, 31 tensors / 60.36 MiB tensor footprint
- contains `lm_in/full_only_seq`, layer-0 input, RoPE, VFM tokens, action
  tokens, timestep, and `noise_x`
- does not store layer-0 output or step-0 velocity; the older step-0 handoff
  boundary was 87.77 MiB / 36 tensors because it archived those outputs
- run action gate: cos 0.9999959 / rel_l2 0.291% / max_abs 0.00687
- `generate_batch`: 26.70s; native UniPC scheduler: 30 steps

This is the input contract for a future native VLM/prefill-to-denoise engine.
It still relies on official upstream code to synthesize `lm_in/full_only_seq`,
RoPE, VFM tokens, and action tokens from the raw request.

To replay that pre-layer boundary as the live denoise engine input:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_live_boundary_in \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-boundary-in dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_boundary.safetensors \
  --live-handoff-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_boundary_in_handoff_trace.json \
  --upstream-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_boundary_in_upstream_trace.json
```

Archived `official_action_only_live_boundary_in_outputs` metrics:

- `OmniInference.generate_batch`: 26.03s
- official velocity calls: 0; FlashRT velocity calls: 30
- boundary load/validate: 0.8ms; engine construct: 1.29s
- native UniPC scheduler: 1 run / 30 steps, scheduler step total 0.0071s
- final action cos 0.9999959 / rel_l2 0.291% / max_abs 0.00687

This is still artifact-backed. Its purpose is to make the native prefill target
executable: once native VAE/SigLIP/VLM code can synthesize the same boundary
tensors live, the denoise engine no longer needs official transformer capture.

The boundary-in path can also be combined with the current warm repeated-request
prepare cache:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_live_boundary_in_warm_prepare_cache \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-boundary-in dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_boundary.safetensors \
  --live-warm-request \
  --cache-warmup-prepare \
  --prepare-slim-no-raw-state-vision \
  --prepare-slim-derive-condition-reference \
  --prepare-slim-derive-initial-noise \
  --live-handoff-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_boundary_in_warm_request_prepare_cache_slim_derive_noise_handoff_trace.json \
  --upstream-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_boundary_in_warm_request_prepare_cache_slim_derive_noise_upstream_trace.json
```

Archived metrics for this service-style path:

- measured `OmniInference.generate_batch`: 12.08s; denoise: 12.07s
- warmup `OmniInference.generate_batch`: 26.08s
- official velocity calls: 0; FlashRT velocity calls: 60
- measured `_prepare_inference_data`: 0.042s; prepare hit/store: 1/1
- measured VAE encode calls: 0
- native UniPC scheduler: 2 runs / 60 steps, scheduler step total 0.0134s
- final action cos 0.9999959 / rel_l2 0.291% / max_abs 0.00687

The helper `_derive_step0_vfm_boundary_from_prepare_payload` formalizes the
part of the pre-layer boundary that is already implied by the slim prepare
contract. From `official_action_only_live_prelayer_prepare_slim_derive_noise_dump.pt`
and seed 0 plus `embed_tokens.weight` it derives 28 step-0
VFM/noise/pack/position/RoPE/text-causal tensors exactly:

- `steps/00/noise_x`
- VFM vision tokens, condition mask, sequence indexes, and empty no-loss/noisy
  metadata
- VFM action noisy tokens, condition mask, sequence/timestep/loss/noisy indexes,
  domain id, and raw action dim
- fixed packed metadata: causal/full indices, offsets, sample offsets, and
  `position_ids`
- layer-0 RoPE cos/sin tensors for causal and full-only sequences
- causal LM hidden state from `cond_text_tokens + eos + <|vision_start|>` and
  `embed_tokens.weight`

Verifier metrics: 28 tensors / 10.65 MiB derived exactly. The layer-0 input
tensors are checked as aliases of `lm_in` (2 tensors / 25.10 MiB). The remaining
1 tensor / 24.61 MiB is the full-only LM packed hidden state. The verifier now
probes that tensor too and requires all 6300 rows to be bit-exact; the action
rows require the reference action projection/timestep path to run with CUDA
matmul TF32 disabled. This narrows the remaining native VLM/prefill task to
synthesizing the same prepare/prefill state from live raw request data rather
than resolving an unresolved packed-hidden-state formula gap.

To run the live denoise engine from the 9.18 MiB slim prepare artifact instead
of the 60.36 MiB pre-layer boundary artifact:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_live_prepare_boundary_in \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --prepare-replay-in dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_noise_dump.pt \
  --live-boundary-prepare-in dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_noise_dump.pt \
  --live-boundary-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prepare_boundary_in_derived_boundary.safetensors \
  --live-handoff-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prepare_boundary_in_handoff_trace.json \
  --upstream-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prepare_boundary_in_upstream_trace.json
```

Archived `official_action_only_live_prepare_boundary_in_outputs` metrics:

- input artifact: 9.18 MiB slim-derived-noise prepare payload
- derived debug boundary: 62.65 MiB / 31 tensors; larger than the 60.36 MiB
  pre-layer artifact only because the debug save clones shared tensor views
- boundary derive: 0.258s; engine construct: 1.074s
- `OmniInference.generate_batch`: 22.24s; denoise: 22.22s; decode: 0.0084s
- official velocity calls: 0; FlashRT velocity calls: 30
- measured `_prepare_inference_data`: 0.101s; measured VAE encode calls: 0
- native UniPC scheduler: 1 run / 30 steps
- final action cos 0.9999959 / rel_l2 0.291% / max_abs 0.00687

This removes the 60.36 MiB executable boundary artifact from the runtime input
contract for replay-style testing. It is still artifact-backed prepare replay:
raw-video no-artifact live inference still needs native VAE/SigLIP/VLM prefill.

To run the same executable-boundary synthesis from this request's in-memory
prepared state, without reading a prepare artifact:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prepare_live_outputs \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-boundary-prepare-live \
  --live-boundary-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prepare_live_derived_boundary.safetensors \
  --live-handoff-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prepare_live_handoff_trace.json \
  --upstream-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prepare_live_upstream_trace.json \
  --prepare-slim-no-raw-state-vision \
  --prepare-slim-derive-condition-reference \
  --prepare-slim-derive-initial-noise
```

Archived `official_action_only_live_prepare_live_outputs` metrics:

- runtime source: in-memory `_prepare_inference_data` payload, not a prepare or
  pre-layer artifact
- derived debug boundary: 62.65 MiB / 31 tensors; debug save clones shared views
- boundary derive: 0.143s; engine construct: 1.267s
- `OmniInference.generate_batch`: 25.93s; denoise: 25.91s; decode: 0.0088s
- official velocity calls: 0; FlashRT velocity calls: 30
- measured `_prepare_inference_data`: 3.88s; measured VAE encode: 3.66s; text tokenization: 30.8ms; pack: 2.65ms
- native UniPC scheduler: 1 run / 30 steps
- final action cos 0.9999959 / rel_l2 0.291% / max_abs 0.00687

This is the first no-prepare-artifact live denoise handoff. It still uses the
official raw-video prepare path, so strict native E2E still requires native
VAE/SigLIP/VLM prefill to produce the same prepared state.

The same bridge can be combined with upstream Wan2.2 AOT encode using
`--vae-compile-encode`. This is not a FlashRT native VAE implementation, but it
is the current no-prepare-artifact service-warmup target line for native VAE
work:

- `official_action_only_live_prepare_live_vae_compile_encode_outputs`
- AOT setup compile/load: 149.39s; loaded AOT functions: 5
- measured `OmniInference.generate_batch`: 24.33s; denoise: 24.32s; decode:
  0.0085s
- measured `_prepare_inference_data`: 2.65s; measured VAE encode: 2.43s
- boundary derive: 0.146s; engine construct: 1.714s
- official velocity calls: 0; FlashRT velocity calls: 30; native UniPC 30 steps
- final action cos 0.9999948 / rel_l2 0.322% / max_abs 0.00802

For the current no-prepare/pre-layer-artifact service floor, combine the live
prepare bridge with one warmup request plus the VAE and prepare caches, and keep
`--live-boundary-out` unset:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prepare_live_warm_cache_nosave_outputs \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --warmup 1 \
  --cache-warmup-vae \
  --cache-warmup-prepare \
  --live-boundary-prepare-live \
  --live-handoff-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prepare_live_warm_cache_nosave_handoff_trace.json \
  --upstream-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prepare_live_warm_cache_nosave_upstream_trace.json \
  --prepare-slim-no-raw-state-vision \
  --prepare-slim-derive-condition-reference \
  --prepare-slim-derive-initial-noise
```

Archived `official_action_only_live_prepare_live_warm_cache_nosave_outputs`
metrics:

- measured `OmniInference.generate_batch`: 13.33s; denoise: 13.32s; decode:
  0.00020s
- warmup `OmniInference.generate_batch`: 25.77s
- measured `_prepare_inference_data`: 0.075s; measured VAE encode calls: 0
- official velocity calls: 0; FlashRT velocity calls: 60; native UniPC 60 steps
- measured native UniPC run: 13.23s
- final action cos 0.9999959 / rel_l2 0.291% / max_abs 0.00687

A debug-save variant with `--live-boundary-out` measured 22.04s because the
archival path perturbs the service run, so production/service floor measurements
should leave that flag off.

The `--live-warm-request` path moves cold FA4/engine setup and repeat VAE encode
out of the measured batch for the same sample by enabling pre-layer bootstrap,
one official warmup request, and a FlashRT warmup VAE latent cache. FlashRT
patches warmup to clone `data_batch` first, because upstream mutates video
tensors in-place during preprocessing.

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_live_warm_request_vae_cache \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-warm-request \
  --live-handoff-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_warm_request_vae_cache_handoff_trace.json
```

Archived `official_action_only_live_warm_request_vae_cache_outputs` metrics:

- `OmniInference.generate_batch`: 12.12s
- `OmniMoTModel.generate_samples_from_batch`: 12.11s
- `OmniMoTModel.decode`: 0.00012s
- live native UniPC scheduler: 2 runs / 60 steps, native scheduler step total
  0.0128s
- final action cos 0.9999958 / rel_l2 0.291% / max_abs 0.00687

The warmup batch still pays cold cost: `[warmup] OmniInference.generate_batch`
25.95s. The VAE cache key includes shape, dtype, device, element count, and a
small sampled content fingerprint so same-shape different-video tensors do not
reuse the warmup latent silently.

For P4 upstream breakdown, add `--upstream-trace-out`:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_live_warm_request_vae_cache \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-warm-request \
  --live-handoff-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_warm_request_vae_cache_handoff_trace.json \
  --upstream-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_warm_request_vae_cache_upstream_trace.json
```

Archived uncached upstream trace metrics:

- measured `OmniMoTModel.encode`: 3.41s
- measured `OmniMoTModel.get_data_and_condition`: 3.49s
- measured `OmniMoTModel._prepare_inference_data`: 3.52s
- input batch preparation: 0.57s total
- warmup official `Cosmos3VFMNetwork.forward`: 9.79s

With warmup VAE cache enabled by `--live-warm-request`, the measured upstream
trace changes to:

- measured `OmniMoTModel.encode`: 0.00051s
- measured `OmniMoTModel._prepare_inference_data`: 0.1068s
- measured cache hits: 1, warmup cache stores: 1, measured cache misses: 0
- warmup `OmniMoTModel.encode`: 3.65s
- warmup prelayer handoff trace events are marked as expected/ok

For the no-warmup VAE replacement boundary, dump the measured VAE encode latent
and replay it:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_live_prelayer_vae_boundary \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-prelayer-bootstrap \
  --vae-encode-dump-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_vae_encode.safetensors \
  --live-handoff-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_vae_boundary_handoff_trace.json \
  --upstream-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_vae_boundary_upstream_trace.json

python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_live_prelayer_vae_latent_replay \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-prelayer-bootstrap \
  --vae-latent-in dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_vae_encode.safetensors \
  --live-handoff-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_vae_latent_replay_handoff_trace.json \
  --upstream-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_vae_latent_replay_upstream_trace.json
```

Archived VAE boundary/replay metrics:

- boundary dump: `official_action_only_live_prelayer_vae_encode.safetensors`,
  4.57 MiB, tensor `vae_encode/output` shape `[1,48,16,30,52]`, plus input
  signature metadata
- boundary run measured `OmniMoTModel.encode`: 3.72s;
  `_prepare_inference_data`: 3.86s; `generate_batch`: 26.22s
- latent replay measured `OmniMoTModel.encode`: 0.0108s;
  `vae_latent_replay_hit`: 0.0050s; `_prepare_inference_data`: 0.141s;
  `generate_batch`: 22.55s
- latent replay final action cos 0.9999959 / rel_l2 0.291% / max_abs 0.00687

This is not the final no-dump VAE path. It is the correctness and latency
boundary a native Wan2.2 VAE encoder must match to remove the remaining
no-warmup official encode.

For a wider no-warmup prepare/prefill replacement boundary, dump and replay the
full `_prepare_inference_data` return tuple:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_boundary_outputs \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-prelayer-bootstrap \
  --prepare-dump-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare.pt \
  --live-handoff-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_boundary_handoff_trace.json \
  --upstream-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_boundary_upstream_trace.json

python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_replay_v2_outputs \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-prelayer-bootstrap \
  --prepare-replay-in /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare.pt \
  --live-handoff-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_replay_v2_handoff_trace.json \
  --upstream-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_replay_v2_upstream_trace.json

python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_inventory_replay_outputs \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-prelayer-bootstrap \
  --prepare-replay-in /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare.pt \
  --prepare-inventory-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_inventory.json \
  --live-handoff-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_inventory_handoff_trace.json \
  --upstream-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_inventory_upstream_trace.json

python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_dump_outputs \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-prelayer-bootstrap \
  --prepare-dump-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_dump.pt \
  --prepare-inventory-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_dump_inventory.json \
  --prepare-slim-no-raw-state-vision \
  --live-handoff-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_dump_handoff_trace.json \
  --upstream-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_dump_upstream_trace.json

python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_dump_replay_outputs \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-prelayer-bootstrap \
  --prepare-replay-in /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_dump.pt \
  --prepare-inventory-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_dump_replay_inventory.json \
  --live-handoff-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_dump_replay_handoff_trace.json \
  --upstream-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_dump_replay_upstream_trace.json

python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_dump_outputs \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-prelayer-bootstrap \
  --prepare-dump-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_dump.pt \
  --prepare-inventory-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_dump_inventory.json \
  --prepare-slim-no-raw-state-vision \
  --prepare-slim-derive-condition-reference \
  --live-handoff-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_dump_handoff_trace.json \
  --upstream-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_dump_upstream_trace.json

python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_dump_replay_outputs \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-prelayer-bootstrap \
  --prepare-replay-in /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_dump.pt \
  --prepare-inventory-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_dump_replay_inventory.json \
  --live-handoff-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_dump_replay_handoff_trace.json \
  --upstream-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_dump_replay_upstream_trace.json

python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_noise_dump_outputs \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-prelayer-bootstrap \
  --prepare-dump-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_noise_dump.pt \
  --prepare-inventory-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_noise_dump_inventory.json \
  --prepare-slim-no-raw-state-vision \
  --prepare-slim-derive-condition-reference \
  --prepare-slim-derive-initial-noise \
  --live-handoff-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_noise_dump_handoff_trace.json \
  --upstream-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_noise_dump_upstream_trace.json

python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_noise_dump_replay_outputs \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-prelayer-bootstrap \
  --prepare-replay-in /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_noise_dump.pt \
  --prepare-inventory-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_noise_dump_replay_inventory.json \
  --live-handoff-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_noise_dump_replay_handoff_trace.json \
  --upstream-trace-out /work/official/flashrt-public/dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_noise_dump_replay_upstream_trace.json
```

Archived prepare boundary/replay metrics:

- prepare artifact: `official_action_only_live_prelayer_prepare.pt`, 297 MiB,
  with input signature and the 7-tuple consumed by `generate_samples_from_batch`
- dump run measured `_prepare_inference_data`: 4.40s; `prepare_boundary_dump`
  event: 3.87s
- replay measured `_prepare_inference_data`: 0.232s; `prepare_replay_hit`:
  0.182s; `generate_batch`: 22.66s
- replay final action cos 0.9999959 / rel_l2 0.291% / max_abs 0.00687
- inventory replay writes
  `official_action_only_live_prelayer_prepare_inventory.json` using schema
  `flashrt_cosmos3_edge_prepare_inventory_v1`: 11 tensors, 297.13 MiB total.
  The largest tensor is `gen_data_clean.raw_state_vision[0]`
  `[1,3,61,480,832]` at 278.79 MiB; the denoise seed/reference/mask vectors
  are `initial_noise[0]`, `condition_reference[0]`, and `condition_mask[0]`,
  each `[1201920]` float32 at 4.58 MiB. This is the current machine-readable
  native prefill contract.
- inventory replay measured `_prepare_inference_data`: 0.207s;
  `prepare_replay_hit`: 0.159s; `generate_batch`: 22.75s; final action
  cos 0.9999958 / rel_l2 0.291% / max_abs 0.00687.
- `--prepare-slim-no-raw-state-vision` proves the post-prepare denoise path
  does not need the 278.79 MiB RGB `raw_state_vision` tensor once
  `temporal_positions_vision` and `x0_tokens_vision` are present. The archived
  slim dump is 18.35 MiB, 10 tensors / 18.34 MiB total, with `x0_tokens_vision`,
  action state, and the three 4.58 MiB noise/reference/mask vectors intact.
  The slim dump run keeps action cos 0.9999958 / rel_l2 0.291%; replaying the
  slim `.pt` gives `prepare_replay_hit` 0.026s, `_prepare_inference_data`
  0.075s, `generate_batch` 22.62s, and unchanged action metrics.
- `--prepare-slim-derive-condition-reference` proves `condition_reference` is
  exactly derivable from `gen_data_clean.x0_tokens_vision/action` and
  `raw_action_dim` after raw video has been dropped. The derived slim artifact
  stores `condition_reference=None`, is 13.76 MiB, and has 9 tensors /
  13.75 MiB. Replay restores `condition_reference` before entering the official
  sampler, so the runtime payload inventory returns to 10 tensors / 18.34 MiB.
  The archived replay has `prepare_replay_hit` 0.026s,
  `_prepare_inference_data` 0.074s, `generate_batch` 22.56s, final action cos
  0.9999959 / rel_l2 0.291% / max_abs 0.00687, and native UniPC 1 run /
  30 steps.
- `--prepare-slim-derive-initial-noise` further proves `initial_noise` is
  bit-exactly derivable from the request seed, `x0_tokens_vision/action`,
  `condition_mask`, and `raw_action_dim`. The slim-derived-noise artifact stores
  `raw_state_vision=None`, `condition_reference=None`, and `initial_noise=None`;
  it is 9.18 MiB with 8 tensors / 9.17 MiB, consisting of the prepared latent
  state plus `condition_mask`. Replay restores both `initial_noise` and
  `condition_reference`, so runtime inventory returns to 10 tensors /
  18.34 MiB. The archived replay has `prepare_replay_hit` 0.051s,
  `_prepare_inference_data` 0.100s, `generate_batch` 22.59s, final action cos
  0.9999959 / rel_l2 0.291% / max_abs 0.00687, and native UniPC 1 run /
  30 steps.

This is a full prepare/prefill correctness boundary for future native
VAE/SigLIP/VLM-prefill replacement. It is not a production path: the artifact is
input-specific and still loaded via `torch.save`/`torch.load`. The derived slim
result shows the native denoise contract can be reduced to latent/action state,
temporal positions, condition mask, request seed/signature metadata, and enough
metadata to reconstruct condition references and initial noise.

For the first native Wan2.2 VAE encode subgraph, enable native BF16
`RMS_norm -> SiLU` inside encode `ResidualBlock`s:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_live_prelayer_vae_native_rms_silu \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-prelayer-bootstrap \
  --vae-native-rms-silu \
  --vae-encode-profile-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_vae_native_rms_silu_profile.json \
  --live-handoff-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_vae_native_rms_silu_handoff_trace.json \
  --upstream-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_vae_native_rms_silu_upstream_trace.json
```

Archived native RMS+SiLU metrics:

- profile inventory artifact:
  `official_action_only_live_prelayer_vae_encode_profile.json` shows one
  `WanVAE_.encode` call from BF16 `[1,3,61,480,832]` to BF16
  `[1,48,16,30,52]`, 27 `CausalConv3d`, 22 `RMS_norm`, 21 `SiLU`, and
  top time in the early high-resolution 160-channel residual convs.
- baseline profile-only `WanVAE_.encode`: 3.778s; native RMS+SiLU profile
  `WanVAE_.encode`: 3.252s. The refreshed profile artifacts now include
  `shape_summary` and `native_candidate_summary`; the top native candidate is
  `encoder.downsamples.0.downsamples.0.residual.2` as
  `steady_cached_causal_conv3d` on `[1,160,24,240,416]`.
- upstream measured `OmniMoTModel.encode`: 4.197s -> 3.295s; measured
  `_prepare_inference_data`: 4.348s -> 3.436s.
- native RMS+SiLU run `generate_batch`: 25.63s; decode 0.0088s; live native
  UniPC scheduler 1 run / 30 steps; final action cos 0.9999955 /
  rel_l2 0.299% / max_abs 0.00929.

The remaining VAE bottleneck is still the early CausalConv3d chain:
`encoder.downsamples.0` and `encoder.downsamples.1` dominate the profile. The
next native VAE step should fuse/quantize the `RMS_norm -> SiLU -> CausalConv3d`
triplet for those fixed shapes, reusing the Motus VAE FP8/FP4 conv swap
structure but recalibrated against the Cosmos3-Edge action gate.

An additional no-go probe rewrites no-cache single-frame Wan2.2
`CausalConv3d` calls to mathematically equivalent `Conv2d` via
`--vae-t1-conv2d`. This is correct but slower on Thor when combined with native
RMS+SiLU:

- `WanVAE_.encode`: 3.291s -> 3.507s
- measured `OmniMoTModel.encode`: 3.295s -> 3.511s
- `OmniInference.generate_batch`: 25.63s -> 25.92s
- final action cos 0.9999960 / rel_l2 0.282% / max_abs 0.00799

The T=1 Conv2d rewrite is therefore kept default-off as an opt-in probe. The
steady `T=24` cached chunks are still the important target.

Another opt-in native VAE probe, `--vae-native-avgdown3d`, replaces Wan2.2
`AvgDown3D` shortcut pooling with `cosmos3_edge_avgdown3d_bf16`. The kernel
canary is bit-exact against the upstream reshape/permute/group-mean semantics,
including the temporal front-pad case. In the real no-warmup pre-layer run,
local `AvgDown3D` profile time moves from about 77ms to 67ms, but the complete
path is not better when combined with native RMS+SiLU:

- `WanVAE_.encode`: 3.252s -> 3.504s
- measured `OmniMoTModel.encode`: 3.295s -> 3.508s
- `OmniInference.generate_batch`: 25.63s -> 26.03s
- final action cos 0.9999955 / rel_l2 0.299% / max_abs 0.00929

This is kept as an opt-in correctness scaffold and default-off/no-go. The
worthwhile native VAE target remains the high-resolution CausalConv3d chain.

The steady cached `T=24` Conv3d-to-Conv2d-slices decomposition has also been
probed via `dev_log_cosmos3_thor/probe_vae_conv2d_decompose.py`. It is local
math-equivalent enough for the action gate, but not a useful default path:

- stage0 160-channel `[1,160,24,240,416]`: cos 0.999995 /
  rel_l2 0.303%, but 0.185s vs BF16 cuDNN Conv3d 0.112s.
- stage1 320-channel `[1,320,24,120,208]`: cos 0.999995 /
  rel_l2 0.311%, but 0.099s vs BF16 cuDNN Conv3d 0.086s.
- stage2 640-channel `[1,640,12,60,104]`: cos 0.999996 /
  rel_l2 0.293%, 0.0294s vs 0.0296s, only ~1.01x and not the dominant site.

This keeps the decomposition default-off too. The remaining viable path is a
single Thor-specific CausalConv3d kernel for the early steady chunks, not a
multi-call Torch/cuDNN decomposition.

A first Thor-specific BF16 MMA CausalConv3d bring-up kernel is also wired as
`bf16_conv3d_v0_ndhwc_bf16out` for `sm_110a`. It mirrors the Motus v17 virtual
cache concat/direct causal-output structure, but uses BF16 tensor cores and
stays probe-only. The small-shape canary passes, and the real VAE shape probe
is accurate enough for the local gate, but still slower than cuDNN:

- stage0 160-channel `[1,160,24,240,416]`: cos 0.999996 /
  rel_l2 0.282%, kernel-only 0.275s vs BF16 cuDNN 0.112s.
- stage1 320-channel `[1,320,24,120,208]`: cos 0.999996 /
  rel_l2 0.284%, kernel-only 0.214s vs BF16 cuDNN 0.087s.
- stage2 640-channel `[1,640,12,60,104]`: cos 0.999996 /
  rel_l2 0.301%, kernel-only 0.089s vs BF16 cuDNN 0.029s.

The v0 kernel is therefore default-off/no-go. It is useful as a native
correctness scaffold, but the next attempt needs better tiling/persistence or
triplet fusion, not this one-kernel v0 swap.

The first steady cached chunk also has a cache-shape corner case: upstream
passes a one-frame cache into `CausalConv3d.forward`, which then prepends one
temporal zero via `F.pad`. A cache2 normalization probe confirms that replacing
that with an explicit `[zero, cache]` two-frame cache is bit-exact, but not a
useful cuDNN optimization by itself:

- stage0 160-channel: prebuilt cache2 Conv3d 0.11231s vs 0.11247s reference;
  building cache2 in Python plus Conv3d is 0.11308s.
- stage1 320-channel: prebuilt cache2 Conv3d 0.08665s vs 0.08669s reference;
  building cache2 in Python plus Conv3d is 0.08697s.
- stage2 640-channel: prebuilt cache2 Conv3d 0.02971s vs 0.02960s reference;
  building cache2 in Python plus Conv3d is 0.02976s.
- native `update_cache2_ncdhw_bf16` matches Python cache2 exactly, but still
  adds 0.30-1.19ms per call at these shapes.

This remains a native-kernel interface fact, not a standalone default path.
Cache2 is only worth standardizing if a future fused CausalConv3d/triplet
kernel consumes it directly and wins the full-site A/B.

The `vae_channels_last3d_probe.json` artifact tests cuDNN BF16 Conv3d with
`channels_last_3d` tensors at the same three steady sites. It is bit-exact and
shows a useful preconverted-layout signal, but not a default-ready per-conv
rewrite:

- stage0 160-channel: total conversion path 0.119s vs 0.112s reference
  (0.95x), preconverted-layout conv 0.073s (1.55x).
- stage1 320-channel: total conversion path 0.076s vs 0.086s reference
  (1.13x), preconverted-layout conv 0.053s (1.62x).
- stage2 640-channel: total conversion path 0.031s vs 0.030s reference
  (0.95x), preconverted-layout conv 0.026s (1.16x).

Conclusion: `channels_last_3d` is worth pursuing only if the VAE encode segment
can keep activations and weights in that layout across adjacent residual blocks
or if a fused native triplet consumes the layout directly. Per-conv conversion
stays default-off.

An opt-in full-run patch, `--vae-channels-last3d-conv320`, applies this only to
steady cached 320-channel CausalConv3d sites and returns contiguous tensors to
preserve the upstream cache contract. Combined with `--vae-native-rms-silu`, it
is correct but slower in the real no-warmup pre-layer run: measured
`OmniMoTModel.encode` 4.04s vs 3.29s for native RMS+SiLU alone,
`_prepare_inference_data` 4.18s, `generate_batch` 26.41s, final action cos
0.9999955 / rel_l2 0.299% / max_abs 0.00929. The profile shows the 320-channel
cached sites become the top regression at 0.13-0.24s each, so the patch remains
default-off/no-go.

The follow-up `vae_resblock_channels_last3d_probe.json` artifact tests a
block-local layout propagation variant across three real steady
`ResidualBlock`s. It keeps the second half of each block in `channels_last_3d`
instead of returning to contiguous after every conv. The probe remains correct,
but all three blocks are slower than the official block:

- stage0 160-channel block: 0.441s vs 0.323s reference (0.73x).
- stage1 320-channel block: 0.259s vs 0.222s reference (0.85x).
- stage2 640-channel block: 0.077s vs 0.072s reference (0.93x).

This closes the small-step `channels_last_3d` path: block-local propagation is
not enough. Future VAE work should use a fused RMS/SiLU/CausalConv3d triplet or
a stronger Thor-specific steady CausalConv3d kernel that consumes a stable
layout directly.

For a non-native service-warmup comparison, `--vae-compile-encode` enables the
upstream Wan2.2 chunk-level AOT `compile_encode` path for 480p 16:9. The
archived run compiled 5 loadable AOT functions in 133.7s during setup, then
reduced measured no-warmup pre-layer VAE encode:

- measured `OmniMoTModel.encode`: 4.197s eager baseline -> 2.428s AOT
- measured `_prepare_inference_data`: 2.554s
- `OmniInference.generate_batch`: 24.43s, decode 0.0085s
- final action cos 0.9999958 / rel_l2 0.293% / max_abs 0.00553

This is useful as an opt-in long-lived-service baseline and a latency target for
native VAE work. It is not the final FlashRT native implementation because it
pays large startup compile cost and still relies on Torch Inductor/AOT.

The direct Motus FP8 conv3d reuse path has been probed on Thor:

```bash
python dev_log_cosmos3_thor/probe_vae_triplet_fp8.py \
  --json-out dev_scratch_cosmos3_thor/edge_av_inverse_0/vae_triplet_fp8_probe_site0.json
```

The SM120-origin v17/v18 FP8 conv3d kernels compile for `sm_110a`; CMake now
exposes `fp8_conv3d_v17_ndhwc_bf16out`,
`fp8_conv3d_v17_anyco_ndhwc_bf16out`, and
`fp8_conv3d_v18_ncdhw_res_bf16out` on Thor. The small-shape kernel canary
matches torch exactly, but the real Cosmos VAE triplet A/B is negative:

- stage0 160-channel `[1,160,24,240,416]`: FP8 v18 cos 0.999868 /
  rel_l2 1.69%, but 0.245s vs BF16 cuDNN 0.113s.
- stage1 320-channel `[1,320,24,120,208]`: FP8 v18 cos 0.999816 /
  rel_l2 1.92%, but 0.182s vs BF16 cuDNN 0.086s.
- stage2 640-channel `[1,640,12,60,104]`: FP8 v18 cos 0.999931 /
  rel_l2 1.38%, but 0.082s vs BF16 cuDNN 0.030s.

Therefore the Motus v17/v18 FP8 conv3d path is kept default-off for
Cosmos3-Edge. The first Thor-specific BF16 v0 kernel above is also slower, so
the next VAE conv step needs a better Thor-specific BF16/FP8 CausalConv3d
tiling/fusion strategy, not a direct Motus conv swap or the v0 BF16 probe.

To also reuse the whole prepared inference state from the warmup request:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint /work/models/Cosmos3-Edge \
  --input-json /work/.tmp_cosmos_edge_inputs/av_inverse_0.json \
  --output-dir /work/.tmp_cosmos_edge_outputs_live_warm_request_prepare_cache \
  --vae-path /work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --cosmos-root /work/external/cosmos-framework \
  --backend official_action_only \
  --live-warm-request \
  --cache-warmup-prepare \
  --live-handoff-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_warm_request_prepare_cache_handoff_trace.json \
  --upstream-trace-out dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_warm_request_prepare_cache_upstream_trace.json
```

Archived `official_action_only_live_warm_request_prepare_cache_outputs` metrics:

- `OmniInference.generate_batch`: 11.98s
- `OmniMoTModel.generate_samples_from_batch`: 11.97s
- `OmniMoTModel.decode`: 0.00015s
- measured `OmniMoTModel._prepare_inference_data`: 0.0185s
- measured prepare cache hits: 1, stores: 1, misses: 0
- measured VAE encode calls: 0
- live native UniPC scheduler: 2 runs / 60 steps, native scheduler step total
  0.0133s
- final action cos 0.9999959 / rel_l2 0.291% / max_abs 0.00687
- Adding `--prepare-slim-no-raw-state-vision` slims the cached prepare payload
  by dropping raw RGB state after warmup prepare. Archived slim cache metrics:
  `generate_batch` 12.08s, denoise 12.07s, measured prepare 0.0146s, prepare
  hit/store 1/1 with `slim_no_raw_state_vision=true`, measured VAE encode calls
  0, native UniPC 2 runs / 60 steps, final action cos 0.9999958 / rel_l2 0.291%.
- Adding `--prepare-slim-derive-condition-reference` on top stores the warmup
  prepare payload without raw RGB and without `condition_reference`; the
  measured cache hit derives `condition_reference` before the official sampler.
  Archived derived slim cache metrics: `generate_batch` 12.00s, denoise 11.99s,
  measured prepare 0.0149s, prepare hit/store 1/1 with both slim flags recorded,
  measured VAE encode calls 0, native UniPC 2 runs / 60 steps, final action cos
  0.9999959 / rel_l2 0.291%.
- Adding `--prepare-slim-derive-initial-noise` on top stores the repeated-request
  prepare payload without raw RGB, stored `condition_reference`, or stored
  `initial_noise`; the measured cache hit derives noise from seed/x0/mask and
  then derives the reference. Archived slim-derived-noise cache metrics:
  `generate_batch` 12.05s, denoise 12.04s, measured prepare 0.0382s, prepare
  hit/store 1/1 with all three slim flags recorded, measured VAE encode calls 0,
  native UniPC 2 runs / 60 steps, final action cos 0.9999959 / rel_l2 0.291%.

## P1 Replay Scaffold

The replay scaffold validates the denoise boundary without rerunning the
transformer. It reads the P0 dump, replays the recorded velocity tensors through
FlashRT's local UniPC scheduler, and splits the final flat latent into:

- vision latent: `[1,48,16,30,52]`
- action model latent: `[60,64]`

Run it inside the container/venv:

```bash
python - <<'PY'
from flash_rt.frontends.torch.cosmos3_edge_thor import Cosmos3EdgeTorchFrontendThor

pipe = Cosmos3EdgeTorchFrontendThor("/work/models/Cosmos3-Edge", hardware="thor")
result = pipe.infer(
    backend="replay",
    output_dir="/work/.tmp_cosmos_edge_replay",
    reference_dump=(
        "/work/official/flashrt-public/dev_scratch_cosmos3_thor/"
        "edge_av_inverse_0/tensors.safetensors"
    ),
    replay_device="cuda",
)
print(result)
PY
```

Expected validation facts for the current dump:

- `num_steps`: 30
- `flat_dim`: 1201920
- `timesteps`: `999 ... 256`
- CUDA replay `max_input_abs_diff`: 0.0

## P1 Torch Reference

The native bring-up path now has two local correctness gates:

- `EdgeLayer0TorchReference`: layer 0 causal/full outputs match the official
  boundary dump exactly after correcting the public checkpoint tower mapping.
- `EdgeTransformerTorchReference`: all 28 layers plus the action head match the
  step-0 official action velocity.
- `EdgeDenoiseTorchReference`: the full 30-step UniPC loop computes action
  velocity live from the Torch reference instead of replaying recorded velocity.
- `EdgeDenoiseFlashRT`: the public optimized eager engine used by
  `backend="flashrt"`. The velocity path is the static FlashRT engine rather
  than the two-tower reference replay path, and the fixed UniPC latent update
  uses the native scheduler step when available. The frontend writes an
  action-only `sample_outputs.json` for this backend and intentionally does not
  emit decoded vision files.
- `EdgeStaticBufferEngine`: owns the static AV geometry, static vision tokens,
  static und K/V cache, action slots, and the flat velocity contract. Its action
  in/out projections use `GemmRunner.bf16_nn` when `flash_rt_kernels` is
  available. The action input bias and modality embedding are precombined once
  during static engine initialization.
- `EdgeTransformerFvkLinearReference`: routes cached-path BF16 linear projections
  through `GemmRunner.bf16_nn` and cached-path RMSNorm through `fvk.rms_norm`;
  cached-path main q/k RoPE through `qwen36_partial_rope_qk_bf16`, gen q/k
  norm+RoPE through `cosmos3_edge_qk_norm_rope_bf16`, and FFN activation through
  `relu2_inplace_bf16`. On Thor, non-causal gen attention uses the vendored FA4
  CuTe path when the `thor-fa4` extra is installed. `flash_rt_fa2` is not
  expected on Thor.
- Hot-path weights/transposes are resident in the static engine cache. The
  current checkpoint keeps 532 bf16 tensors, 4 fp32 timestep tensors, 4 boundary
  tensors, and 336 transposed GEMM weights resident.
- `EdgeStaticBufferEngine.flat_velocity_for_step` uses
  `cosmos3_edge_fill_flat_velocity_bf16` when the binding is available. This
  replaces the Torch `zero_()` plus action-tail assignment for the flat velocity
  output buffer.
- `EdgeStaticBufferEngine._decode_action_velocity` uses
  `cosmos3_edge_add_bias_zero_action_tail_bf16` when the binding is available.
  This fuses action output bias add and invalid action-dimension clearing.
- `EdgeStaticBufferEngine` uses `cosmos3_edge_scatter_rows_bf16` and
  `cosmos3_edge_gather_rows_bf16` when available for action token row movement
  between the compact action buffer and full sequence buffer.
- `EdgeStaticBufferEngine.full_sequence_for_step` uses
  `cosmos3_edge_copy_action_tail_f32_to_bf16` when available for CUDA float32
  latents, avoiding the Torch action-tail slice and BF16 cast.
- `EdgeStaticBufferEngine._encode_action_tokens` uses
  `cosmos3_edge_add_action_bias_timestep_bf16` when available. The timestep
  embedding is computed once for the scalar timestep and broadcast natively
  across the 60 action rows.
- `EdgeDenoiseFlashRT` precomputes all dump timesteps into
  `EdgeStaticBufferEngine.timestep_embed_cache`, so measured denoise steps reuse
  resident BF16 timestep embeddings instead of recomputing the Torch
  cos/sin/MLP path.

Install the FA4 dependency set in the isolated container venv before measuring
the optimized eager path:

```bash
cd /work/official/flashrt-public
source /work/.venv_cosmos_thor/bin/activate
python -m pip install -e ".[thor-fa4]"
```

Run the full reference denoise check inside the container/venv:

```bash
python - <<'PY'
from flash_rt.frontends.torch.cosmos3_edge_thor import Cosmos3EdgeTorchFrontendThor

pipe = Cosmos3EdgeTorchFrontendThor("/work/models/Cosmos3-Edge", hardware="thor")
result = pipe.infer(
    backend="flashrt",
    output_dir="/work/.tmp_cosmos_edge_flashrt",
    reference_dump=(
        "/work/official/flashrt-public/dev_scratch_cosmos3_thor/"
        "edge_av_inverse_0/tensors.safetensors"
    ),
    boundary_dump=(
        "/work/official/flashrt-public/dev_scratch_cosmos3_thor/"
        "edge_av_inverse_0_boundary_step0/tensors.safetensors"
    ),
    replay_device="cuda",
)
print(result)
PY
```

Current 30-step optimized eager result with FA4 attention:

- und-cache precompute/init: 1.44s
- 30-step run wall time after one warm step: P50 11.998s over 10 iters
- speedup vs official `OmniMoTModel.generate_samples_from_batch`: 2.762x
- `max_input_abs_diff`: 0.00711
- `max_velocity_abs_diff`: 0.046875
- final action vs official: cos 0.9999959, rel_l2 0.286%, max_abs 0.00734
- peak allocated VRAM during warm run: 11.02 GiB
- estimated action-only E2E with official upstream, skipped vision decode, and
  FlashRT denoise: 14.52s, or 3.26x vs official 47.32s E2E
- hard gate run: `flashrt_benchmark_gate_10iter.json` with `gates_passed=true`
- native flat velocity fill canary:
  `flashrt_benchmark_fill_velocity_native_3iter.json`, P50 11.97s, speedup
  2.77x, with the same action accuracy gate.
- native action bias+tail-clear canary:
  `flashrt_benchmark_action_tail_native_3iter.json`, P50 12.02s, speedup
  2.76x, with the same action accuracy gate.
- native action row scatter/gather canary:
  `flashrt_benchmark_action_rows_native_3iter.json`, P50 12.03s, speedup
  2.75x, with the same action accuracy gate.
- native action input tail-copy canary:
  `flashrt_benchmark_action_input_native_3iter.json`, P50 12.02s, speedup
  2.76x, with the same action accuracy gate.
- native action bias+timestep canary:
  `flashrt_benchmark_action_bias_timestep_native_3iter.json`, P50 12.03s,
  speedup 2.75x, with the same action accuracy gate.
- timestep cache canary:
  `flashrt_benchmark_timestep_cache_3iter.json`, P50 12.04s, speedup 2.75x,
  with the same action accuracy gate.
- native UniPC scheduler canary:
  `flashrt_benchmark_native_unipc_3iter.json`, P50 12.00s, speedup 2.76x,
  final action cos 0.9999959, rel_l2 0.286%. This is enabled by default when
  the binding is available.
- opt-in native residual add canary:
  `flashrt_benchmark_residual_add_native_3iter.json`, P50 12.24s, speedup
  2.71x, with the same action accuracy gate. This kernel is correct but is not
  default because the extra launches are slower than Torch add in the current
  eager stack.
- opt-in CUDA graph run:
  `flashrt_benchmark_graph_10iter.json` with `use_cuda_graphs=true`,
  `graph_attention=true`, `native_scheduler=true`, P50 12.01s, speedup 2.76x,
  final action cos 0.9999962, rel_l2 0.279%. Graph replay is verified, but
  eager remains the default because it is marginally faster on this Thor run.
- graph loop breakdown:
  `flashrt_benchmark_graph_loop_profile_3iter.json` adds CUDA event timing for
  one profiled 30-step graph run. Velocity graph replay accounts for 11.966s of
  the 11.970s step total; native UniPC scheduler accounts for only 0.0041s
  total, with p50 0.141ms/step. Capturing the Python 30-step loop as one larger
  graph is therefore not the next useful latency target; the remaining time is
  inside per-step velocity compute.
- additional seed check: seed=1 dump
  `dev_scratch_cosmos3_thor/edge_av_inverse_1/tensors.safetensors`, final
  action cos 0.9999956 / rel_l2 0.300% / max_abs 0.00923. This run skips the
  speedup gate because the official seed=1 denoise baseline was a single faster
  run at 29.16s; it is used as the anti-overfit accuracy check.

Reproduce the number:

```bash
python benchmarks/cosmos3_edge_thor_denoise.py \
  --iters 10 \
  --warmup-steps 1 \
  --enforce-gates \
  --json-out dev_scratch_cosmos3_thor/edge_av_inverse_0/flashrt_benchmark_gate_10iter.json
```

Reproduce the opt-in CUDA graph number:

```bash
FLASHRT_COSMOS3_EDGE_FA4_FWD=1 \
python benchmarks/cosmos3_edge_thor_denoise.py \
  --iters 10 \
  --warmup-steps 1 \
  --use-cuda-graphs \
  --enforce-gates \
  --json-out dev_scratch_cosmos3_thor/edge_av_inverse_0/flashrt_benchmark_graph_10iter.json
```

Verify the existing delivery artifacts without rerunning GPU benchmarks:

```bash
python dev_log_cosmos3_thor/verify_delivery.py
```

Attention notes:

- FA4 (`flash_rt.hardware.thor.fa4_backend`) is the default native eager
  attention path for gen/full GQA. It is accurate at the Cosmos3 shape and avoids
  the long-context failure seen in the cuBLAS MHA fallback. FA4 accepts BF16
  Q/K/V directly, so the hot path avoids per-layer BF16→FP16 casts.
- `fmha_strided_full` remains behind `FLASHRT_COSMOS3_EDGE_FMHA=1`; it is
  accurate on some large random shapes but increased full step-0 velocity error
  in the Cosmos3 stack, so it is not the default.
- `attention_mha_bf16` remains behind `FLASHRT_COSMOS3_EDGE_MHA=1`; its
  `softmax_bf16` path is not reliable for Cosmos3's long `S_kv=6425`.
- Velocity-only CUDA graph capture is available behind
  `FLASHRT_COSMOS3_EDGE_FA4_FWD=1` plus `use_cuda_graphs=True`. The failure was
  narrowed to the final gen norm bypassing the weight cache; after routing
  `norm_moe_gen.weight` through the resident cache, the 28-layer
  `flat_velocity` probe and the 10-iter denoise graph benchmark both pass. Graph
  replay is opt-in rather than default because the measured P50 is 12.01s vs
  12.00s for eager in the current artifact set.
- The graph loop profile shows native scheduler and Python-loop glue are no
  longer material at the 12s denoise scale: scheduler event time is 4.1ms across
  all 30 steps. This closes the immediate "capture the outer loop" question as
  a no-go for latency; further denoise work must reduce the per-layer velocity
  kernels themselves.
- A forced graph experiment with SDPA attention also exited with code 139, so
  there is no SDPA graph fallback enabled in this Python eager stack.
- `FLASHRT_COSMOS3_EDGE_FA4_FWD=1` switches eager attention to the lower-level
  FA4 `_flash_attn_fwd` entry with preallocated output/LSE buffers. It is also
  required for the opt-in graph path. As an eager-only experiment, seed0 3-iter
  P50 was 12.008s with the same action accuracy, which does not beat the
  default gate result.
- A/B attempts to replace per-layer `torch.cat` with manual static K/V copies
  and precompute action timestep embeddings did not improve P50 latency on Thor,
  so those overrides are not in the default path. The earlier two-op
  `relu_inplace_bf16 + square_` experiment was also not useful, but the fused
  `relu2_inplace_bf16` kernel is now enabled by default. Seed0 3-iter A/B:
  native relu2 P50 12.274s vs Torch relu2 P50 12.926s, with identical final
  action metrics.
- `cosmos3_edge_qk_norm_rope_bf16` fuses gen q/k RMSNorm and RoPE and is enabled
  by default. Seed0 3-iter A/B: fused P50 11.938s vs two-step native path P50
  12.255s, with identical final action metrics.
- `cosmos3_edge_fill_flat_velocity_bf16` is enabled by default when
  `flash_rt_kernels` exposes the binding. It fills the `[1201920]` flat velocity
  buffer in one native launch, zeroing the vision segment and copying the
  `[60,64]` action tail. The CUDA canary in
  `tests/test_cosmos3_edge_kernel_bindings.py` checks exact equality with the
  Torch construction.
- `cosmos3_edge_add_bias_zero_action_tail_bf16` is enabled by default when
  available. It adds the action output bias for the valid action dimensions and
  clears columns `raw_action_dim:64` in one native launch. The CUDA canary checks
  exact equality with the prior Torch `add_` plus slice-clear construction.
- `cosmos3_edge_scatter_rows_bf16` and `cosmos3_edge_gather_rows_bf16` are
  enabled by default when available. They replace the action-row advanced
  indexing operations used to install encoded action tokens into the full
  sequence and gather decoded action hidden states back out. The CUDA canary
  checks exact equality with PyTorch indexed row copy.
- `cosmos3_edge_copy_action_tail_f32_to_bf16` is enabled by default when
  available and the latent is CUDA float32. It copies the action tail from the
  flat denoise latent into the BF16 action input buffer in one native launch.
  The CUDA canary checks exact equality with PyTorch tail slicing plus BF16
  conversion.
- `cosmos3_edge_add_action_bias_timestep_bf16` is enabled by default when
  available. It fuses the action static bias add and timestep embedding add,
  preserving the prior two-step BF16 rounding order. The timestep embedding is
  computed as one row instead of 60 duplicate rows; this changes final action
  metrics only in the last few decimals and remains well inside the reference
  gate.
- `EdgeDenoiseFlashRT` now precomputes the 30 fixed timestep embeddings from
  the dump schedule. The measured denoise loop indexes this cache by step, so
  the per-step action encode path no longer calls the Torch timestep
  `cos/sin/linear/silu/linear` sequence.
- `cosmos3_edge_unipc_step_f32_bf16` is enabled by default when available. It
  folds the fixed 30-step UniPC x0 conversion, corrector, and predictor update
  into one native launch per denoise step. The kernel preserves the prior
  BF16 `sigma * velocity` rounding order before updating the float32 latent.
- `cosmos3_edge_add_bf16` is available as an opt-in native residual add
  experiment with `FLASHRT_COSMOS3_EDGE_NATIVE_ADD=1`. The CUDA canary checks
  exact equality with Torch BF16 add. It remains disabled by default because the
  measured 3-iter canary regressed P50 to 12.24s versus the 12.00s default.

P2 NVFP4 canary:

- SM110 exposes the NVFP4-named bindings in `flash_rt_kernels`, and minimal
  `quantize_bf16_to_nvfp4_swizzled` plus `fp4_w4a16_gemm_sm120_bf16out`
  canaries execute.
- A real Cosmos3 layer-0 gen `add_q_proj` canary with correct `[N,K]` weight
  layout and `bf16_weight_to_nvfp4_swizzled` alpha produced cos 0.9974,
  rel_l2 10.3%, and was slower than BF16 GEMM when activation quantization was
  included. This does not pass the P2 accuracy/perf gate, so NVFP4 is not wired
  into the default backend.

P4 E2E note:

- Official benchmark for `av_inverse_0` reports `OmniMoTModel.decode` at
  11.65s. Since inverse dynamics only emits action in the final JSON, this is a
  strong candidate for an E2E skip in a future no-dump path.
- The external Cosmos Framework path confirms the skip opportunity:
  `inference.py` unconditionally does `outputs.pop("vision") -> decode_vision`
  and saves `vision.mp4` before writing `content["action"]`; the action payload
  itself is taken from `outputs["action"]` and does not depend on decoded vision.
  The external checkout remains unmodified.
- FlashRT-owned `backend="official_action_only"` validates that skip on the
  real no-dump official path: `OmniMoTModel.decode` is 0.00025s, generated
  action is bit-exact to the official baseline JSON, `files=[]`, and no
  `vision*` artifact remains. This still uses the official eager denoise path;
  the remaining P4 gap is wiring the FlashRT denoise engine to live upstream
  conditioning instead of captured dumps.
- The opt-in live capture path produced
  `official_action_only_live_denoise.safetensors` from the real official run:
  30 steps, valid `EdgeDenoiseDump` geometry, action-only JSON with `files=[]`,
  decode 0.00048s, and selected tensors match the original reference dump with
  max_abs 0.0. This proves a FlashRT-owned live capture boundary, but not yet
  live FlashRT denoise handoff.
- The live FlashRT handoff prototype produced
  `official_action_only_live_handoff_boundary.safetensors` and an action-only
  run with `OmniInference.generate_batch` 26.10s, decode 0.0083s, files `[]`,
  final action cos 0.9999959 / rel_l2 0.292% / max_abs 0.00729 vs official
  baseline. This removes captured denoise dumps from the online path after
  step 0; remaining work is removing the official step-0 bootstrap and reducing
  cold handoff overhead.
- The pre-layer bootstrap variant removes the official step-0 decoder forward:
  `prelayer_aborted=true`, 30 FlashRT velocity calls, `generate_batch` 26.24s,
  decode 0.0090s, final action cos 0.9999955 / rel_l2 0.308% / max_abs 0.00780.
  It confirms the remaining one-shot latency is cold FA4/engine setup rather
  than official decoder compute.
- The archived pre-layer boundary contract
  `official_action_only_live_prelayer_boundary.safetensors` is the smaller
  native-denoise/VLM input boundary: 60.36 MiB, 31 tensors, no layer-0 output,
  and no step-0 velocity. The older step-0 handoff boundary was 87.77 MiB /
  36 tensors because it also stored layer-0 output and velocity. The run keeps
  the same action gate (cos 0.9999959 / rel_l2 0.291% / max_abs 0.00687) and
  native UniPC covers all 30 steps. This boundary defines what native prefill
  must synthesize live from raw input before the denoise engine can be fully
  detached from official upstream code.
- `--live-boundary-in` can now use that 60.36 MiB boundary artifact to
  initialize the live FlashRT denoise engine directly. The archived replay run
  has `official_velocity_call_count=0`, 30 FlashRT velocities, native UniPC 30
  steps, `generate_batch` 26.03s, and the same action gate. It is a runnable
  native prefill target contract, not the final no-artifact live path.
- Combined with `--live-warm-request --cache-warmup-prepare` and all slim derive
  flags, boundary-in gives the current artifact/cache service path:
  measured `generate_batch` 12.08s, official velocity calls 0, FlashRT velocity
  calls 60, measured VAE encode calls 0, measured prepare 0.042s, and the same
  action gate.
- The prepare-to-boundary derive helper proves 28 VFM/noise/pack/position/RoPE/text
  tensors / 10.65 MiB are already implied by the 9.18 MiB slim prepare contract
  plus `embed_tokens.weight`,
  and verifies 2 layer-0 input tensors / 25.10 MiB as `lm_in` aliases. The
  remaining pre-layer boundary gap is 1 tensor / 24.61 MiB:
  `lm_in/full_only_seq`; the verifier now requires the derived full 6300-row
  tensor to be bit-exact, including the 60 action rows.
- `--live-boundary-prepare-in` now derives the executable 31-tensor boundary
  in-process from the same 9.18 MiB slim prepare artifact plus checkpoint
  weights, then drives the live FlashRT denoise engine with 0 official velocity
  calls. The archived run measured `generate_batch` 22.24s, denoise 22.22s,
  boundary derive 0.258s, engine construct 1.074s, native UniPC 30 steps, and
  the same action gate. The optional `--live-boundary-out` debug artifact is
  62.65 MiB because shared views are cloned for safetensors archival; it is not
  the runtime input.
- With `--live-warm-request`, the real measured pre-layer batch drops to
  `generate_batch` 12.12s / denoise 12.11s while preserving the action gate;
  the trace shows 60 FlashRT velocity calls and native UniPC scheduler coverage
  for both warmup and measured sampler runs. Only the first warmup velocity call
  pays cold compile, and a measured VAE cache hit removes the previous 3.41s
  encode from the real request. The warmup VAE cache key includes a sampled
  content fingerprint in addition to tensor metadata.
- The opt-in `--cache-warmup-prepare` path reuses the full prepared inference
  state from warmup for a repeated request: `generate_batch` 11.98s / denoise
  11.97s, measured `_prepare_inference_data` 0.0185s, prepare hit/store 1/1,
  measured misses 0, and measured VAE encode calls 0. The live sampler now uses
  the native FlashRT UniPC step for this fixed Edge schedule: 2 native scheduler
  runs / 60 steps, 0.0133s total scheduler step time. With
  `--prepare-slim-no-raw-state-vision`, the same cache stores a slim post-prepare
  payload without raw RGB frames: measured `generate_batch` 12.08s, prepare
  0.0146s, and trace hit/store events both mark
  `slim_no_raw_state_vision=true`. With
  `--prepare-slim-derive-condition-reference`, the cached payload also omits
  `condition_reference` and restores it from `x0_tokens_vision/action` on hit:
  measured `generate_batch` 12.00s, prepare 0.0149s, and both cache events mark
  `slim_derive_condition_reference=true`. With
  `--prepare-slim-derive-initial-noise`, the cache also omits `initial_noise`
  and restores it from seed/x0/mask metadata on hit: measured `generate_batch`
  12.05s, prepare 0.0382s, and cache events mark all three slim flags. This is
  a service-style repeated-request prefill cache, not the final no-warmup live
  VAE/SigLIP/VLM replacement.
- The opt-in upstream trace without VAE cache identified measured VAE
  encode/get-data as the next P4 target; the current warmup cache moves that
  repeated work out of the measured request for repeated-sample/service-style
  warm paths. Native live VAE/SigLIP/VLM prefill remains the broader no-warmup
  P4 target.

Important mapping correction from the layer-0 gate:

- `to_q/to_k/to_v/to_out` = und/causal tower
- `add_q_proj/add_k_proj/add_v_proj/to_add_out` = gen/full tower
- `norm_added_q/norm_added_k` belong to gen/full
- `k_norm_und_for_gen` is required for gen attending to und keys

## Optimization Port

The Cosmos3-Nano FlashRT structure is reusable, but not drop-in compatible:

- Reuse: hardware dispatch, frontend/pipeline split, UniPC loop shape, static
  conditioning cache, CUDA graph capture, TeaCache step reuse, and model-local
  kernels.
- Change: Edge uses hidden size 2048, 28 layers, 16/8 GQA heads, `relu2`, and
  Nemotron Dense VL reasoner/processor instead of Nano's 4096/36-layer Qwen path.
- Thor work: add a model-local `pipeline_thor.py`, SM110 joint-attention/mrope
  kernels, per-layer BF16 reference taps, then FP8/NVFP4 calibration.

Keep this official runner as the accuracy and benchmark gate while the optimized
Thor pipeline is filled in.

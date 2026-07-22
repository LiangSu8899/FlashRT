# Cosmos3-Edge AV on Jetson AGX Thor

FlashRT provides an optimized 30-step denoise engine for the Cosmos3-Edge AV
inverse-dynamics policy on Jetson AGX Thor (`sm_110`). The integration keeps
the official Cosmos preprocessing and action contract while replacing the
denoise boundary with a static, quantized FlashRT graph.

## Reasoner (VLM chat) engine — `flash_rt/models/cosmos3_reasoner/`

`CosmosReasonerThor` runs the checkpoint's understanding view (und text tower
28L/2048/GQA 16:8/relu^2 MLP + SigLIP2 vision + PatchMerger + lm_head) for
batch-1 greedy chat over text, image, and video inputs. Two paths share the
weights:

- vision + prefill: exact torch replica of the official
  `nemotron_3_dense_vl` modules (parity-first; one pass per prompt).
- decode (`quant="bf16"`): SM110 BF16 M=1 GEMV, fused RoPE + BF16 KV writes,
  split-KV BF16 attention, merged q/k/v projection, and in-graph greedy sampling
  are captured as one whole-step CUDA graph.
- decode (`quant="fp4"`): every weight in W4A16 (e2m1 codes packed 2/byte +
  per-16 bf16 scales) with q/k/v rows merged into one wide GEMV per layer, a
  model-local GEMV using native Blackwell e2m1x2 conversion with FP32
  accumulation (`cosmos3_reasoner_gemv_w4a16_bf16`), a fused
  RoPE + e4m3 KV-cache append (`cosmos3_reasoner_rope_kv_fp8_bf16`,
  device-side position/slot scalars), a split-KV single-query GQA flash-decode
  attention over the FP8 KV cache with a device-side length
  (`cosmos3_reasoner_decode_attn_fp8kv_bf16` — no fixed-window mask,
  CUDA-graph safe on growing sequences), and in-graph greedy sampling with
  device-side loop-state advance, so the host decode loop is nothing but
  `graph.replay()` calls on one captured token step.

Parity: both optimized paths retain FP32 accumulation inside GEMV and attention.
The BF16 path is validated against eager logits below. Greedy output is not
claimed to be token-for-token identical because near-ties may choose a different
token after valid floating-point reduction-order changes.

Thor T5000 decode throughput (batch 1, greedy, 128 forced output tokens). Values
are P50 over five measured runs after one warmup. The reproducible profile uses
the public Cosmos cookbook image/video assets and prompt lengths of 1705, 911,
and 1263 tokens for text, image, and video respectively.

| decode tok/s | Text | Image | Video |
|---|---|---|---|
| FlashRT BF16 + CUDA graph | **68.3** | **71.6** | **70.1** |
| FlashRT NVFP4 + CUDA graph | **104.3** | **112.6** | **108.7** |

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/NVIDIA/cosmos.git nvidia-cosmos
git -C nvidia-cosmos sparse-checkout set cookbooks/cosmos3/reasoner
python benchmarks/cosmos3_reasoner_thor.py --modes text,image,video \
  --quant bf16 --warmup-iters 1 --iters 5 \
  --nvidia-assets-dir nvidia-cosmos/cookbooks/cosmos3/reasoner/assets \
  --json-out bf16.json

python benchmarks/cosmos3_reasoner_thor.py --modes text,image,video \
  --quant fp4 --warmup-iters 1 --iters 5 \
  --nvidia-assets-dir nvidia-cosmos/cookbooks/cosmos3/reasoner/assets \
  --json-out nvfp4.json
```

The text case uses deterministic synthetic padding to reach the published
1705-token ISL. The official cookbook image and four-frame video requests
naturally produce 911 and 1263 tokens with the checkpoint processor. The
profile fixes the hardware, model, batch, greedy decode, input lengths, and
output length. The BF16 row uses BF16 weights, activations, and KV cache.
Against the eager implementation, the first decode-step logits measured
cosine 0.999937 and 1.12% relative L2 with the same argmax; later greedy tokens
can diverge because the custom GEMV uses a different valid FP32 reduction order.
The NVFP4 row instead uses NVFP4 weights, BF16 activations, and an FP8 KV cache.

Replay-only profiling attributes about 79% of FP4 decode kernel time to W4A16
GEMV (~138 GB/s effective) and 17% to FP8-KV attention. Material gains beyond
this point require a precision-preserving native-scale/tensor-core W4 path and
an XQA build specialized for head_dim 128/group 2; small elementwise fusions
cannot close that remaining gap. Prefill is still the torch path.

`config="cosmos3_edge"` is the Thor baseline runner for NVIDIA's official
Cosmos Framework inference path. It is intentionally upstream-first: run this
baseline, capture latency/outputs, then port the FlashRT optimized denoise/action
path behind the same config.

This page covers the supported path, measured performance, usage, and
correctness requirements. The latency table measures denoise only. It does not
include tokenization, video preprocessing, VAE encode, or response handling.

## Performance

The following results use the same `av_inverse_0` 30-step denoise boundary and
15 back-to-back hot iterations. Accuracy is measured against the official
final action with the gates `cosine >= 0.999` and `relative L2 < 3%`. Both
non-cached engines and the 3- and 2-compute TeaCache schedules were also
checked with a second seed.

| Configuration | Denoise P50 | Speedup vs Official | Cosine | Rel-L2 | Peak memory (GiB) |
|---|---:|---:|---:|---:|---:|
| Official Cosmos3-Edge (eager) | 33.14 s | 1.00x | reference | reference | not recorded |
| FlashRT FP8, no step cache | 5.792 s | **5.72x** | 0.999983 | 0.59% | 7.85 |
| FlashRT FP8 + NVFP4 FFN, no step cache | 5.020 s | **6.60x** | 0.999771 | 2.15% | 7.54 |
| FlashRT FP8 + TeaCache, 15 computes | 2.877 s | **11.52x** | 0.999985 | 0.56% | 7.85 |
| FlashRT FP8 + TeaCache, 6 computes | 1.153 s | **28.74x** | 0.999984 | 0.57% | 7.85 |
| FlashRT FP8 + TeaCache, 3 computes | 0.576 s | **57.56x** | 0.999986 | 0.54% | 7.85 |
| FlashRT FP8 + TeaCache, 2 computes | 0.384 s | **86.36x** | 0.999983 | 0.59% | 7.85 |
| FlashRT FP8 + NVFP4 FFN + TeaCache, 2 computes | **0.325 s** | **102.06x** | 0.999720 | 2.38% | 7.54 |

The production optimization claim is the no-cache result: **5.72x with FP8**
or **6.60x with FP8 plus NVFP4 FFN**. Those rows execute all 30 velocity
evaluations and isolate the benefit of the FlashRT engine from step caching.

TeaCache is an optional approximation and its usable schedule is workload
dependent. The values above are benchmark observations for two validated
samples, not a dataset-level task-success claim. Camera placement, robot
dynamics, task distribution, conditioning strength, and user fine-tuning can
all change how quickly the velocity field varies between steps. Qualify each
TeaCache schedule against the target checkpoint and deployment data before
using it. The 2-compute rows are aggressive benchmark operating points.

For the measured workload, **FP8 + TeaCache with 3 computes** is the suggested
starting point: 0.576 s, 57.56x over official eager denoise, and 0.54% relative
L2. Start with more computes for a new task and reduce the count only after
task-level validation.

## Optimization structure

The no-cache 6x-class path combines several changes rather than relying on one
kernel:

- Static AV buffers and a pre-installed und-conditioned K/V prefix remove
  repeated allocation and invariant work.
- FP8 E4M3 weights and activations cover the six generated-stream projections.
  Activation scales are calibrated once with an amax margin.
- The optional NVFP4 path replaces the FFN up/down projections with W4A4
  CUTLASS kernels.
- Fused residual, RMSNorm, quantization, ReLU2, Q/K normalization, and RoPE
  kernels reduce launches and intermediate traffic.
- FA4 handles generated-stream attention and a native UniPC kernel advances
  the scheduler.
- The final layer computes only the 60 action rows while retaining full K/V
  context.
- All 30 denoise steps are captured in one CUDA Graph.

TeaCache adds a fixed compute schedule to that graph. A skipped denoise step
still runs native UniPC but reuses the most recent velocity buffer. The schedule
contains no runtime branch and requires no additional kernel.

## Requirements

- Jetson AGX Thor with CUDA compute capability 11.0.
- A FlashRT build containing the Thor kernels and FA4 backend.
- The public Cosmos3-Edge checkpoint.
- An importable NVIDIA Cosmos Framework checkout for the official baseline and
  live preprocessing path.
- The Wan2.2 VAE checkpoint used by Cosmos3-Edge.

Use arguments or environment-specific configuration to point at these assets;
FlashRT does not assume a host directory layout.

```bash
export COSMOS_EDGE_CHECKPOINT=/path/to/Cosmos3-Edge
export COSMOS_EDGE_INPUT=/path/to/av_inverse_input.json
export COSMOS_FRAMEWORK_ROOT=/path/to/cosmos-framework
export WAN_VAE_CHECKPOINT=/path/to/Wan2.2_VAE.pth
export COSMOS_EDGE_OUTPUT=/path/to/output
```

Build the extensions for Thor using the normal repository build flow:

FP8 is the recommended default (large precision margin); `--ffn-fp4` trades
about 13% latency for a still-passing but thinner gate margin. Measured negatives,
kept default-off or not landed: a fused QKV wide GEMM (cuBLASLt N=4096 tactic
regression, behind `qkv_fused=`) and SageAttention2 int8 gen attention (the
SM80-era int8 mma core is both numerically wrong and slower than FA4 on
SM110). Extending W4A4 to the final O projections also passed the accuracy gate
but did not improve latency because activation quantization and output casting
offset the smaller GEMM. The largest remaining denoise lever is a Thor-native
int8/fp8 attention kernel.

```bash
cmake -B build -S . -DGPU_ARCH=110
cmake --build build -j"$(nproc)" --target flash_rt_kernels flash_rt_fp4
```

## Official baseline

Run the official action-only path first. This establishes the output and
end-to-end baseline without decoding or saving generated video:

```bash
python examples/cosmos3_edge_thor_baseline.py \
  --checkpoint "$COSMOS_EDGE_CHECKPOINT" \
  --input-json "$COSMOS_EDGE_INPUT" \
  --output-dir "$COSMOS_EDGE_OUTPUT/official" \
  --vae-path "$WAN_VAE_CHECKPOINT" \
  --cosmos-root "$COSMOS_FRAMEWORK_ROOT" \
  --backend official_action_only
```

The full official backend remains available with `--backend official` when
decoded vision output is needed.

## FlashRT denoise benchmark

The optimized benchmark consumes the captured denoise and step-0 boundary
fixtures. All asset paths are explicit so a run cannot silently use a stale
local artifact.

```bash
export COSMOS_EDGE_REFERENCE=/path/to/denoise/tensors.safetensors
export COSMOS_EDGE_BOUNDARY=/path/to/step0_boundary/tensors.safetensors
export COSMOS_EDGE_OFFICIAL_BENCHMARK=/path/to/official/benchmark.json
```

FP8, all 30 computes:

```bash
python benchmarks/cosmos3_edge_thor_denoise.py \
  --checkpoint "$COSMOS_EDGE_CHECKPOINT" \
  --reference-dump "$COSMOS_EDGE_REFERENCE" \
  --boundary-dump "$COSMOS_EDGE_BOUNDARY" \
  --official-benchmark "$COSMOS_EDGE_OFFICIAL_BENCHMARK" \
  --engine fp8 \
  --iters 15 \
  --warmup-steps 30 \
  --enforce-gates
```

FP8 plus NVFP4 FFN, all 30 computes:

```bash
python benchmarks/cosmos3_edge_thor_denoise.py \
  --checkpoint "$COSMOS_EDGE_CHECKPOINT" \
  --reference-dump "$COSMOS_EDGE_REFERENCE" \
  --boundary-dump "$COSMOS_EDGE_BOUNDARY" \
  --official-benchmark "$COSMOS_EDGE_OFFICIAL_BENCHMARK" \
  --engine fp8 \
  --ffn-fp4 \
  --iters 15 \
  --warmup-steps 30 \
  --enforce-gates
```

## TeaCache

`--teacache-computes K` selects `K` evenly spaced velocity evaluations and
always includes steps 0 and 29. `--teacache-steps` accepts an explicit
comma-separated compute set and takes precedence.

```bash
python benchmarks/cosmos3_edge_thor_denoise.py \
  --checkpoint "$COSMOS_EDGE_CHECKPOINT" \
  --reference-dump "$COSMOS_EDGE_REFERENCE" \
  --boundary-dump "$COSMOS_EDGE_BOUNDARY" \
  --official-benchmark "$COSMOS_EDGE_OFFICIAL_BENCHMARK" \
  --engine fp8 \
  --teacache-computes 3 \
  --iters 15 \
  --warmup-steps 30 \
  --enforce-gates
```

For qualification, retain a no-cache output for every evaluated seed and
report both action-space metrics and downstream task success. Passing the
cosine and relative-L2 gates is necessary for this benchmark but does not by
itself establish policy quality.

## Python API

The stable model registration exposes the official pipeline through
`flash_rt.load_model`:

```python
import flash_rt

model = flash_rt.load_model(
    checkpoint="/path/to/Cosmos3-Edge",
    config="cosmos3_edge",
    framework="torch",
    hardware="thor",
)
model.set_prompt(input_json="/path/to/av_inverse_input.json")
result = model.infer(
    output_dir="/path/to/output",
    backend="official_action_only",
    cosmos_root="/path/to/cosmos-framework",
    vae_path="/path/to/Wan2.2_VAE.pth",
    benchmark=True,
)
```

`backend="flashrt"` is the fixture-backed correctness and denoise benchmark
route. Supply `reference_dump` and `boundary_dump` to that backend. The live
official-to-FlashRT handoff remains an integration path for development; it is
not the source of the denoise-only numbers above.

## Scope and limitations

- The performance table is denoise P50 in a hot process, not cold-start or raw
  request end-to-end latency.
- The no-cache engine is the portable optimization result. TeaCache schedules
  require case-by-case validation and can differ substantially after
  fine-tuning.
- Official preprocessing, including the Wan2.2 VAE encode, is outside the
  optimized denoise boundary. On the measured sample VAE encode was about
  3.7 s, so it is the next major end-to-end optimization target.
- The current route is registered only for `framework="torch"` and
  `hardware="thor"`; unsupported hardware fails during model resolution.

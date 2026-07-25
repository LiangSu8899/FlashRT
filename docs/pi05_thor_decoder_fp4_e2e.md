# Pi0.5 Thor Decoder NVFP4 End-to-End Result

This document records the first strict end-to-end result for the Pi0.5
action-expert decoder NVFP4 path on NVIDIA Thor SM110. It follows the separately
recorded M=10 primitive baseline and includes activation preprocessing, every
decoder layer and denoising step, CUDA Graph replay, image upload/preprocessing,
synchronization, and action postprocessing.

## Implemented Path

The explicit `use_fp4_decoder=True` path replaces all four decoder projection
GEMMs at the production `M=10` shape:

| Projection | M | N | K | CUTLASS variant |
|---|---:|---:|---:|---:|
| `qkv` | 10 | 2560 | 1024 | v7 |
| `o` | 10 | 1024 | 2048 | v7 |
| `gate_up` | 10 | 8192 | 1024 | v9 |
| `down` | 10 | 1024 | 4096 | v7 |

Weights are loaded directly from the FP16 safetensors checkpoint, transformed
with the same Q/K head interleave and Gate+Up concatenation as the FP8 loader,
then quantized once to NVFP4 E2M1 plus CUTLASS SFB. There is no FP8-dequantized
weight path.

Runtime activation preprocessing is CUDA-Graph capturable and uses:

- Pi0.5 AdaRMSNorm + gate output + NVFP4/SFA in one launch for QKV.
- Dynamic NVFP4/SFA quantization for the attention output before O.
- Gated residual update + Pi0.5 AdaRMSNorm + gate output + NVFP4/SFA in one
  launch before Gate+Up and the next layer's QKV.
- Existing fused GeGLU + NVFP4/SFA before Down.

The production FP8 frontend remains the default. The opt-in is exposed through
`load_model(..., use_fp4=True, use_fp4_decoder=True)`. Decoder-only A/B runs can
keep the encoder in FP8 with `fp4_layers=()`, `use_awq=False`, and
`use_p1_split_gu=False`.

The decoder FP4 path currently supports standard Torch B=1 inference only.
CFG, batched inference, and model-runtime export raise explicit errors when the
path is enabled. Unsupported hardware, shapes, missing NVFP4 kernels, invalid
variants, or failed GEMM launches also raise; none select FP8 implicitly.

## Verification Contract

The committed harness is `tests/bench_pi05_decoder_fp4_e2e.py`. The official
run used:

- Commit `bc070ae5ae3764d872efced263c401d3c05f91fb`.
- NVIDIA Thor, compute capability 11.0, MAXN.
- GPC min/max/current 1.575 GHz.
- NVD min/max/current 1.692 GHz.
- EMC cap 4.266 GHz.
- Torch 2.10.0 with CUDA 13.0.
- Two camera views and the 13-token prompt encoded in the harness.
- Eight LIBERO observations with N=8, percentile 99.9 calibration.
- Matched NumPy noise seeds for action comparison.
- Separate FP8 and FP4 processes.
- 20 warmup calls and 100 complete `infer()` latency samples per mode.

The suite requires a clean tracked worktree and fails if clocks, hardware,
finite outputs, action fidelity, or FP4 p50 speedup do not meet the contract.

## Locked End-to-End Result

| Metric | FP8 | Decoder FP4 | Change |
|---|---:|---:|---:|
| p50 latency | 44.7473 ms | 43.1039 ms | -1.6433 ms |
| p95 latency | 44.9634 ms | 43.2874 ms | -1.6760 ms |
| p50 speedup | 1.0000x | 1.0381x | +3.81% |

Matched-noise fidelity across all eight observations:

| Metric | Result |
|---|---:|
| Raw 32D action cosine | 0.99956287 |
| Worst raw per-sample cosine | 0.99921848 |
| Raw max absolute difference | 0.09716797 |
| Final 7D action cosine | 0.99980575 |
| Worst final-action per-sample cosine | 0.99975416 |
| Final-action max absolute difference | 0.04827201 |

All correctness and latency gates passed. These action metrics establish
numerical fidelity against the production FP8 decoder; they are not a robot
task-success-rate claim.

The measured extension hashes were:

- `flash_rt_fp4`: `a944449f4a1f763461fb92b6e87d3796c6b6dfbde58e8550eaa62bf15d61a345`
- `flash_rt_kernels`: `c16f817c9ea924b1d88c97e9b510bd61cdbecf3422f483609bb9de8e38b0292b`
- `result.json`: `cf64f3e470448881a35dfbcb7219609413633ca55197e07379f929999492fc83`

The full local result contains all 200 latency samples and is intentionally not
committed because it records machine-local paths. The reproducible method and
all acceptance thresholds are in the committed harness.

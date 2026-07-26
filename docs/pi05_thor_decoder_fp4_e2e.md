# Pi0.5 Thor NVFP4 End-to-End Results

This document records the strict end-to-end development of the Pi0.5 NVFP4
path on NVIDIA Thor SM110. The first run isolated the action-expert decoder;
the current candidate combines all 17 live encoder FFN NVFP4 layers with the
decoder NVFP4 path. For 1/2/3 views, p50 must be at least two milliseconds
faster than the published encoder-FP4 + decoder-FP8 results: at most
28.5/34.3/40.8 ms respectively. The earlier explicit 3-view 40 ms target is
also retained.

Latency covers the full `infer()` call: image preprocessing and upload, SigLIP,
encoder, all 18 decoder layers across 10 denoising steps, CUDA Graph replay,
synchronization, action download, and postprocessing.

## Implemented Path

The explicit `use_fp4_decoder=True` path replaces all four decoder projection
GEMMs at the production `M=10` shape:

| Projection | M | N | K | CUTLASS variant |
|---|---:|---:|---:|---:|
| `qkv` | 10 | 2560 | 1024 | v7 |
| `o` | 10 | 1024 | 2048 | v7 |
| `gate_up` | 10 | 8192 | 1024 | v10 |
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

The current candidate additionally uses register-only decoder AdaRMSNorm
preprocessing and native SM110 E2M1x2 conversion. The encoder P1 path uses two
FP4-output Gate/Up GEMMs, a gate LUT plus native E2M1x2 combiner, and encoder
Down variant v7. Native FP4 conversion uses round-to-nearest-even, so it is an
explicit numerical mode rather than a bit-exact alias for the historical
midpoint implementation.

The candidate also enables the established 17-layer encoder FFN NVFP4 preset
with AWQ and P1 split-GU. Its AWQ exponent is 0.8. The standard uint8 image hot
path uses a precomputed 256-entry FP16 normalization table and a reused host
buffer. This replaces per-frame uint8-to-FP32-to-FP16 allocations while
producing bit-identical normalized images and bit-identical model outputs.

The production FP8 frontend remains the default. The opt-in is exposed through
`load_model(..., use_fp4=True, use_fp4_decoder=True)`. The decoder FP4 path
currently supports standard Torch B=1 inference only. CFG, batched inference,
and model-runtime export raise explicit errors when enabled. Unsupported
hardware, shapes, missing kernels, invalid variants, or failed launches also
raise; none select FP8 implicitly.

## Verification Contract

The committed harness is `tests/bench_pi05_decoder_fp4_e2e.py`. The current
multi-view run used:

- NVIDIA Thor, compute capability 11.0, MAXN.
- GPC min/max/current 1.575 GHz.
- NVD min/max/current 1.692 GHz.
- EMC cap 4.266 GHz.
- Torch 2.10.0 with CUDA 13.0.
- Production graph autotune level 3.
- One, two, or three camera views and the explicit 13-token prompt in the
  harness.
- Eight LIBERO observations with N=8, percentile 99.9 calibration.
- Matched NumPy noise seeds for action comparison.
- Separate FP8 and FP4 processes.
- 20 warmup calls and 100 complete `infer()` samples per mode.

The suite requires a clean tracked worktree and fails unless clocks and device
identity match, all outputs are finite, FP4 is faster than FP8, and the
per-view published-minus-2-ms p50 target passes. It also requires 2-view p95
at most 40 ms and 3-view p50 at most 40 ms. Final 7D action cosine must be at
least 0.999 globally and 0.995 for every sample; internal raw cosine must be at
least 0.995 globally and for every sample.

## Encoder Down v7 Multi-View Result (commit `8424808`, 2026-07-26)

The encoder Down GEMM was re-swept with complete `infer()` calls at the actual
per-view encoder shapes. Variant v7 (`tile128x128x256`, cluster `1x1x1`) was
the fastest end-to-end choice. It changes only the CUTLASS tile schedule; the
FP4 inputs, weights, scale layouts, and outputs are unchanged.

Locked-clock results use 20 warmups and 100 retained samples per mode:

| Views | Same-run FP8 p50 / p95 | FP4+FP4 p50 / p95 | Published FP4+FA4 p50 | Gain vs published |
|---:|---:|---:|---:|---:|
| 1 | 35.315 / 35.588 ms | **28.035 / 28.269 ms** | 30.5 ms | **2.465 ms** |
| 2 | 41.485 / 41.652 ms | **32.986 / 33.138 ms** | 36.3 ms | **3.314 ms** |
| 3 | 49.461 / 49.726 ms | **39.393 / 39.743 ms** | 42.8 ms | **3.407 ms** |

The same matched-noise run recorded the following raw and final-action
cosines, measured against the FP16 reference path; the Down variant changes
only the GEMM schedule.

| Views | Raw cosine / worst sample | Final action cosine / worst sample |
|---:|---:|---:|
| 1 | 0.954346 / 0.720291 | 0.870413 / 0.021278 |
| 2 | 0.997625 / 0.995526 | 0.999380 / 0.998825 |
| 3 | 0.997289 / 0.995878 | 0.999062 / 0.996514 |

The variant sweep used complete FP4 child processes with the 1-view fixture,
5 warmups, and 20 retained calls. The screened p50 values were v0 28.352 ms,
v1 28.686 ms, v3 28.791 ms, v4 28.358 ms, v6 28.897 ms, v7 28.022 ms,
v8 28.962 ms, and v10 28.541 ms. The formal multi-view measurements above
then confirmed v7 with the full sampling contract. The tree also retains the decoder MSE
quantization diagnostic used for that precision comparison; it does not add
a runtime fallback.

## 40 ms Production Result

| Metric | Production FP8 | Full NVFP4 | Change |
|---|---:|---:|---:|
| p50 latency | 44.1315 ms | 39.1045 ms | -5.0270 ms |
| p95 latency | 44.4170 ms | 39.2420 ms | -5.1749 ms |
| p50 speedup | 1.0000x | 1.1286x | +12.86% |

Both absolute latency gates passed with 0.8955 ms of p50 headroom and 0.7580 ms
of p95 headroom.

Matched-noise fidelity across all eight observations:

| Metric | Result |
|---|---:|
| Internal raw 32D action cosine | 0.99764686 |
| Worst raw per-sample cosine | 0.99506149 |
| Raw max absolute difference | 0.24609375 |
| Final returned 7D action cosine | 0.99913207 |
| Worst final-action per-sample cosine | 0.99635148 |
| Final-action max absolute difference | 0.16438568 |

The full encoder FP4 preset has a documented raw-output cosine around 0.998.
The final 7D action is the API output consumed by the LIBERO robot, so it keeps
the stricter 0.999 global gate; the full 32D tensor remains a recorded internal
diagnostic. These checks establish matched-input numerical fidelity, not robot
task success rate.

The measured artifacts were:

- `flash_rt_fp4`: `a944449f4a1f763461fb92b6e87d3796c6b6dfbde58e8550eaa62bf15d61a345`
- `flash_rt_kernels`: `c16f817c9ea924b1d88c97e9b510bd61cdbecf3422f483609bb9de8e38b0292b`
- `result.json`: `0bc9e539cd0225d254cff4f674e8befdcd00acae8264af28a336d1ddb66bbcb3`

## Decoder-Isolation Baseline

The earlier run at commit `bc070ae5ae3764d872efced263c401d3c05f91fb`
kept the encoder in FP8 and changed only the decoder. It established the
decoder contribution independently of the production encoder preset:

| Metric | FP8 | Decoder FP4 | Change |
|---|---:|---:|---:|
| p50 latency | 44.7473 ms | 43.1039 ms | -1.6433 ms |
| p95 latency | 44.9634 ms | 43.2874 ms | -1.6760 ms |
| p50 speedup | 1.0000x | 1.0381x | +3.81% |

Its final 7D action cosine was 0.99980575 and raw 32D cosine was 0.99956287.
The result JSON SHA-256 was
`cf64f3e470448881a35dfbcb7219609413633ca55197e07379f929999492fc83`.

The local result files contain all 200 retained latency samples and are not
committed because they include machine-local paths. The reproducible method,
precision configuration, and acceptance thresholds are committed in the
harness.

## FP4+FP4 Multi-View Candidate (2026-07-25)

The formal runs use commit
`8c09371586b70e7a0c53fb79cc017f16100cbeab` and the strict default
configuration in `tests/bench_pi05_decoder_fp4_e2e.py`:

- Encoder layers 0-16: NVFP4 Gate, Up, and Down FFN projections with AWQ
  alpha 0.8 and P1 split-GU. Encoder attention projections remain FP8.
- Decoder layers 0-17 across all 10 denoising steps: NVFP4 QKV, O, Gate+Up,
  and Down projections. No decoder projection selects FP8 implicitly.
- FA4 is active for SigLIP and encoder attention.
- Each view count uses its matching eight-observation fixture, 20 warmups, and
  100 complete `infer()` samples in separate FP8 and FP4 processes.

Locked-clock latency:

| Views | Same-run FP8 p50 / p95 | FP4+FP4 p50 / p95 | Published FP4+FA4 p50 | Published delta | Gate |
|---:|---:|---:|---:|---:|---|
| 1 | 35.614 / 35.878 ms | **28.884 / 29.025 ms** | 30.5 ms | -1.616 ms | **fail**, target <=28.5 ms |
| 2 | 41.727 / 41.948 ms | **33.095 / 33.340 ms** | 36.3 ms | **-3.205 ms** | pass |
| 3 | 49.816 / 50.076 ms | **39.821 / 39.985 ms** | 42.8 ms | **-2.979 ms** | pass, including <40 ms |

Matched-noise fidelity across eight observations per view count:

| Views | Raw cosine / worst sample | Raw max abs | Final action cosine / worst sample | Action max abs | Gate |
|---:|---:|---:|---:|---:|---|
| 1 | 0.902456 / 0.480090 | 2.005371 | 0.697944 / -0.358415 | 1.887612 | **fail** |
| 2 | 0.997764 / 0.995729 | 0.380005 | 0.999295 / 0.998271 | 0.096136 | pass |
| 3 | 0.998375 / 0.997610 | 0.211670 | 0.999444 / 0.998854 | 0.095402 | pass |

The 1-view failure is reproducible and is not caused by an unstable FP8
reference: an independent FP8 rerun was elementwise identical. A decoder-only
FP4 diagnostic, with every encoder FFN left on FP8, still produced final-action
cosine 0.859115 because one sample's gripper sign changed. Quantizing encoder
layers 0-15 or all 0-16 produced nearly the same failure, so excluding the last
live encoder layer is not a fix. AWQ alpha 0.5 also failed. These diagnostic
configurations are not runtime fallback paths and are not the proposed preset.

This candidate must not merge yet. Two blockers remain:

1. Fix 1-view FP4+FP4 numerical fidelity without relaxing the cosine gates.
2. Run task-level LIBERO rollouts. The local environments do not currently
   contain the `libero` Python package, so this validation has not been run.

Reproduction command, repeated with `--num-views 1`, `2`, and `3`:

```bash
PYTHONPATH=<repo-root> python tests/bench_pi05_decoder_fp4_e2e.py \
  --num-views 2 \
  --output-dir <output-dir>
```

The FA4 path additionally needs its runtime dependencies on the
interpreter's path (the `thor-fa4` pip extra) and the CUDA runtime
libraries of the active torch install on `LD_LIBRARY_PATH`.

Current shared artifacts:

- `flash_rt_fp4`: `2c66b308661a142765af9cad8ee6a54eff465665829964359d0cada1c4a0ec96`
- `flash_rt_kernels`: `30270002a9646ec230fd69f2cb76ef33acbb5d683872c5833796aa15e10c0c91`
- 1-view `result.json`: `9a5c911dd3a867d7b58abf25bcfeb7201e3a6649baafc7d529a2c0f92bd53267`
- 2-view `result.json`: `43d7c80ca06528a76c183e3e51a018b27eabbb4cb44f38406fd8dacf3b0e4df1`
- 3-view `result.json`: `0780386b4281bf057425a710fcb821dee0dd8cc552d045e3426cf56f38fb6ade`

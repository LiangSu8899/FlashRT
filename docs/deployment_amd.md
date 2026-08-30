# FlashRT on AMD (ROCm / CDNA4)

Status: **pi05 production-ready on MI350-series (gfx950)**. The AMD backend
is a self-contained tree — the CUDA tree and root CMake are untouched, and
the two never build together.

Measured on an MI350X (gfx950, ROCm 7.x, PyTorch ROCm 2.10), pi05 with
2 camera views, 10-step denoise, FP8 default:

| Arm | median / inference | vs torch.compile |
|---|---|---|
| PyTorch eager (openpi reference) | 116.9 ms | 0.35× |
| `torch.compile` max-autotune | 40.5 ms | 1.00× |
| **FlashRT AMD, FP8 (default)** | **16.4 ms** | **2.47×** |
| FlashRT AMD, BF16 (`use_fp8=False`) | ~30 ms | 1.34× |

Output cosine vs the FP32 openpi reference on real LIBERO frames: 0.9993
(FP8), 0.9992 (BF16). For context, the same protocol on an RTX 5090
(FlashRT FP8, CUDA) measures 19.7 ms — the CDNA4 port is currently the
fastest FlashRT pi05 target.

## Layout

| Path | Role |
|---|---|
| `csrc/amd/` | HIP kernels + hipBLASLt GemmRunner + MFMA GEMM/attention + pybind module `flash_rt_amd_kernels` (standalone CMake project) |
| `flash_rt/amd/core/` | `HipBuffer` / `HipGraph` — ctypes twins of the CUDA runtime seam over `libamdhip64` |
| `flash_rt/amd/models/pi05/pipeline.py` | Mirror of `pipeline_rtx.py` for CDNA4 (BF16 + FP8 branches) |
| `flash_rt/amd/frontends/torch/pi05.py` | `Pi05TorchFrontendAmd` — weights, quantization, prompt, capture driving |
| `flash_rt/amd/hardware/cdna4/` | Attention backends (aiter CK-tile + hand-written split-KV decoder) |

The pybind ABI is identical to the CUDA module: every kernel entry takes
`uintptr_t` device pointers plus a `uintptr_t` stream. Pipeline code is
portable text between the two trees.

Hot-path composition at the defaults: full HIP-graph capture-then-replay,
static per-tensor FP8 (OCP e4m3) with real-data calibration, hand-written
MFMA small-M GEMMs with setup-time weight repacking for the decoder
(qkv / attn-out / ffn-gate-up; the K=4096 down-projection stays on the
measured-faster hipBLASLt), aiter CK-tile attention for vision/encoder,
hand-written split-KV flash attention with a fused FP8-quantize epilogue
for the decoder cross-attention, and CDNA-tuned elementwise/norm kernels.

## Requirements

- ROCm 7.x with hipBLASLt (`gfx950` validated; `GPU_ARCH` overridable)
- PyTorch ROCm build (`torch.version.hip` set)
- pybind11 (auto-installed to a local target dir by the build wrapper if
  the interpreter lacks it)
- optional: [aiter](https://github.com/ROCm/aiter) for the vision/encoder
  attention backend (the default). Without it, set `FVK_AMD_ATTN=sdpa`
  for the torch-sdpa fallback (slower, still correct). aiter JIT-compiles
  against the running torch ABI — keep one torch version per environment.

## Build

```bash
cmake -B build-amd -S csrc/amd [-DGPU_ARCH=gfx950]
cmake --build build-amd -j 8
# or, in cmake-less environments (falls back to a one-shot hipcc):
bash scripts/amd/build_amd.sh gfx950
```

Output: `flash_rt/amd/flash_rt_amd_kernels*.so`. Release flags only
(`-O3 -ffp-contract=fast`); no third-party kernel library is vendored —
hipBLASLt ships with ROCm, and everything else is hand-written HIP/MFMA.

## Run (pi05)

Through the stable door (`amd_cdna4` is auto-detected on ROCm builds):

```python
import flash_rt
model = flash_rt.load_model(checkpoint_dir, config="pi05",
                            framework="torch", hardware="amd_cdna4",
                            num_views=2, use_fp8=True)
fe = model.pipeline
fe.set_prompt(prompt_text, state=state)   # builds + captures the HIP graph
fe.calibrate(observation)                 # real-data FP8 calibration
result = fe.infer({"image": img, "wrist_image": wrist})
actions = result["actions"]               # (chunk, 7) unnormalized
```

Or construct `flash_rt.amd.frontends.torch.pi05.Pi05TorchFrontendAmd`
directly with the same keyword surface as `Pi05TorchFrontendRtx`. A
runnable end-to-end script with timing lives at
`examples/pi05_amd_quickstart.py`. The PaliGemma tokenizer model is
resolved as documented in `flash_rt/utils/paligemma_tokenizer.py`
(`FLASH_RT_PALIGEMMA_TOKENIZER` override supported).

### Precision tiers

- **FP8 (default, `use_fp8=True`)** — OCP FP8 E4M3 weights + activations
  for the large GEMMs, static per-tensor scales from real-data
  calibration (`calibrate(...)`, percentile 99.9 by default — the same
  contract as `docs/calibration.md`). MI350's FP8 is OCP e4m3, never the
  CDNA3 `fnuz` variant.
- **BF16 (`use_fp8=False`, or env `FVK_PI05_AMD_FORCE_BF16=1`)** — the
  unquantized baseline; no calibration required.
- **FP4 (MXFP4)** — the GEMM kernels are unlocked and parity-verified
  (`GemmRunner.mxfp4_nt_dev`, hipBLASLt `HIP_R_4F_E2M1` + UE8M0 vec32
  block scales), but there is deliberately **no end-to-end FP4 tier
  yet**: on the current ROCm stack no available fp4 path beats FP8 at
  these shapes, so a user-facing knob would only select a slower path.
  `load_model(use_fp4=True)` remains Thor-only and raises on AMD.

### Prompt-length graph strategies

Pi0.5 renders robot state into the prompt, so token length drifts.
Both RTX strategies are supported, same semantics:

- `state_prompt_mode="exact"` (default): one captured graph per exact
  prompt length, cached; pair with `warm_state_prompt_buckets(...)`.
- `state_prompt_mode="fixed"`: ONE padded graph serves every length
  (masked prefix + runtime `devpos`/`seqused` K/V append) — no mid-loop
  captures. Latency follows the PADDED length, so right-size the
  capacity to your deployment's real prompt+state length with
  `state_prompt_fixed_max_len=<tokens>` (env
  `FLASHRT_PI05_STATE_PROMPT_FIXED_MAX_LEN`; default is the 200-token
  ceiling, and a prompt exceeding the capacity raises instead of
  recapturing). The decoder runs the same custom split-KV kernel as
  exact mode (seqused pointer, FP8-out epilogue included); the encoder
  runs the seqused variant of the MFMA flash kernel (masked sdpa stays
  available via `FVK_AMD_FIXED_ENC_ATTN=sdpa`). At a right-sized
  capacity the premium over exact mode is well under 1 ms, and accuracy
  holds (cos vs the FP32 reference 0.9993, on par with exact).

### Environment knobs

| Env | Default | Meaning |
|---|---|---|
| `FVK_AMD_ATTN` | `aiter` | vision/encoder attention backend: `aiter` (CK-tile) or `sdpa` (torch fallback) |
| `FVK_AMD_DEC_ATTN` | `custom` | decoder cross-attention: `custom` (hand-written split-KV flash, fastest) or backend default |
| `FVK_AMD_ATTN_FP8OUT` | `1` | fuse the decoder attention output's FP8 quantize into the attention epilogue (bit-identical to the standalone quantize) |
| `FVK_AMD_FIXED_ENC_ATTN` | `flash` | fixed-mode encoder attention: `flash` (MFMA flash kernel, seqused pointer) or `sdpa` (masked torch fallback) |
| `FVK_AMD_CALIB_DET_ATTN` | `flash` | route the encoder site through the deterministic MFMA flash kernel during FP8 calibration so the collected scales are run-to-run stable; `off` calibrates on the library path |
| `FVK_AMD_DEC_GEMM` | `mfma` | decoder small-M GEMMs: `mfma` (packed-weight MFMA kernels where measured faster) or `hipblaslt` |
| `FLASHRT_FP8_NT_AUTOTUNE` | `auto` | timed hipBLASLt algorithm selection for the FP8 GEMMs at setup |
| `FLASHRT_FP8_ALGO_POOL` | `16` | heuristic pool depth for the timed selection; deeper pools (64/128) find faster encoder algos but widen run-to-run pick variance |
| `FVK_PI05_AMD_FORCE_BF16` | `0` | force the BF16 baseline regardless of `use_fp8` |
| `FLASHRT_PI05_STATE_PROMPT_MODE` | — | overrides the `state_prompt_mode` constructor arg |
| `FLASHRT_PI05_STATE_PROMPT_FIXED_MAX_LEN` | `200` | fixed-mode padded capacity in tokens; overrides `state_prompt_fixed_max_len` |

`FRT_ATTN_NSPLIT` / `FRT_ATTN_FUSED` / `FRT_ATTN_REDUCE_ALT` are
attention micro-bench knobs (A/B sweeps); leave them unset in production.

### Feature matrix vs the RTX frontend

| Surface | AMD |
|---|---|
| `set_prompt` / `warm_state_prompt_buckets` | ✅ |
| `calibrate` / `calibrate_with_real_data` (single + multi-frame percentile) | ✅ |
| `infer` / `precision_spec` / `get_latency_stats` | ✅ |
| `state_prompt_mode` `"exact"` + `"fixed"` (devpos/seqused) | ✅ |
| Temporal K/V caching (`cache_frames`), vision pooling/truncation knobs | ✅ |
| `set_rl_mode` (advantage-conditioned RL) | ❌ raises `NotImplementedError` |
| CFG-batched mode (`set_batched_mode` / `*_batch`) | ❌ raises `NotImplementedError` |
| INT8 vision/decoder legacy tiers (RTX sm87 era) | ❌ not applicable |

## Reproducing the numbers

1. Build (above) on a gfx950 part with ROCm 7.x + torch-ROCm.
2. `python examples/pi05_amd_quickstart.py --checkpoint <pi05_ckpt>`
   → expect ~16-17 ms median FP8 after warmup (add `--bf16` for ~30 ms).
3. For a judged comparison, run the identical loop on the CUDA build
   (RTX frontend) — protocols are the same: 50-iteration median after
   5 warmup replays, real image observations, `calibrate` before timing.

Cross-run medians move ±0.2-0.35 ms with hipBLASLt's timed autotune
picks; compare arms inside one process where possible.

When judging output cosine against a saved reference, pin the denoise
noise via `infer(obs, noise=...)` to the exact array the reference was
generated with — a fresh random draw shifts cos by ~1e-3, which swamps
real numerics differences. With pinned noise the FP8 band is
0.9992-0.9994 across processes (residual: library-attention
nondeterminism and autotune algorithm picks).

## HIP-vs-CUDA notes

- Wavefront is 64: reductions use `__shfl_down` wave64 helpers
  (`csrc/amd/kernels/common_hip.h`); there are no `*_sync` shuffle variants.
- The 3-arg graph instantiate is `hipGraphInstantiateWithFlags`.
- Memcpy-kind and capture-mode enums match CUDA numerically (validated on
  hardware by the runtime-seam smoke).
- FP8 storage is OCP `e4m3` (`__hip_fp8_e4m3`, `HIP_R_8F_E4M3`) — never the
  CDNA3 `fnuz` variant.
- hipBLASLt matmul is column-major; the GemmRunner uses the operand-swap
  form (`D_col = B_col @ A_col`), the same trick the CUDA FP8 paths use.
- gfx950 exposes `V_MFMA_F32_16X16X32_FP8_FP8` / `..._BF16` builtins with
  per-lane contiguous 8 B fragments; the decoder GEMMs repack weights at
  setup into the exact per-lane consumption order so each workgroup
  streams its tile as one linear slab (`csrc/amd/gemm/smallm_mfma.h`).
- Static `__shared__` is capped at 64 KB; larger LDS (up to gfx950's
  160 KB) needs `hipFuncSetAttribute(MaxDynamicSharedMemorySize)` +
  dynamic shared memory (see `csrc/amd/attention/encoder_flash.hip`).
- The default stream cannot be captured (`hipErrorStreamCaptureUnsupported`);
  capture on a dedicated stream (the pipeline uses a torch side stream).

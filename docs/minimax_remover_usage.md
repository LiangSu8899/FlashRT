# MiniMax-Remover — FlashRT Inference Pipeline

MiniMax-Remover video inpainting (subtitle / object removal) with FlashRT
kernelized inference on Blackwell SM120. Two precision paths ship under
`flash_rt.models.minimax_remover`:

| Entrypoint | Precision | Recommended use |
|------------|-----------|-----------------|
| `MiniMaxRemoverPipelineFP8` | FP8 (W8A8) | **Default.** Full-frame inpainting; end-to-end cosine >= 0.999, PSNR ~35-41 dB vs fp16 (see Performance). |
| `MiniMaxRemoverPipeline` | NVFP4 (W4A4) | Cropped small regions only. Full-frame large latents drift/blacken due to FP4 error accumulation. |

Both reuse the **generic** FlashRT kernels — the package ships **no
model-specific CUDA operators** — and rewrite the transformer Linears as
quantized GEMMs, fuse norm/gate/residual/gelu ops, and use kernel attention
(FA2 / SageAttention). The NVFP4 path additionally captures the N-step
flow-matching loop as a single CUDA Graph; the FP8 path is graph-compatible
(stable static scales, no host sync in steady state) but does not itself
capture a graph.

## Build

```bash
cd FlashRT
cmake -S . -B build -DGPU_ARCH=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target flash_rt_kernels
pip install -e ".[torch,minimax-remover]"
```

`GPU_ARCH=120` (RTX 5090) or `121` selects the Blackwell target; the NVFP4
surface is compiled in automatically (internally gated by
`ENABLE_CUTLASS_SM120_NVFP4_W4A16`, which is set from `GPU_ARCH`, not a flag
users pass). The FP8 symbols are part of the default build. Then install the
runtime extras:

```bash
pip install -e ".[minimax-remover]"   # diffusers + einops + scipy + sageattention
```

Importing `flash_rt.models.minimax_remover` always succeeds — it needs
**none** of `diffusers` / `einops` / `scipy` / `triton` / `sageattention`. The kernel
surface is validated lazily in each pipeline's `__init__` via
`load_nvfp4_kernels()` / `load_fp8_kernels()` (`flash_rt/models/minimax_remover/_utils.py`),
and the runtime deps are resolved at construction via `_import_runtime()`. If
a required symbol or dep is missing the constructor raises a clear
`RuntimeError` with the rebuild/install hint, so a non-matching build or a
bare environment fails fast instead of crashing mid-run.

### Required kernel symbols

NVFP4 path (Blackwell-only, auto-enabled by `GPU_ARCH=120/121`):

| Symbol | Role |
|--------|------|
| `nvfp4_sf_swizzled_bytes` | block-scale-factor byte layout helper |
| `bf16_weight_to_nvfp4_swizzled` | one-shot weight -> NVFP4 quantise |
| `quantize_bf16_to_nvfp4_swizzled` | per-call dynamic activation quantise |
| `fp4_w4a16_gemm_sm120_bf16out_pingpong` | SM120-native W4A4 MMA -> bf16 |
| `add_bias_bf16` | in-place bias add on bf16 GEMM output |
| `fp4_w4a16_gemm_bias_gelu_fp4out_sm120` | fused FFN-up GEMM + bias + GELU -> FP4 |

FP8 path (default build):

| Symbol | Role |
|--------|------|
| `quantize_fp8_static_fp16` | weight + static activation FP8 quantise |
| `fp8_gemm_descale_fp16` | FP8 W8A8 MMA -> fp16 with scale descale |
| `add_bias_fp16` | in-place bias add on fp16 GEMM output |

Shared block-fusion symbols (default `gelu_mode="inplace"`, default build, required by both paths):

| Symbol | Role |
|--------|------|
| `gelu_inplace` | in-place tanh-approximate GELU on bf16 FFN-up output |
| `gelu_inplace_fp16` | in-place tanh-approximate GELU on fp16 FFN-up output |

The attention backend (`FLASHRT_ATTN_MODE`) optionally pulls in
`sageattention` (Sage); `fa2` uses the vendored `flash_rt_fa2.so` and is the
dependency-light fallback. The fused norm / RoPE / Euler-step elementwise
kernels are self-contained Triton JIT kernels shipped in the package
(`_kernels.py`) and need no build step.

## Pipelines

### FP8 — `MiniMaxRemoverPipelineFP8` (default, full-frame)

`flash_rt/models/minimax_remover/_fp8_pipeline.py`. Uses static calibration:
the first inference call runs in dynamic-FP8 calibration mode (accumulating
activation amax on GPU), then freezes to a static `act_scale` for all
subsequent calls (zero CPU sync overhead in the steady state, suitable for
CUDA Graph capture). **The frozen scale is calibrated to the first call's
input; if the input resolution/shape changes, construct a new pipeline so
the scale is re-calibrated.**

- every eligible transformer Linear -> FP8 W8A8 GEMM (weight quantised once
  at load time; activation quantised with a calibrated static scale);
- per-block LayerNorm + adaLN modulation + gate-residual fused into Triton
  kernels (fp32 statistics);
- `torch.nn.functional.scaled_dot_product_attention` -> FA2 / SageAttention.

### NVFP4 — `MiniMaxRemoverPipeline` (small-region only)

`flash_rt/models/minimax_remover/pipeline.py`. It wraps a loaded diffusers
MiniMax-Remover `pipe` and consumes it in place:

- every eligible transformer Linear -> NVFP4 W4A4 GEMM (weight quantised
  once at load time; activation quantised **dynamically** per call with
  per-16-element UE4M3 block scales computed on-GPU — no offline
  calibration, no CPU sync);
- transformer switched to bf16 (NVFP4-native, eliminates the fp16<->bf16
  cast pair);
- RoPE freqs cached as complex<float>;
- per-block LayerNorm + adaLN modulation + gate-residual fused into a
  single fp32-stat Triton kernel;
- the N-step flow-matching denoise loop replaced by a manual, graph-
  capturable pointer-based loop (`ManualRemoverPipeline`). QKV quantises
  the norm output **once** and reuses it for all three projections; the
  FFN-up GEMM fuses bias + GELU straight to FP4 output so the FFN-down
  projection skips re-quantisation. With `FLASHRT_MANUAL_GRAPH=1` the
  whole N-step x N-block loop is captured as a single CUDA Graph;
  inside the captured graph there are **zero** torch elementwise ops —
  every operation is a kernel launch.

Both paths run the VAE encode / decode unchanged from the loaded diffusers
model (one-shot per segment, outside the graph). No MiniMax-Remover source
is imported; the `pipe` is duck-typed through `.transformer` / `.vae` /
`.scheduler` / `.video_processor` and the `expand_masks` / `resize` helpers.

## Performance (RTX 5060 Ti, SM120, CUDA 13)

All numbers below are reproducible with the quickstart:

```bash
python3 examples/minimax_remover_quickstart.py \
    --model-dir ./minimax-remover \
    --frames-dir ./object_removal_data/<frames> \
    --masks-dir  ./object_removal_data/<masks> \
    --output-dir ./out                          # FP8 (default)
python3 examples/minimax_remover_quickstart.py ... --use-fp4   # NVFP4
python3 examples/minimax_remover_quickstart.py ... --no-flashrt  # fp16 reference
```

Wall time is a single end-to-end segment (load -> encode -> denoise loop ->
decode -> save). FP8 numbers include the one-time calibration pass on the
first call. Correctness (PSNR / cosine) is the FP8/NVFP4 output compared
against the `--no-flashrt` fp16 reference output on the same input.

### End-to-end, full-frame (single segment, all frames at once)

All rows compare against the non-FlashRT `--no-flashrt` fp16 reference on the
**same** input clip (same seed, same frames, same masks).

| Clip (frames, resolution) | Stack | Wall time | Speedup vs fp16 ref | PSNR mean / worst vs fp16 ref | cosine mean |
|--------------------------|-------|-----------|---------------------|-------------------------------|-------------|
| tennis (70 frames, 432x240) | fp16 reference (`--no-flashrt`) | 17.31 s | 1.0x | — | — |
| tennis (70 frames, 432x240) | FlashRT FP8 (default) | 11.76 s | **1.47x** | 40.8 / 37.4 dB | 0.99981 |
| tennis (70 frames, 432x240) | FlashRT NVFP4 (`--use-fp4`) | 9.52 s | 1.82x | 7.0 / 6.2 dB | 0.00000 (broken) |
| bmx-trees (80 frames, 432x240) | fp16 reference (`--no-flashrt`) | 19.76 s | 1.0x | — | — |
| bmx-trees (80 frames, 432x240) | FlashRT FP8 (default) | 13.24 s | **1.49x** | 35.1 / 32.0 dB | 0.99912 |
| bmx-trees (80 frames, 432x240) | FlashRT NVFP4 (`--use-fp4`) | 10.72 s | 1.84x | 7.3 / 7.0 dB | 0.00000 (broken) |

Takeaways:

- **FP8 is the correct default for full-frame inpainting**: ~1.5x faster than
  the fp16 reference with cosine >= 0.999 and PSNR 35-41 dB.
- **NVFP4 is faster but unusable on full-frame latents**: cosine collapses to
  ~0.0 and PSNR to ~7 dB (median per-pixel deviation ~85/255). The FP4
  quantisation error accumulates over the large full-frame activations and the
  output drifts to black — exactly why FP8 is the default. NVFP4 is only
  appropriate for small cropped regions, where its per-block error stays
  bounded.

### Transformer GEMM (NVFP4 vs fp16 matmul, single layer)

| Linear | fp16 matmul | NVFP4 W4A4 | per-layer speedup |
|--------|-------------|------------|-------------------|
| FFN up [5120 -> 13824] | 1.095 ms | 0.840 ms | 1.30× |
| FFN down [13824 -> 5120] | 1.020 ms | 0.864 ms | 1.18× |
| QKV / out [5120 -> 5120] | 0.409 ms | 0.359 ms | 1.14× |

Including cast + quantise overhead, the isolated FP4 GEMM is 4–9× faster
than the fp16 matmul on the large FFN projections (e.g. ffn_up
3.95 ms -> 0.47 ms). NVFP4 is also 1.14–1.30× faster per layer than the
static-quant FP8 GEMM.

### Precision specification

The pipelines keep the math reference-equivalent on the precision-critical
path (fp32-stat LayerNorm / RMSNorm, interleaved RoPE) and confine the loss
to the quantised GEMMs and the attention backend.

| Component | Metric | Value |
|-----------|--------|-------|
| Attention — SageAttention QK-int8 PV-fp8 (`sage_fp8`, default) | cosine vs SDPA | 0.9993 |
| Attention — SageAttention QK-int8 PV-fp16 (`sage_fp16`) | cosine vs SDPA | 0.9999 |
| NVFP4 W4A4 GEMM | cosine vs fp16 matmul | >= 0.999 |
| FP8 W8A8 GEMM | cosine vs fp16 matmul | >= 0.999 |
| End-to-end FP8 (full-frame) | PSNR vs fp16 ref | 35-41 dB (mean) / >= 32 dB (worst frame) |
| End-to-end FP8 (full-frame) | cosine vs fp16 ref | >= 0.999 |
| End-to-end NVFP4 (full-frame) | cosine vs fp16 ref | ~0.0 — **broken**, output drifts to black (median per-pixel deviation ~85 / 255) |
| End-to-end NVFP4 (small cropped region only) | PSNR vs fp16 ref | ~52 dB (mean) / ~45 dB (worst frame); per-block FP4 error stays bounded only when activations are small |

The default `sage_fp8` attention gives the best latency at cosine 0.9993;
switch to `FLASHRT_ATTN_MODE=sage_fp16` for cosine 0.9999 at a small
latency cost. NVFP4 needs no calibration, so the first call is already in
the steady state; the FP8 path calibrates on the first call then freezes.

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `FLASHRT_ATTN_MODE` | `sage_fp8` | attention backend (`sage_fp8`/`sage_fp16`/`sage`/`sage_triton`/`triton_fp8`/`triton_fp16`/`fa2`) |
| `FLASHRT_FP4_GEMM` | `pingpong` | GEMM kernel variant (`pingpong`/`plain`/`widen`) (NVFP4 path) |
| `FLASHRT_FUSED_BLOCK` | `1` | fused QKV-quant-once + fused FFN-up GEMM+bias+gelu block (`0` = per-projection re-quant debug path) (NVFP4 path) |
| `FLASHRT_MANUAL_GRAPH` | `0` | capture the whole denoise loop as one CUDA Graph (NVFP4 path) |
| `FLASHRT_NUM_STEPS` | unset | override the denoise step count (default 12) |
| `FLASHRT_FP8_TARGET` | `all` | FP8 Linear scope (`all` / `ffn_only`) (FP8 path) |
| `FLASHRT_NORM_MODE` | `triton` | per-block LayerNorm kernel: `triton` (fp32-stat Triton, bit-exact) / `fp16` (FlashRT `ada_layer_norm_fp16`, lower precision, debug only) |
| `FLASHRT_GELU_MODE` | `inplace` | FFN GELU kernel: `inplace` (FlashRT fused `gelu_inplace*`) / `torch` (original `F.gelu`, debug only) |

## Usage

### FP8 (recommended default — full-frame)

```python
from flash_rt.models.minimax_remover import MiniMaxRemoverPipelineFP8

# `pipe` is a loaded diffusers Minimax_Remover_Pipeline (transformer + vae +
# scheduler). The FlashRT pipeline consumes it in place. The first call
# calibrates the FP8 act_scale; subsequent calls reuse the frozen scale.
pipeline = MiniMaxRemoverPipelineFP8(pipe)
output = pipeline(
    images=frames,        # [F, H, W, 3] uint8/np, 0..255
    masks=masks,          # [F, H, W, 1] np, 0/1
    num_frames=len(frames),
    height=720, width=1280,
    num_inference_steps=12,
)
video = output.frames
```

### NVFP4 (small cropped regions only)

```python
from flash_rt.models.minimax_remover import MiniMaxRemoverPipeline

pipeline = MiniMaxRemoverPipeline(pipe)
output = pipeline(
    images=frames, masks=masks, num_frames=len(frames),
    height=720, width=1280, num_inference_steps=12,
)
video = output.frames
```

## Model weights

MiniMax-Remover checkpoint + the `Transformer3DModel` / `AutoencoderKLWan`
definitions are loaded by the reference project (unmodified, via the loaded
diffusers `pipe`). This FlashRT pipeline module imports no MiniMax-Remover
source.

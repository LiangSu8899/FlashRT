# LTX-2.5 (22B distilled, audio+video) on FlashRT — RTX SM120

FlashRT integration of the official [LTX-2](https://github.com/Lightricks/LTX-2)
`ltx-pipelines` two-stage distilled pipeline, with FlashRT compute swaps.

Status: attention backend swap shipped; fused NVFP4 epilogues and CUDA graph
capture over the transformer loop are in progress on this branch.

## Requirements

- RTX 5090 (SM120), 32GB. Peak allocated ≈ 23.2GB at 1536×1024×121f.
- An environment with the official LTX-2 packages (`ltx-core`,
  `ltx-pipelines`, and `ltx-kernels` for the NVFP4 path), either installed or
  reachable through `FLASH_RT_LTX2_ROOT` (path to an LTX-2 monorepo checkout).
- The LTX-2.5 split checkpoint pack (one safetensors per component):

```
<pack>/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors
<pack>/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
<pack>/vae/ltx-2.5-video-vae-bf16.safetensors
<pack>/vae/ltx-2.5-audio-vae-bf16.safetensors
<pack>/model_patches/ltx-2.5-duration-head-bf16.safetensors        (optional)
<pack>/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
```

The NVFP4 transformer ships prequantized block-16 weights with calibrated
static activation scales, so there is no calibration pass.

## Quickstart

```python
import flash_rt

pipe = flash_rt.load_model(
    checkpoint="/path/to/LTX-2.5",   # the split pack root
    config="ltx25",
)
pipe.set_prompt("A golden retriever running through a sunny meadow")
stats = pipe.infer(seed=42, output_path="out.mp4")
print(stats)
```

`infer` accepts `height`, `width` (multiples of 32), `num_frames`
(`k*8+1`), `frame_rate`, `seed`, and `output_path` (omit to skip mp4 encode).

## Attention backends

Selected by the `attention=` constructor argument or `FLASH_RT_LTX25_ATTN`:

| value | path | notes |
|---|---|---|
| `auto` (default) | `sage2-fvk` when available | |
| `sage2-fvk` | FlashRT raw-pointer sage2 qk-int8/pv-fp8 kernels | graph-capture safe; video d128 sites |
| `sage2` | upstream `sageattention` package | needs the package built for SM120 |
| `sage3` | `sageattn3` FP4 Blackwell package | fastest, lower per-call accuracy; opt-in |
| `sdpa` | torch SDPA | baseline |

Audio-branch attention (head_dim 64, ~1k tokens) always runs SDPA: measured on
5090, the quantized paths lose to SDPA at that shape.

Measured on 5090 at 1536×1024×121f (median, video self-attention site,
S=24576): SDPA-cudnn 42.4ms, sage2 17.4ms, sage3 13.0ms. End-to-end stage-2
denoise per step: 5.11s (SDPA) → 3.84s (sage2), with output quality equivalent
under matched-input single-forward cosine and frame inspection.

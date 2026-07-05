"""FlashRT -- MiniMax-Remover video inpainting pipeline.

MiniMax-Remover is a flow-matching video inpainting Transformer used for
subtitle / object removal. This package ships two kernelized inference
pipelines:

* ``MiniMaxRemoverPipeline`` (NVFP4 W4A4): For cropped small regions only.
  Full-frame large latents produce black/drift outputs due to FP4 quantization
  error accumulation.

* ``MiniMaxRemoverPipelineFP8`` (FP8 W8A8): Recommended default for full-frame
  inpainting. End-to-end cosine >= 0.999 and PSNR ~35-41 dB vs the fp16
  reference (measured on full-frame clips), at ~1.5x wall-clock speedup.

Both pipelines rewrite transformer Linears as quantized GEMMs over FlashRT
SM120 kernels, fuse norm/gate/residual/gelu ops, and use kernel attention
(FA2 / SageAttention). The NVFP4 path captures the N-step flow-matching loop
as a single CUDA Graph; the FP8 path is stream-safe and graph-compatible
(stable static scales, no host sync in steady state) but does not itself
capture a graph.
"""

from flash_rt.models.minimax_remover.pipeline import MiniMaxRemoverPipeline
from flash_rt.models.minimax_remover._fp8_pipeline import MiniMaxRemoverPipelineFP8
from flash_rt.models.minimax_remover._utils import load_nvfp4_kernels, load_fp8_kernels

__all__ = ["MiniMaxRemoverPipeline", "MiniMaxRemoverPipelineFP8"]

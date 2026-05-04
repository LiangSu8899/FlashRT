"""FlashRT -- RTX SM120 Qwen3.6-27B inference pipeline.

Phase 1 implementation: PyTorch-eager wrapper around HF
AutoModelForCausalLM. The class shape, file layout, and frontend
contract are real; the compute path is a thin shim over HF until
Phase 2 starts replacing it kernel-by-kernel with fvk calls.

Why ship a Phase-1 shim that doesn't use fvk yet:
  * Locks the file layout and class names expected by
    ``flash_rt.hardware._PIPELINE_MAP`` so the (config, framework, arch)
    triple resolves.
  * Lets us write the cosine regression test (vs the Phase-0 fixture)
    against the same Pipeline + Frontend objects we'll keep through
    Phase 2/3/4 -- only the internals get swapped, never the seams.
  * Establishes the PyTorch eager reference path inside the FlashRT
    tree (separate from HF transformers' integrations.finegrained_fp8
    monkey-patch — see FLASHRT_QWEN36_HF_PATCH env var) so later phases
    can bisect a regression to "ours" vs "HF's" without leaving the repo.

What this file does NOT do yet (intentionally, by Phase plan):
  * No fvk kernel calls -- those land in Phase 2.
  * No CUDA Graph capture, no FP8 calibration -- Phase 4.
  * No KV cache management beyond what HF .generate() provides
    internally -- Phase 4.
  * No pointer-interface contract on the forward path -- the
    contract is enforced once we go fvk in Phase 2 (see
    docs/adding_new_model.md).

Architecture summary (Qwen3.6-27B = registered as model_type qwen3_5)::

    [input_ids]
        |
        v  embed_tokens (BF16, vocab=248320, hidden=5120)
        v
    64 decoder layers, alternating linear-attn (3) + full-attn (1):
        layer 0,1,2:   linear_attention   (Gated DeltaNet, conv1d k=4,
                                            16 K-heads / 48 V-heads)
        layer 3:       full_attention     (GQA 24Q/4KV, head_dim=256,
                                            output_gate=True)
        layer 4..63:   same pattern repeats (linear x3, full x1) ...
        |
        v  per layer:  RMSNorm -> attn (linear or full)
        v              + residual -> RMSNorm -> SwiGLU MLP -> residual
        v
        v  final RMSNorm -> lm_head (BF16, tied or untied)
        v
    [logits: (B, S, 248320)]

    Plus an MTP (multi-token-prediction) head with 1 full-attn layer,
    used for speculative decoding in Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Qwen36Dims:
    """Static dimension constants for Qwen3.6-27B.

    Source: config.json:text_config (model_type=qwen3_5_text). These are
    fixed for the 27B variant; if a different size is added later, this
    becomes a per-checkpoint loader instead of a class-level constant.
    """
    hidden: int = 5120
    num_layers: int = 64
    full_attn_period: int = 4          # full at indices 3, 7, ..., 63
    vocab_size: int = 248320

    # full-attention sites (16 layers)
    full_q_heads: int = 24
    full_kv_heads: int = 4
    full_head_dim: int = 256

    # linear-attention sites (48 layers, Gated DeltaNet)
    lin_k_heads: int = 16
    lin_v_heads: int = 48
    lin_head_dim: int = 128
    lin_conv_kernel: int = 4

    # MLP
    intermediate: int = 17408

    # MTP head
    mtp_layers: int = 1


class Qwen36Pipeline:
    """Framework-agnostic Qwen3.6 inference pipeline (RTX SM120).

    Phase-1 implementation: hosts an HF AutoModelForCausalLM and
    delegates ``forward(input_ids)``. The class signature is the
    Phase-2+ target -- only the internals will change.

    Future shape (Phase 2+):
        gemm:  fvk.GemmRunner
        fvk:   flash_rt_kernels module
        attn:  AttentionBackend (RtxFlashAttnBackendQwen36)
        bufs:  pre-allocated CudaBuffer dict
        weights: fp8-quantized + bf16 device pointers
    """

    DIMS = Qwen36Dims()

    def __init__(self, hf_model: Any) -> None:
        """Wrap an HF model object.

        Args:
            hf_model: Output of ``AutoModelForCausalLM.from_pretrained``.
                In Phase 1 we own the reference; in Phase 2+ we'll
                ingest only the safetensors path and own weight loading
                ourselves.
        """
        self.hf = hf_model
        self.config = hf_model.config
        # Sanity-check the dim assumptions.
        assert self.config.hidden_size == self.DIMS.hidden, (
            f'expected hidden={self.DIMS.hidden}, '
            f'got {self.config.hidden_size}'
        )
        assert self.config.num_hidden_layers == self.DIMS.num_layers
        assert self.config.head_dim == self.DIMS.full_head_dim
        assert (
            self.config.layer_types.count('full_attention')
            == self.DIMS.num_layers // self.DIMS.full_attn_period
        )

    def forward(self, input_ids):
        """Single forward pass: token IDs -> logits. Phase-1 thin shim.

        Args:
            input_ids: (B, S) torch.long on cuda.

        Returns:
            logits: (B, S, vocab_size) bf16 on cuda.
        """
        import torch  # local import; pipeline_rtx is import-time-light.
        with torch.no_grad():
            out = self.hf(
                input_ids=input_ids, use_cache=False, return_dict=True,
            )
        return out.logits

    def generate(self, input_ids, *, max_new_tokens: int, do_sample: bool = False):
        """Greedy/sampled autoregressive generate. Phase-1 delegates to HF.

        Phase-4 will replace this with a C++-driven decode loop that
        captures CUDA Graphs and bypasses HF's generate() entirely.
        """
        import torch
        with torch.no_grad():
            return self.hf.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True,
            )

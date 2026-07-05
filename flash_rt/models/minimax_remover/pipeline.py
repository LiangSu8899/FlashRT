"""FlashRT -- MiniMax-Remover NVFP4 kernelized inference pipeline.

MiniMax-Remover is a flow-matching video inpainting model (a
Transformer3DModel + an AutoencoderKLWan VAE). This pipeline rewrites the
transformer denoise path onto NVFP4 (W4A4) GEMMs driven by the generic
FlashRT kernels, fused fp32-stat Triton norm/RoPE, kernel attention and
a graph-capturable manual N-step flow-matching loop.

For full-frame inpainting prefer the FP8 (W8A8) sibling
``MiniMaxRemoverPipelineFP8`` in ``_fp8_pipeline.py`` (near-fp16 precision,
recommended default); this NVFP4 path is calibrated only for small cropped
regions.

What is replaced (vs the diffusers reference ``Minimax_Remover_Pipeline``):

* every eligible transformer Linear -> NVFP4 W4A4 GEMM (weight quantised
  once at load time, activation quantised dynamically per call -- no
  offline calibration).
* per-block norm/gate/residual/gelu elementwise -> single fused kernels.
* ``torch.nn.functional.scaled_dot_product_attention`` -> FA2 / SageAttention.
* the N-step Python denoise loop -> a single captured CUDA Graph
  (``FLASHRT_MANUAL_GRAPH=1``); inside the graph there are zero torch
  elementwise ops, only kernel launches.

The loaded diffusers ``pipe`` is consumed in place: it is duck-typed
(``.transformer`` / ``.vae`` / ``.scheduler`` / ``.video_processor`` and
the ``expand_masks`` / ``resize`` helpers). No MiniMax-Remover source is
imported -- the VAE encode / decode run unchanged from the loaded model.

The pipeline requires the generic SM120 NVFP4 kernels in
``flash_rt_kernels``. They are gated by the Blackwell NVFP4 build option
(``ENABLE_CUTLASS_SM120_NVFP4_W4A16``); ``load_nvfp4_kernels`` (defined in
``_utils``, the single source of truth for the kernel surface) validates
them and raises a clear ``RuntimeError`` if they are absent, so a
non-NVFP4 build fails fast instead of crashing mid-quantisation.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

from flash_rt.models.minimax_remover._fp8_pipeline import MiniMaxRemoverPipelineFP8
from flash_rt.models.minimax_remover._utils import (
    _REQUIRED_NVFP4_SYMBOLS,
    _load_kernels,
    load_nvfp4_kernels,
)

# Re-exported here for backward compatibility (docs / tests historically import
# the kernel surface from this module). The single source of truth lives in
# ``_utils``; the two must never diverge.
__all__ = ["MiniMaxRemoverPipeline", "MiniMaxRemoverPipelineFP8",
           "_REQUIRED_NVFP4_SYMBOLS", "_load_kernels"]

_RUNTIME_DEPS = ("diffusers", "einops", "triton")


def _import_runtime():
    """Lazily import the optional runtime deps + the pipeline submodules."""
    missing = []
    for dep in _RUNTIME_DEPS:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    if missing:
        raise RuntimeError(
            "MiniMax-Remover runtime requires "
            f"{', '.join(missing)} which {'is' if len(missing) == 1 else 'are'} "
            "not installed. Install the model extra:\n"
            "    pip install -e \".[minimax-remover]\"\n"
            "('triton' normally ships with torch on CUDA; 'diffusers' and "
            "'einops' are in the extra.)")
    from ._nvfp4_linear import install_flashrt_nvfp4
    from ._attention import install_attention
    from ._manual_denoise import ManualRemoverPipeline
    return install_flashrt_nvfp4, install_attention, ManualRemoverPipeline


class _FrozenMarker:
    """NVFP4 needs no calibration (per-call dynamic quantisation).

    This marker reports ``is_frozen() == True`` so a smart ``__call__``
    routes straight to the manual graph pipeline on the first call.
    """

    def __init__(self):
        self._frozen = True

    def is_frozen(self):
        return self._frozen


class MiniMaxRemoverPipeline:
    """NVFP4 kernelized inference pipeline for MiniMax-Remover.

    Wraps a loaded diffusers MiniMax-Remover ``pipe`` and rewrites the
    transformer denoise path onto NVFP4 GEMMs, fused Triton norm/RoPE,
    kernel attention and a graph-capturable manual flow-matching loop.
    The pipe is consumed in place; transformer Linears referenced by the
    patched path are quantised to NVFP4 and the original weights deleted.

    Args:
        pipe: a loaded diffusers pipeline exposing ``.transformer``,
            ``.vae``, ``.scheduler``, ``.video_processor``,
            ``_execution_device``, ``expand_masks``, ``resize`` and the
            ``vae_scale_factor_temporal`` / ``vae_scale_factor_spatial``
            properties.
        num_inference_steps: default denoise step count (12).
        fp4_target: ``"all"`` (default, every attention/ffn/proj Linear)
            or ``"ffn_only"`` (FFN up/down only).
        use_bf16: run the transformer in bf16 (NVFP4-native, no cast).
        use_manual_pipeline: install the graph-capturable manual denoise
            loop (default ``True``).
    """

    def __init__(self, pipe, num_inference_steps=12, fp4_target="all",
                 use_bf16=True, use_manual_pipeline=True):
        self.fvk = load_nvfp4_kernels()
        # Optional runtime deps (diffusers / einops / triton) and the pipeline
        # submodules are resolved here, at construction time -- importing the
        # model package never touches them.
        install_flashrt_nvfp4, install_attention, ManualRemoverPipeline = _import_runtime()
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.num_inference_steps = num_inference_steps

        n_lin = install_flashrt_nvfp4(self.transformer, self.fvk,
                                      verbose=False, target=fp4_target)
        logger.info("MiniMax-Remover: target=%r, %d Linears -> NVFP4 W4A4 GEMM",
                    fp4_target, n_lin)

        if use_bf16:
            self.transformer.to(torch.bfloat16)
            logger.info("MiniMax-Remover: transformer -> bf16 (NVFP4-native)")

        self._optimize_rope()

        n_attn = install_attention(self.transformer)
        logger.info("MiniMax-Remover: %d attention blocks -> kernel backend", n_attn)

        pipe._flashrt_wrapper = _FrozenMarker()

        self._manual = None
        if use_manual_pipeline:
            self._manual = ManualRemoverPipeline(pipe, self.fvk)
            self._install_smart_call()
            logger.info("MiniMax-Remover: manual denoise pipeline installed "
                        "(CUDA-graph capturable, NVFP4 dynamic quant -- no calibration)")

    def _optimize_rope(self):
        """Cache the RoPE freqs as complex<float>.

        The reference caches complex128/float64 which is slow on consumer
        GPUs. Patched on the loaded rope module's class (no model import).
        """
        rope_module = getattr(self.transformer, "rope", None)
        if rope_module is None:
            return
        rope_cls = type(rope_module)
        orig = rope_cls.forward

        def rope_fwd_c64(self, hidden_states):
            out = orig(self, hidden_states)
            return out.to(torch.complex64) if out.dtype == torch.complex128 else out

        rope_cls.forward = rope_fwd_c64

    def _install_smart_call(self):
        """Route ``pipe.__call__`` to the manual pipeline (frozen, no calibration)."""
        _orig_call = self.pipe.__class__.__call__

        @torch.no_grad()
        def _smart_call(self, *args, **kwargs):
            w = getattr(self, "_flashrt_wrapper", None)
            m = getattr(self, "_manual_pipeline", None)
            ns = os.environ.get("FLASHRT_NUM_STEPS")
            if ns:
                kwargs["num_inference_steps"] = int(ns)
            if w is not None and w.is_frozen() and m is not None:
                return m(*args, **kwargs)
            return _orig_call(self, *args, **kwargs)

        self.pipe.__class__.__call__ = _smart_call
        self.pipe._manual_pipeline = self._manual

    def __call__(self, images, masks, num_frames, height, width,
                 num_inference_steps=None, generator=None, iterations=16,
                 output_type="np"):
        """Run the NVFP4 denoise loop.

        Delegates to the manual pipeline when installed (the optimised
        path), otherwise to the patched diffusers ``pipe.__call__``.
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if self._manual is not None:
            return self._manual(
                images, masks, num_frames, height, width,
                num_inference_steps=num_inference_steps, generator=generator,
                iterations=iterations, output_type=output_type)
        return self.pipe(
            images=images, masks=masks, num_frames=num_frames,
            height=height, width=width,
            num_inference_steps=num_inference_steps, generator=generator,
            iterations=iterations, output_type=output_type)

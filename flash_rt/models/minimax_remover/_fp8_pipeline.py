"""FlashRT -- MiniMax-Remover FP8 kernelized inference pipeline.

FP8 (W8A8) version for full-frame inpainting. Unlike NVFP4 (W4A4) which
produces black/drift outputs on full-frame large latents, FP8 stays close
to the fp16 reference: end-to-end cosine >= 0.999 and PSNR ~35-41 dB vs
fp16 on full-frame clips.

Uses static calibration: the first inference call runs in dynamic-FP8
calibration mode (accumulating activation amax on GPU), then freezes to a
static act_scale for all subsequent calls (zero CPU sync overhead in the
steady state).
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

from flash_rt.models.minimax_remover._utils import load_fp8_kernels
from flash_rt.models.minimax_remover._kernels import mask_mul


def _import_runtime_fp8():
    """Lazy import FP8 runtime dependencies."""
    missing = []
    for dep in ("diffusers", "einops", "triton"):
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    if missing:
        raise RuntimeError(
            f"MiniMax-Remover FP8 requires {', '.join(missing)}. "
            "Install: pip install -e '.[minimax-remover]'"
        )
    from ._fp8_linear import install_flashrt_fp8, set_calibration, freeze_calibration
    from ._kern_block import install_fused_blocks, install_fa2_attention
    return install_flashrt_fp8, set_calibration, freeze_calibration, \
           install_fused_blocks, install_fa2_attention


class MiniMaxRemoverPipelineFP8:
    """FP8 (W8A8) kernelized inference pipeline for full-frame inpainting.

    Unlike NVFP4 which is calibrated only for small cropped regions, FP8
    works on full-frame large latents: end-to-end cosine >= 0.999 and PSNR
    ~35-41 dB vs the fp16 reference on full-frame clips.

    The first ``__call__`` runs in calibration mode (dynamic FP8 + amax
    accumulation). At the end of that call the static act_scale is frozen
    and all subsequent calls use the frozen scale (zero CPU sync, suitable
    for CUDA Graph capture).

    Args:
        pipe: loaded diffusers pipeline
        num_inference_steps: denoise steps (12)
        fp8_target: "all" or "ffn_only"
        use_bf16: run transformer in bf16 (default False, keeps fp16)
        calib_margin: act_scale margin multiplier (1.1)
    """

    def __init__(self, pipe, num_inference_steps=12, fp8_target="all",
                 use_bf16=False, calib_margin=1.1):
        self.fvk = load_fp8_kernels()
        (install_flashrt_fp8, set_calibration, freeze_calibration,
         install_fused_blocks, install_fa2_attention) = _import_runtime_fp8()

        self.pipe = pipe
        self.transformer = pipe.transformer
        self.num_inference_steps = num_inference_steps
        self.calib_margin = calib_margin
        self._calibrated = False

        self._set_calibration = lambda on: set_calibration(self.transformer, on)
        self._freeze_calibration = lambda: freeze_calibration(
            self.transformer, margin=self.calib_margin)

        fp8_target_env = os.environ.get("FLASHRT_FP8_TARGET", fp8_target)
        n_lin = install_flashrt_fp8(self.transformer,
                                    verbose=True, target=fp8_target_env)
        logger.info("MiniMax-Remover FP8: target=%r, %d Linears -> FP8 W8A8 GEMM",
                    fp8_target_env, n_lin)

        if use_bf16:
            self.transformer.to(torch.bfloat16)
            logger.info("MiniMax-Remover FP8: transformer -> bf16")

        n_block = install_fused_blocks(self.transformer)
        logger.info("MiniMax-Remover FP8: %d blocks -> fused norm/gate/gelu kernels",
                    n_block)

        n_attn = install_fa2_attention(self.transformer)
        logger.info("MiniMax-Remover FP8: %d attention blocks -> kernel backend",
                    n_attn)

        self._orig_pipe_call = self.pipe.__call__
        from flash_rt.models.minimax_remover._fp8_manual_denoise import (
            FP8ManualDenoise,
        )

        # Manual graph-capturable denoise (used once calibrated + when
        # FLASHRT_FP8_GRAPH=1). Lazily captures a CUDA Graph per latent shape.
        self._graph_denoise = FP8ManualDenoise(self.pipe, self.transformer)
        # Transformer compute dtype. ``next(transformer.parameters())`` is
        # unreliable here because scale_shift_table / time_embedder are kept
        # in fp32 (via _keep_in_fp32_modules). The diffusers reference path
        # hardcodes fp16 (bf16 only when use_bf16).
        self._dtype = torch.bfloat16 if use_bf16 else torch.float16
        self._vae_dtype = next(self.pipe.vae.parameters()).dtype

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        """Run the wrapped pipe, calibrating FP8 scales on the first call.

        On the first call, a one-shot forward hook on the transformer
        freezes the FP8 act_scales immediately after the FIRST denoise
        step completes.  This lets steps 2..N (and the fused FFN epilogue
        kernel) run with static scales, so a single-call invocation
        benefits from the fused path instead of only multi-call ones.
        The cost is a single CPU sync (~1 ms) after step 1.

        When ``FLASHRT_FP8_GRAPH=1`` and scales are frozen (call 2+), the
        denoise loop runs via the manual graph-capturable path
        (``_manual_call`` -> ``FP8ManualDenoise``). The first call always
        uses the diffusers path (calibration); the graph is captured on
        the second call and replayed thereafter.
        """
        use_graph = os.environ.get("FLASHRT_FP8_GRAPH", "0") == "1"
        if not self._calibrated:
            logger.info("MiniMax-Remover FP8: calibration mode "
                        "(first call, dynamic FP8 + amax accumulation; "
                        "freezes after step 1)")
            self._set_calibration(True)
            # One-shot hook: freeze after the first transformer forward.
            fired = [False]

            def _freeze_after_step1(_module, _inp, _out):
                if fired[0]:
                    return
                fired[0] = True
                n = self._freeze_calibration()
                self._calibrated = True
                logger.info("MiniMax-Remover FP8: mid-inference freeze "
                            "after step 1 — %d act_scales frozen "
                            "(margin=%.2f); steps 2+ now use static FP8 "
                            "+ fused FFN epilogue", n, self.calib_margin)

            handle = self.transformer.register_forward_hook(
                _freeze_after_step1)
            try:
                result = self._orig_pipe_call(*args, **kwargs)
            finally:
                handle.remove()
        elif use_graph:
            # Frozen scales + graph requested: manual graph-capturable path.
            result = self._manual_call(*args, use_graph=True, **kwargs)
        elif os.environ.get("FLASHRT_FP8_EAGER_MANUAL", "1") == "1":
            # Steady-state: eager manual denoise (avoids the per-step
            # torch.cat of [latents, masked, masks] and the scheduler.step
            # CPU sync of the diffusers path). masked/masks latents are
            # constant across steps; _denoise_loop_body copies only the
            # changing latents slice into a persistent concat buffer.
            result = self._manual_call(*args, use_graph=False, **kwargs)
        else:
            result = self._orig_pipe_call(*args, **kwargs)
        return result

    @torch.no_grad()
    def _manual_call(self, images, masks, num_frames, height, width,
                     num_inference_steps=12, generator=None, iterations=16,
                     output_type="np", use_graph=False):
        """Manual encode + graph-denoise + decode (mirrors the diffusers
        ``MinimaxRemoverPipeline.__call__`` but replaces the denoise loop
        with the CUDA-graph-capturable ``FP8ManualDenoise``). Requires
        frozen FP8 scales (caller guarantees calibration is done).
        """
        pipe = self.pipe
        device = self.transformer.device

        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        num_channels_latents = 16
        vsft = pipe.vae_scale_factor_temporal
        vsfs = pipe.vae_scale_factor_spatial
        num_latent_frames = (num_frames - 1) // vsft + 1
        shape = (1, num_channels_latents, num_latent_frames,
                 height // vsfs, width // vsfs)
        from diffusers.utils.torch_utils import randn_tensor
        latents = randn_tensor(shape, generator=generator, device=device,
                               dtype=self._dtype)

        masks_t = pipe.expand_masks(masks, iterations)
        masks_t = pipe.resize(masks_t, height, width).to(device).to(self._vae_dtype)
        masks_t[masks_t > 0] = 1
        from einops import rearrange
        images_t = rearrange(images, "f h w c -> c f h w")
        images_t = pipe.resize(images_t[None, ...], height, width).to(device).to(self._vae_dtype)
        masked_images = mask_mul(images_t, masks_t)

        latents_mean = (torch.tensor(pipe.vae.config.latents_mean)
                        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                        .to(device, self._vae_dtype))
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1).to(device, self._vae_dtype)

        masked_latents = pipe.vae.encode(masked_images.to(self._vae_dtype)).latent_dist.mode()
        masks_latents = pipe.vae.encode((2 * masks_t - 1.0).to(self._vae_dtype)).latent_dist.mode()
        # Per-channel normalize (matches diffusers exactly). Done outside the
        # graph; the latent_normalize() Triton helper collapses latents_std to
        # a scalar via .max() which is wrong for per-channel stats.
        masked_latents = ((masked_latents - latents_mean) * latents_std).to(self._dtype)
        masks_latents = ((masks_latents - latents_mean) * latents_std).to(self._dtype)

        result_latents = self._graph_denoise.denoise(
            latents, masked_latents, masks_latents, num_inference_steps,
            use_graph=use_graph)

        result_latents = (result_latents.to(self._vae_dtype) / latents_std
                          + latents_mean)
        video = pipe.vae.decode(result_latents, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type=output_type)

        from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
        return WanPipelineOutput(frames=video)

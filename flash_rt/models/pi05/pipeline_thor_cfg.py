"""FlashRT — Pi0.5 Thor SM110 classifier-free-guidance inference pipeline.

Backend-agnostic per-chunk CFG class for the Pi0.5 Thor frontend
(torch + JAX), mirroring the RTX file structure
(``flash_rt/models/pi05/pipeline_rtx_cfg.py``).

The B=2 fused-CFG companion lives in
:mod:`flash_rt.models.pi05.pipeline_thor_cfg_batched`
(``Pi05ThorCFGBatchedPipeline``) and mirrors RTX's
``pipeline_rtx_cfg_batched.py`` split.

Architectural notes
-------------------

  * **Backend-agnostic.** Thor has two frontends (torch and JAX) with
    incompatible buffer types (``torch.Tensor`` vs ``CudaBuffer``).
    Rather than build twin classes, we accept a small set of
    primitives as constructor callables — each frontend wires them to
    its own buffer layer. The pipeline class then orchestrates the
    dual-replay + combine without knowing whether the underlying
    storage is torch or CudaBuffer.

  * **No new kernels.** The combine uses the already-shipped
    ``fvk.cfg_combine_into_residual_fp16`` (Thor SM110's
    ``flash_rt_kernels.so``). All elementwise / GEMM primitives Thor
    already exposes are reused.
"""
from __future__ import annotations

import logging
from typing import Callable

from .pipeline_thor import Pi05ThorPipeline

logger = logging.getLogger(__name__)


class Pi05ThorCFGPipeline(Pi05ThorPipeline):
    """Per-chunk classifier-free guidance pipeline for Pi0.5 Thor SM110.

    Args:
        fvk: ``flash_rt.flash_rt_kernels`` module — the Thor kernel
            library. Must export ``cfg_combine_into_residual_fp16``.
        cfg_beta: Guidance strength. Must be ``>= 1.0``. ``1.0`` collapses
            to the cond-only output (after the zero-residual prep below).
            Common deployment range ``[1.5, 2.5]`` per the π0.6 RECAP
            paper.
        Sa: Action chunk length (number of decoder rows; Pi0.5 = 10).
        replay_siglip: Callable ``() -> None`` that replays the captured
            SigLIP graph. The frontend must have written the language
            slot of ``enc_x`` before calling this — typically by uploading
            ``lang_emb_cond`` or ``lang_emb_uncond`` into the shared
            ``lang_emb`` buffer immediately before each call.
        replay_enc_ae: Callable ``() -> None`` that replays the captured
            encoder + decoder graph. Reads from ``enc_x`` and the noise
            buffer; writes the action chunk back into the noise buffer.
        upload_cond_lang_emb / upload_uncond_lang_emb: Callables
            ``() -> None`` that write the conditioned / unconditioned
            language embeddings into the captured graph's ``lang_emb``
            slot. The pipeline holds the cond/uncond embeds as opaque
            objects and lets the frontend choose how to upload them.
        snapshot_noise: Callable ``() -> None`` that copies the current
            ``g_noise`` contents into a private "noise R" snapshot. The
            pipeline calls this once after the frontend seeds noise but
            before the first encoder+decoder replay.
        restore_noise: Callable ``() -> None`` that copies the snapshot
            back into ``g_noise``. Called between cond and uncond
            replays so both branches start from identical R.
        snapshot_g_noise_to_v_cond / snapshot_g_noise_to_v_uncond:
            Callables ``() -> None`` that copy the current ``g_noise``
            (= the action chunk produced by the just-completed replay)
            into the v_cond / v_uncond holding buffers. The pipeline
            then issues the combine kernel against these two pointers
            plus the zeroed ``g_noise``.
        zero_g_noise: Callable ``() -> None`` that zeroes ``g_noise``.
            Called immediately before the combine kernel so the
            kernel's accumulator semantics
            (``residual += v_uncond + β·(v_cond - v_uncond)``) yield
            the per-chunk CFG assignment we want.
        g_noise_ptr / v_cond_ptr / v_uncond_ptr: Callables
            ``() -> int`` returning the device pointer (as Python int)
            for the respective buffer. Lazy getters so the frontend
            can resolve ``self._g_noise.data_ptr()`` /
            ``self.g_noise.ptr.value`` at call time.
        sync: Callable ``() -> None`` that synchronizes the inference
            stream. Called once after the combine kernel so the
            frontend can read the result host-side.
        stream_int: CUDA stream as a Python int. Defaults to 0
            (default stream).
    """

    def __init__(
        self,
        fvk,
        *,
        cfg_beta: float = 1.5,
        Sa: int,
        replay_siglip: Callable[[], None],
        replay_enc_ae: Callable[[], None],
        upload_cond_lang_emb: Callable[[], None],
        upload_uncond_lang_emb: Callable[[], None],
        snapshot_noise: Callable[[], None],
        restore_noise: Callable[[], None],
        snapshot_g_noise_to_v_cond: Callable[[], None],
        snapshot_g_noise_to_v_uncond: Callable[[], None],
        zero_g_noise: Callable[[], None],
        g_noise_ptr: Callable[[], int],
        v_cond_ptr: Callable[[], int],
        v_uncond_ptr: Callable[[], int],
        sync: Callable[[], None],
        stream_int: int = 0,
    ):
        if cfg_beta < 1.0:
            raise ValueError(
                f"cfg_beta must be >= 1.0 (1.0 disables CFG); got {cfg_beta}")
        # Per-chunk CFG runs the B=1 graphs twice; report B=1 to the
        # parent so callers querying ``batch_size`` see the actual
        # graph contract. The Stage 3 batched-CFG subclass advertises
        # B=2 instead.
        super().__init__(batch_size=1)
        self.fvk = fvk
        self.cfg_beta = float(cfg_beta)
        self.Sa = int(Sa)

        self._replay_siglip = replay_siglip
        self._replay_enc_ae = replay_enc_ae
        self._upload_cond_lang = upload_cond_lang_emb
        self._upload_uncond_lang = upload_uncond_lang_emb
        self._snapshot_noise = snapshot_noise
        self._restore_noise = restore_noise
        self._snap_to_v_cond = snapshot_g_noise_to_v_cond
        self._snap_to_v_uncond = snapshot_g_noise_to_v_uncond
        self._zero_g_noise = zero_g_noise
        self._g_noise_ptr = g_noise_ptr
        self._v_cond_ptr = v_cond_ptr
        self._v_uncond_ptr = v_uncond_ptr
        self._sync = sync
        self.stream_int = int(stream_int)

        # Set by ``set_language_embeds_pair`` (frontend builds them).
        self._lang_emb_cond = None
        self._lang_emb_uncond = None

        # Calibration mode flag — when set, ``run_pipeline`` runs the
        # cond branch only (single forward) so FP8 scale collection sees
        # the same activation magnitudes the production cond pass does.
        # The frontend's lazy-recalibrate step must consult this and
        # skip the second branch when it is true. Mirrors RTX
        # ``Pi05CFGPipeline._calibration_mode`` (pipeline_rtx_cfg.py:89).
        self._calibration_mode = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_language_embeds_pair(self, cond_embeds, uncond_embeds) -> None:
        """Stash the cond / uncond embeddings.

        The pipeline does not interpret these — it just hands them back
        to the frontend's ``upload_*_lang_emb`` callbacks. Whatever
        type the frontend chose for the embeddings (a torch.Tensor
        on the torch side, a numpy array on the JAX side) is fine.

        Mirrors :meth:`flash_rt.models.pi05.pipeline_rtx_cfg.Pi05CFGPipeline
        .set_language_embeds_pair` (pipeline_rtx_cfg.py:159).
        """
        self._lang_emb_cond = cond_embeds
        self._lang_emb_uncond = uncond_embeds

    def run_pipeline(self) -> None:
        """Run the per-chunk CFG forward.

        Sequence (per Section 1.3 of the Stage 0 design notes):

          1. Cond branch
             a. ``upload_cond_lang_emb`` — write cond into shared lang slot
             b. ``replay_siglip``       — vision + cond_lang → enc_x
             c. ``snapshot_noise``      — R = current g_noise
             d. ``replay_enc_ae``       — full encoder + 10-step decoder
             e. ``snapshot_to_v_cond``  — v_cond = g_noise (= A_cond)

          2. (Calibration mode short-circuit: if ``self._calibration_mode``
             is True, return now. The cond pass is the calibration sample.)

          3. Uncond branch
             a. ``upload_uncond_lang_emb``
             b. ``replay_siglip``     — vision + uncond_lang → enc_x
             c. ``restore_noise``     — g_noise = R (so uncond starts from
                                        the same R as cond)
             d. ``replay_enc_ae``
             e. ``snapshot_to_v_uncond``

          4. CFG combine
             ``g_noise.zero_()``
             ``fvk.cfg_combine_into_residual_fp16(g_noise, v_cond,
                 v_uncond, β, n)`` — kernel does
                 ``residual += v_uncond + β·(v_cond - v_uncond)``,
                 so with the zero-pre-write residual ends up as
                 ``v_uncond + β·(v_cond - v_uncond)``, the per-chunk
                 CFG assignment we want.

          5. ``sync`` — block until combine is done so the frontend can
             read the result host-side.
        """
        # ── Cond branch ──
        self._upload_cond_lang()
        self._replay_siglip()
        self._snapshot_noise()
        self._replay_enc_ae()
        self._snap_to_v_cond()

        if self._calibration_mode:
            # Calibration uses the cond branch only — skip the uncond
            # forward (no FP8 magnitude information would change) and
            # leave g_noise holding A_cond so the frontend can read it
            # if needed.
            self._sync()
            return

        # ── Uncond branch ──
        self._upload_uncond_lang()
        self._replay_siglip()
        self._restore_noise()
        self._replay_enc_ae()
        self._snap_to_v_uncond()

        # ── CFG combine ──
        # The fvk kernel is an accumulator
        # (``residual += v_uncond + β·(v_cond - v_uncond)``); we zero
        # ``g_noise`` first so the assignment we want falls out.
        self._zero_g_noise()
        n = self.Sa * 32
        self.fvk.cfg_combine_into_residual_fp16(
            self._g_noise_ptr(),
            self._v_cond_ptr(),
            self._v_uncond_ptr(),
            self.cfg_beta, n, self.stream_int)
        self._sync()

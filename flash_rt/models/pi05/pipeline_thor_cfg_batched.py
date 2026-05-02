"""FlashVLA — Pi0.5 Thor SM110 fused B=2 classifier-free-guidance pipeline.

Single-inheritance subclass of
:class:`flash_vla.models.pi05.pipeline_thor_batched.Pi05ThorBatchedPipeline`
that runs the conditioned + unconditioned CFG branches as the two
slots of a single B=2 fused forward, mirroring the RTX layout
(``flash_vla/models/pi05/pipeline_rtx_cfg_batched.py``).

Slot 0 carries the conditioned prompt; slot 1 carries the
unconditioned prompt. Both contexts ride the same B=2 ``enc_ae_b2``
graph replay; the **per-step** CFG combine + noise mirror happen
inside that captured graph (see
:func:`flash_vla.models.pi05.pipeline_thor_batched.decoder_forward_b2` with
``cfg_beta`` set), matching RTX
:meth:`Pi05CFGBatchedPipeline.transformer_decoder_batched` and
arXiv:2511.14759 Appendix E:

    for each denoise step:
        v_b2 ← action_head(x_b2)        (per-slot velocity, B=2)
        noise[cond] += v_uncond + beta * (v_cond - v_uncond)
        noise[uncond] ← noise[cond]      (mirror — both slots see
                                          the same x_t at step k+1)

This replaces the previous per-chunk CFG combine (one combine on the
final action chunks), which was an approximation that diverged from
paper-correct CFG at beta > 1 because the cond and uncond
trajectories integrated independently through the 10-step decoder.
"""
from __future__ import annotations

import logging
from typing import Callable

from .pipeline_thor_batched import Pi05ThorBatchedPipeline

logger = logging.getLogger(__name__)


class Pi05ThorCFGBatchedPipeline(Pi05ThorBatchedPipeline):
    """B=2 fused classifier-free guidance pipeline for Pi0.5 Thor SM110.

    Mirrors RTX
    :class:`flash_vla.models.pi05.pipeline_rtx_cfg_batched.Pi05CFGBatchedPipeline`:
    slot 0 carries the conditioned prompt, slot 1 carries the
    unconditioned prompt, both ride a single ``enc_ae_b2`` graph
    replay, and the per-step CFG combine + noise mirror live INSIDE
    that captured graph (driven by ``cfg_beta`` plumbed through to
    :func:`flash_vla.models.pi05.pipeline_thor_batched.decoder_forward_b2`).

    The frontend's
    :meth:`_capture_enc_ae_graph_b2` recaptures the graph with
    ``cfg_beta`` set when this pipeline is constructed, so changing
    ``cfg_beta`` requires rebuilding the pipeline (same contract as
    :class:`Pi05ThorCFGPipeline` and the RTX equivalent).

    Architectural notes
        * **Calibration scales come from the B=1 calibration pass.**
          The parent's contract is "B=1 calibration transfers to
          B=N". RTX's
          :meth:`Pi05CFGBatchedPipeline.calibrate_fp8` overrides for
          a joint cond+uncond pass to clip uncond's tails — that's a
          follow-on optimization here.
        * **Slot conventions** follow RTX exactly: slot 0 = cond,
          slot 1 = uncond.
        * **The frontend owns the buffers.** This class is a thin
          orchestrator: stages cond/uncond language tokens into the
          two ``_enc_x_b2`` slots via callbacks, mirrors the noise
          into both slots so the diffusion sees identical R, and
          replays the B=2 graph (which carries the per-step CFG
          combine internally).
    """

    # Slot conventions — fixed so the pipeline, the cfg_combine kernel
    # call (inside the captured graph), and the action readback all
    # agree across torch / JAX / RTX.
    COND_SLOT = 0
    UNCOND_SLOT = 1

    def __init__(
        self,
        fvk,
        *,
        cfg_beta: float = 1.5,
        Sa: int,
        replay_siglip_for_cond: Callable[[], None],
        replay_siglip_for_uncond: Callable[[], None],
        replay_enc_ae_b2: Callable[[], None],
        seed_b2_noise_from_R: Callable[[], None],
        sync: Callable[[], None],
        stream_int: int = 0,
        outer_graph_replay: "Callable[[], None] | None" = None,
    ):
        if cfg_beta < 1.0:
            raise ValueError(
                f"cfg_beta must be >= 1.0 (1.0 disables CFG); got {cfg_beta}")
        super().__init__(batch_size=2)
        self.fvk = fvk
        self.cfg_beta = float(cfg_beta)
        self.Sa = int(Sa)

        # Stage cond into slot 0 of _enc_x_b2 by running the B=1 SigLIP
        # graph after writing cond_lang into _lang_emb, then copying
        # _enc_x[:Se] → _enc_x_b2[0:Se]. ``replay_siglip_for_cond``
        # encapsulates this whole sequence; uncond is symmetric.
        self._replay_siglip_cond = replay_siglip_for_cond
        self._replay_siglip_uncond = replay_siglip_for_uncond
        # Single B=2 graph replay — runs encoder + decoder for both
        # slots in one pass AND drives the per-step CFG combine + noise
        # mirror via ``decoder_forward_b2(cfg_beta=...)`` which the
        # frontend bakes into the graph at capture time.
        self._replay_enc_ae_b2 = replay_enc_ae_b2
        # Seeds ``_g_noise_b2[0:Sa]`` and ``_g_noise_b2[Sa:2*Sa]`` with
        # the SAME draw so cond and uncond start from identical R.
        self._seed_b2_noise_from_R = seed_b2_noise_from_R
        self._sync = sync
        self.stream_int = int(stream_int)
        # When the frontend has captured the entire fused-CFG pipeline
        # into a single outer CUDA Graph, this callable triggers that
        # one replay (matching RTX
        # :meth:`Pi05CFGBatchedPipeline.forward` /
        # ``self._graph.replay(...)``). When None, ``forward()`` falls
        # back to eager callback orchestration via ``run_pipeline``.
        self._outer_graph_replay = outer_graph_replay

        # Calibration mode flag — when True, ``run_pipeline`` runs cond
        # only (single B=1 forward via the serial graph) so the FP8
        # scale collection sees standard cond activation magnitudes.
        # Stage 4+ may swap this for a joint B=2 cond+uncond pass to
        # mirror RTX's calibrate_fp8 override.
        self._calibration_mode = False

    def forward(self) -> None:
        """One fused replay of the captured outer CFG graph.

        When the frontend has captured the entire fused-CFG pipeline
        (lang swap + SigLIP×2 + enc_ae_b2 with per-step CFG) into a
        single outer CUDA Graph, ``forward()`` is one ``graph.replay``
        + final sync — matching RTX
        :meth:`Pi05CFGBatchedPipeline.forward` at
        ``pipeline_rtx_cfg_batched.py:419``.

        The frontend stages observation images and the noise R into
        the captured input slots BEFORE calling ``forward()``. After
        the replay, ``_g_noise_b2[0:Sa]`` holds the final guided
        action chunk.

        When no outer graph is available (older capture path or eager
        debug mode) this falls back to the eager callback
        orchestration in :meth:`run_pipeline`.
        """
        if self._outer_graph_replay is None:
            self.run_pipeline()
            return
        self._seed_b2_noise_from_R()
        self._outer_graph_replay()
        self._sync()

    def run_pipeline(self) -> None:
        """Eager callback orchestration of the B=2 fused per-step CFG forward.

        Sequence:
          1. ``replay_siglip_for_cond`` — vision + cond_lang → enc_x_b2 slot 0
          2. ``replay_siglip_for_uncond`` — vision + uncond_lang → slot 1
          3. ``seed_b2_noise_from_R`` — both slots get the SAME R
          4. ``replay_enc_ae_b2`` — single B=2 replay. The captured
             graph carries the 10-step decoder loop AND, at the end of
             every step, the cfg_combine kernel + cudaMemcpyAsync
             noise mirror (slot 0 = guided, slot 1 = mirror of slot 0).
             At graph end, ``_g_noise_b2[0:Sa]`` holds the final
             guided action chunk (slot 1 holds a copy and is unused).
          5. ``sync`` — host-side wait so the frontend can read slot 0.

        Used as the eager fallback path; the production hot path is
        :meth:`forward` once the outer graph has been captured.
        """
        self._replay_siglip_cond()
        self._replay_siglip_uncond()
        self._seed_b2_noise_from_R()
        self._replay_enc_ae_b2()
        self._sync()

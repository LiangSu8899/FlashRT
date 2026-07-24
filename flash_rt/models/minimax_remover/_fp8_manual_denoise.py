"""FlashRT -- MiniMax-Remover FP8 manual graph-capturable denoise pipeline.

Replaces the diffusers ``MinimaxRemoverPipeline.__call__`` denoise loop
with a pointer-based implementation that mirrors ``_manual_denoise.py``
(NVFP4) but calls the **installed FP8 block forwards** from
``_kern_block.install_fused_blocks``.

Why a manual loop is needed for CUDA Graph capture:
  * ``condition_embedder`` calls ``Timesteps`` which uses ``torch.arange``
    (a CPU op) -- breaks capture.
  * ``FlowMatchEulerDiscreteScheduler.step`` mutates ``self.step_index``
    (a Python int) and does CPU indexing into ``sigmas`` -- breaks capture.

This module:
  1. Pre-computes ALL per-step time embeddings + norm_out modulation
     BEFORE capture (condition_embedder runs once per step, outside the
     graph).
  2. Pre-computes RoPE freqs (fixed for a given latent shape).
  3. Pre-computes the Euler flow-matching ``dt`` values.
  4. Runs a manual denoise loop of pre-allocated buffer copies + the
     installed FP8 transformer forward + an in-place Triton euler step --
     every operation is a kernel launch, fully CUDA-graph capturable.
  5. Captures the entire N-step loop as ONE graph and replays it on
     subsequent calls with the same latent shape.

The FP8 act_scales MUST be frozen (static calibration) before capture --
the parent ``MiniMaxRemoverPipelineFP8`` guarantees this (mid-inference
freeze after step 1 on the first call). Capture therefore happens on the
second call; the first call runs the diffusers (calibration) path.

--------------------------------------------------------------------
Performance finding (measured on RTX 5060 Ti, 70-frame tennis clip)
--------------------------------------------------------------------
Graph capture is **technically feasible and bit-correct** (with a
graph-safe attention backend, the captured graph reproduces the eager
result exactly). However it is **not a net win** for this workload:

  * The denoise loop is GPU-bound -- the kernels are large and the GPU
    stays saturated, so Python/launch overhead is negligible
    (graph replay saves only ~20 ms vs eager, both using triton_fp8).
  * The fast default attention backend (``sage_fp8`` = SageAttention
    QK-int8/PV-fp8) is **not** CUDA-graph safe -- it produces garbage
    inside a captured graph (likely a CPU-syncing absmax reduction).
    Switching to a graph-safe backend (``triton_fp8``/``triton_fp16``)
    costs ~1.1 s, far more than the graph saves.

Result: graph replay (triton_fp8) = 7.87 s vs the default sage_fp8
eager path = 6.76 s (2nd call). The graph loses by ~1.1 s.

The code is retained (gated behind ``FLASHRT_FP8_GRAPH=1``, default off)
because it is correct and would become worthwhile if a fast graph-safe
attention backend is added (e.g. FlashRT's vendored ``flash_rt_fa2``,
which is pointer-based and graph-safe but not built in this tree).

No MiniMax-Remover imports: tensors + the loaded diffusers ``pipe`` only.
"""

import logging
import os

import torch

from ._kernels import ada_layernorm_fp16_io

logger = logging.getLogger(__name__)


def transformer_forward_fp8(transformer, hidden_states, tproj_step,
                            mod_out_step, rotary_emb, eps):
    """FP8 transformer forward with pre-computed time projection + norm_out mod.

    Calls the installed FP8 block forwards (``block(hs, tproj, rope)``).
    Avoids ``condition_embedder`` (torch.arange) and uses ``ada_layernorm``
    for norm_out, so the whole forward is CUDA-graph capturable.

    Args:
        transformer: Transformer3DModel with FP8-patched blocks + attention.
        hidden_states: [B, 3*C, T, H, W] concat latent (fp16).
        tproj_step: [1, 6, D] pre-computed time projection (unflattened).
        mod_out_step: [2, D] fp32 (shift, scale) for norm_out.
        rotary_emb: pre-computed RoPE freqs.
        eps: layer-norm epsilon.
    Returns:
        [B, C_out, T, H, W] noise prediction (fp16).
    """
    B, C_in, T, H, W = hidden_states.shape
    p_t, p_h, p_w = transformer.config.patch_size
    post_t, post_h, post_w = T // p_t, H // p_h, W // p_w

    hs = transformer.patch_embedding(hidden_states)
    hs = hs.flatten(2).transpose(1, 2)

    for block in transformer.blocks:
        hs = block(hs, tproj_step, rotary_emb)

    S, D = hs.shape[1], hs.shape[2]
    hs_2d = hs.contiguous().view(S, D)
    # norm_out: original is (FP32LayerNorm(hs) * (1+scale) + shift).type_as;
    # ada_layernorm_fp16_io is the fp32-stat reference-equivalent single kernel
    # (same kernel used by every block's norm1/norm2). mod_out_step[0]=shift,
    # mod_out_step[1]=scale.
    hs_2d = ada_layernorm_fp16_io(hs_2d, mod_out_step[1], mod_out_step[0], eps)
    hs = transformer.proj_out(hs_2d.view(1, S, D))

    hs = hs.reshape(B, post_t, post_h, post_w, p_t, p_h, p_w, -1)
    hs = hs.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hs.flatten(6, 7).flatten(4, 5).flatten(2, 3)


class FP8ManualDenoise:
    """Manual graph-capturable denoise for the FP8 (W8A8) path.

    Owned by ``MiniMaxRemoverPipelineFP8``. ``denoise(...)`` runs the full
    N-step loop, capturing a CUDA Graph on the first invocation for a given
    latent shape and replaying it thereafter. FP8 scales must be frozen
    before the first ``denoise`` call.
    """

    def __init__(self, pipe, transformer):
        self.pipe = pipe
        self.transformer = transformer
        self.eps = float(transformer.config.eps)
        self._dtype = next(transformer.parameters()).dtype
        self._vae_dtype = next(pipe.vae.parameters()).dtype
        # Per-shape cache: (graph, lat_buf, masked_buf, masks_buf,
        #                   tproj_all, mod_out, dt_all, rotary_emb)
        self._graphs = {}

    # ------------------------------------------------------------------ #
    # Static (per-shape, per-step) pre-computation -- runs OUTSIDE graph. #
    # ------------------------------------------------------------------ #
    def _precompute_static(self, latents, num_steps):
        """Pre-compute time embeddings, norm_out modulation, RoPE, dt.

        These are identical for every call with the same (latent shape,
        num_steps), so they are cached per shape key.
        """
        device = latents.device
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps
        sigmas = scheduler.sigmas
        dt_all = [float(sigmas[i + 1] - sigmas[i]) for i in range(num_steps)]

        tr = self.transformer
        D = tr.scale_shift_table.shape[-1]
        temb_all, tproj_all = [], []
        mod_out = torch.empty(num_steps, 2, D, dtype=torch.float32,
                              device=device)
        with torch.no_grad():
            for i in range(num_steps):
                t_step = timesteps[i:i + 1]
                temb_i, tproj_i = tr.condition_embedder(t_step)
                tproj_i = tproj_i.unflatten(1, (6, -1))
                temb_all.append(temb_i)
                tproj_all.append(tproj_i)
                temb_f = temb_i.float().unsqueeze(1)
                mod_out[i] = (tr.scale_shift_table + temb_f).squeeze(0)

        rotary_emb = tr.rope(latents)
        return tproj_all, mod_out, dt_all, rotary_emb

    # ------------------------------------------------------------------ #
    # The graph-capturable N-step loop.                                   #
    # ------------------------------------------------------------------ #
    def _denoise_loop_body(self, lat_buf, masked_buf, masks_buf, concat_buf,
                           tproj_all, mod_out, dt_all, rotary_emb, num_steps,
                           skip_steps=None):
        """Zeroth-order TeaCache: at skip steps reuse the cached noise_pred
        (the velocity from the last COMPUTED step) and only run the cheap
        euler step, mirroring the Motus/Cosmos3 TeaCache mechanism."""
        C = lat_buf.shape[1]
        tr = self.transformer
        eps = self.eps
        skip_set = set(skip_steps) if skip_steps else set()
        skip_set.discard(0)
        if num_steps > 1:
            skip_set.discard(num_steps - 1)
        cached_noise_pred = None
        for step in range(num_steps):
            if step in skip_set and cached_noise_pred is not None:
                noise_pred = cached_noise_pred
            else:
                concat_buf[:, :C].copy_(lat_buf)
                concat_buf[:, C:2 * C].copy_(masked_buf)
                concat_buf[:, 2 * C:3 * C].copy_(masks_buf)
                noise_pred = transformer_forward_fp8(
                    tr, concat_buf, tproj_all[step], mod_out[step],
                    rotary_emb, eps)
                cached_noise_pred = noise_pred
            # Euler flow-matching step: latents += dt * noise_pred.
            # PyTorch in-place add_ replaces the Triton ``euler_step_inplace``
            # kernel to avoid Triton JIT cold-start (~0.3s on the first call).
            # The fp16 accumulation difference is negligible (verified
            # opt-vs-Triton PSNR >= 46 dB over 8 denoise steps).
            lat_buf.add_(noise_pred, alpha=dt_all[step])

    def _capture_graph(self, latents, masked_latents, masks_latents,
                       tproj_all, mod_out, dt_all, rotary_emb, num_steps,
                       skip_steps=None):
        """Capture the full N-step denoise loop as one CUDA Graph.

        The TeaCache skip schedule is baked into the captured graph (the
        Python branch is resolved at capture time), matching how Motus
        bakes its skip set before graph capture."""
        device = latents.device
        C = latents.shape[1]
        B, _, T, H, W = latents.shape
        dtype = latents.dtype

        lat_buf = latents.clone()
        masked_buf = masked_latents.clone()
        masks_buf = masks_latents.clone()
        concat_buf = torch.empty(B, 3 * C, T, H, W, dtype=dtype, device=device)

        def denoise():
            self._denoise_loop_body(
                lat_buf, masked_buf, masks_buf, concat_buf,
                tproj_all, mod_out, dt_all, rotary_emb, num_steps,
                skip_steps=skip_steps)

        # Warmup on a side stream to compile all kernels / init cuBLASLt
        # workspaces before capture. The FP8 scales are already frozen, so
        # the warmup runs with the exact same kernels as capture/replay.
        s = torch.cuda.Stream()
        n_warmup = int(os.environ.get("FLASHRT_FP8_GRAPH_WARMUP", "1"))
        with torch.cuda.stream(s):
            for _ in range(n_warmup):
                denoise()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Reset the latent buffer to the caller's input before capture (the
        # warmup mutated it); the capture pass will produce the final result.
        lat_buf.copy_(latents)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            denoise()

        logger.info("MiniMax-Remover FP8: CUDA Graph captured for shape %s "
                    "(%d steps, warmup=%d, skip=%s)", tuple(latents.shape),
                    num_steps, n_warmup,
                    sorted(skip_steps) if skip_steps else None)
        return (graph, lat_buf, masked_buf, masks_buf,
                tproj_all, mod_out, dt_all, rotary_emb)

    # ------------------------------------------------------------------ #
    # Public entry: run the denoise, capture-or-replay per shape.         #
    # ------------------------------------------------------------------ #
    def denoise(self, latents, masked_latents, masks_latents, num_steps,
                use_graph=True, skip_steps=None):
        """Run the N-step denoise; capture+replay a graph when use_graph.

        ``skip_steps`` (optional list of int) enables zeroth-order
        TeaCache: the listed steps reuse the cached noise prediction
        instead of running the transformer forward. Step 0 and the last
        step are never skipped. When ``use_graph`` the skip schedule is
        baked into the captured graph (per-shape, per-schedule cache).

        Returns the final latents tensor (same shape/dtype as ``latents``).
        """
        skip_key = tuple(sorted(skip_steps)) if skip_steps else ()
        shape_key = tuple(latents.shape) + (num_steps,) + skip_key
        entry = self._graphs.get(shape_key) if use_graph else None

        if entry is None:
            tproj_all, mod_out, dt_all, rotary_emb = self._precompute_static(
                latents, num_steps)
            if use_graph:
                entry = self._capture_graph(
                    latents, masked_latents, masks_latents,
                    tproj_all, mod_out, dt_all, rotary_emb, num_steps,
                    skip_steps=skip_steps)
                self._graphs[shape_key] = entry
            else:
                # Eager manual path (no graph) -- still avoids condition_embedder
                # / scheduler CPU ops; useful for PSNR validation vs graph.
                C = latents.shape[1]
                concat_buf = torch.empty(
                    latents.shape[0], 3 * C, *latents.shape[2:],
                    dtype=latents.dtype, device=latents.device)
                lat_buf = latents.clone()
                self._denoise_loop_body(
                    lat_buf, masked_latents, masks_latents, concat_buf,
                    tproj_all, mod_out, dt_all, rotary_emb, num_steps,
                    skip_steps=skip_steps)
                return lat_buf

        (graph, lat_buf, masked_buf, masks_buf,
         tproj_all, mod_out, dt_all, rotary_emb) = entry
        # Copy this call's inputs into the captured buffers, then replay.
        lat_buf.copy_(latents)
        masked_buf.copy_(masked_latents)
        masks_buf.copy_(masks_latents)
        graph.replay()
        torch.cuda.synchronize()
        return lat_buf.clone()

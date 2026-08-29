"""FlashRT AMD — CDNA4 attention backend v2: aiter asm flash-attention.

Drop-in replacement for :class:`Cdna4AttnBackend` that dispatches the
per-site attention math to ``aiter`` (AMD's asm flash-attention library,
the vLLM-ROCm production attention path) instead of torch SDPA.

Construction signature, buffer ownership/shapes/dtypes, ``get_ptrs()``,
``run()`` dispatch and the fixed-shape surface are IDENTICAL to the sdpa
backend (this class subclasses it and overrides only the three per-site
attention calls), so the frontend can swap backends via a single
parameter without touching the pipeline.

WHY AITER
---------
rocprof on the FP8 graph shows sdpa's attn_fwd at ~44us/call x ~225
calls = ~11.1ms/inference = 32.6% of GPU time — the #1 kernel bucket.
Isolated-call survey (bench_amd_attention.py) on the exact pi05 shapes:

    vision   aiter 25.0us  vs sdpa 34.7us
    encoder  aiter 72.9us  vs sdpa 122.7us
    decoder  aiter 68.0us  vs sdpa  62.4us   (sdpa wins tiny-M)

CALL CONTRACT
-------------
``aiter.mha_fwd`` takes (B, S, H, D) tensors and supports GQA natively
(K/V may carry a single KV head against 8 Q heads — no expand, no
materialized repeat). Our owned buffers already match that layout as
zero-copy views:

    vision   Q/K/V (nv, 256, 16, 72)         — used directly (B=nv)
    encoder  Q (seq, 8, 256).unsqueeze(0), K/V (seq, 1, 256).unsqueeze(0)
    decoder  Q (chunk, 8, 256).unsqueeze(0), K/V (total_kv, 1, 256).unsqueeze(0)

All slices are along leading dims of contiguous buffers, so every view
handed to aiter is contiguous. ``softmax_scale = 1/sqrt(head_dim)`` is
passed explicitly (matches the RTX FA2 default and the sdpa backend).
``window_size_left = window_size_right = -1`` disables local windowing
(FA2-family "no window" sentinel); ``is_causal=False`` — all Pi0.5
sites are bidirectional.

POINTER STABILITY / CAPTURE SAFETY
----------------------------------
``mha_fwd(..., out=...)`` writes the result into a caller-provided
tensor, so each site passes a view of the SAME pre-allocated O buffer
the sdpa backend owns — the returned pointer is stable across graph
replays and the sdpa backend's final ``copy_`` disappears entirely.

``return_softmax_lse=False`` / ``return_dropout_randval=False`` are
requested so no result tensors need to be returned, but aiter may still
allocate the LSE (and any workspace) internally on every call. Those
per-call allocations go through torch's caching allocator, so they rely
on the SAME warmup-then-capture contract the interim sdpa path already
uses: the pipeline runs 3 warmup iterations on the capture stream before
``hipStreamBeginCapture`` so the allocator reaches steady state and
in-capture allocations are served from cached blocks. The dedicated
capture gate in the dev tests (test_amd_attn_aiter.py) verifies this
holds — it is the make-or-break check for this backend.

Like the sdpa backend, aiter launches on torch's CURRENT stream; the
frontend wraps calibration + capture + replay in
``torch.cuda.stream(...)`` so all work lands on the capture stream.

FIXED-SHAPE MODE
----------------
aiter exposes varlen/``cu_seqlens`` entry points that could express the
runtime-valid-length contract, but v2 keeps it simple: when
``set_fixed_shape(True)`` is active, the encoder/decoder sites delegate
to the sdpa base-class implementation, whose boolean key masks (driven
by :meth:`set_fixed_valid_len`) are read by pointer at replay — masking
by slicing is NOT an option in-graph because the valid length changes
without recapture. The vision site has no mask in either mode and stays
on aiter. Exact-shape mode (one graph per prompt length) runs fully on
aiter.
"""

from __future__ import annotations

import math

from flash_rt.amd.hardware.cdna4.attn_backend import Cdna4AttnBackend


def _resolve_mha_fwd():
    """Import aiter and resolve its ``mha_fwd`` entry point.

    Raises ImportError with a clear message when aiter is unavailable so
    the frontend can fall back to the sdpa backend.
    """
    try:
        import aiter  # noqa: F401
    except ImportError as ex:
        raise ImportError(
            "Cdna4AiterAttnBackend requires the 'aiter' package "
            "(AMD asm flash-attention); import failed — install aiter "
            "or use the sdpa backend (FVK_AMD_ATTN=sdpa)."
        ) from ex
    fn = getattr(aiter, "mha_fwd", None)
    if fn is None:
        # Older/newer layouts keep mha_fwd under aiter.ops.mha.
        try:
            from aiter.ops.mha import mha_fwd as fn  # type: ignore
        except ImportError as ex:
            raise ImportError(
                "aiter imported but exposes no 'mha_fwd' (checked "
                "aiter.mha_fwd and aiter.ops.mha.mha_fwd) — aiter "
                "version mismatch; use the sdpa backend "
                "(FVK_AMD_ATTN=sdpa)."
            ) from ex
    return fn


class Cdna4AiterAttnBackend(Cdna4AttnBackend):
    """Pi0.5 attention backend for AMD CDNA4 dispatching to aiter mha_fwd.

    Same construction signature and consumed surface as
    :class:`Cdna4AttnBackend`; only the three per-site attention calls
    are overridden. See the module docstring for the call contract,
    capture-safety notes and the fixed-shape delegation policy.
    """

    def __init__(self, num_views: int, encoder_seq_max: int, chunk_size: int,
                 num_encoder_layers: int = 18, dtype=None):
        super().__init__(num_views, encoder_seq_max, chunk_size,
                         num_encoder_layers=num_encoder_layers, dtype=dtype)
        # Resolve at construction so an unusable environment fails fast
        # (the frontend catches ImportError and falls back to sdpa).
        self._mha_fwd = _resolve_mha_fwd()
        # Explicit softmax scales (mha_fwd has no "default scale" arg on
        # the positional signature; match FA2's 1/sqrt(head_dim)).
        self._vis_scale = 1.0 / math.sqrt(self.vis_Q.shape[-1])   # D=72
        self._enc_scale = 1.0 / math.sqrt(self.enc_Q.shape[-1])   # D=256

    # ── aiter dispatch ──

    def _aiter_attn(self, q, k, v, out, scale):
        """mha_fwd with the pipeline's fixed contract: (B, S, H, D)
        layout, no dropout, bidirectional (no causal mask, no window),
        native GQA (Hkv may be 1), result written in-place into ``out``
        (a view of the pre-allocated, pointer-stable O buffer)."""
        self._mha_fwd(
            q, k, v,
            0.0,          # dropout_p
            scale,        # softmax_scale (explicit 1/sqrt(head_dim))
            False,        # is_causal
            -1,           # window_size_left  (no local window)
            -1,           # window_size_right (no local window)
            False,        # return_softmax_lse
            False,        # return_dropout_randval
            out=out,
        )

    # ── Attention calls (override sdpa math; same pointers returned) ──

    def vision_attn(self, stream: int = 0) -> int:
        # Buffers are (nv, 256, 16, 72) = (B, S, H, D) already — no
        # views needed on either side. Vision has no mask in fixed-shape
        # mode either, so this site always runs on aiter.
        self._aiter_attn(self.vis_Q, self.vis_K, self.vis_V,
                         self._vis_O, self._vis_scale)
        return self._vis_O.data_ptr()

    def encoder_attn(self, layer_idx: int, seq: int, stream: int = 0) -> int:
        if self._fixed_shape:
            # Fixed-shape masking needs the pointer-read boolean key mask
            # (valid length changes without recapture) — delegate to the
            # sdpa base implementation. See module docstring.
            return super().encoder_attn(layer_idx, seq, stream=stream)
        # (1, seq, 8, 256) vs (1, seq, 1, 256): native GQA, no expand.
        q = self.enc_Q[:seq].unsqueeze(0)
        k = self.enc_K[layer_idx, :seq].unsqueeze(0)
        v = self.enc_V[layer_idx, :seq].unsqueeze(0)
        self._aiter_attn(q, k, v, self._enc_O[:seq].unsqueeze(0),
                         self._enc_scale)
        return self._enc_O.data_ptr()

    def decoder_attn(self, layer_idx: int, enc_seq: int, dec_seq: int,
                     stream: int = 0) -> int:
        if self._fixed_shape:
            # Same delegation rationale as encoder_attn.
            return super().decoder_attn(layer_idx, enc_seq, dec_seq,
                                        stream=stream)
        total_kv = enc_seq + dec_seq
        # Chunk queries cross-attend the shared [encoder + appended chunk]
        # K/V range bidirectionally — same semantics as the sdpa path.
        q = self.dec_Q[:dec_seq].unsqueeze(0)                 # (1, chunk, 8, 256)
        k = self.enc_K[layer_idx, :total_kv].unsqueeze(0)     # (1, kv, 1, 256)
        v = self.enc_V[layer_idx, :total_kv].unsqueeze(0)
        self._aiter_attn(q, k, v, self._dec_O[:dec_seq].unsqueeze(0),
                         self._enc_scale)
        return self._dec_O.data_ptr()

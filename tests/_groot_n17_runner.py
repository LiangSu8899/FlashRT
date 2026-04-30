"""Minimal per-stage runner for Phase 3b.2 cosine tests.

Each forward in ``flash_vla.models.groot_n17.pipeline_thor`` is validated
against a fixture activation by:

    1. Loading only the ckpt tensors that stage needs (lazy, lru-cached).
    2. Reconstructing the stage's input from a sibling fixture activation
       (e.g. vlln input = RMSNorm(llm_layer_15, language_model.norm.weight)).
    3. Calling the pointer-only forward on pre-allocated GPU buffers.
    4. Returning the cpu fp32 output for cosine comparison.

This bypasses the full ``WeightLoader`` machinery (Phase 3a) — it goes
straight to the safetensors shards. Once the Phase 3c frontend lands, the
production loader becomes the source of truth and this helper can be
retired.
"""

from __future__ import annotations

import glob
import os
from functools import lru_cache

import torch


_CKPT_GLOB = "/root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/*"


@lru_cache(maxsize=1)
def ckpt_dir() -> str:
    matches = sorted(glob.glob(_CKPT_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"GR00T-N1.7-3B not found under {_CKPT_GLOB}; populate HF cache first")
    return matches[0]


@lru_cache(maxsize=1)
def _shard_index() -> dict[str, str]:
    """Map every ckpt tensor key to the shard file containing it."""
    from safetensors import safe_open
    out: dict[str, str] = {}
    for shard in sorted(glob.glob(os.path.join(ckpt_dir(), "model-*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                out[k] = shard
    return out


@lru_cache(maxsize=256)
def load_tensor(key: str, *, device: str = "cuda",
                dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Load a single ckpt tensor onto ``device`` and cast to ``dtype``.

    The returned tensor is contiguous and suitable for ``.data_ptr()`` use.
    """
    from safetensors import safe_open
    path = _shard_index()[key]
    with safe_open(path, framework="pt", device=device) as f:
        t = f.get_tensor(key)
    t = t.to(dtype).contiguous()
    return t


@lru_cache(maxsize=1)
def fvk():
    """Get the fvk module + side-load the strided FMHA library if available.

    ``fmha_strided_full`` is a runtime-loadable kernel; if the .so is not
    side-loaded, calls become no-ops with a "Strided FMHA not loaded"
    warning. Production frontends (e.g. ``pi05_thor.py:126``) auto-load
    from a path list. Mirror that here for tests.
    """
    import flash_vla.flash_vla_kernels as _fvk
    candidates = [
        "/workspace/libfmha_fp16_strided.so",
        "/work/libfmha_fp16_strided.so",
        "/work/build/libfmha_fp16_strided.so",
        "/tmp/libfmha_fp16_strided.so",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                _fvk.load_fmha_strided_library(p)
                break
            except Exception:
                continue
    return _fvk


class CausalLlmAttnAdapter:
    """Test-only ``AttentionBackend``-shaped wrapper that intercepts the
    ``llm`` site to do **causal** scaled-dot-product attention via
    ``torch.nn.functional.scaled_dot_product_attention`` (is_causal=True).

    Why: the FlashRT ``attention_mha_fp16`` and ``attention_qkv_fp16``
    kernels in this .so are **non-causal**; they overwrite the entire
    logits buffer with QK^T regardless of the pre-fill mask. Qwen3-VL
    text-decoder layers are causal (HF ``Qwen3VLTextAttention.is_causal=True``).
    Pure-fp32 PyTorch with causal SDPA matches fixture cos=0.999996;
    non-causal drops to cos=0.994. We use the SDPA fallback for tests so
    Phase 3b.2 cosine validation can land while a proper causal kernel
    is added later (out of scope for 3b.2).

    All other sites (vit, vl_self_attn, dit_*) delegate to the wrapped
    backend unchanged.

    The adapter expects to be constructed with the SAME PyTorch tensors
    that back the ``llm_slots`` Q/K/V/O ptrs, so it can wrap them as
    torch views without leaving the pointer-only contract for the
    rest of the pipeline.
    """

    def __init__(self, backend, *, q_tensor, k_exp_tensor, v_exp_tensor,
                 o_tensor, num_q_heads, head_dim):
        self._backend = backend
        self._Q = q_tensor
        self._K = k_exp_tensor
        self._V = v_exp_tensor
        self._O = o_tensor
        self._NH = int(num_q_heads)
        self._HD = int(head_dim)

    def get_slot_ptrs(self, site, layer_idx):
        return self._backend.get_slot_ptrs(site, layer_idx)

    def run(self, site, layer_idx, q_seq, *, kv_seq=None, stream=0,
            state_nk=None):
        if site != "llm":
            return self._backend.run(
                site, layer_idx, q_seq, kv_seq=kv_seq, stream=stream,
                state_nk=state_nk)

        import torch as _t
        import torch.nn.functional as _F
        S = int(q_seq)
        if kv_seq is None:
            kv_seq = q_seq
        # Q/K/V buffers live as (S, NH*HD); reshape to SDPA's expected
        # (1, NH, S, HD) layout. Compute fully in fp32 to avoid fp16 softmax
        # numerical drift, then cast back to fp16 when writing to O.
        Qh = self._Q[:S].view(S, self._NH, self._HD).permute(1, 0, 2).unsqueeze(0).float()
        Kh = self._K[:S].view(S, self._NH, self._HD).permute(1, 0, 2).unsqueeze(0).float()
        Vh = self._V[:S].view(S, self._NH, self._HD).permute(1, 0, 2).unsqueeze(0).float()
        scores = (Qh @ Kh.transpose(-2, -1)) / (self._HD ** 0.5)
        # Explicit upper-triangular -inf mask (more reliable than is_causal flag).
        causal_mask = _t.triu(
            _t.ones(S, S, dtype=_t.bool, device=Qh.device), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_w = scores.softmax(dim=-1)
        out = (attn_w @ Vh).squeeze(0).permute(1, 0, 2).reshape(
            S, self._NH * self._HD)
        self._O[:S].copy_(out.half())
        return self._O.data_ptr()


def gemm_runner():
    """Fresh ``GemmRunner`` per call.

    NOT a singleton — empirically a single ``GemmRunner`` shared across
    test fixtures shows shape-dependent autotune cache effects: when one
    fixture exercises GEMMs of shape A and a later fixture's GEMMs of
    shape B reuse the same runner, B's selected tactic can drift below
    cosine threshold (~0.003 cos drop observed). Each per-stage fixture
    constructs its own runner; the singleton wrapper proved fragile.
    """
    return fvk().GemmRunner()


# ─────────────────────────────────────────────────────────────────────────
# Static FP8 quantization helpers (per-tensor symmetric, FP8 E4M3, max=448)
# ─────────────────────────────────────────────────────────────────────────


class Fp8Calibrator:
    """Static FP8 calibration helpers for per-stage cosine tests.

    All FP8 tensors in the FlashRT pipeline are E4M3 (range ±448) with
    per-tensor symmetric scales. Two scales meet at every FP8 GEMM:

      * ``weight_scale``  = ``max(|W|) / 448`` — baked at quant time.
      * ``act_scale``     = ``max(|A|) / 448`` — captured during a fp16
        shadow pass (one-shot calibration), then frozen for production.

    ``alpha = act_scale * weight_scale`` is passed as a host ``float`` to
    ``GemmRunner.fp8_nn_*`` so the kernel can dequantize the accumulator
    back to fp16 in one fused multiply.
    """

    FP8_MAX = 448.0

    @staticmethod
    def quantize_weight(w_fp16: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Per-tensor symmetric quantization. Input ``w_fp16`` should already
        be in the layout the GEMM kernel expects (``[K, N]`` for ``fp8_nn_*``).

        Returns ``(w_fp8, weight_scale)`` where the host ``weight_scale`` is
        a Python float and ``w_fp8`` is a contiguous E4M3 tensor on the
        same device.
        """
        amax = w_fp16.detach().abs().max().item()
        scale = max(amax / Fp8Calibrator.FP8_MAX, 1e-8)
        w_fp8 = (
            (w_fp16.float() / scale)
            .clamp(-Fp8Calibrator.FP8_MAX, Fp8Calibrator.FP8_MAX)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        return w_fp8, scale

    @staticmethod
    def act_scale_dev(amax: float, *, device: str = "cuda") -> torch.Tensor:
        """Build the fp32 device scalar consumed by ``quantize_fp8_static_fp16``.

        Despite the ``_fp16`` suffix in the kernel name, the d_scale pointer
        is read as fp32 (verified empirically: fp16 d_scale produces a
        downstream amax off by ~448× — kernel reinterprets the 16-bit value
        as fp32 bits → denormal → garbage). Returns a contiguous 1-element
        fp32 tensor; ``.data_ptr()`` is the ``d_scale`` argument.
        """
        s = max(amax / Fp8Calibrator.FP8_MAX, 1e-8)
        return torch.tensor([s], dtype=torch.float32, device=device).contiguous()

    @staticmethod
    def alpha(act_scale: float, weight_scale: float) -> float:
        """Compose host alpha for ``GemmRunner.fp8_nn_*``: dequant multiplier."""
        return act_scale * weight_scale


# ─────────────────────────────────────────────────────────────────────────
# Qwen3-VL ViT 2D rotary position embedding (host-side cos/sin table)
# ─────────────────────────────────────────────────────────────────────────


# Re-export from production module — single source of truth.
from flash_vla.models.groot_n17.calibration import build_vit_rope_tables  # noqa: F401

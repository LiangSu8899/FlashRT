"""attention_core — the sage2 (Blackwell INT8-QK) dense form.

The consumer-Blackwell quantized twin of the dense attention family: the
same stateless seam — complete Q/K/V every call, host SDPA layout — executed
by the ``flashrt/sageattention2-blackwell`` kernel: per-warp (or per-thread)
INT8 quantization of Q and K, FP8 per-channel (or FP16) V, one fused
attention, BF16 out. Where the bf16 forms preserve numerics exactly, this
form trades a bounded quantization error for roughly half the attention
time on long unmasked sites; the parity gates downstream judge that trade
on real captures like any other impl's.

Two precision variants, both from the same artifact:

- ``pv_fp8``  — INT8 QK / FP8 V. The speed point of the family.
- ``pv_fp16`` — INT8 QK / FP16 V. Recovers most of the quantization error
  at ~60% more attention time; the option for hosts whose gates reject the
  FP8 point.

Qualification, decided from the artifact and the captures, refusal legible:

- head_dim must be advertised by the artifact (128 today); other dims
  return no binding so the host keeps its own attention,
- masked sites are not claimed: a mask has no form this kernel accepts, and
  the packed-KV plan of the FA2 form does not transfer (the quantizers
  consume dense NHD), so any allowed-ranges request refuses here,
- the workspace (INT8/FP8 staging, scales, output) is caller-owned and
  allocated once per shape at bind — call sequences are pointer-stable and
  the artifact declares itself CUDA-graph safe.
"""

from __future__ import annotations

import torch

from .. import hub_kernel
from ...guard import PROCEED, GuardedSeam

KERNEL_DEP = {
    "provider": "huggingface_kernels",
    "repo": "flashrt/sageattention2-blackwell",
    "version": ">=1",
}

_VARIANTS = ("pv_fp8", "pv_fp16")


def _artifact():
    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


def supported_head_dims() -> tuple[int, ...]:
    """Executable envelope, read from the artifact — never duplicated here."""
    caps = _artifact().capabilities()
    dims = tuple(sorted(int(d) for d in caps["head_dims"]))
    if not dims:
        raise ValueError(
            "attention_core sage2: artifact advertised no head dims")
    return dims


class DenseAttentionSage2(GuardedSeam, torch.nn.Module):
    """sage2 replacement for an ordinary dense unmasked SDPA call.

    Inputs and outputs use the host SDPA layout ``[B, H, S, D]``; the kernel
    consumes NHD ``[B, S, H, D]``. Quantization runs per call inside the
    artifact against the caller-owned workspace, so repeated calls launch an
    identical sequence on identical pointers.
    """

    def __init__(self, q_shape, kv_shape, dtype: torch.dtype, device,
                 variant: str = "pv_fp8",
                 qk_quant_granularity: str = "per_warp"):
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError(
                f"attention_core sage2: unknown variant {variant!r} "
                f"(expected one of {_VARIANTS})")
        b, heads, seq_q, head_dim = q_shape
        kb, kv_heads, seq_kv, kv_dim = kv_shape
        if kb != b or kv_dim != head_dim:
            raise ValueError(
                "attention_core sage2: Q and KV batch/head dims differ")
        if kv_heads != heads:
            raise ValueError(
                "attention_core sage2: GQA sites are not claimed by this "
                "form yet; query and KV head counts must match")
        if head_dim not in supported_head_dims():
            raise ValueError(
                f"attention_core sage2: head_dim {head_dim} outside the "
                f"artifact envelope {supported_head_dims()}")
        if dtype != torch.bfloat16:
            raise ValueError(
                "attention_core sage2: the artifact consumes bf16 inputs")
        self.q_shape = tuple(q_shape)
        self.kv_shape = tuple(kv_shape)
        self.variant = variant
        self.granularity = qk_quant_granularity

        art = _artifact()
        self._fn = (art.sage2_prefill_fp8v_bf16_d128 if variant == "pv_fp8"
                    else art.sage2_prefill_f16_bf16_d128)
        # NHD staging + caller-owned workspace, one set per bound shape.
        self.register_buffer("_q_nhd", torch.empty(
            b, seq_q, heads, head_dim, dtype=dtype, device=device))
        self.register_buffer("_k_nhd", torch.empty(
            b, seq_kv, kv_heads, head_dim, dtype=dtype, device=device))
        self.register_buffer("_v_nhd", torch.empty_like(self._k_nhd))
        self.register_buffer("_out_nhd", torch.empty_like(self._q_nhd))
        self._workspace = art.allocate_workspace(
            self._q_nhd, self._k_nhd, self._v_nhd,
            fp8v=(variant == "pv_fp8"),
            qk_quant_granularity=qk_quant_granularity)
        self._frt_arm(
            dtypes=(dtype,), device=torch.device(device),
            k=int(head_dim), rows=int(b * heads * seq_q))

    def forward(self, query, key, value, *, scale=None):
        admitted = self._frt_admit(query)
        if admitted is not PROCEED:
            return admitted
        # BHSD -> NHD staging copies (fused away once the host adopts the
        # NHD projection layout; kept explicit and pointer-stable here).
        self._q_nhd.copy_(query.transpose(1, 2))
        self._k_nhd.copy_(key.transpose(1, 2))
        self._v_nhd.copy_(value.transpose(1, 2))
        self._fn(
            self._q_nhd, self._k_nhd, self._v_nhd,
            softmax_scale=scale, out=self._out_nhd,
            workspace=self._workspace,
            qk_quant_granularity=self.granularity)
        return self._out_nhd.transpose(1, 2)


def bind_sage2_dense_attention(captures, *, variant: str = "pv_fp8",
                               qk_quant_granularity: str = "per_warp"):
    """Bind the sage2 dense form from one real capture set, or refuse.

    ``captures`` follows the family convention: an object carrying
    ``q_shape``, ``kv_shape``, ``dtype``, ``device``, and optionally
    ``allowed_ranges`` / ``mask``. Returns ``None`` (with the reason as an
    attribute on the function, mirroring the family's refusal trail) when
    the site is outside this form's envelope.
    """
    def refuse(reason: str):
        bind_sage2_dense_attention.last_refusal = reason
        return None

    mask = getattr(captures, "mask", None)
    ranges = tuple(getattr(captures, "allowed_ranges", ()) or ())
    if mask is not None or ranges:
        return refuse("masked/packed sites are not claimed by sage2")
    b, heads, seq_q, head_dim = captures.q_shape
    try:
        dims = supported_head_dims()
    except (ValueError, OSError, RuntimeError) as exc:
        return refuse(f"artifact unavailable: {exc}")
    if head_dim not in dims:
        return refuse(
            f"head_dim {head_dim} outside artifact envelope {dims}")
    if captures.dtype != torch.bfloat16:
        return refuse(f"dtype {captures.dtype} outside envelope (bf16)")
    try:
        return DenseAttentionSage2(
            captures.q_shape, captures.kv_shape, captures.dtype,
            captures.device, variant=variant,
            qk_quant_granularity=qk_quant_granularity)
    except (ValueError, RuntimeError) as exc:
        return refuse(str(exc))

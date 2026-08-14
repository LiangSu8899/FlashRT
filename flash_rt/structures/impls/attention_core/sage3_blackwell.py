"""attention_core — the sage3 (Blackwell FP4-QK/PV) dense form.

The speed point below the sage2 family: the same stateless seam — complete
Q/K/V every call, host SDPA layout — executed by the
``flashrt/sageattention3-blackwell`` artifact's fused entry: K centered and
FP4-quantized in one kernel, Q group-mean-centered and quantized likewise,
the mean contribution restored exactly through a small delta GEMM, then one
blockscaled E2M1 attention, BF16 out. Centering is a mathematical
invariant; the accuracy trade lives entirely in the FP4 quantization of
QK and PV, and the artifact declares it: ``accuracy_profile`` is read at
bind and carried on the module, so downstream gates judge this form
against the speed-first band it claims rather than the band the INT8
forms occupy. Family order by accuracy is sage2 pv_fp16, sage2 pv_fp8,
then this form; by speed the reverse.

Qualification, decided from the artifact and the captures, refusal legible:

- the artifact's attention is self-attention: one shape for Q, K, and V.
  Cross-attention and GQA sites return no binding,
- masked sites are not claimed, same reasoning as the sage2 form: the
  quantizers consume dense NHD and a packed-KV plan does not transfer,
- head_dim must be advertised by the artifact; other dims return no
  binding so the host keeps its own attention,
- sequence padding to the artifact's token alignment happens inside the
  fused workspace — any length binds, the padded tail never escapes,
- the workspace (FP4 staging, scales, centered-K, group means, delta,
  output) is caller-owned and allocated once at bind — call sequences
  are pointer-stable and the artifact declares itself CUDA-graph safe.
"""

from __future__ import annotations

import torch

from .. import hub_kernel
from ...guard import PROCEED, GuardedSeam

KERNEL_DEP = {
    "provider": "huggingface_kernels",
    "repo": "flashrt/sageattention3-blackwell",
    "version": ">=1",
}


def _artifact():
    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


def supported_head_dims() -> tuple[int, ...]:
    """Executable envelope, read from the artifact — never duplicated here."""
    caps = _artifact().capabilities()
    dims = tuple(sorted(int(d) for d in caps["head_dims"]))
    if not dims:
        raise ValueError(
            "attention_core sage3: artifact advertised no head dims")
    return dims


class DenseAttentionSage3(GuardedSeam, torch.nn.Module):
    """sage3 replacement for an ordinary dense unmasked self-attention call.

    Inputs and outputs use the host SDPA layout ``[B, H, S, D]``; the
    artifact consumes NHD ``[B, S, H, D]``. Centering, quantization, the
    delta correction, and attention all run inside the artifact's fused
    entry against the caller-owned workspace, so repeated calls launch an
    identical sequence on identical pointers.
    """

    def __init__(self, q_shape, kv_shape, dtype: torch.dtype, device):
        super().__init__()
        b, heads, seq_q, head_dim = q_shape
        if tuple(kv_shape) != tuple(q_shape):
            raise ValueError(
                "attention_core sage3: the artifact is a self-attention "
                "form; Q and KV shapes must match (cross-attention and "
                "GQA sites are not claimed)")
        if head_dim not in supported_head_dims():
            raise ValueError(
                f"attention_core sage3: head_dim {head_dim} outside the "
                f"artifact envelope {supported_head_dims()}")
        if dtype != torch.bfloat16:
            raise ValueError(
                "attention_core sage3: the artifact consumes bf16 inputs")
        art = _artifact()
        caps = art.capabilities()
        if not caps.get("fused_prep"):
            raise ValueError(
                "attention_core sage3: installed artifact predates the "
                "fused-prep entry this form is built on")
        self.q_shape = tuple(q_shape)
        self.kv_shape = tuple(kv_shape)
        #: accuracy band the artifact claims for itself; gates judge this
        #: form against the band it declares, not the INT8 family's.
        self.accuracy_profile = str(caps.get("accuracy_profile", ""))
        self._fn = art.sage3_prefill_fp4_bf16
        # NHD staging + the fused caller-owned workspace, one set per
        # bound shape. Padding to the artifact's token alignment lives
        # inside the workspace; the entry returns the unpadded view.
        self.register_buffer("_q_nhd", torch.empty(
            b, seq_q, heads, head_dim, dtype=dtype, device=device))
        self.register_buffer("_k_nhd", torch.empty_like(self._q_nhd))
        self.register_buffer("_v_nhd", torch.empty_like(self._q_nhd))
        self._workspace = art.allocate_fused_workspace(
            self._q_nhd, self._k_nhd, self._v_nhd)
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
        out = self._fn(
            self._q_nhd, self._k_nhd, self._v_nhd,
            softmax_scale=scale, workspace=self._workspace)
        return out.transpose(1, 2)


def bind_sage3_dense_attention(captures):
    """Bind the sage3 dense form from one real capture set, or refuse.

    ``captures`` follows the family convention: an object carrying
    ``q_shape``, ``kv_shape``, ``dtype``, ``device``, and optionally
    ``allowed_ranges`` / ``mask``. Returns ``None`` (with the reason as an
    attribute on the function, mirroring the family's refusal trail) when
    the site is outside this form's envelope.
    """
    def refuse(reason: str):
        bind_sage3_dense_attention.last_refusal = reason
        return None

    mask = getattr(captures, "mask", None)
    ranges = tuple(getattr(captures, "allowed_ranges", ()) or ())
    if mask is not None or ranges:
        return refuse("masked/packed sites are not claimed by sage3")
    if tuple(captures.kv_shape) != tuple(captures.q_shape):
        return refuse(
            "self-attention form: Q and KV shapes differ at this site")
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
        return DenseAttentionSage3(
            captures.q_shape, captures.kv_shape, captures.dtype,
            captures.device)
    except (ValueError, RuntimeError) as exc:
        return refuse(str(exc))

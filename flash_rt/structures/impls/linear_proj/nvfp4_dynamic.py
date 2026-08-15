"""NVFP4 (W4A4, dynamic activation scales) ``linear_proj`` implementation.

Weights are packed to NVFP4 (E2M1 data plus per-16-element-block scale
factors) at bind time; activations are quantized to the same format at
runtime, per call, with dynamically computed block scales — no
calibration data at either end. This is the execution form behind the
27B enablement line: checkpoints whose upstream loader decompresses
4-bit weights to BF16 inside ``forward`` (and therefore cannot fit the
card) run on the same card once their projections consume the packed
layout directly.

The ``flashrt/fp4-gemm`` entry point ``fp4_w4a16_linear_bf16`` takes the
pre-quantized activation tensor plus its scale factors — despite the
``a16`` in its historical name, the GEMM it runs is W4A4. ``variant=2``
is the qualified dispatch across the decode and short-prefill shapes
this impl serves (same-token 1.0000 against an exact reference on the
27B host, decode and prefill both through this path).

There is no host fallback: the module this replaces holds packed
weights the host cannot execute. The guard therefore refuses instead of
falling back, and adoption of a whole checkpoint is a load-time
transform, not a reversible attachment.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam

KERNEL_DEP = {
    "provider": "huggingface_kernels",
    "repo": "flashrt/fp4-gemm",
    "version": ">=1",
}

#: mirrors the kernel's own shape checks (``torch_binding.cpp``: K
#: divisible by 16 for the per-block scale factors, positive dims) —
#: no invented size walls: the adoption path serves whatever the
#: checkpoint author packed, and the 27B host's 17408-wide FFN is a
#: qualified shape, not an edge case
SUPPORT = {
    "K": {"min": 16, "multiple_of": 16},
    "N": {"min": 1},
}


@lru_cache(maxsize=1)
def _kernel():
    from flash_rt.structures.impls import hub_kernel

    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


@lru_cache(maxsize=1)
def _native_mrows():
    """The locally built native extension's small-M warp-split GEMM.

    The 16x8x64 block-scaled MMA atom computes a full 16-row tile, so
    M<=16 rows cost the same weight stream as one — the spec-verify
    rows (draft block + 1, and the shorter re-advance prefixes) are
    the customers. Absence is not a refusal: the tiled GEMM serves
    every shape correctly, this tier just reads the weights once for
    all rows where the build carries it.

    The pointer-style native entry registers as a torch custom op
    (with a fake shim) so the compiled multi-row passes trace through
    it instead of tripping over the raw stream handle.
    """
    try:
        from flash_rt import flash_rt_kernels as _fk
    except ImportError:
        return None
    fn = getattr(_fk, "fp4_w4a4_mma_sm120_warpsplit_mrows_bf16out",
                 None)
    if fn is None:
        return None

    from torch.library import custom_op

    @custom_op("flashrt_native::warpsplit_mrows", mutates_args=())
    def _op(a_packed: torch.Tensor, w_packed: torch.Tensor,
            a_sfa: torch.Tensor, w_sfb: torch.Tensor, n: int, k: int,
            warps: int, stages: int) -> torch.Tensor:
        m = a_packed.shape[0]
        y = torch.empty(m, n, device=a_packed.device,
                        dtype=torch.bfloat16)
        rc = fn(a_packed.data_ptr(), w_packed.data_ptr(), y.data_ptr(),
                m, n, k, a_sfa.data_ptr(), w_sfb.data_ptr(), 1.0,
                warps, stages,
                torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(
                f"warpsplit_mrows refused rc={rc} for M={m} N={n} K={k}")
        return y

    @_op.register_fake
    def _(a_packed, w_packed, a_sfa, w_sfb, n, k, warps, stages):
        return a_packed.new_empty((a_packed.shape[0], n),
                                  dtype=torch.bfloat16)

    return _op


def _check(weights: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    w = weights["w"]
    if w.dim() != 2:
        raise ValueError(f"w must be [N, K], got {tuple(w.shape)}")
    n, k = w.shape
    for name, dim in (("K", k), ("N", n)):
        bounds = SUPPORT[name]
        if dim < bounds["min"]:
            raise ValueError(
                f"{name}={dim} outside support envelope "
                f"(min {bounds['min']})")
        if bounds.get("multiple_of") and dim % bounds["multiple_of"]:
            raise ValueError(
                f"{name}={dim} must be a multiple of "
                f"{bounds['multiple_of']}")
    b = weights.get("b")
    if b is not None and tuple(b.shape) != (n,):
        raise ValueError(
            f"bias shape {tuple(b.shape)} does not match N={n}")
    return n, k


def _quantize_activation(kern, flat: torch.Tensor):
    """Use the direct BF16 producer when the installed artifact carries it."""
    if flat.dtype is torch.bfloat16:
        direct = getattr(kern, "quantize_fp4_sfa_bf16", None)
        if direct is not None:
            return direct(flat.contiguous())
    return kern.quantize_fp4_sfa_fp16(
        flat.to(torch.float16).contiguous())


class _ShareCell:
    """One activation-quantization seat shared by sibling projections.

    Reuse is keyed on tensor *identity*: a sibling consumes the stored
    quantization only when its input is the very object that produced
    it, so a fresh activation can never be served a stranger's data —
    at worst the cell misses and the seam quantizes for itself. Inside
    one traced graph the identity resolves at trace time, which bakes
    the single-quantize dataflow into the compiled prefill and the
    captured decode step alike; in eager it holds per call because the
    host hands every sibling the same normed hidden.

    Contract: activations must be functional — a caller that mutates a
    tensor *in place* and feeds the same object again would hit the
    cell with stale contents. This host family never does (every step's
    layernorm output is a fresh allocation, and the captured paths
    re-run the quantize inside the graph), which is why the linking is
    an explicit opt-in per adopted model rather than ambient behavior.
    """

    __slots__ = ("x", "a", "sfa")

    def __init__(self):
        self.x = None
        self.a = None
        self.sfa = None


#: sibling groups that consume the same activation in this host family:
#: the attention trio reads the input layernorm's output, the MLP pair
#: reads the post-attention layernorm's output. down/o are not grouped —
#: their inputs are their own.
_SHARE_GROUPS = (
    ("self_attn", ("q_proj", "k_proj", "v_proj")),
    ("mlp", ("gate_proj", "up_proj")),
)


def link_shared_producers(root: torch.nn.Module) -> int:
    """Link sibling NVFP4 seams so each shared activation quantizes once.

    Walks ``root`` for the host family's sibling groups and hands every
    group one :class:`_ShareCell`. Returns the number of groups linked.
    Additive and reversible: seams without a cell behave exactly as
    before, and a group only forms when at least two members are bound.
    """
    n_groups = 0
    for name, mod in root.named_modules():
        for tag, members in _SHARE_GROUPS:
            if not name.endswith(tag):
                continue
            seams = [getattr(mod, m, None) for m in members]
            seams = [s for s in seams
                     if isinstance(s, LinearProjNvfp4Dynamic)]
            if len(seams) < 2:
                continue
            cell = _ShareCell()
            for s in seams:
                s._share = cell
            n_groups += 1
    return n_groups


class LinearProjNvfp4Dynamic(GuardedSeam, torch.nn.Module):
    """Packed-weight projection: FP4 GEMM with runtime activation scales."""

    _frt_can_fallback = False

    def __init__(self, w_packed, w_sfb, bias, n, k):
        super().__init__()
        self.register_buffer("_w_packed", w_packed)
        self.register_buffer("_w_sfb", w_sfb)
        self._bias = bias
        self._n = n
        self._k = k
        kern = _kernel()
        self._kern = kern
        self._gemm = kern.fp4_w4a16_linear_bf16
        # M=1 decode rows route to the warp-split GEMV where the build
        # carries it and the shape qualifies (its own contract: N%8,
        # K a multiple of 64*warps). Absence is not a refusal - the
        # tiled GEMM serves every shape correctly, the GEMV just fills
        # the SMs it underfills at long-K decode shapes.
        gemv = getattr(kern, "fp4_w4a4_gemv_warpsplit_bf16", None)
        self._gemv = (gemv if gemv is not None
                      and n % 8 == 0 and k % (64 * 4) == 0 else None)
        # 2<=M<=16 rows route to the native multi-row warp-split tier
        # where the local build carries it. Per-shape launch config:
        # deeper stages hide the strided-B latency the extra A-row
        # loads expose — the wide/long-K families take (2,6), the
        # short-K o-class (2,4), tiny-N (4,4); each config's K
        # divisibility is its own gate.
        # hub artifact first (its ops are compile-safe torch ops),
        # local native build second, tiled GEMM always the floor
        hub_mrows = getattr(kern, "fp4_w4a4_gemm_warpsplit_mrows_bf16",
                            None)
        mrows = hub_mrows if hub_mrows is not None else _native_mrows()
        self._mrows_hub = hub_mrows is not None
        self._mrows = None
        if mrows is not None and n % 8 == 0:
            if (k >= 8192 or n >= 8192) and k % 128 == 0:
                self._mrows, self._mr_cfg = mrows, (2, 6)
            elif n <= 2048 and k % 256 == 0:
                self._mrows, self._mr_cfg = mrows, (4, 4)
            elif k % 128 == 0:
                self._mrows, self._mr_cfg = mrows, (2, 4)
        # M>=512 prefill slabs take the cooperative 256-tile tier
        # where the artifact carries it - wins every measured prefill
        # family over the base tile (the wrapper owns its workspace)
        self._m256 = getattr(kern, "nvfp4_gemm_m256_bf16", None)
        self._frt_arm(dtypes=CAST_OK, device=w_packed.device, k=int(k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        shape = x.shape
        cell = getattr(self, "_share", None)
        if cell is not None and cell.x is x:
            a_packed, a_sfa = cell.a, cell.sfa
        else:
            flat = x.reshape(-1, shape[-1])
            a_packed, a_sfa = _quantize_activation(self._kern, flat)
            if cell is not None:
                cell.x, cell.a, cell.sfa = x, a_packed, a_sfa
        m = a_packed.shape[0]
        if m == 1 and self._gemv is not None:
            y = self._gemv(a_packed, self._w_packed, a_sfa, self._w_sfb)
        elif 2 <= m <= 16 and self._mrows is not None:
            w_, s_ = self._mr_cfg
            if self._mrows_hub:
                y = self._mrows(a_packed, self._w_packed, a_sfa,
                                self._w_sfb, warps=w_, stages=s_)
            else:
                y = self._mrows(a_packed, self._w_packed, a_sfa,
                                self._w_sfb, self._n, self._k, w_, s_)
        elif m >= 512 and self._m256 is not None:
            y = self._m256(a_packed, self._w_packed, a_sfa,
                           self._w_sfb)
        else:
            y = self._gemm(a_packed, self._w_packed, a_sfa, self._w_sfb,
                           variant=2)
        if self._bias is not None:
            y = y + self._bias
        return y.reshape(*shape[:-1], self._n).type_as(x)


@torch.no_grad()
def bind_proj_seam(
    weights: Mapping[str, torch.Tensor],
) -> tuple[LinearProjNvfp4Dynamic, float]:
    """Bind one projection from a dense ``[N, K]`` weight.

    The weight is packed to the Hub kernel's NVFP4 layout on the GPU;
    the returned float is the pack-and-unpack relative L2 against the
    input weight — the *conversion* cost of regridding into this layout,
    reported so a caller adopting a whole checkpoint can put it in the
    receipt instead of losing it.
    """
    n, k = _check(weights)
    kern = _kernel()
    w = weights["w"].to("cuda", torch.float16).contiguous()
    w_packed, w_sfb = kern.quantize_fp4_sfa_fp16(w, is_sfb=True)
    # the conversion check accumulates in row slabs: a whole-tensor
    # FP32 dequant doubles the bind's transient footprint, and on
    # head-class weights that spike is what fails under a tight budget
    deq = kern.dequantize_fp4_sfa_fp16(w_packed, w_sfb)
    num_sq = den_sq = 0.0
    for i in range(0, n, 4096):
        diff = deq[i:i + 4096].float() - w[i:i + 4096].float()
        num_sq += float(diff.square().sum())
        den_sq += float(w[i:i + 4096].float().square().sum())
    del deq
    rel = (num_sq ** 0.5) / max(den_sq ** 0.5, 1e-12)
    bias = weights.get("b")
    if bias is not None:
        bias = bias.detach().to("cuda", torch.bfloat16)
    bound = LinearProjNvfp4Dynamic(w_packed, w_sfb, bias, n, k)
    # bind-time smoke: one M=1 launch through the real entry point before
    # the seam is handed out — a stale build or missing symbol surfaces
    # as a clean bind refusal, not later inside the host's forward
    probe = bound(torch.zeros(1, k, device=w_packed.device,
                              dtype=torch.bfloat16))
    if probe.shape != (1, n) or not torch.isfinite(probe).all():
        raise ValueError(
            f"refused: nvfp4 bind smoke produced shape "
            f"{tuple(probe.shape)}, finite={bool(torch.isfinite(probe).all())}")
    return bound, rel

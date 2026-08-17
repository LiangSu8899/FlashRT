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

import os
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


#: seams whose tier dispatch runs at call time, by index. A host that
#: compiles one graph for a whole range of row counts cannot carry a
#: Python-level tier branch (it would freeze the tracing sample's
#: choice), so those seams register here and dispatch inside a custom
#: op instead: the trace sees one opaque call, and the branch runs on
#: the shape the call actually receives — at capture, that is the size
#: being captured; outside a capture, it is the live batch.
_RT_SEATS: dict[int, object] = {}


def register_runtime_dispatch(seam) -> int:
    """Give ``seam`` a call-time dispatching entry; returns its index."""
    idx = len(_RT_SEATS)
    _RT_SEATS[idx] = seam
    seam._rt_idx = idx
    return idx


# Registered at import, never on first use: schema inference inside a
# traced region graph-breaks the host's compiled forward, and the first
# call of a lazily registered op lands exactly there.
@torch.library.custom_op("flash_rt_structures::nvfp4_linear_rt",
                         mutates_args=())
def _nvfp4_linear_rt(a_packed: torch.Tensor, a_sfa: torch.Tensor,
                     idx: int) -> torch.Tensor:
    return _RT_SEATS[idx]._mm_packed_impl(a_packed, a_sfa)


@_nvfp4_linear_rt.register_fake
def _(a_packed, a_sfa, idx):
    return a_packed.new_empty((a_packed.shape[0], _RT_SEATS[idx]._n),
                              dtype=torch.bfloat16)


def _runtime_dispatch():
    return _nvfp4_linear_rt


#: opt-in census of which tier each call actually lands in, keyed by
#: (row count, tier). Dispatch is data-dependent and lives inside a
#: custom op, so a host's own profile attributes every tier to the same
#: opaque call — reading the choice off the kernel names is exactly
#: what this makes unnecessary.
_TIER_CENSUS: dict | None = ({} if os.environ.get("FRT_TIER_CENSUS")
                             else None)

#: whether an adopted pack's per-tensor factor rides each tier's alpha
#: instead of a separate pass over the result. The routes are equivalent
#: (2.3e-3 apart at the seam, both the same distance from BF16), and on
#: real input streams a speculative host's acceptance length is
#: indifferent between them — each numeric path lands somewhere in a
#: ±0.2 content-dependent band around the host's own, with no
#: systematically better draw. On degenerate repeated-sentence prompts
#: the same choice swings acceptance 14%, which is a fact about that
#: protocol, not about the numerics: never judge this switch (or any
#: speculative A/B) on synthetic repetition. Folding is the default
#: because it is free — the separate pass costs ~10ms of 2K TTFT.
_ALPHA_FOLD = os.environ.get("FRT_NVFP4_ALPHA_FOLD", "1") != "0"


def _tier_census_note(seam, m) -> None:
    if not isinstance(m, int):
        tier = "sym->gemm"
    elif m == 1 and seam._gemv is not None:
        tier = "gemv"
    elif 2 <= m <= 16 and seam._mrows is not None:
        tier = "mrows"
    elif m >= 512 and seam._m256 is not None:
        tier = "m256"
    else:
        tier = "gemm"
    key = (m if isinstance(m, int) else -1, tier)
    _TIER_CENSUS[key] = _TIER_CENSUS.get(key, 0) + 1


def tier_census() -> dict:
    """The census so far, or an empty mapping when it is not armed."""
    return dict(_TIER_CENSUS or {})


if _TIER_CENSUS is not None:
    import atexit

    @atexit.register
    def _dump_tier_census():
        # printed from whichever process ran the seams: on a serving
        # host that is the engine worker, not the caller
        rows = sorted(_TIER_CENSUS.items(), key=lambda kv: -kv[1])
        print("[linear_proj.nvfp4] tier census (M, tier): calls",
              flush=True)
        for (m, tier), n in rows:
            print(f"[linear_proj.nvfp4]   M={m:<6} {tier:<10} {n}",
                  flush=True)


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

    def __init__(self, w_packed, w_sfb, bias, n, k, global_scale=None):
        super().__init__()
        self.register_buffer("_w_packed", w_packed)
        self.register_buffer("_w_sfb", w_sfb)
        self._bias = bias
        self._n = n
        self._k = k
        #: a per-tensor factor sitting outside the block scales. Weights
        #: this seam packs itself fold everything into the block scale
        #: and leave this None; weights adopted from a checkpoint that
        #: stores a separate global scale carry it here rather than
        #: having the block scales rescaled to absorb it — rescaling
        #: would re-round every E4M3 scale and put a lossy step into
        #: what is otherwise a pure relayout.
        self._w_gs = (None if global_scale is None
                      else float(global_scale))
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
        # measured per-shape launch config (512MB-rotation protocol,
        # bench_v1_sweep): small-N underfilled shapes want max warps
        # (+29%), long-K reads take (8,3) (+4%), the widest tall rows
        # edge to (2,3); everything else stays the entry default (4,4).
        # Each pick honors the kernel's (K/64)%warps==0 contract.
        kt = k // 64
        if os.environ.get("FRT_GEMV_CFG", "1") == "0":
            self._gemv_cfg = (4, 4)
        elif n <= 2048 and kt % 8 == 0:
            self._gemv_cfg = (8, 4)
        elif k >= 16384 and kt % 8 == 0:
            self._gemv_cfg = (8, 3)
        elif n >= 17000 and kt % 2 == 0:
            self._gemv_cfg = (2, 3)
        else:
            self._gemv_cfg = (4, 4)
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
        y = self._mm_packed(a_packed, a_sfa)
        return y.reshape(*shape[:-1], self._n).type_as(x)

    def _mm_packed(self, a_packed: torch.Tensor,
                   a_sfa: torch.Tensor) -> torch.Tensor:
        """Tier-dispatched matmul, through the runtime op when armed."""
        if getattr(self, "_rt_idx", None) is not None:
            # armed for a host that reuses one compiled graph across row
            # counts: whether this trace shows a symbolic M or a
            # specialized one, the branch it records is not the branch
            # the replay needs, so every call goes through the op whose
            # body runs on the shape actually received
            return _runtime_dispatch()(a_packed, a_sfa, self._rt_idx)
        return self._mm_packed_impl(a_packed, a_sfa)

    def _mm_packed_impl(self, a_packed: torch.Tensor,
                        a_sfa: torch.Tensor) -> torch.Tensor:
        """Tier-dispatched matmul over a pre-quantized activation.

        The same dispatch the seam's own forward uses, exposed so a
        producer that already holds the packed activation (a fused
        silu-mul epilogue, a sibling seam's shared quantization) can
        feed the weights without a decode round-trip through BF16.
        Returns the (m, n) BF16 product with bias applied.
        """
        m = a_packed.shape[0]
        if _TIER_CENSUS is not None:
            _tier_census_note(self, m)
        # every tier's entry takes the output scale as its own alpha, so
        # an adopted checkpoint's per-tensor factor rides the epilogue
        # that is already writing the result. Applying it afterwards
        # instead costs a full-size read-modify-write per projection —
        # invisible at M=1, and the whole prefill regression at M=2048.
        al = 1.0 if (self._w_gs is None or not _ALPHA_FOLD) else self._w_gs
        if not isinstance(m, int):
            # a symbolic row count with no runtime op armed: a tier
            # branch here would freeze the tracing sample's choice, and
            # the tiled GEMM is the one tier that serves every M
            # this branch returns before the shared epilogue below, so
            # it always folds — the switch exists to A/B the dispatched
            # tiers, and leaving a path that drops the factor entirely
            # would be a bug wearing an experiment's clothes
            y = self._gemm(a_packed, self._w_packed, a_sfa, self._w_sfb,
                           1.0 if self._w_gs is None else self._w_gs,
                           variant=2)
            if self._bias is not None:
                y = y + self._bias
            return y
        if m == 1 and self._gemv is not None:
            gw, gs = self._gemv_cfg
            y = self._gemv(a_packed, self._w_packed, a_sfa, self._w_sfb,
                           alpha=al, warps=gw, stages=gs)
        elif 2 <= m <= 16 and self._mrows is not None:
            w_, s_ = self._mr_cfg
            if self._mrows_hub:
                y = self._mrows(a_packed, self._w_packed, a_sfa,
                                self._w_sfb, alpha=al, warps=w_,
                                stages=s_)
            else:
                # the local native op fixes alpha at 1.0 in its schema;
                # scaling after it is the only route on that build
                y = self._mrows(a_packed, self._w_packed, a_sfa,
                                self._w_sfb, self._n, self._k, w_, s_)
                if self._w_gs is not None and _ALPHA_FOLD:
                    y = y * self._w_gs
        elif m >= 512 and self._m256 is not None:
            y = self._m256(a_packed, self._w_packed, a_sfa,
                           self._w_sfb, alpha=al)
        else:
            y = self._gemm(a_packed, self._w_packed, a_sfa, self._w_sfb,
                           al, variant=2)
        if self._w_gs is not None and not _ALPHA_FOLD:
            y = y * self._w_gs
        if self._bias is not None:
            y = y + self._bias
        return y


@torch.no_grad()
def bind_proj_seam_packed(
    w_packed: torch.Tensor,
    w_sfb: torch.Tensor,
    n: int,
    k: int,
    *,
    global_scale=None,
    bias: torch.Tensor | None = None,
) -> LinearProjNvfp4Dynamic:
    """Adopt a projection that is *already* NVFP4, without re-gridding.

    The regridding entry (:func:`bind_proj_seam`) exists for a host
    holding dense rows. A host holding a packed checkpoint does not need
    it, and should not pay it: dequantizing someone else's grid and
    quantizing it again with ours replaces their calibration with our
    packer's rounding, which is a change of model wearing the costume of
    an acceleration. The block-scale layout is the only thing that
    differs between the two conventions, and a relayout is a
    permutation — it loses nothing.

    ``w_packed`` is ``[N, K/2]`` E2M1 nibble pairs; ``w_sfb`` is the
    block scales already in this kernel's atom layout; ``global_scale``
    is the checkpoint's per-tensor factor, applied at the output.
    Tensors are adopted by reference: the caller's copy *is* the seam's,
    so seating a whole model costs no additional weight memory.
    """
    if w_packed.dtype is not torch.uint8:
        raise ValueError(
            f"packed weight must be uint8 nibble pairs, got "
            f"{w_packed.dtype}")
    if w_packed.shape != (n, k // 2):
        raise ValueError(
            f"packed weight {tuple(w_packed.shape)} does not match "
            f"N={n} K={k} (expected {(n, k // 2)})")
    bound = LinearProjNvfp4Dynamic(
        w_packed, w_sfb.view(torch.uint8).reshape(-1),
        (None if bias is None else bias.detach().to(torch.bfloat16)),
        n, k, global_scale=global_scale)
    probe = bound(torch.zeros(1, k, device=w_packed.device,
                              dtype=torch.bfloat16))
    if probe.shape != (1, n) or not torch.isfinite(probe).all():
        raise ValueError(
            f"refused: nvfp4 pack-adopt smoke produced shape "
            f"{tuple(probe.shape)}, finite="
            f"{bool(torch.isfinite(probe).all())}")
    return bound


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

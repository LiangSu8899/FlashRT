"""vLLM engine family: explicit assembly onto a serving host's model.

A vLLM model is an ``nn.Module`` tree living inside the engine process,
but its seams do not match the static module patterns discovery reads:
projections are merged/parallel classes whose ``forward`` returns
``(out, bias)``, expert weights live stacked on a routed-experts child,
and the LM head is consulted through ``quant_method.apply`` rather than
module forward. This adapter recognises those seams by structure —
2-D ``weight`` plus a ``quant_method`` slot for projections, a
``w13_weight``/``w2_weight`` pair for an expert bank, a ``lm_head``
whose vocabulary row count may exceed a quantize entry's grid limit —
and never by class or model name.

Three engine facts shape the assembly, each carried here so callers do
not rediscover them:

- **Seats must be installed after weights load and before the engine's
  first trace.** vLLM's compiled artifact resolves parameters by tree
  path; a post-compile swap either raises ``KeyError`` or is silently
  bypassed. ``install_load_hook`` patches the model runner's
  ``load_model`` for exactly this window.
- **A Python shape branch dies in the compiled form.** vLLM traces with
  guard evaluation off, so band dispatch (decode rows to the packed
  bank, prefill rows to the retained host) lives inside a custom op:
  its body re-runs at capture (decode sizes bake the seam branch into
  the graphs) and eagerly at prefill.
- **The head is intercepted at ``quant_method``**, and binds as row
  slabs when the vocabulary exceeds the quantize entry's row support.

Everything installed through :func:`attach_engine` goes through
``swap.attach``; the returned handle detaches bit-exactly. The expert
bank and head interceptions are host mutations recorded as ``revert``
callables on the same handle, so one ``detach`` restores all of it.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from .. import swap as _swap
from ..impls.linear_proj import nvfp4_dynamic as _linear
from ..impls.moe_experts import nvfp4_w4a16 as _experts_w4a16
from ..impls.moe_experts import nvfp4_dynamic as _experts_w4a4

#: dense projection seams, by dataflow position suffix. These are
#: positions in the qwen3_5 family's dataflow, not module identities;
#: a host that lacks one simply contributes no seat.
DENSE_SEAT_SUFFIXES = (
    "linear_attn.out_proj", "linear_attn.in_proj_qkvz",
    "self_attn.qkv_proj", "self_attn.o_proj",
    "shared_expert.gate_up_proj", "shared_expert.down_proj",
    "mlp.gate_up_proj", "mlp.down_proj",
)

_SEATS_BY_IDX: dict[int, Any] = {}


# Registered at import: the registration itself must never sit on a
# traced path — a lazy first call lands inside dynamo and the schema
# inference graph-breaks the host's compiled forward.
@torch.library.custom_op("flash_rt_structures::vllm_moe_seat",
                         mutates_args=())
def _vllm_moe_seat_op(hidden: torch.Tensor, router_logits: torch.Tensor,
                      top_idx: torch.Tensor, top_w: torch.Tensor,
                      idx: int) -> torch.Tensor:
    return _SEATS_BY_IDX[idx].run(hidden, router_logits, top_idx, top_w)


@_vllm_moe_seat_op.register_fake
def _(hidden, router_logits, top_idx, top_w, idx):
    return torch.empty_like(hidden)


_E2M1_LUT = None
_SRC_INDEX = None

#: engine fused-projection composition, in the engine's own concat
#: order (the split sites in its forwards are the receipts)
_FUSE = {
    "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "in_proj_ba": ("in_proj_b", "in_proj_a"),
}


def _source_ckpt_weight(seat_name):
    """Full-precision rows for a seat, read from ``FRT_SOURCE_CKPT``.

    ``seat_name`` is the engine module path; the checkpoint key drops
    the engine's leading ``language_model.`` and prefixes ``model.``.
    Fused engine projections concat their checkpoint constituents in
    the engine's own order. Returns None when the env is unset or the
    key cannot be resolved — the caller falls through to pack-specific
    dequant, and its bind probe stays the last word either way.
    """
    import json
    import os

    root = os.environ.get("FRT_SOURCE_CKPT")
    if not root or not seat_name:
        return None
    global _SRC_INDEX
    if _SRC_INDEX is None:
        from safetensors import safe_open
        idx = json.load(open(os.path.join(
            root, "model.safetensors.index.json")))
        _SRC_INDEX = (idx["weight_map"], {}, root, safe_open)
    wmap, handles, root, safe_open = _SRC_INDEX

    def read(key):
        fn = wmap.get(key)
        if fn is None:
            return None
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(root, fn),
                                    framework="pt", device="cpu")
        return handles[fn].get_tensor(key)

    cands = [seat_name]
    if seat_name.startswith("language_model.model."):
        cands.append("model.language_model."
                     + seat_name[len("language_model.model."):])
    if seat_name.startswith("model."):
        cands.append("model.language_model."
                     + seat_name[len("model."):])
    cands.append("model." + seat_name)
    for path in cands:
        leaf = path.rsplit(".", 1)[-1]
        parts = _FUSE.get(leaf)
        if parts is None:
            t = read(path + ".weight")
            if t is not None:
                return t.to("cuda", torch.bfloat16)
            continue
        base = path.rsplit(".", 1)[0]
        pieces = [read(f"{base}.{p}.weight") for p in parts]
        if all(p is not None for p in pieces):
            return torch.cat(
                [p.to("cuda", torch.bfloat16) for p in pieces],
                dim=0).contiguous()
    return None


#: FP8 weight dtypes a checkpoint may store a projection in
_FP8_DTYPES = tuple(
    d for d in (getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e5m2", None)) if d is not None)


def _host_precision(mod) -> str:
    """Which precision the *checkpoint* put this projection in.

    A mixed-precision checkpoint is a per-projection decision that was
    made with calibration data — attention projections held at FP8 while
    the FFN goes to NVFP4, a draft head excluded from quantization
    entirely. Seating every position at the adapter's favourite width
    overrides that decision silently: the weight stream shrinks, the
    step gets faster, and the accuracy it cost shows up somewhere no
    throughput number looks. Read the choice off what the host is
    actually holding rather than off module names, which is the same
    discipline the rest of this adapter uses for seam recognition.
    """
    w = getattr(mod, "weight", None)
    if w is None:
        return "none"
    dt = w.data.dtype
    if dt is torch.uint8:
        return "nvfp4"
    if dt in _FP8_DTYPES:
        return "fp8"
    return "unquantized"


def _projection_weight(mod) -> torch.Tensor:
    """The projection's dense rows, whatever the checkpoint stored.

    A bf16/fp16 weight is the rows. A uint8 weight is a modelopt NVFP4
    pack (e2m1 nibble pairs + fp8 block-16 scales + a global scalar):
    dequantize it here so the seam re-grids from real values. Nibble
    order inside the byte is resolved by the caller's bind probe — this
    helper emits the low-nibble-first convention, and a probe failure
    is a refusal, never silent garbage.
    """
    w = mod.weight.data
    if w.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return w
    if w.dtype in _FP8_DTYPES:
        # an FP8 host needs no source checkpoint: every FP8 code is
        # exact in BF16, so the host's own tensors are the rows —
        # dequantization here loses nothing. The engine stores this
        # weight transposed ([K, N], its cutlass B), so orientation
        # comes from the layer's declaration, same as the adopt path.
        ws = getattr(mod, "weight_scale", None)
        if ws is None:
            raise ValueError("fp8 weight without weight_scale")
        n = int(getattr(mod, "output_size_per_partition", 0) or 0)
        k = int(getattr(mod, "input_size_per_partition", 0) or 0)
        rows = w.to(torch.float32) * ws.data.reshape(-1)[0].float()
        if n and k and tuple(w.shape) == (k, n):
            rows = rows.t()
        return rows.to(torch.bfloat16).contiguous()
    src = _source_ckpt_weight(getattr(mod, "_frt_seat_name", None))
    if src is not None:
        # a quantized runtime weight, but the caller pointed
        # FRT_SOURCE_CKPT at the full-precision checkpoint: re-grid
        # from the real rows (the adopt door's own discipline) instead
        # of dequantizing whatever runtime pack the engine chose
        return src
    if w.dtype is not torch.uint8:
        raise ValueError(f"unrecognised weight dtype {w.dtype}")
    global _E2M1_LUT
    if _E2M1_LUT is None:
        _E2M1_LUT = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            device=w.device, dtype=torch.float32)
    ws = mod.weight_scale.data.to(torch.float32)      # [N, K/16]
    ws2 = getattr(mod, "weight_scale_2", None)
    g = (float(ws2.data.reshape(-1)[0]) if ws2 is not None else 1.0)
    n = w.shape[0]
    lo = (w & 0xF).long()
    hi = (w >> 4).long()
    codes = torch.stack([lo, hi], dim=-1).reshape(n, -1)
    vals = _E2M1_LUT[codes]                            # [N, K]
    scale = ws.repeat_interleave(16, dim=1) * g
    return (vals * scale).to(torch.bfloat16).contiguous()


def _adopt_fp8_pack(mod, rows_hint):
    """A seat on the host's own FP8 tensors, or None.

    Same hot-plug contract as the NVFP4 adopt: the engine already holds
    the weight as ``float8_e4m3fn`` with the checkpoint's calibrated
    per-tensor scales, and that per-tensor W8A8 scheme is exactly what
    ``fp8_static`` executes — so the seat adopts the storage by
    reference and changes only which kernel reads it. No dequantize, no
    re-quantize, no second copy on the card.
    """
    from ..impls.linear_proj import fp8_static as _fp8

    w = getattr(mod, "weight", None)
    ws = getattr(mod, "weight_scale", None)
    xs = getattr(mod, "input_scale", None)
    if w is None or ws is None or xs is None:
        return None
    if w.data.dtype not in _FP8_DTYPES:
        return None
    # the engine stores this weight transposed — [K, N], a column-major
    # B for its cutlass entry — so the projection's dims must come from
    # the layer's own declaration, not the storage order
    n = int(getattr(mod, "output_size_per_partition", 0) or 0)
    k = int(getattr(mod, "input_size_per_partition", 0) or 0)
    if not n or not k:
        return None
    for name, dim in (("K", k), ("N", n)):
        lo = _fp8.SUPPORT[name]["min"]
        hi = _fp8.SUPPORT[name]["max"]
        if not lo <= dim <= hi:
            raise ValueError(
                f"fp8 adopt: {name}={dim} outside support envelope")
    if tuple(w.data.shape) == (k, n):
        # one transposed copy, then the original storage goes: after
        # seating, this seam is the weight's only consumer, so total
        # memory is unchanged and the transient peak is one projection.
        # The host module keeps a [K, N] *view* of the new storage —
        # same logical content, so detach still reads correct values.
        w_nk = w.data.t().contiguous()
        mod.weight.data = w_nk.t()
    elif tuple(w.data.shape) == (n, k):
        w_nk = w.data
    else:
        raise ValueError(
            f"fp8 adopt: storage {tuple(w.data.shape)} matches neither "
            f"orientation of N={n} K={k}")
    ms = sorted(int(m) for m in rows_hint)
    form = _fp8._form_for(None, "bf16",
                          float(ms[len(ms) // 2]) * n * k)
    # the form bands were measured against a BF16-Linear host; against
    # an engine whose FP8 path fuses its own quantize they are not
    # gospel, so an explicit override stays available for A/B
    form = os.environ.get("FRT_FP8_FORM", form)
    bias = torch.zeros(n, device=w_nk.device, dtype=torch.bfloat16)
    return _fp8.FusedLinearProj(
        w_nk, bias,
        xs.data.reshape(-1)[:1].to(torch.float32),
        ws.data.reshape(-1)[:1].to(torch.float32),
        original=None, form=form)


def _adopt_nvfp4_pack(mod):
    """Build a seam on the host's own packed weights, or return None.

    This is the hot-plug form: the engine already holds NVFP4, so the
    seat changes which kernel reads it and nothing else. Numerics stay
    the checkpoint's — which for a speculative host means acceptance
    stays the checkpoint's too, and every millisecond the seat saves is
    kept rather than paid back as a lower accept rate.

    Returns None when this host's pack is not in a shape the seam can
    adopt, so the caller falls back to regridding and the difference is
    a reported choice rather than a silent one.
    """
    w = getattr(mod, "weight", None)
    ws = getattr(mod, "weight_scale", None)
    if w is None or ws is None or w.data.dtype is not torch.uint8:
        return None
    w = w.data
    # the engine's own kernel may pad the packed columns for its tile
    # shape; those columns are not part of the projection
    pad = int(getattr(mod, "weights_padding_cols", 0) or 0)
    if pad:
        w = w[:, :w.shape[1] - pad]
    gs = None
    for attr in ("weight_global_scale", "weight_scale_2"):
        v = getattr(mod, attr, None)
        if v is not None:
            flat = v.data.reshape(-1).float()
            if flat.numel() > 1 and not bool(
                    (flat == flat[0]).all()):
                # a merged projection may carry one global factor per
                # constituent; a single alpha can only stand in for
                # them when they agree, and pretending otherwise would
                # scale one half by the other's factor
                raise ValueError(
                    "per-part global scales differ; cannot adopt "
                    "under one alpha")
            gs = float(flat[0])
            break
    n, k = w.shape[0], w.shape[1] * 2
    return _linear.bind_proj_seam_packed(
        w.contiguous() if pad else w, ws.data, n, k, global_scale=gs)


def _bind_fp8_seam(mod, w, rows_hint):
    """An FP8 seat for a projection the checkpoint quantized to FP8.

    The input scale is the checkpoint's own calibrated one where the
    host carries it; a projection whose activations the checkpoint
    scaled dynamically has none, and the amax of its bind probe would be
    a statistic invented here rather than one measured on data, so that
    case is refused instead of guessed.
    """
    from ..impls.linear_proj import fp8_static as _fp8

    s = getattr(mod, "input_scale", None)
    if s is None:
        raise ValueError(
            "FP8 seat needs the checkpoint's calibrated input scale; "
            "this projection carries none")
    return _fp8.bind_proj_seam(
        {"w": w}, input_scale=float(s.data.reshape(-1)[0]),
        row_profile=list(rows_hint), original=mod)


class _ProjSeat(nn.Module):
    """Preserves the engine's ``(out, bias)`` projection contract.

    Attribute reads fall through to the replaced module: an engine that
    planned a fusion around this projection reads its scale attributes
    off the module object itself, and a seat that answers only
    ``forward`` turns that read into a startup crash far from here.
    """

    def __init__(self, seam, host=None):
        super().__init__()
        self.seam = seam
        if host is not None:
            object.__setattr__(self, "_frt_host", host)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            host = self.__dict__.get("_frt_host")
            if host is None:
                raise
            return getattr(host, name)

    def forward(self, x, *args, **kwargs):
        return self.seam(x), None


class _FusedMlpSeat(nn.Module):
    """Stands where the host's SwiGLU MLP stood.

    The host's own dataflow is ``gate_up -> SiLU·mul -> down``, and it
    already merges gate and up into one projection — so once both
    projections carry seats, the activation between them is the only
    step that still round-trips through BF16 and re-quantizes for the
    down projection. This seat runs the merged projection through its
    seam, collapses activation + quantization into the fused producer,
    and hands the packed rows straight to the down seam. Anything the
    host did around that (an expert gate) is kept by delegating to the
    retained module for the parts this seat does not own.
    """

    def __init__(self, host, gate_up_seam, down_seam, silu_mul):
        super().__init__()
        self.host_mlp = host
        self.gate_up_seam = gate_up_seam
        self.down_seam = down_seam
        self._silu_mul = silu_mul

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        a_packed, a_sfa = _linear._quantize_activation(
            self.gate_up_seam._kern, flat)
        merged = self.gate_up_seam._mm_packed(a_packed, a_sfa)
        p2, s2 = self._silu_mul(merged.contiguous())
        out = self.down_seam._mm_packed(p2, s2)
        out = out.reshape(*shape[:-1], self.down_seam._n).type_as(x)
        gate = getattr(self.host_mlp, "expert_gate", None)
        if gate is not None:
            out = torch.sigmoid(gate(x)[0]) * out
        return out


class _MoESeat(nn.Module):
    """Stands where the fused-MoE module stood: routing here, bank in
    the seam, the host's own shared-expert module added back (it owned
    it too, and its projections may themselves carry seats), and a
    declared band — decode rows walk the packed bank, prefill rows go
    to the retained host module."""

    #: rows above which the retained host module serves the batch.
    #: Measured, not assumed. A routed-MoE decode does not amortise the
    #: way a dense one does — eight tokens pick their own top-8 experts,
    #: so expert traffic grows with the batch instead of being shared,
    #: and a packed bank keeps paying well past batch one. Measured on
    #: Thor against vLLM 0.26 (35B-A3B, 128-token generations): 2.10x at
    #: batch 1, 2.45x at 4, 2.51x at 8, 1.65x at 16. The earlier value
    #: of 8 handed batch-16 traffic back to the host and threw that
    #: 1.65x away — the arm measured 0.98x, because at that batch the
    #: dense seats alone are worth nothing while the MoE seat is worth
    #: everything. 16 is the largest batch measured to pay; sweep with
    #: ``FRT_MOE_BAND_T`` before raising it further.
    BAND_T = int(os.environ.get("FRT_MOE_BAND_T", "16"))

    def __init__(self, seam, top_k, renormalize, shared, host):
        super().__init__()
        self.seam = seam
        self.top_k = top_k
        self.renormalize = renormalize
        self.shared = shared
        self.host = host
        self.host_internal = bool(getattr(host, "is_internal_router", False))
        self.is_internal_router = False   # the host block branches on this
        self._frt_host_serving = True     # prefill band runs through host
        self.idx = len(_SEATS_BY_IDX)
        _SEATS_BY_IDX[self.idx] = self

    def run(self, hidden_states, router_logits, top_idx, top_w):
        if hidden_states.shape[0] > self.BAND_T and self.host is not None:
            logits = (hidden_states if self.host_internal else router_logits)
            return self.host(hidden_states=hidden_states,
                             router_logits=logits)
        out = self.seam(hidden_states, top_idx, top_w)
        if self.shared is not None:
            out = out + self.shared(hidden_states)
        return out.to(hidden_states.dtype)

    def forward(self, hidden_states, router_logits):
        # routing stays in the traced region so inductor fuses the
        # softmax/topk/renormalize chain; the opaque op keeps only what
        # tracing would freeze — the band branch and the bank walk
        w = torch.softmax(router_logits.float(), dim=-1)
        tw, ti = torch.topk(w, self.top_k, dim=-1)
        if self.renormalize:
            tw = tw / tw.sum(dim=-1, keepdim=True)
        return _vllm_moe_seat_op(hidden_states, router_logits, ti, tw,
                                 self.idx)


class _SlabbedHeadMethod:
    """Stands in for the LM head's quant method: the engine computes
    logits through ``quant_method.apply``, never module forward.

    The vocabulary projection stays weight-only INT8 rather than joining
    the FP4 band: it is the logits family, where the decision on this
    model line has been W8 from the start. One seam covers the whole
    vocabulary when the entry's row support reaches it; otherwise the
    rows split into slabs and concatenate.
    """

    def __init__(self, seams, orig):
        self.seams = seams
        self.orig = orig

    def apply(self, layer, x, bias=None):
        xb = x.to(torch.bfloat16)
        y = (self.seams[0](xb) if len(self.seams) == 1
             else torch.cat([s(xb) for s in self.seams], dim=-1))
        if bias is not None:
            y = y + bias
        return y.to(x.dtype)

    def __getattr__(self, name):
        return getattr(self.orig, name)


def _is_projection(module) -> bool:
    w = getattr(module, "weight", None)
    return (isinstance(w, torch.Tensor) and w.dim() == 2
            and hasattr(module, "quant_method"))


def _arm_runtime_dispatch(seam) -> None:
    """Move this seam's tier dispatch to call time.

    This engine compiles a seam forward once per shape *range* and
    replays it without re-evaluating shape guards, so the row count a
    trace observed is not the row count a replay carries: a Python-level
    ``if m >= N`` freezes the tracing sample's choice and then runs it
    for every M (measured as a hard refusal from the M256 tier at engine
    start, and — quieter and more expensive — as every decode row taking
    the tiled GEMM instead of the GEMV). Registering the seam for
    runtime dispatch puts the branch inside a custom op, so the trace
    sees one opaque call and the tier is chosen on the shape the call
    actually receives. Every tier stays available.
    """
    _linear.register_runtime_dispatch(seam)


def _expert_holder(module):
    for _, child in module.named_modules():
        if torch.is_tensor(getattr(child, "w13_weight", None)):
            return child
    return None


class _NoSeats:
    """The handle shape for a host where nothing could be seated.

    Every seat refusing is a normal outcome, not an error: the hub may be
    unreachable, this architecture may have no build, the memory budget
    may leave no room. What must not happen is the engine dying because
    its accelerator was absent — a server that fails to start is worse
    than one that starts unaccelerated. So the refusals are reported and
    the host is handed back untouched, with a handle of the same shape so
    callers need no special case.
    """

    def __init__(self, refused, reverts):
        self.notes = {"refused": refused, "head_slabs": 0, "seated": 0}
        self._reverts = list(reverts)

    def detach(self):
        for fn in reversed(self._reverts):
            fn()
        self._reverts.clear()

    def report(self):
        return {}

    def summary(self):
        return {"seams": 0, "guarded_calls": 0, "fallbacks": 0,
                "seams_fell_back": [], "seams_self_detached": [],
                "seams_never_called": [], "clean": True,
                "refused": len(self.notes["refused"])}


def attach_engine(model, *, seats=DENSE_SEAT_SUFFIXES, experts=True,
                  head=True, use_gemv=None, verbose=True, strict=False,
                  fused_mlp=True, consume=False, ckpt_rename=None,
                  tag="model", precision="nvfp4", rows_hint=(1, 8),
                  adopt_pack=True):
    """Seat a vLLM model: dense projections, expert banks, LM head.

    Call between weight load and the engine's first trace (see
    :func:`install_load_hook`). Returns the ``swap.attach`` handle;
    ``handle.detach()`` restores the module tree, the expert modules
    and the head's quant method.
    """
    if use_gemv is None:
        cc = torch.cuda.get_device_capability()
        use_gemv = cc >= (12, 0)   # the warp-split GEMV entry's own arch
    if not use_gemv:
        orig_init = _linear.LinearProjNvfp4Dynamic.__init__

        def _init(self, *a, **kw):
            orig_init(self, *a, **kw)
            self._gemv = None
        _linear.LinearProjNvfp4Dynamic.__init__ = _init

    swaps: dict[str, nn.Module] = {}
    reverts: list = []
    refused: list = []
    parked_tiers: dict[str, int] = {}
    fused_mlps = 0
    staged = 0
    probe_cos: list[tuple[float, str]] = []
    #: the packer's own pack-and-unpack relative L2, which ``bind_proj_seam``
    #: returns and callers have been discarding. It is the grid's quality
    #: with no activation and no host in the way — the one number that
    #: says whether a seat reproduces the checkpoint's precision or
    #: quietly lowers it.
    pack_rels: list[float] = []
    kinds: dict[str, int] = {}
    adopted = 0
    requant = 0
    skipped: list[str] = []
    modules = dict(model.named_modules())

    # The head binds first. Its quantize needs an fp32 transient the
    # width of the vocabulary, and by the time the dense seats are in
    # place this process holds both the engine's original weights and
    # the seats' packed copies — the transient is exactly what is no
    # longer there. Bound first, it runs while the engine's weights are
    # the only thing resident.
    head_slabs = 0
    if head:
        lm = next((m for n, m in modules.items()
                   if n.endswith("lm_head")
                   and isinstance(getattr(m, "weight", None), torch.Tensor)),
                  None)
        if lm is not None:
            try:
                from ..impls.linear_proj import w8a16_static as _w8
                rows = lm.weight.shape[0]
                # name it so a quantized runtime head can be re-gridded
                # from the checkpoint's own rows rather than unpacked
                lm._frt_seat_name = "lm_head"
                w = _projection_weight(lm)
                # slabs, even where the entry's row support would take
                # the whole vocabulary in one bind: the engine has
                # already claimed its memory fraction by this point, so
                # the transient a whole-vocabulary quantize needs is
                # exactly what is not there. Four slabs put the peak
                # inside what the engine leaves behind.
                cap = min(_w8.SUPPORT["N"]["max"], -(-rows // 4))
                slab = -(-cap // 64) * 64
                seams = [_w8.bind_proj_seam({"w": w[lo:lo + slab]})
                         for lo in range(0, rows, slab)]
                orig_method = lm.quant_method
                lm.quant_method = _SlabbedHeadMethod(seams, orig_method)
                reverts.append(
                    lambda lm=lm, m=orig_method: setattr(
                        lm, "quant_method", m))
                head_slabs = len(seams)
            except Exception as e:
                refused.append(("lm_head", repr(e)[:200]))
                if verbose:
                    print(f"[structures.vllm] lm_head refused: "
                          f"{repr(e)[:200]}", flush=True)



    # dense projections, smallest first: on tight cards early frees
    # make room for the big binds
    targets = [(n, m) for n, m in modules.items()
               if any(n.endswith(s) for s in seats) and _is_projection(m)]
    targets.sort(key=lambda t: t[1].weight.numel())
    for name, mod in targets:
        try:
            mod._frt_seat_name = (ckpt_rename(name) if ckpt_rename
                                  else name)
            kind = (_host_precision(mod)
                    if precision in ("mirror", "auto") else "nvfp4")
            if precision == "auto" and kind == "fp8":
                # the auto tier's one opinion: a W8 position is carried
                # to W4 (the measured arbitrage: smaller weight stream,
                # acceptance unmoved on real streams), sourced from the
                # host's own FP8 rows — no external checkpoint. This is
                # a precision change and the tier says so in its report.
                kind = "nvfp4"
                requant += 1
            if kind == "unquantized":
                # the checkpoint held this projection out of its own
                # quantization; a seat here is not an acceleration of
                # the host's decision, it is a replacement of it
                skipped.append(name)
                continue
            seam, seam_shares_host, k_in = None, False, None
            if kind == "fp8" and adopt_pack:
                seam = _adopt_fp8_pack(mod, rows_hint)
                if seam is not None:
                    adopted += 1
                    seam_shares_host = True
                    # the engine stores FP8 as [K, N] (its cutlass B),
                    # so neither storage axis can be assumed; the seam's
                    # own [N, K] weight is the one orientation-safe place
                    # to read the projection's input width
                    k_in = int(seam._w_fp8.shape[1])
            if kind == "nvfp4" and adopt_pack:
                # try the host's own pack first: regridding is for a
                # host holding dense rows, and paying it here would
                # substitute our packer's rounding for the checkpoint's
                seam = _adopt_nvfp4_pack(mod)
                if seam is not None:
                    adopted += 1
                    # the seam holds the engine's own tensors, so the
                    # engine's copy is the seam's copy: releasing it
                    # would release the weights the seat executes
                    seam_shares_host = True
            if seam is None:
                w_bind = _projection_weight(mod)
                if kind == "fp8":
                    seam = _bind_fp8_seam(mod, w_bind, rows_hint)
                else:
                    seam, pack_rel = _linear.bind_proj_seam(
                        {"w": w_bind})
                    pack_rels.append(pack_rel)
                k_in = w_bind.shape[1]
            elif k_in is None:
                # the probe width is the projection's K, and the adopt
                # path has no dense rows to read it off — the NVFP4 host
                # holds packed bytes, whose second dim is K/2. A probe
                # built at that width fails inside the *host's* forward,
                # which reads as the seam being rejected when nothing
                # about the seam was tested at all.
                k_in = int(seam._k)
            # bind acceptance: the seam must reproduce the host module
            # on a live probe — this is what catches a wrong weight
            # layout (a packed checkpoint mistaken for dense rows) at
            # bind time instead of as garbage tokens later
            probe = torch.randn(4, k_in, device="cuda",
                                dtype=torch.bfloat16)
            with torch.no_grad():
                host_out = mod(probe)
                host_out = (host_out[0] if isinstance(host_out, tuple)
                            else host_out)
                cos = torch.nn.functional.cosine_similarity(
                    seam(probe).float().reshape(-1),
                    host_out.float().reshape(-1), dim=0)
            if float(cos) < 0.98:
                raise ValueError(
                    f"bind probe cos {float(cos):.4f} < 0.98")
            # the threshold is an admission gate, not a verdict: a tree
            # of seats all sitting just above it is a different model
            # from one sitting at 0.9999, and only the distribution says
            # which this is
            probe_cos.append((float(cos), name))
            # the dense rows were only ever the seam's input; holding
            # them across the next bind doubles the transient peak
            del probe, host_out
            if kind == "nvfp4":
                _arm_runtime_dispatch(seam)
            kinds[kind] = kinds.get(kind, 0) + 1
            swaps[name] = _ProjSeat(seam, host=mod)
            if consume and kind == "nvfp4" and not seam_shares_host:
                # nvfp4 only: the FP8 seam retains the host module for
                # its own fallback form, and releasing rows it may still
                # execute would turn a fallback into a device mismatch.
                # Release the engine's copy of these rows now, not after
                # the whole tree is seated: the seat owns the packed
                # form from here on, and holding both copies is exactly
                # what makes the *next* bind fail. Staged to host memory
                # so the handle's restore path still has them.
                dev = mod.weight.data.device
                cpu_rows = mod.weight.data.to("cpu")
                # the revert closure may capture only the host copy: a
                # default argument holding the device tensor pins the
                # very allocation this is releasing, and the release
                # then measures as a no-op.
                mod.weight.data = cpu_rows
                reverts.append(
                    lambda m=mod, w=cpu_rows, d=dev: setattr(
                        m.weight, "data", w.to(d)))
                del cpu_rows
                staged += 1
                if staged % 16 == 0:
                    torch.cuda.empty_cache()
                if verbose and staged % 48 == 0:
                    free_b, _ = torch.cuda.mem_get_info()
                    print(f"[structures.vllm] staged {staged} seats, "
                          f"{free_b / 2**30:.2f} GiB free", flush=True)
        except Exception as e:
            # a bare exception type carries no diagnosis; the first few
            # refusals keep their frames so "refused" names a line
            if len(refused) < 3:
                import traceback
                refused.append((name, repr(e)[:80] + " | "
                                + traceback.format_exc()[-1600:]))
            else:
                refused.append((name, repr(e)[:120]))

    if fused_mlp:
        from ..impls.decoder_ffn import nvfp4_fused as _ffn

        silu_mul = _ffn._native_silu_mul()
        if silu_mul is not None:
            for name, mod in modules.items():
                gu = swaps.get(f"{name}.gate_up_proj")
                dn = swaps.get(f"{name}.down_proj")
                if gu is None or dn is None:
                    continue
                if not hasattr(mod, "act_fn"):
                    continue
                # the fused producer emits packed FP4 for the down seam:
                # a pair the checkpoint put in different widths has no
                # such handoff
                if not all(isinstance(s.seam, _linear.LinearProjNvfp4Dynamic)
                           for s in (gu, dn)):
                    continue
                swaps[name] = _FusedMlpSeat(mod, gu.seam, dn.seam,
                                            silu_mul)
                swaps.pop(f"{name}.gate_up_proj")
                swaps.pop(f"{name}.down_proj")
                fused_mlps += 1

    if experts:
        impl = (_experts_w4a4 if use_gemv else _experts_w4a16)
        for name, mod in modules.items():
            if not name.endswith("mlp.experts"):
                continue
            holder = _expert_holder(mod)
            if holder is None:
                continue
            try:
                seam, _ = impl.bind_experts_seam(
                    {"gate_up_proj": holder.w13_weight.data,
                     "down_proj": holder.w2_weight.data},
                    act_fn=torch.nn.functional.silu)
                top_k = (getattr(mod, "top_k", None)
                         or getattr(getattr(mod, "moe_config", None),
                                    "experts_per_token", None) or 8)
                renorm = getattr(mod, "renormalize", None)
                parent = modules[name.rsplit(".experts", 1)[0]]
                swaps[name] = _MoESeat(
                    seam, int(top_k),
                    True if renorm is None else bool(renorm),
                    getattr(parent, "shared_expert", None), mod)
            except Exception as e:
                refused.append((name, repr(e)[:120]))

    model.eval()
    if not swaps:
        if strict:
            raise RuntimeError(
                "refused: no seat could be bound on this host (%d refusals; "
                "first: %s). Pass strict=False to let the engine start "
                "unaccelerated." % (len(refused),
                                    refused[0][1] if refused else "none"))
        if verbose:
            print(f"[structures.vllm] 0 seats, {len(refused)} refused — "
                  f"host runs unmodified", flush=True)
            for name, why in refused[:3]:
                print(f"[structures.vllm]   {name}: {why}", flush=True)
        handle = _NoSeats(refused, reverts)
        handle.notes["refused"] = refused
        return handle
    handle = _swap.attach(model, swaps, revert=reverts)
    if consume:
        # Until this runs the engine's own weights and the seats' packed
        # copies are both resident, and on a card sized for the model
        # alone that doubling is what makes the later binds fail — the
        # measured refusals were 80 MiB allocations against a full card.
        # Consuming moves each replaced module's truth to the weight
        # store; fallback and detach survive as restore-from-store.
        freed = handle.consume().get("freed_bytes", 0)
        if verbose:
            print(f"[structures.vllm] consumed host weights: "
                  f"{freed / 2**30:.2f} GiB freed", flush=True)
    if verbose:
        parked = ""
        fused = f", {fused_mlps} fused MLPs" if fused_mlps else ""
        by = (", ".join(f"{k}x{v}" for k, v in sorted(kinds.items()))
              if precision == "mirror" else "")
        skip = (f", {len(skipped)} left unquantized (checkpoint's own "
                f"exclusion)" if skipped else "")
        print(f"[structures.vllm] {tag}: {len(swaps)} seats "
              f"({head_slabs} head slabs), {len(refused)} refused"
              f"{parked}{fused}"
              f"{(' [' + by + ']') if by else ''}{skip}", flush=True)
        if adopted:
            print(f"[structures.vllm] {tag}: {adopted} seats adopted "
                  f"the host's own pack (no regrid, no extra weight "
                  f"memory)", flush=True)
        if requant:
            print(f"[structures.vllm] {tag}: {requant} seats carried "
                  f"W8->W4 (auto tier: precision change, task-level "
                  f"validation is the caller's gate)", flush=True)
        if pack_rels:
            pr = sorted(pack_rels)
            print(f"[structures.vllm] {tag}: pack relL2 median "
                  f"{pr[len(pr) // 2]:.5f}, worst {pr[-1]:.5f}",
                  flush=True)
        if probe_cos:
            cs = sorted(probe_cos)
            print(f"[structures.vllm] {tag}: bind cos min "
                  f"{cs[0][0]:.5f} ({cs[0][1].rsplit('.', 2)[-2:] and '.'.join(cs[0][1].split('.')[-3:])}), "
                  f"p10 {cs[len(cs) // 10][0]:.5f}, "
                  f"median {cs[len(cs) // 2][0]:.5f}", flush=True)
        seen = set()
        for nm, why in refused:
            key = why[:60]
            if key in seen:
                continue
            seen.add(key)
            print(f"[structures.vllm]   refused {nm}: {why[:1800]}",
                  flush=True)
    handle.notes = {"refused": refused, "head_slabs": head_slabs,
                    "parked_tiers": parked_tiers,
                    "fused_mlps": fused_mlps}
    return handle


def _draft_ckpt_rename(name: str) -> str:
    """Draft module path -> its key in the full-precision checkpoint.

    The proposer holds its own model, so its paths restart at ``model.``
    and collide with the target's layer 0 — a source lookup on the raw
    path silently reads the *target's* rows, and only the bind probe
    stands between that and a wrong seat. The checkpoint files the draft
    under its own ``mtp.`` subtree; naming it that way is what makes the
    lookup mean what it says.
    """
    for lead in ("model.model.", "model."):
        if name.startswith(lead):
            return "mtp." + name[len(lead):]
    return "mtp." + name


def install_load_hook(*, on_attached=None, seat_draft=True,
                      **attach_kwargs):
    """Patch every importable vLLM model-runner so :func:`attach_engine`
    runs after weights load and before the engine's first trace. Set
    ``VLLM_DISABLE_COMPILE_CACHE=1``: the engine's compile cache key
    does not see the module tree, and a stale artifact resolves
    parameters that the seats replaced."""
    import importlib
    import os

    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    patched = []
    for modname in ("vllm.v1.worker.gpu.model_runner",
                    "vllm.v1.worker.gpu_model_runner",
                    "vllm.v2.worker.gpu_model_runner"):
        try:
            module = importlib.import_module(modname)
        except ImportError:
            continue
        runner = getattr(module, "GPUModelRunner", None)
        if runner is None or not hasattr(runner, "load_model"):
            continue
        orig = runner.load_model

        def load_model(self, *a, __orig=orig, **kw):
            __orig(self, *a, **kw)
            handle = attach_engine(self.model, **attach_kwargs)
            # A speculative engine is two models, and its throughput is
            # the product of acceptance and step rate. Seating only the
            # target re-grids one side of the agreement test: the draft
            # still proposes from the checkpoint's own grid, the target
            # now judges from ours, and the measured cost is acceptance
            # — a loss no kernel win can pay back. The draft's seats are
            # bound from the same source rows for that reason first, and
            # for its own speed second.
            draft = getattr(getattr(self, "drafter", None), "model", None)
            if draft is not None and isinstance(draft, nn.Module) and seat_draft:
                kw2 = dict(attach_kwargs)
                kw2["head"] = False        # the draft shares the target's
                kw2["ckpt_rename"] = _draft_ckpt_rename
                kw2["tag"] = "draft"
                try:
                    dh = attach_engine(draft, **kw2)
                    # one detach undoes both models: the draft handle's
                    # undo list rides on the target's
                    handle._revert.append(dh.detach)
                except Exception as e:  # noqa: BLE001
                    if attach_kwargs.get("verbose", True):
                        print(f"[structures.vllm] draft not seated: "
                              f"{repr(e)[:160]}", flush=True)
            if on_attached is not None:
                on_attached(handle)
        runner.load_model = load_model
        patched.append(modname)
    if not patched:
        raise RuntimeError(
            "refused: no vLLM model runner found to hook; the engine "
            "layout is outside this adapter's profile")
    return patched

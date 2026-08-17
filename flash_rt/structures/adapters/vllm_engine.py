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


class _ProjSeat(nn.Module):
    """Preserves the engine's ``(out, bias)`` projection contract."""

    def __init__(self, seam):
        super().__init__()
        self.seam = seam

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
    logits through ``quant_method.apply``, never module forward."""

    def __init__(self, seams, orig):
        self.seams = seams
        self.orig = orig

    def apply(self, layer, x, bias=None):
        xb = x.to(torch.bfloat16)
        y = torch.cat([s(xb) for s in self.seams], dim=-1)
        if bias is not None:
            y = y + bias
        return y.to(x.dtype)

    def __getattr__(self, name):
        return getattr(self.orig, name)


def _is_projection(module) -> bool:
    w = getattr(module, "weight", None)
    return (isinstance(w, torch.Tensor) and w.dim() == 2
            and hasattr(module, "quant_method"))


def _park_m_threshold_tiers(seam) -> list[str]:
    """Park the tiers whose selection depends on the row count.

    This engine compiles a seam forward once per shape *range* and
    replays it without re-evaluating shape guards, so the row count a
    trace observed is not the row count a replay carries. A Python-level
    ``if m >= N`` therefore bakes in whichever side the tracing sample
    took and then runs it for every M — measured as a hard refusal from
    the M256 tier at engine start. The two tiers that carry an M
    threshold (the large-M cooperative tile, the small-M multi-row arm)
    are parked here; the M=1 GEMV stays because this host's decode
    graphs are captured at fixed batch sizes, and the tiled GEMM serves
    every other M correctly. A capability parked, never a refusal.
    """
    parked = []
    if getattr(seam, "_m256", None) is not None:
        seam._m256 = None
        parked.append("m256")
    if getattr(seam, "_mrows", None) is not None:
        seam._mrows = None
        parked.append("mrows")
    return parked


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
                  fused_mlp=True):
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
    modules = dict(model.named_modules())

    # dense projections, smallest first: on tight cards early frees
    # make room for the big binds
    targets = [(n, m) for n, m in modules.items()
               if any(n.endswith(s) for s in seats) and _is_projection(m)]
    targets.sort(key=lambda t: t[1].weight.numel())
    for name, mod in targets:
        try:
            mod._frt_seat_name = name
            w_bind = _projection_weight(mod)
            seam, _ = _linear.bind_proj_seam({"w": w_bind})
            # bind acceptance: the seam must reproduce the host module
            # on a live probe — this is what catches a wrong weight
            # layout (a packed checkpoint mistaken for dense rows) at
            # bind time instead of as garbage tokens later
            probe = torch.randn(4, w_bind.shape[1], device="cuda",
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
            for tier in _park_m_threshold_tiers(seam):
                parked_tiers[tier] = parked_tiers.get(tier, 0) + 1
            swaps[name] = _ProjSeat(seam)
        except Exception as e:
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

    head_slabs = 0
    if head:
        lm = next((m for n, m in modules.items()
                   if n.endswith("lm_head")
                   and isinstance(getattr(m, "weight", None), torch.Tensor)),
                  None)
        if lm is not None:
            try:
                rows = lm.weight.shape[0]
                slab = -(-rows // 4) // 64 * 64
                seams = [
                    _linear.bind_proj_seam(
                        {"w": lm.weight.data[lo:lo + slab]})[0]
                    for lo in range(0, rows, slab)]
                orig_method = lm.quant_method
                lm.quant_method = _SlabbedHeadMethod(seams, orig_method)
                reverts.append(
                    lambda lm=lm, m=orig_method: setattr(
                        lm, "quant_method", m))
                head_slabs = len(seams)
            except Exception as e:
                refused.append(("lm_head", repr(e)[:120]))

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
    if verbose:
        parked = (", parked " + ", ".join(
            f"{t}x{n}" for t, n in sorted(parked_tiers.items()))
            if parked_tiers else "")
        fused = f", {fused_mlps} fused MLPs" if fused_mlps else ""
        print(f"[structures.vllm] {len(swaps)} seats "
              f"({head_slabs} head slabs), {len(refused)} refused"
              f"{parked}{fused}", flush=True)
    handle.notes = {"refused": refused, "head_slabs": head_slabs,
                    "parked_tiers": parked_tiers,
                    "fused_mlps": fused_mlps}
    return handle


def install_load_hook(*, on_attached=None, **attach_kwargs):
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
            if on_attached is not None:
                on_attached(handle)
        runner.load_model = load_model
        patched.append(modname)
    if not patched:
        raise RuntimeError(
            "refused: no vLLM model runner found to hook; the engine "
            "layout is outside this adapter's profile")
    return patched

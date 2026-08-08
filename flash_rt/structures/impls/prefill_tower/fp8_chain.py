"""The fused static-FP8 launch chain over a plain-norm prefill tower.

Per layer: RMSNorm→FP8 (the host's ``(1+w)`` folded into the kernel's
weight), one merged-QKV FP8 GEMM, split+RoPE into a chain-owned
cache, dense non-causal attention over exactly the used keys (pads
trail, so a used-key count carries the host's mask), FP8 output
projection, and a fused residual+RMSNorm→FP8 into the merged gate/up
GEMM. The keys are un-permuted back to host rotate-half layout and
appended to the host's own cache object — the tower's whole product
is that cache, and every downstream reader keeps its contract.

Same equivalences as the sibling conditioned-stack chain (shared
helpers): the rotate-half↔adjacent-pair permutation on q/k rows, the
probe-calibrated activation scales, the attention ladder. The mask
fact here is simpler: one valid-query row names the used-key run;
pad-query rows are garbage-in-garbage-out by construction (their
hidden states feed only pad positions, their keys are masked by
every downstream mask).
"""

from __future__ import annotations

import types
from typing import Any, Callable

import torch

from .. import KernelUnavailable, hub_kernel
from ...guard import GuardedSeam
from ..adarms_stack.fp8_chain import (
    ATTN_RUNGS, FP8_MAX, _attention_rungs, _cache_kv, _fp8_weight,
    _gelu_tanh_like, _interleave_rows, _make_attend)

GEMM_PACKAGE = "flashrt/fp8-gemm"
NORM_PACKAGE = "flashrt/flashrt-residual-norm-quant"
ROPE_PACKAGE = "flashrt/flashrt-qkv-cache-rope"
GEMM_SYMBOLS = ("fp8_linear_bf16",)
NORM_SYMBOLS = ("rms_norm_quant_fp8_static_bf16",
                "residual_add_rms_norm_quant_fp8_static_bf16")
ROPE_SYMBOLS = ("qkv_split_rope_kvcache_bf16",)

SMOKE_FLOOR = 0.985


def missing_symbols() -> list[str]:
    gaps: list[str] = []
    for repo, symbols in ((GEMM_PACKAGE, GEMM_SYMBOLS),
                          (NORM_PACKAGE, NORM_SYMBOLS),
                          (ROPE_PACKAGE, ROPE_SYMBOLS)):
        try:
            kern = hub_kernel(repo, ">=1")
        except KernelUnavailable:
            gaps.append(repo)
            continue
        gaps.extend(f"{repo}:{s}" for s in symbols
                    if not hasattr(kern, s))
    if not _attention_rungs():
        gaps.append("attention: " + " or ".join(
            r[1] for r in ATTN_RUNGS))
    return gaps


class BoundPrefillFp8Chain(GuardedSeam, torch.nn.Module):
    """Bind-time state: FP8 weights, folded norm weights, caches."""

    _frt_can_fallback = False

    def __init__(self) -> None:
        super().__init__()
        self.table: list[dict] = []
        self.dims: dict = {}
        self.buf: dict = {}
        self.rope = None
        self.scaling = 1.0
        self.eps = 1e-6
        self.s_used = 0
        self.out_ctor = None
        self.kperm = None
        self.kperm_inv = None
        self.final_w = None
        self.cache_type = None
        self.kernels: dict = {}


def _plain_norm_weight(norm) -> torch.Tensor | None:
    """The affine vector of an unconditioned RMS norm, or None."""
    if getattr(norm, "dense", None) is not None:
        return None
    w = getattr(norm, "weight", None)
    if w is None or getattr(w, "ndim", 0) != 1:
        return None
    return w


def _stack_parts(stack):
    layers = list(stack.layers)
    attn = layers[0].self_attn
    head_dim = getattr(attn, "head_dim", None)
    if not isinstance(head_dim, int):
        raise ValueError("attention exposes no integer head_dim")
    nh = attn.q_proj.out_features // head_dim
    kv = attn.k_proj.out_features // head_dim
    dim = attn.q_proj.in_features
    hidden = layers[0].mlp.gate_proj.out_features
    return layers, nh, kv, head_dim, dim, hidden


def _prefill_mask_facts(mask: torch.Tensor, seq: int) -> int | None:
    """One valid-query row names the used-key run ``[0, s_used)``."""
    if mask.dim() != 4 or mask.shape[0] != 1 or mask.shape[-1] != seq:
        return None
    rows = mask[0, 0] if mask.shape[1] == 1 else mask[0, :1][0]
    if rows.shape[-2] != seq:
        return None
    row = rows[0] == 0
    s_used = int(row.sum())
    if s_used < 1 or not bool(row[:s_used].all()):
        return None
    return s_used


def _build_rope(bound, stack, position_ids) -> bool:
    hd = bound.dims["hd"]
    half = hd // 2
    dummy = torch.zeros(1, position_ids.shape[1], hd, device="cuda",
                        dtype=torch.float32)
    cos, sin = stack.rotary_emb(dummy, position_ids.to("cuda"))
    cos, sin = cos[0].float(), sin[0].float()
    if not torch.allclose(cos[:, :half], cos[:, half:], atol=1e-5):
        return False
    rope = torch.empty(position_ids.shape[1], hd, device="cuda",
                       dtype=torch.bfloat16)
    rope[:, 0::2] = cos[:, :half].to(torch.bfloat16)
    rope[:, 1::2] = sin[:, :half].to(torch.bfloat16)
    bound.rope = rope
    return True


@torch.no_grad()
def _quantize(bound, layers, amax) -> None:
    nh, kv, hd = (bound.dims[k] for k in ("nh", "kv", "hd"))
    for i, ly in enumerate(layers):
        attn, mlp = ly.self_attn, ly.mlp
        a_qkv, a_o, a_gu, a_dn = (amax[(i, s)] / FP8_MAX for s in
                                  ("qkv", "o", "gu", "dn"))
        qkv_w = torch.cat([
            _interleave_rows(attn.q_proj.weight, nh, hd),
            _interleave_rows(attn.k_proj.weight, kv, hd),
            attn.v_proj.weight], dim=0)
        gu_w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight],
                         dim=0)
        entry: dict[str, Any] = {}
        for name, w, act in (("qkv", qkv_w, a_qkv),
                             ("o", attn.o_proj.weight, a_o),
                             ("gu", gu_w, a_gu),
                             ("dn", mlp.down_proj.weight, a_dn)):
            packed, w_scale = _fp8_weight(w)
            entry[name] = packed
            entry[f"a_{name}"] = act * w_scale
        entry["sc_qkv"] = torch.tensor([a_qkv], device="cuda",
                                       dtype=torch.float32)
        entry["sc_gu"] = torch.tensor([a_gu], device="cuda",
                                      dtype=torch.float32)
        entry["inv_o"] = 1.0 / a_o if a_o > 0 else 1.0
        entry["inv_dn"] = 1.0 / a_dn if a_dn > 0 else 1.0
        # the host norm scales by (1 + weight); the kernel scales by
        # its weight verbatim, so the fold happens here, once
        entry["nw_in"] = (1.0 + ly.input_layernorm.weight.detach()
                          .float()).to("cuda", torch.bfloat16)
        entry["nw_post"] = (1.0 + ly.post_attention_layernorm.weight
                            .detach().float()).to("cuda", torch.bfloat16)
        bound.table.append(entry)


def _alloc(bound) -> None:
    S, D, nh, kv, hd, H = (bound.dims[k] for k in
                           ("seq", "dim", "nh", "kv", "hd", "hidden"))
    dev, bf = "cuda", torch.bfloat16
    b = bound.buf
    b["res"] = torch.empty(S, D, device=dev, dtype=bf)
    b["xn8"] = torch.empty(S, D, device=dev, dtype=torch.float8_e4m3fn)
    b["qkv"] = torch.empty(S, (nh + 2 * kv) * hd, device=dev, dtype=bf)
    b["q"] = torch.empty(1, S, nh, hd, device=dev, dtype=bf)
    kc = torch.zeros(1, S, kv, hd, device=dev, dtype=bf)
    vc = torch.zeros(1, S, kv, hd, device=dev, dtype=bf)
    L = bound.dims["layers"]
    b["kc"] = [kc] * L
    b["vc"] = [vc] * L
    b["fg"] = torch.empty(S, D, device=dev, dtype=bf)
    b["gu"] = torch.empty(S, 2 * H, device=dev, dtype=bf)
    b["seqused"] = torch.full((1,), bound.s_used, device=dev,
                              dtype=torch.int32)
    b["att"] = torch.empty(1, S, nh, hd, device=dev, dtype=bf)


def _make_run(bound: BoundPrefillFp8Chain):
    kg = bound.kernels["kg"]
    kn = bound.kernels["kn"]
    kr = bound.kernels["kr"]
    attend = bound.kernels["attend"]
    S, D, nh, kv, hd, H, L = (bound.dims[k] for k in
                              ("seq", "dim", "nh", "kv", "hd",
                               "hidden", "layers"))
    b = bound.buf
    table = bound.table
    eps = bound.eps
    qkv3 = b["qkv"].view(1, S, (nh + 2 * kv) * hd)
    fp8 = torch.float8_e4m3fn
    kperm_inv = bound.kperm_inv
    kh = b["kc"][0].view(S, hd)

    def run(x2d, pkv):
        res = b["res"]
        res.copy_(x2d)
        for l in range(L):
            e = table[l]
            if l == 0:
                kn.rms_norm_quant_fp8_static_bf16(
                    res, e["nw_in"], e["sc_qkv"], eps, out=b["xn8"])
            kg.fp8_linear_bf16(b["xn8"], e["qkv"], alpha=e["a_qkv"],
                               out=b["qkv"])
            kr.qkv_split_rope_kvcache_bf16(
                qkv3, bound.rope, nh, kv, hd, 0, q_out=b["q"],
                k_cache=b["kc"][0], v_cache=b["vc"][0], max_seq_len=S)
            if pkv is not None:
                # fresh tensors per layer: the cache keeps references,
                # and the shared buffers are overwritten next layer
                k_host = torch.index_select(kh, -1, kperm_inv)
                pkv.update(k_host.view(1, 1, S, hd),
                           b["vc"][0].reshape(1, 1, S, hd).clone(), l)
            att2 = attend(l)
            o8 = (att2.float() * e["inv_o"]).clamp(
                -FP8_MAX, FP8_MAX).to(fp8)
            kg.fp8_linear_bf16(o8, e["o"], alpha=e["a_o"], out=b["fg"])
            res.add_(b["fg"])
            kn.rms_norm_quant_fp8_static_bf16(
                res, e["nw_post"], e["sc_gu"], eps, out=b["xn8"])
            kg.fp8_linear_bf16(b["xn8"], e["gu"], alpha=e["a_gu"],
                               out=b["gu"])
            hid = torch.nn.functional.gelu(
                b["gu"][:, :H].float(), approximate="tanh") \
                * b["gu"][:, H:].float()
            h8 = (hid * e["inv_dn"]).clamp(-FP8_MAX, FP8_MAX).to(fp8)
            kg.fp8_linear_bf16(h8, e["dn"], alpha=e["a_dn"], out=b["fg"])
            if l < L - 1:
                nxt = table[l + 1]
                kn.residual_add_rms_norm_quant_fp8_static_bf16(
                    res, b["fg"], nxt["nw_in"], nxt["sc_qkv"], eps,
                    out=b["xn8"])
            else:
                res.add_(b["fg"])
        fin = res.float()
        fin = fin * torch.rsqrt(fin.square().mean(-1, keepdim=True)
                                + eps)
        out = (fin * bound.final_w).to(torch.bfloat16)
        return bound.out_ctor(last_hidden_state=out.view(1, S, D),
                              past_key_values=pkv)

    return run


def bind_prefill_fp8_chain(model, root: str,
                           probe: Callable[[], Any]) -> dict:
    """Bind the chain onto the tower at ``root``; adapter contract out."""
    try:
        kg = hub_kernel(GEMM_PACKAGE, ">=1")
        kn = hub_kernel(NORM_PACKAGE, ">=1")
        kr = hub_kernel(ROPE_PACKAGE, ">=1")
    except KernelUnavailable as exc:
        return {"refused": f"prefill_fp8_chain: {exc}"}
    rungs = _attention_rungs()
    gaps = missing_symbols()
    if gaps:
        return {"refused": "prefill_fp8_chain missing: "
                           f"{', '.join(gaps)}"}

    stack = model.get_submodule(root) if root else model
    layers, nh, kv, hd, dim, hidden = _stack_parts(stack)
    if kv != 1:
        return {"refused": f"prefill_fp8_chain: kv_heads {kv} outside "
                           "the single-KV band"}
    if not _gelu_tanh_like(layers[0].mlp.act_fn):
        return {"refused": "prefill_fp8_chain: FFN activation is not "
                           "tanh-GELU"}
    scalings = {float(ly.self_attn.scaling) for ly in layers}
    if len(scalings) != 1:
        return {"refused": "prefill_fp8_chain: per-layer attention "
                           "scaling differs"}
    final_w = _plain_norm_weight(stack.norm)
    if final_w is None or any(
            _plain_norm_weight(ly.input_layernorm) is None
            or _plain_norm_weight(ly.post_attention_layernorm) is None
            for ly in layers):
        return {"refused": "prefill_fp8_chain: a norm is not a plain "
                           "affine RMS norm"}

    bound = BoundPrefillFp8Chain()
    bound.kernels = {"kg": kg, "kn": kn, "kr": kr}
    bound.scaling = scalings.pop()
    bound.dims = {"nh": nh, "kv": kv, "hd": hd, "dim": dim,
                  "hidden": hidden, "layers": len(layers)}
    bound.eps = float(getattr(stack.norm, "eps", 1e-6))
    bound.final_w = (1.0 + final_w.detach().float()).to("cuda")

    # ---- one probe: the prefill call, mask fact, amax sites ----
    calls: list[dict] = []
    amax: dict[tuple[int, str], float] = {}

    def note(site):
        def hook(_m, args):
            peak = float(args[0].detach().abs().amax())
            amax[site] = max(amax.get(site, 0.0), peak)
        return hook

    hooks = []
    for i, ly in enumerate(layers):
        hooks.append(ly.self_attn.q_proj.register_forward_pre_hook(
            note((i, "qkv"))))
        hooks.append(ly.self_attn.o_proj.register_forward_pre_hook(
            note((i, "o"))))
        hooks.append(ly.mlp.gate_proj.register_forward_pre_hook(
            note((i, "gu"))))
        hooks.append(ly.mlp.down_proj.register_forward_pre_hook(
            note((i, "dn"))))

    saved_probe = stack.__dict__.get("forward")
    host_forward = stack.forward

    def capturing(_self, *args, **kwargs):
        out = host_forward(*args, **kwargs)
        embs = kwargs.get("inputs_embeds")
        pkv = getattr(out, "past_key_values", None)
        hidden_out = getattr(out, "last_hidden_state", None)
        if (embs is not None and kwargs.get("use_cache")
                and hidden_out is not None
                and embs.dim() == 3 and embs.shape[0] == 1
                and kwargs.get("adarms_cond") is None):
            entry = {
                "x": embs.detach().clone(),
                "mask": kwargs.get("attention_mask"),
                "pos": kwargs.get("position_ids"),
                "out": hidden_out.detach().clone(),
                "out_type": type(out),
                "cache_type": type(pkv) if pkv is not None else None,
                "kv": [tuple(t.detach().clone()
                             for t in _cache_kv(pkv, i))
                       for i in range(len(layers))] if pkv is not None
                      else None,
            }
            entry["mask"] = (entry["mask"].detach().clone()
                             if entry["mask"] is not None else None)
            entry["pos"] = (entry["pos"].detach().clone()
                            if entry["pos"] is not None else None)
            calls.append(entry)
        return out

    stack.forward = types.MethodType(capturing, stack)
    try:
        with torch.inference_mode():
            probe()
    finally:
        for hook in hooks:
            hook.remove()
        if saved_probe is not None:
            stack.forward = saved_probe
        else:
            stack.__dict__.pop("forward", None)

    if not calls:
        return {"refused": "prefill_fp8_chain: probe never made a "
                           "prefill call"}
    first = calls[0]
    if first["mask"] is None or first["pos"] is None:
        return {"refused": "prefill_fp8_chain: probe call carried no "
                           "mask or positions"}
    S = first["x"].shape[1]
    s_used = _prefill_mask_facts(first["mask"], S)
    if s_used is None:
        return {"refused": "prefill_fp8_chain: mask outside the "
                           "[valid|pad] band"}
    if any((i, s) not in amax or amax[(i, s)] <= 0.0
           for i in range(len(layers))
           for s in ("qkv", "o", "gu", "dn")):
        return {"refused": "prefill_fp8_chain: calibration saw a dead "
                           "quantizer site"}

    bound.dims["seq"] = S
    bound.s_used = s_used
    bound.out_ctor = first["out_type"]
    bound.cache_type = first["cache_type"]
    # the GEMM entry's row band is a runtime fact, not a symbol fact:
    # probe it at the bound shape before spending any weight packing
    try:
        kg.fp8_linear_bf16(
            torch.zeros(S, dim, device="cuda",
                        dtype=torch.float8_e4m3fn),
            torch.zeros(dim, dim, device="cuda",
                        dtype=torch.float8_e4m3fn), alpha=1.0)
    except Exception as exc:  # noqa: BLE001 — a band fact, not a crash
        return {"refused": "prefill_fp8_chain: the FP8 GEMM entry "
                           f"refuses {S} rows ({exc})"}
    if not _build_rope(bound, stack, first["pos"]):
        return {"refused": "prefill_fp8_chain: rotary table is not "
                           "half-duplicated"}
    _quantize(bound, layers, amax)
    _alloc(bound)
    half = hd // 2
    kperm = torch.empty(hd, dtype=torch.long, device="cuda")
    kperm[0::2] = torch.arange(half, device="cuda")
    kperm[1::2] = torch.arange(half, hd, device="cuda")
    bound.kperm = kperm
    bound.kperm_inv = torch.argsort(kperm)

    attend, attn_mode = None, None
    rung_trail = []
    for mode, kern in rungs:
        try:
            candidate = _make_attend(bound, mode, kern)
            candidate(0)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 — a dead rung, next
            rung_trail.append(f"{mode}: {type(exc).__name__}")
            continue
        attend, attn_mode = candidate, mode
        break
    if attend is None:
        return {"refused": "prefill_fp8_chain: no attention rung "
                           f"executes here ({'; '.join(rung_trail)})"}
    bound.kernels["attend"] = attend

    run = _make_run(bound)
    guard = bound._frt_arm(dtypes=(torch.bfloat16,),
                           device=torch.device("cuda"))
    guard.notes["n_layers"] = len(layers)
    guard.notes["s_used"] = s_used
    guard.notes["attention"] = attn_mode

    # ---- smoke: hidden states and the cache the tower left behind ----
    worst = None
    with torch.inference_mode():
        for c in calls:
            fresh = c["cache_type"]() if c["cache_type"] else None
            got = run(c["x"][0].to(torch.bfloat16), fresh)
            valid = slice(0, s_used)
            cos = torch.nn.functional.cosine_similarity(
                got.last_hidden_state[0, valid].float().flatten(),
                c["out"][0, valid].float().flatten(), dim=0)
            worst = float(cos) if worst is None else min(worst,
                                                         float(cos))
            if fresh is not None and c["kv"] is not None:
                for l in range(len(layers)):
                    gk, gv = _cache_kv(fresh, l)
                    hk, hv = c["kv"][l]
                    for a, bb in ((gk, hk), (gv, hv)):
                        cc = torch.nn.functional.cosine_similarity(
                            a[..., :s_used, :].float().flatten(),
                            bb[..., :s_used, :].float().flatten(),
                            dim=0)
                        worst = min(worst, float(cc))
    if worst is None or worst < SMOKE_FLOOR:
        return {"refused": f"prefill_fp8_chain smoke cos {worst} < "
                           f"{SMOKE_FLOOR} across {len(calls)} probe "
                           "call(s)"}
    guard.notes["smoke_cos"] = round(worst, 6)

    # ---- route ----
    saved = stack.__dict__.get("forward")
    x_shape = tuple(first["x"].shape)
    mask_shape = tuple(first["mask"].shape)

    def routed(_self, *args, **kwargs):
        compiling = torch.compiler.is_compiling()
        capturing_now = (False if compiling
                         else torch.cuda.is_current_stream_capturing())
        eager = not compiling and not capturing_now
        if eager:
            guard.calls += 1
        embs = kwargs.get("inputs_embeds")
        mask = kwargs.get("attention_mask")
        pkv = kwargs.get("past_key_values")
        ok = (not args and embs is not None
              and kwargs.get("use_cache")
              and kwargs.get("adarms_cond") is None
              and tuple(embs.shape) == x_shape
              and (mask is None or tuple(mask.shape) == mask_shape))
        if not ok:
            if not eager:
                raise RuntimeError(
                    "prefill_fp8_chain: out-of-contract call during "
                    "capture/compile — fix the eager path first")
            guard.fallbacks += 1
            guard.last_reason = "call outside the routed contract"
            return host_forward(*args, **kwargs)
        if pkv is None and bound.cache_type is not None:
            pkv = bound.cache_type()
        return run(embs[0].to(torch.bfloat16), pkv)

    def enable() -> None:
        stack.forward = types.MethodType(routed, stack)

    def disable() -> None:
        if saved is not None:
            stack.forward = saved
        elif "forward" in stack.__dict__:
            del stack.forward

    def revert() -> None:
        disable()
        bound.table.clear()
        bound.buf.clear()

    enable()
    return {
        "observed": {f"{root}::prefill_fp8_chain": bound},
        "revert": [revert],
        "toggle": (enable, disable),
        "smoke_cos": worst,
    }

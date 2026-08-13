"""The fused static-FP8 launch chain over a biased vision tower.

Per layer: LayerNorm→FP8 with the affine pair in the kernel, one
merged-QKV FP8 GEMM with the bias in the epilogue, dense per-call
attention (no cache, no mask — the patch sequence is full and
unpadded), and the output/down projections carry their bias *and*
the residual add in the GEMM epilogue — the native vision form,
kernel for kernel. The activation quantizer sites are calibrated on
the probe run against the pristine host.

The attention element is a ladder: the house FA4 entry probed at the
bound head shape first, plain SDPA as the floor — SDPA is a single
capture-safe primitive and this tower's attention is a small slice
of its time; the GEMM epilogues are where the native form wins.
"""

from __future__ import annotations

import types
from typing import Any, Callable

import torch

from .. import KernelUnavailable, hub_kernel
from ...guard import GuardedSeam
from ..chain_elements import (
    fp8_weight as _fp8_weight, gelu_tanh_like as _gelu_tanh_like)

GEMM_PACKAGE = "flashrt/fp8-gemm"
FUSE_PACKAGE = "flashrt/transformer-fused-ops"
GEMM_SYMBOLS = ("fp8_linear_bias_bf16", "fp8_linear_bias_residual_bf16",
                "fp8_linear_bias_gelu_bf16")
FP4_GEMM_PACKAGE = "flashrt/fp4-gemm"
FP4_FUSE_PACKAGE = "flashrt/fp4-fused-ops"

#: the band table IS the recipe (house convention). The ``fp4`` row is
#: the native SigLIP preset: the FFN pair rides NVFP4 — the LN
#: producer emits packed FP4, FC1 fuses bias+GELU and emits packed FP4
#: straight into FC2's residual GEMM. The attention half stays FP8.
BANDS: dict[str, dict] = {
    "fp8": {"packages": (), "precision_rank": 0},
    "fp4": {"packages": (
        (FP4_GEMM_PACKAGE, ("nvfp4_gemm_bias_bf16",
                            "nvfp4_gemm_bias_residual_bf16",
                            "quantize_fp4_sfa_bf16",
                            "pack_nvfp4_weight_bf16")),
        (FP4_FUSE_PACKAGE, ("layer_norm_nvfp4_bf16",))),
        "precision_rank": 1, "awq": 0.8},
}

FUSE_SYMBOLS = ("layer_norm_quant_fp8_static_bf16",
                "quantize_fp8_static_bf16")
FA4_REPO = "flashrt/fa4-cute-runtime"

SMOKE_FLOOR = 0.97
FP8_MAX = 448.0


def missing_symbols(band: str = "fp8") -> list[str]:
    gaps: list[str] = []
    for repo, symbols in ((GEMM_PACKAGE, GEMM_SYMBOLS),
                          (FUSE_PACKAGE, FUSE_SYMBOLS)
                          ) + BANDS[band]["packages"]:
        try:
            kern = hub_kernel(repo, ">=1")
        except KernelUnavailable:
            gaps.append(repo)
            continue
        gaps.extend(f"{repo}:{s}" for s in symbols
                    if not hasattr(kern, s))
    return gaps


class BoundVisionFp8Chain(GuardedSeam, torch.nn.Module):
    """Bind-time state: FP8 weights with epilogue biases, buffers."""

    _frt_can_fallback = False

    def __init__(self) -> None:
        super().__init__()
        self.table: list[dict] = []
        self.dims: dict = {}
        self.buf: dict = {}
        self.scaling = 1.0
        self.out_ctor = None
        self.out_dtype = None
        self.kernels: dict = {}
        self.band = "fp8"


def _stack_parts(stack):
    layers = list(stack.layers)
    attn = layers[0].self_attn
    dim = attn.q_proj.in_features
    heads = getattr(attn, "num_heads", None)
    if not isinstance(heads, int):
        head_dim = getattr(attn, "head_dim", None)
        if not isinstance(head_dim, int):
            raise ValueError("attention exposes neither num_heads "
                             "nor head_dim")
        heads = dim // head_dim
    hidden = layers[0].mlp.fc1.out_features
    return layers, heads, dim // heads, dim, hidden


def _awq_scale(chan_amax: torch.Tensor, alpha: float) -> torch.Tensor:
    """Native per-input-channel AWQ pre-scale, verbatim:
    s = (a / a.mean())^alpha clamped to [0.25, 4]."""
    a = chan_amax.float().clamp(min=1e-6)
    return (a / a.mean()).pow(alpha).clamp(min=0.25, max=4.0)


@torch.no_grad()
def _quantize(bound, layers, amax, chan=None) -> None:
    alpha = BANDS[bound.band].get("awq", 0.0)
    for i, ly in enumerate(layers):
        attn, mlp = ly.self_attn, ly.mlp
        a_qkv, a_o, a_fc1, a_fc2 = (amax[(i, s)] / FP8_MAX for s in
                                    ("qkv", "o", "fc1", "fc2"))
        qkv_w = torch.cat([attn.q_proj.weight, attn.k_proj.weight,
                           attn.v_proj.weight], dim=0)
        qkv_b = torch.cat([attn.q_proj.bias, attn.k_proj.bias,
                           attn.v_proj.bias], dim=0)
        entry: dict[str, Any] = {}
        fp4 = BANDS[bound.band]["packages"] != ()
        pack4 = (bound.kernels["kg4"].pack_nvfp4_weight_bf16
                 if fp4 else None)
        for name, w, bias, act in (
                ("qkv", qkv_w, None, a_qkv),
                ("o", attn.out_proj.weight, None, a_o),
                ("fc1", mlp.fc1.weight, mlp.fc1.bias, a_fc1),
                ("fc2", mlp.fc2.weight, mlp.fc2.bias, a_fc2)):
            if fp4 and name in ("fc1", "fc2"):
                # the padded pack carries SigLIP's logical 4304 as the
                # physical aligned width; FC1's FP4 output is born at
                # that width, so FC2 consumes it with zero glue. The
                # native tier rides AWQ on the up (FC1) weight only:
                # s into the columns, 1/s carried by the LN producer.
                w = w.detach().to("cuda", torch.float32)
                if (alpha and name == "fc1" and chan is not None
                        and (i, "fc1") in chan):
                    s = _awq_scale(chan[(i, "fc1")].to("cuda"), alpha)
                    w = w * s[None, :]
                    entry["inv_s_fc1"] = (1.0 / s).to(
                        torch.bfloat16).contiguous()
                wp, wsf, pb, _ = pack4(
                    w.to(torch.bfloat16).contiguous(),
                    bias.detach().to("cuda", torch.bfloat16)
                    .contiguous(), mse=True)
                entry[name] = (wp, wsf)
                entry[f"{name}_b"] = pb
                continue
            packed, w_scale = _fp8_weight(w)
            entry[name] = packed
            entry[f"a_{name}"] = act * w_scale
        for name, b in (("qkv_b", qkv_b), ("o_b", attn.out_proj.bias),
                        ("fc1_b", mlp.fc1.bias), ("fc2_b", mlp.fc2.bias)):
            if name in entry:
                continue
            entry[name] = b.detach().to("cuda", torch.bfloat16)
        for name, norm in (("ln1", ly.layer_norm1),
                           ("ln2", ly.layer_norm2)):
            entry[f"{name}_w"] = norm.weight.detach().to(
                "cuda", torch.bfloat16)
            entry[f"{name}_b"] = norm.bias.detach().to(
                "cuda", torch.bfloat16)
            entry[f"{name}_eps"] = float(getattr(norm, "eps", 1e-6))
        entry["sc_qkv"] = torch.tensor([a_qkv], device="cuda",
                                       dtype=torch.float32)
        entry["sc_o"] = torch.tensor([a_o], device="cuda",
                                     dtype=torch.float32)
        entry["sc_fc1"] = torch.tensor([a_fc1], device="cuda",
                                       dtype=torch.float32)
        entry["sc_fc2"] = torch.tensor([a_fc2], device="cuda",
                                       dtype=torch.float32)
        bound.table.append(entry)


def _make_attend(bound, mode: str, kern):
    nh, hd = bound.dims["nh"], bound.dims["hd"]
    scaling = bound.scaling
    if mode == "fa4_cute":
        def attend(q, k, v):
            out = torch.empty_like(q)
            kern.forward_static(q, k, v, out, softmax_scale=scaling,
                                causal=False)
            return out
        return attend

    def attend(q, k, v):
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            scale=scaling)
        return o.transpose(1, 2)
    return attend


def _make_run(bound: BoundVisionFp8Chain):
    kg = bound.kernels["kg"]
    kf = bound.kernels["kf"]
    attend = bound.kernels["attend"]
    B, S, nh, hd, D, H = (bound.dims[k] for k in
                          ("batch", "seq", "nh", "hd", "dim",
                           "hidden"))
    b = bound.buf
    table = bound.table
    rf = torch.profiler.record_function

    if BANDS[bound.band]["packages"] != ():
        kg4 = bound.kernels["kg4"]
        kf4 = bound.kernels["kf4"]
        ln4 = kf4.layer_norm_nvfp4_bf16
        gemm_res4 = kg4.nvfp4_gemm_bias_residual_bf16
        quant4 = kg4.quantize_fp4_sfa_bf16
        xp4, xsf4 = b["xp4"], b["xsf4"]
        hp4, hsf4 = b["hp4"], b["hsf4"]
        e0 = table[0]
        # FC1 element ladder, bind-time facts (shape and device, never
        # a device list): the fused FP4-out epilogue, then the fused
        # bf16-GELU epilogue, then the plain bias GEMM with the GELU
        # and quantize as separate elements.
        def _try(fn, *args, **kw):
            try:
                fn(*args, **kw)
                torch.cuda.synchronize()
                return fn
            except Exception:   # noqa: BLE001 — next rung
                return None

        fused4 = getattr(kg4, "nvfp4_gemm_bias_gelu_nvfp4", None)
        if fused4 is not None:
            fused4 = _try(fused4, xp4, e0["fc1"][0], xsf4,
                          e0["fc1"][1], e0["fc1_b"],
                          out_packed=hp4, out_sfa=hsf4)
        fusedb = getattr(kg4, "nvfp4_gemm_bias_gelu_bf16", None)
        if fused4 is None and fusedb is not None:
            fusedb = _try(fusedb, xp4, e0["fc1"][0], xsf4,
                          e0["fc1"][1], e0["fc1_b"], out=b["hid"])
        else:
            fusedb = None

        if fused4 is not None:
            def ffn(res, e):
                ln4(res, e["ln2_w"], e["ln2_b"], e.get("inv_s_fc1"),
                    e["ln2_eps"], packed=xp4, sfa=xsf4)
                fused4(xp4, e["fc1"][0], xsf4, e["fc1"][1],
                       e["fc1_b"], out_packed=hp4, out_sfa=hsf4)
                gemm_res4(hp4, e["fc2"][0], hsf4, e["fc2"][1],
                          e["fc2_b"], res, out=res)
        elif fusedb is not None:
            def ffn(res, e):
                ln4(res, e["ln2_w"], e["ln2_b"], e.get("inv_s_fc1"),
                    e["ln2_eps"], packed=xp4, sfa=xsf4)
                fusedb(xp4, e["fc1"][0], xsf4, e["fc1"][1],
                       e["fc1_b"], out=b["hid"])
                quant4(b["hid"], hp4, hsf4)
                gemm_res4(hp4, e["fc2"][0], hsf4, e["fc2"][1],
                          e["fc2_b"], res, out=res)
        else:
            gemm_bias4 = kg4.nvfp4_gemm_bias_bf16
            gelu = torch.nn.functional.gelu

            def ffn(res, e):
                ln4(res, e["ln2_w"], e["ln2_b"], e.get("inv_s_fc1"),
                    e["ln2_eps"], packed=xp4, sfa=xsf4)
                gemm_bias4(xp4, e["fc1"][0], xsf4, e["fc1"][1],
                           e["fc1_b"], out=b["hid"])
                quant4(gelu(b["hid"], approximate="tanh"), hp4, hsf4)
                gemm_res4(hp4, e["fc2"][0], hsf4, e["fc2"][1],
                          e["fc2_b"], res, out=res)
    else:
        def ffn(res, e):
            kf.layer_norm_quant_fp8_static_bf16(
                res, e["ln2_w"], e["ln2_b"], e["sc_fc1"],
                eps=e["ln2_eps"], out=b["xn8"])
            kg.fp8_linear_bias_gelu_bf16(
                b["xn8"], e["fc1"], e["fc1_b"], alpha=e["a_fc1"],
                out=b["hid"])
            kf.quantize_fp8_static_bf16(b["hid"], e["sc_fc2"],
                                        out=b["h8"])
            kg.fp8_linear_bias_residual_bf16(
                b["h8"], e["fc2"], e["fc2_b"], res,
                alpha=e["a_fc2"])

    def run(x3d):
        res = b["res"]
        res.copy_(x3d.reshape(B * S, D))
        for e in table:
            with rf("vi:qkv"):
                kf.layer_norm_quant_fp8_static_bf16(
                    res, e["ln1_w"], e["ln1_b"], e["sc_qkv"],
                    eps=e["ln1_eps"], out=b["xn8"])
                kg.fp8_linear_bias_bf16(b["xn8"], e["qkv"], e["qkv_b"],
                                        alpha=e["a_qkv"], out=b["qkv"])
            with rf("vi:attn"):
                q = b["qkv"][:, :D].view(B, S, nh, hd)
                k = b["qkv"][:, D:2 * D].view(B, S, nh, hd)
                v = b["qkv"][:, 2 * D:].view(B, S, nh, hd)
                att = attend(q, k, v)
            with rf("vi:o"):
                kf.quantize_fp8_static_bf16(
                    att.reshape(B * S, D), e["sc_o"], out=b["o8"])
                kg.fp8_linear_bias_residual_bf16(
                    b["o8"], e["o"], e["o_b"], res, alpha=e["a_o"])
            with rf("vi:ffn"):
                ffn(res, e)
        return bound.out_ctor(
            last_hidden_state=res.view(B, S, D)
            .to(bound.out_dtype).clone())

    return run


def bind_vision_fp8_chain(model, root: str,
                          probe: Callable[[], Any],
                          band: str = "fp8") -> dict:
    """Bind the chain onto the tower at ``root``; adapter contract out."""
    try:
        kg = hub_kernel(GEMM_PACKAGE, ">=1")
        kf = hub_kernel(FUSE_PACKAGE, ">=1")
        kg4 = kf4 = None
        if BANDS[band]["packages"]:
            kg4 = hub_kernel(FP4_GEMM_PACKAGE, ">=1")
            kf4 = hub_kernel(FP4_FUSE_PACKAGE, ">=1")
    except KernelUnavailable as exc:
        return {"refused": f"vision_{band}_chain: {exc}"}
    gaps = missing_symbols(band=band)
    if gaps:
        return {"refused": f"vision_fp8_chain missing: "
                           f"{', '.join(gaps)}"}

    stack = model.get_submodule(root) if root else model
    layers, nh, hd, dim, hidden = _stack_parts(stack)
    if not _gelu_tanh_like(layers[0].mlp.activation_fn
                           if hasattr(layers[0].mlp, "activation_fn")
                           else layers[0].mlp.act_fn):
        return {"refused": "vision_fp8_chain: MLP activation is not "
                           "tanh-GELU"}
    scale_attr = getattr(layers[0].self_attn, "scale", None)
    scaling = float(scale_attr) if scale_attr else hd ** -0.5

    bound = BoundVisionFp8Chain()
    bound.kernels = {"kg": kg, "kf": kf, "kg4": kg4, "kf4": kf4}
    bound.band = band
    bound.scaling = scaling
    bound.dims = {"nh": nh, "hd": hd, "dim": dim, "hidden": hidden,
                  "layers": len(layers)}

    calls: list[dict] = []
    amax: dict = {}
    chan: dict = {}

    def note(site):
        def hook(_m, args):
            peak = float(args[0].detach().abs().amax())
            amax[site] = max(amax.get(site, 0.0), peak)
        return hook

    def cnote(site):
        # per-input-channel amax at the FC1 input — the native SigLIP
        # AWQ statistic (collected on the LN output the producer emits)
        def hook(_m, args):
            v = args[0].detach().float().abs()
            v = v.reshape(-1, v.shape[-1]).amax(0)
            prev = chan.get(site)
            chan[site] = v if prev is None else torch.maximum(prev, v)
        return hook

    hooks = []
    for i, ly in enumerate(layers):
        hooks.append(ly.self_attn.q_proj.register_forward_pre_hook(
            note((i, "qkv"))))
        hooks.append(ly.self_attn.out_proj.register_forward_pre_hook(
            note((i, "o"))))
        if BANDS[band]["packages"] and BANDS[band].get("awq"):
            hooks.append(ly.mlp.fc1.register_forward_pre_hook(
                cnote((i, "fc1"))))
        hooks.append(ly.mlp.fc1.register_forward_pre_hook(
            note((i, "fc1"))))
        hooks.append(ly.mlp.fc2.register_forward_pre_hook(
            note((i, "fc2"))))

    saved_probe = stack.__dict__.get("forward")
    host_forward = stack.forward

    def capturing(_self, *args, **kwargs):
        out = host_forward(*args, **kwargs)
        embs = kwargs.get("inputs_embeds",
                          args[0] if args else None)
        hidden_out = getattr(out, "last_hidden_state", None)
        if (embs is not None and hidden_out is not None
                and embs.dim() == 3
                and kwargs.get("attention_mask") is None):
            calls.append({"x": embs.detach().clone(),
                          "out": hidden_out.detach().clone(),
                          "out_type": type(out)})
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
        return {"refused": "vision_fp8_chain: probe never made an "
                           "unmasked encoder call"}
    first = calls[0]
    if any(tuple(c["x"].shape) != tuple(first["x"].shape)
           for c in calls[1:]):
        return {"refused": "vision_fp8_chain: probe calls disagree "
                           "on shape"}
    if any((i, s) not in amax or amax[(i, s)] <= 0.0
           for i in range(len(layers))
           for s in ("qkv", "o", "fc1", "fc2")):
        return {"refused": "vision_fp8_chain: calibration saw a dead "
                           "quantizer site"}

    B, S, _ = first["x"].shape
    bound.dims["batch"], bound.dims["seq"] = B, S
    bound.out_ctor = first["out_type"]
    bound.out_dtype = first["out"].dtype
    _quantize(bound, layers, amax, chan)
    dev, bf = "cuda", torch.bfloat16
    b = bound.buf
    b["res"] = torch.empty(B * S, dim, device=dev, dtype=bf)
    b["xn8"] = torch.empty(B * S, dim, device=dev,
                           dtype=torch.float8_e4m3fn)
    b["qkv"] = torch.empty(B * S, 3 * dim, device=dev, dtype=bf)
    b["o8"] = torch.empty(B * S, dim, device=dev,
                          dtype=torch.float8_e4m3fn)
    if BANDS[bound.band]["packages"] != ():
        kg4 = bound.kernels["kg4"]
        aligned = getattr(kg4, "aligned_fp4_dim",
                          lambda d, alignment=32: -(-d // 32) * 32)
        hidden_p = int(aligned(hidden))
        b["hid"] = torch.empty(B * S, hidden_p, device=dev, dtype=bf)
        quant4 = kg4.quantize_fp4_sfa_bf16
        b["xp4"], b["xsf4"] = quant4(
            torch.zeros(B * S, dim, device=dev, dtype=bf))
        b["hp4"], b["hsf4"] = quant4(
            torch.zeros(B * S, hidden_p, device=dev, dtype=bf))
    else:
        b["hid"] = torch.empty(B * S, hidden, device=dev, dtype=bf)
        b["h8"] = torch.empty(B * S, hidden, device=dev,
                              dtype=torch.float8_e4m3fn)

    attend, attn_mode = None, None
    try:
        ka = hub_kernel(FA4_REPO, ">=1")
        cand = _make_attend(bound, "fa4_cute", ka)
        cand(torch.zeros(B, S, nh, hd, device=dev, dtype=bf),
             torch.zeros(B, S, nh, hd, device=dev, dtype=bf),
             torch.zeros(B, S, nh, hd, device=dev, dtype=bf))
        torch.cuda.synchronize()
        attend, attn_mode = cand, "fa4_cute"
    except Exception:       # noqa: BLE001 — the floor rung serves
        attend, attn_mode = _make_attend(bound, "sdpa", None), "sdpa"
    bound.kernels["attend"] = attend

    run = _make_run(bound)
    guard = bound._frt_arm(dtypes=(torch.bfloat16,),
                           device=torch.device("cuda"))
    guard.notes["n_layers"] = len(layers)
    guard.notes["attention"] = attn_mode

    worst = None
    with torch.inference_mode():
        for c in calls:
            got = run(c["x"].to(torch.bfloat16))
            cos = torch.nn.functional.cosine_similarity(
                got.last_hidden_state.float().flatten(),
                c["out"].float().flatten(), dim=0)
            worst = float(cos) if worst is None else min(worst,
                                                         float(cos))
    if worst is None or worst < SMOKE_FLOOR:
        return {"refused": f"vision_fp8_chain smoke cos {worst} < "
                           f"{SMOKE_FLOOR} across {len(calls)} "
                           "probe call(s)"}
    guard.notes["smoke_cos"] = round(worst, 6)

    saved = stack.__dict__.get("forward")
    x_shape = tuple(first["x"].shape)

    def routed(_self, *args, **kwargs):
        compiling = torch.compiler.is_compiling()
        capturing_now = (False if compiling
                         else torch.cuda.is_current_stream_capturing())
        eager = not compiling and not capturing_now
        if eager:
            guard.calls += 1
        embs = kwargs.get("inputs_embeds",
                          args[0] if args else None)
        ok = (embs is not None
              and kwargs.get("attention_mask") is None
              and tuple(embs.shape) == x_shape)
        if not ok:
            if not eager:
                raise RuntimeError(
                    "vision_fp8_chain: out-of-contract call during "
                    "capture/compile — fix the eager path first")
            guard.fallbacks += 1
            guard.last_reason = "call outside the routed contract"
            return host_forward(*args, **kwargs)
        return run(embs.to(torch.bfloat16))

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
        "observed": {f"{root}::vision_fp8_chain": bound},
        "revert": [revert],
        "toggle": (enable, disable),
        "smoke_cos": worst,
    }

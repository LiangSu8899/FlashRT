"""Auto-assembly: discover seams, calibrate them in one pass, bind them.

This is the distribution layer. Given a host model and a way to run it,
it finds every structure seam (:mod:`.discover`), captures exactly the
calibration each one needs in a single forward pass, and binds each
through its library impl. The caller gets a ``path -> module`` swap map
and any outside-cadence update functions — the same thing the hand
recipes produced, derived from the model object rather than written by
hand. A host integrates by importing and calling; it writes no
per-seam scaffolding.

The calibration each structure needs, captured structure-aware:
  linear_proj / qkv_pack : the shared input the projection(s) see, and
                           its per-tensor amax (the static act scale)
  adaln_producer         : the (cond, style) pairs the conditioning
                           projection emits across the tick, for the
                           step table and its fingerprint locator
  decoder_ffn / vision_ffn : the normed input the MLP sees

Seam negotiation is resolved here: when an adaln_producer feeds a
sibling qkv_pack under the same parent, the producer emits fp8 and the
pack takes the shared act scale, skipping its own input quantization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .discover import (Seam, _resolve, discover, group_families,
                       seam_weights)

_FP8 = torch.float8_e4m3fn
_FP8_CHAIN_MAX_ROWS = 256  # fp8 producer chain qualifies at denoise
                          # M (bandwidth-bound); large-M prefill skips

# Host-family attention adapters. Attention seams are not a static
# module pattern — where the attention math actually runs is
# host-specific (a function in one host, a processor in another), so
# auto-discovery of the attention_core structure is delegated to
# registered adapters. Each adapter, given the model and a way to run
# it, returns (swaps, update) or None (this host is not its family).
_ATTENTION_ADAPTERS: list = []


def register_attention_adapter(adapter) -> None:
    """Register a host-family attention adapter (callable)."""
    _ATTENTION_ADAPTERS.append(adapter)


@dataclass
class AutoPlan:
    """Discovered + calibrated swaps, ready to stage."""

    swaps: dict[str, torch.nn.Module] = field(default_factory=dict)
    updates: list[Callable[[], None]] = field(default_factory=list)
    seams: list[Seam] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


def _amax(tensors) -> float:
    return max(t.float().abs().max().item() for t in tensors)


def _rows(t: torch.Tensor) -> int:
    return int(t.reshape(-1, t.shape[-1]).shape[0])


def auto_swaps(
    model: torch.nn.Module,
    forward: Callable[[], Any],
    *,
    structures: tuple[str, ...] = ("decoder_ffn", "vision_ffn",
                                   "qkv_pack", "adaln_producer",
                                   "linear_proj", "norm_fused"),
    negotiate_fp8: bool = True,
    frames: int = 1,
    verbose: bool = False,
) -> AutoPlan:
    """Discover, calibrate in one pass, and bind every applicable seam."""

    def say(msg: str) -> None:
        if verbose:
            print(f"[autobuild] {msg}", flush=True)

    seams = discover(model, structures)
    # a packed group owns its q/k/v; linear_proj keeps only what the
    # pack does not take (the output projection), so the two structures
    # compose instead of fighting over the same module
    packed = {s.path + "." + a for s in seams
              if s.structure == "qkv_pack" for a in (s.pack_attrs or ())}
    seams = [s for s in seams
             if not (s.structure == "linear_proj" and s.path in packed)]
    say(f"discovered {len(seams)} seam(s)")
    if not seams:
        return AutoPlan()

    # ---- one calibration pass, structure-aware capture ----
    caps: dict[str, dict[str, Any]] = {}
    hooks = []

    def cap_input(path):
        def hook(module, args, kwargs, out):
            x = args[0] if args else next(iter(kwargs.values()))
            caps[path].setdefault("x", []).append(x.detach())
            caps[path]["rows"] = _rows(x)
            return None
        return hook

    def cap_cond(path):
        def hook(module, args, out):
            caps[path].setdefault("pairs", []).append(
                (args[0].detach().clone(), out.detach().clone()))
            return None
        return hook

    for seam in seams:
        caps[seam.path] = {}
        target = _resolve(model, seam.path)
        if seam.structure == "adaln_producer":
            # the norm's own input gives rows; the cond projection gives
            # the (cond, style) pairs the table is built from
            hooks.append(target.register_forward_hook(
                cap_input(seam.path), with_kwargs=True))
            hooks.append(getattr(target, seam.cond_attr)
                         .register_forward_hook(cap_cond(seam.path)))
        elif seam.structure == "qkv_pack":
            first = getattr(target, seam.pack_attrs[0])
            hooks.append(first.register_forward_hook(
                cap_input(seam.path), with_kwargs=True))
        else:
            hooks.append(target.register_forward_hook(
                cap_input(seam.path), with_kwargs=True))

    with torch.no_grad():
        for _ in range(max(1, frames)):
            forward()
    for h in hooks:
        h.remove()
    say("calibration pass done")

    # ---- fp8 seam negotiation: the load-bearing structure combination.
    # A single kernel need not win alone (fp8 qkv at M=50 is marginal,
    # fa2 in a bf16 stack loses); the *chain* wins — an adaln producer
    # that emits fp8 lets the qkv pack skip its own input quantization
    # and hands a clean fp8 seam down to the attention core. Bind the
    # producer→pack pair together with one shared act scale wherever a
    # producer feeds a pack under the same parent layer. ----
    act_scales: dict[str, torch.Tensor] = {}
    negotiated: dict[str, dict[str, Seam]] = {}
    if negotiate_fp8:
        by_parent: dict[str, dict[str, Seam]] = {}
        for seam in seams:
            layer = _layer_of(seam.path)
            if seam.structure == "adaln_producer" and _feeds_attention(
                    seam.path):
                by_parent.setdefault(layer, {})["producer"] = seam
            elif seam.structure == "qkv_pack":
                by_parent.setdefault(layer, {})["pack"] = seam
        # the chain wins at small M (denoise): fp8 is bandwidth-bound and
        # pays there, while a large-M prefill GEMM is compute-bound and
        # fp8 buys little — and an fp8 producer feeding a big compiled
        # prefill region is where the triton fp8 codegen chokes. Qualify
        # on the calibrated row count, not on host names.
        for lay, g in by_parent.items():
            if "producer" not in g or "pack" not in g:
                continue
            pack_cap = caps.get(g["pack"].path, {})
            rows = pack_cap.get("rows", 1 << 30)
            if pack_cap.get("x") and rows <= _FP8_CHAIN_MAX_ROWS:
                # the pack's input == the producer's output; its amax is
                # the one static scale both sides share
                negotiated[lay] = g
                act_scales[lay] = torch.tensor(
                    [max(_amax(pack_cap["x"]) / 448.0, 1e-8)],
                    device=next(model.parameters()).device)

    # ---- the negotiated chain binds as one unit ----
    # producer and consumer must agree on the seam dtype: a pack bound
    # for fp8 input whose producer failed to bind would be handed BF16,
    # and the host would silently grow a quantize fused into whatever
    # produced it. Bind the pair together, or leave both on BF16.
    plan = AutoPlan(seams=seams)
    handled: set[str] = set()
    for lay, g in negotiated.items():
        p_seam, k_seam = g["producer"], g["pack"]
        p_cap, k_cap = caps.get(p_seam.path, {}), caps.get(k_seam.path, {})
        if not (p_cap.get("pairs") and k_cap.get("x")):
            continue
        try:
            pair = _bind_negotiated(model, p_seam, k_seam, p_cap, k_cap,
                                    act_scales[lay], plan)
        except (ValueError, RuntimeError) as refusal:
            plan.notes.setdefault("refused", []).append(
                (lay + " [chain]", str(refusal)[:80]))
            continue
        plan.swaps.update(pair)
        handled.update({p_seam.path, k_seam.path})
    plan.notes["negotiated_layers"] = sorted(
        lay for lay, g in negotiated.items()
        if g["pack"].path in handled)

    # ---- bind the remaining seams individually ----
    for name, members in group_families(seams).items():
        for seam in members:
            if seam.path in handled:
                continue
            cap = caps.get(seam.path, {})
            try:
                bound = _bind_auto(model, seam, cap, plan, act_scales,
                                   negotiate_fp8)
            except (ValueError, RuntimeError) as refusal:
                plan.notes.setdefault("refused", []).append(
                    (seam.path, str(refusal)[:80]))
                continue
            if bound is None:
                continue
            if isinstance(bound, dict):
                plan.swaps.update(bound)
            else:
                plan.swaps[seam.path] = bound
    # ---- attention_core: host-family adapters (fa2 seam) ----
    if "attention_core" in structures:
        from . import adapters as _adapters  # noqa: F401 (registers)
        for adapter in _ATTENTION_ADAPTERS:
            try:
                result = adapter(model, forward)
            except (ValueError, RuntimeError) as refusal:
                plan.notes.setdefault("refused", []).append(
                    ("attention_core", str(refusal)[:80]))
                continue
            if result is None:
                continue
            att_swaps, update = result
            plan.swaps.update(att_swaps)
            if update is not None:
                plan.updates.append(update)
            plan.notes["attention_adapter"] = type(adapter).__name__ \
                if hasattr(adapter, "__name__") else str(adapter)
            break

    say(f"bound {len(plan.swaps)} seam(s), "
        f"{len(plan.notes.get('refused', []))} refused")
    return plan


def _bind_auto(model, seam, cap, plan, act_scales, negotiate_fp8):
    """Route one seam to its impl with the captured calibration."""
    from .handle import get

    if seam.structure in ("decoder_ffn", "vision_ffn"):
        if not cap.get("x"):
            return None
        h = get(seam.structure)
        return h.bind(_resolve(model, seam.path), cap["x"], gate_cos=0.0)

    if seam.structure == "norm_fused":
        from .impls.norm_fused import bind_norm_fused
        return bind_norm_fused(_resolve(model, seam.path),
                               calibration=cap.get("x"))

    if seam.structure == "linear_proj":
        if not cap.get("x"):
            return None
        from .impls.linear_proj import fp8_static as proj_impl
        return proj_impl.bind_proj_seam(
            seam_weights(model, seam), calibration=cap["x"],
            original=_resolve(model, seam.path))

    if seam.structure == "qkv_pack":
        from .impls.qkv_pack import bind_attn_block, bind_qkv_pack
        if not cap.get("x"):
            return None
        block = _resolve(model, seam.path)
        act_scale = torch.tensor(
            [max(_amax(cap["x"]) / 448.0, 1e-8)],
            device=getattr(block, seam.pack_attrs[0]).weight.device)
        if seam.variant.get("bind") == "module":
            # the whole block: packed projections *and* the attention
            # compute dtype (hosts that run SDPA in fp32 pay for it)
            return {seam.path: bind_attn_block(
                block, act_scale, rows=cap["rows"],
                sdpa_dtype=torch.bfloat16)}
        mods = [getattr(block, a) for a in seam.pack_attrs]
        parts = bind_qkv_pack(mods, act_scale, rows=cap["rows"],
                              in_dtype="bf16_fused_quant")
        return {seam.path + "." + a: m
                for a, m in zip(seam.pack_attrs, parts)}

    if seam.structure == "adaln_producer":
        from .impls.adaln_producer import (bind_adaln_producer,
                                           bind_style_table)
        if not cap.get("pairs"):
            return None
        norm = _resolve(model, seam.path)
        proj = getattr(norm, seam.cond_attr)
        loc = plan.notes.setdefault("_locators", {}).get(seam.family)
        table = bind_style_table(proj, cap["pairs"], locator=loc)
        plan.notes["_locators"][seam.family] = table.locator
        return {seam.path + "." + seam.cond_attr: table}

    return None


class _Eager(torch.nn.Module):
    """Wrap a module so its forward runs outside the compiled region.

    An fp8-emitting seam's arithmetic, if traced by inductor, gets fused
    into fp8 math (illegal on sm120 triton) — and the quantize even
    reaches back across the boundary, so inductor casts the host's own
    gated residual to fp8 to feed it. The hand recipes never hit this
    because the whole denoise block froze to eager. A swapped-in module
    does not inherit that freezing, so fp8 seams declare it. Overriding
    the instance ``forward`` is not enough (dynamo inlines the class
    forward); the disable must sit on a class method, which is what this
    wrapper provides. The kernels are opaque either way, so eager here
    is a graph break, not real work.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    @torch._dynamo.disable
    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)


def _eager(module):
    return _Eager(module)


def _bind_negotiated(model, p_seam, k_seam, p_cap, k_cap, scale, plan):
    """Bind an fp8 producer and the pack it feeds as one chain.

    This is the combination the structure layer exists for: neither half
    is worth much alone (a small-M fp8 projection barely beats BF16, a
    producer that only reshapes styles saves nothing), but together the
    producer's fused quantize removes the consumer's input quantization
    entirely and hands a clean fp8 seam downstream.
    """
    from .impls.adaln_producer import bind_adaln_producer
    from .impls.qkv_pack import bind_qkv_pack

    norm = _resolve(model, p_seam.path)
    loc = plan.notes.setdefault("_locators", {}).get(p_seam.family)
    style_width = p_cap["pairs"][0][1].shape[-1]
    prod = bind_adaln_producer(
        norm, p_cap["pairs"], act_scale=scale,
        rows=p_cap.get("rows") or k_cap["rows"],
        dim=getattr(norm, "dim", style_width // 3),
        locator=loc, norm="rms")
    plan.notes["_locators"][p_seam.family] = prod.locator

    block = _resolve(model, k_seam.path)
    mods = [getattr(block, a) for a in k_seam.pack_attrs]
    parts = bind_qkv_pack(mods, scale, rows=k_cap["rows"],
                          in_dtype="fp8_static")
    swaps = {p_seam.path: prod}
    swaps.update({k_seam.path + "." + a: m
                  for a, m in zip(k_seam.pack_attrs, parts)})
    return swaps


def _layer_of(path: str) -> str:
    """The parent layer key: a.layers.7.self_attn -> a.layers.7."""
    import re
    m = re.search(r"(.*\.layers\.\d+)\.", path)
    return m.group(1) if m else path.rsplit(".", 1)[0]


def _feeds_attention(path: str) -> bool:
    """An adaln producer that feeds attention (input_layernorm) rather
    than the MLP (post_attention_layernorm)."""
    leaf = path.rsplit(".", 1)[-1]
    return "input" in leaf or leaf in ("norm1", "ln1")

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
                                   "qkv_pack", "adaln_producer"),
    negotiate_fp8: bool = True,
    frames: int = 1,
    verbose: bool = False,
) -> AutoPlan:
    """Discover, calibrate in one pass, and bind every applicable seam."""

    def say(msg: str) -> None:
        if verbose:
            print(f"[autobuild] {msg}", flush=True)

    seams = discover(model, structures)
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

    # ---- bind each seam via its impl (act_scales carries the shared
    # producer/consumer scale for the fp8 seam negotiation) ----
    act_scales: dict[str, torch.Tensor] = {}
    plan = AutoPlan(seams=seams)
    for name, members in group_families(seams).items():
        for seam in members:
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

    if seam.structure == "linear_proj":
        if not cap.get("x"):
            return None
        from .impls.linear_proj import fp8_static as proj_impl
        return proj_impl.bind_proj_seam(
            seam_weights(model, seam), calibration=cap["x"],
            original=_resolve(model, seam.path))

    if seam.structure == "qkv_pack":
        from .impls.qkv_pack import bind_qkv_pack
        if not cap.get("x"):
            return None
        parent = _resolve(model, seam.path)
        mods = [getattr(parent, a) for a in seam.pack_attrs]
        act_scale = torch.tensor(
            [max(_amax(cap["x"]) / 448.0, 1e-8)],
            device=mods[0].weight.device)
        act_scales[seam.path] = act_scale
        parts = bind_qkv_pack(mods, act_scale, rows=cap["rows"],
                              in_dtype="bf16_fused_quant")
        return {seam.path + "." + a: m
                for a, m in zip(seam.pack_attrs, parts)}

    if seam.structure == "adaln_producer":
        from .impls.adaln_producer import bind_style_table
        if not cap.get("pairs"):
            return None
        norm = _resolve(model, seam.path)
        proj = getattr(norm, seam.cond_attr)
        loc = plan.notes.setdefault("_locators", {}).get(seam.family)
        table = bind_style_table(proj, cap["pairs"], locator=loc)
        plan.notes["_locators"][seam.family] = table.locator
        return {seam.path + "." + seam.cond_attr: table}

    return None

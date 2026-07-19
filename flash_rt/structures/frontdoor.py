"""One-call front door: ``structures.attach(model, forward, ...)``.

The consumption contract mirrors ``kernels.get_kernel``: one import, one
call. Everything the structure layer needs — seam discovery, real
distribution calibration, per-layer parity gates, family-level accuracy
and net-win gates, transactional swap, receipt — runs inside the call.
An attachment that does not both stay accurate and win latency is
refused; the model is left untouched and the refusal is reported, never
silently absorbed.

Calibration entry points (``calibration=``):
  "auto"          capture activations by running ``forward`` under hooks
  "<path>.pt"     load a previously captured calibration file if it
                  exists, otherwise capture then save there
Pass ``forward`` as one callable (optionally repeated with ``frames=N``,
e.g. a closure advancing a dataloader) or a sequence of callables, one
per calibration frame.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch

from .swap import AttachHandle as _AttachHandle, attach as _swap_attach
from .discover import Seam, discover, group_families, seam_weights
from .gates import parity_metrics


def _bind_seam(seam: Seam, weights, calib, original):
    if seam.structure == "decoder_ffn":
        from .impls.decoder_ffn import fp8_static as impl
    elif seam.structure == "vision_ffn":
        from .impls.vision_ffn import fp8_static as impl
    elif seam.structure == "linear_proj":
        from .impls.linear_proj import fp8_static as proj_impl
        return proj_impl.bind_proj_seam(weights, calibration=calib,
                                        original=original)
    else:
        raise ValueError(f"no auto impl for structure {seam.structure!r}")
    return impl.bind_mlp_seam(weights, variant=seam.variant,
                              calibration_normed=calib, original=original)


def _cuda_time_ms(fn: Callable[[], Any], warmup: int = 5,
                  iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    if not torch.cuda.is_available():
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) * 1e3 / iters
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


@dataclass
class Plan:
    """Result of one ``attach`` call: what was activated, why, evidence."""

    activated: dict[str, torch.nn.Module]
    families: dict[str, dict[str, Any]]
    receipt: dict[str, Any]
    _handle: _AttachHandle | None = None
    active: bool = field(init=False)

    def __post_init__(self) -> None:
        self.active = self._handle is not None

    def report(self) -> str:
        lines = [f"structures.attach: {len(self.activated)} seam(s) active"]
        for name, stat in self.families.items():
            lines.append(
                f"  {name}: {stat['layer_pass']}/{stat['layers']} layer-pass"
                f" (worst cos {stat['gate_worst_cos']:.6f}) -> "
                f"{stat['outcome']}"
                + (f" [{stat.get('reason', '')}]"
                   if stat["outcome"] == "refused" else ""))
        e2e = self.receipt.get("e2e")
        if e2e:
            lines.append(
                f"  e2e: {e2e['base_ms']:.2f} -> {e2e['ms']:.2f} ms "
                f"({e2e['speedup']:.3f}x), cos={e2e.get('cos', float('nan')):.6f}")
        return "\n".join(lines)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.detach()
        self.active = False

    def save_receipt(self, directory: str | pathlib.Path) -> pathlib.Path:
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"attach_{self.receipt['digest'][:12]}.json"
        path.write_text(json.dumps(self.receipt, indent=2, default=str))
        return path


def attach(
    model: torch.nn.Module,
    forward: Callable[[], Any] | Sequence[Callable[[], Any]],
    *,
    calibration: str = "auto",
    frames: int = 1,
    structures: tuple[str, ...] = ("decoder_ffn", "vision_ffn"),
    gate_cos: float = 0.999,
    e2e_budget: float = 0.995,
    min_speedup: float = 1.02,
    output: Callable[[], torch.Tensor] | None = None,
    verbose: bool = True,
) -> Plan:
    """Discover, calibrate, gate and activate structures in one call."""

    def say(msg: str) -> None:
        if verbose:
            print(f"[structures] {msg}", flush=True)

    thunks = (list(forward) if isinstance(forward, (list, tuple))
              else [forward] * max(1, frames))
    seams = discover(model, structures)
    families = group_families(seams)
    say(f"discovered {len(seams)} seam(s) in {len(families)} family(ies)")
    if not seams:
        return Plan({}, {}, {"digest": "none", "seams": 0})

    # ---- calibration: load cached captures or run forward under hooks ----
    cache = (pathlib.Path(calibration)
             if calibration not in ("auto",) else None)
    caps: dict[str, dict[str, list[torch.Tensor]]]
    if cache is not None and cache.is_file():
        caps = torch.load(cache, weights_only=True)
        say(f"calibration loaded from {cache} "
            f"({len(next(iter(caps.values()))['in'])} frame(s))")
    else:
        caps = {s.path: {"x": [], "in": [], "out": []} for s in seams}
        hooks = []

        def mlp_hook(path):
            def hook(module, args, out):
                caps[path]["in"].append(args[0].detach().to("cpu"))
                caps[path]["out"].append(out.detach().to("cpu"))
            return hook

        def norm_hook(path):
            def hook(module, args, out):
                caps[path]["x"].append(args[0].detach().to("cpu"))
            return hook

        from .discover import _resolve
        for seam in seams:
            hooks.append(_resolve(model, seam.path).register_forward_hook(
                mlp_hook(seam.path)))
            if seam.norm_attr:
                hooks.append(_resolve(
                    model, seam.parent_path + "." + seam.norm_attr
                ).register_forward_hook(norm_hook(seam.path)))
        with torch.no_grad():
            for thunk in thunks:
                thunk()
        for hook in hooks:
            hook.remove()
        say(f"calibrated on {len(thunks)} frame(s)")
        if cache is not None:
            torch.save(caps, cache)
            say(f"calibration saved to {cache}")

    device = next(model.parameters()).device

    # ---- bind + per-layer parity gate (structure boundary incl. residual) --
    fam_stats: dict[str, dict[str, Any]] = {}
    fam_swaps: dict[str, dict[str, torch.nn.Module]] = {}
    for name, members in families.items():
        swaps, refused, worst = {}, [], 1.0
        for seam in members:
            cap = caps.get(seam.path, {"x": [], "in": [], "out": []})
            if not cap["in"]:
                refused.append((seam.layer_index, "no calibration hits"))
                continue
            seam.m_profile = sorted(
                {int(t.reshape(-1, t.shape[-1]).shape[0])
                 for t in cap["in"]})
            from .discover import _resolve
            try:
                bound = _bind_seam(seam, seam_weights(model, seam),
                                   [t for t in cap["in"]],
                                   original=_resolve(model, seam.path))
            except ValueError as refusal:
                refused.append((seam.layer_index, str(refusal)[:80]))
                continue
            layer_worst = 1.0
            with torch.no_grad():
                for k in range(len(cap["in"])):
                    xin = cap["in"][k].to(device)
                    got = bound(xin)
                    want = cap["out"][k].to(device)
                    if cap["x"]:
                        got = cap["x"][k].to(device) + got
                        want = cap["x"][k].to(device) + want
                    layer_worst = min(
                        layer_worst, parity_metrics(got, want)["cosine"])
            worst = min(worst, layer_worst)
            if layer_worst >= gate_cos:
                swaps[seam.path] = bound
            else:
                refused.append((seam.layer_index, round(layer_worst, 6)))
        fam_swaps[name] = swaps
        fam_stats[name] = {
            "structure": members[0].structure, "dims": members[0].dims,
            "variant": members[0].variant, "layers": len(members),
            "layer_pass": len(swaps), "refused_layers": refused,
            "gate_worst_cos": round(worst, 6),
            "m_profile": sorted({m for s in members for m in s.m_profile}),
            "outcome": "pending"}
        say(f"{name}: {len(swaps)}/{len(members)} layer-pass "
            f"@{gate_cos} (worst {worst:.6f})")

    # ---- family-level e2e gate: accuracy budget AND net win ----
    eval_thunk = thunks[-1]
    get_out = output or (lambda: _first_tensor(eval_thunk()))

    def run_out() -> torch.Tensor | None:
        with torch.no_grad():
            out = get_out()
        return out.detach().float().cpu() if out is not None else None

    base_out = run_out()
    base_ms = _cuda_time_ms(lambda: eval_thunk())
    say(f"baseline {base_ms:.2f} ms")

    active: dict[str, torch.nn.Module] = {}
    for name, swaps in fam_swaps.items():
        stat = fam_stats[name]
        if not swaps:
            stat["outcome"], stat["reason"] = "refused", "no passing layers"
            continue
        handle = _swap_attach(model, swaps)
        cos = (parity_metrics(run_out(), base_out)["cosine"]
               if base_out is not None else None)
        ms = _cuda_time_ms(lambda: eval_thunk())
        handle.detach()
        wins = ((cos is None or cos >= e2e_budget)
                and base_ms / ms >= min_speedup)
        stat["e2e"] = {"cos": cos, "ms": round(ms, 3),
                       "speedup": round(base_ms / ms, 4)}
        if wins:
            stat["outcome"] = "activated"
            active.update(swaps)
        else:
            stat["outcome"] = "refused"
            stat["reason"] = (f"e2e cos {cos:.6f} < {e2e_budget}"
                              if cos is not None and cos < e2e_budget
                              else f"no net win ({base_ms / ms:.3f}x)")
        say(f"{name}: e2e cos="
            f"{'n/a' if cos is None else f'{cos:.6f}'} "
            f"{ms:.2f}ms ({base_ms / ms:.3f}x) -> {stat['outcome']}")

    # ---- union re-check, then commit ----
    e2e_final: dict[str, Any] | None = None
    handle = None
    if active:
        handle = _swap_attach(model, active)
        cos = (parity_metrics(run_out(), base_out)["cosine"]
               if base_out is not None else None)
        ms = _cuda_time_ms(lambda: eval_thunk())
        if ((cos is not None and cos < e2e_budget)
                or base_ms / ms < min_speedup):
            handle.detach()
            handle, active = None, {}
            for stat in fam_stats.values():
                if stat["outcome"] == "activated":
                    stat["outcome"] = "refused"
                    stat["reason"] = "union fails combined budget"
            say("union of activated families fails combined budget -> "
                "refuse all")
        else:
            e2e_final = {"base_ms": round(base_ms, 3), "ms": round(ms, 3),
                         "speedup": round(base_ms / ms, 4),
                         "cos": None if cos is None else round(cos, 7)}
            say(f"active: {len(active)} seam(s), "
                f"{base_ms:.2f} -> {ms:.2f} ms ({base_ms / ms:.3f}x)")
    if not active:
        say("outcome: whole-host refusal — model left untouched")

    receipt = {
        "model": type(model).__name__,
        "frames": len(thunks),
        "calibration": calibration,
        "eval_note": ("eval frame == calibration frame"
                      if len(thunks) == 1 else "last frame held for eval"),
        "gate_cos": gate_cos, "e2e_budget": e2e_budget,
        "min_speedup": min_speedup,
        "families": fam_stats, "e2e": e2e_final,
        "seams_active": sorted(active),
    }
    receipt["digest"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, default=str).encode()
    ).hexdigest()
    return Plan(active, fam_stats, receipt, _handle=handle)

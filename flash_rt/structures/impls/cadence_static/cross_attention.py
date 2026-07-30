"""Cross-attention K/V addressing for the existing cadence-static structure.

Cross attention has a stable structural signature: Q consumes the current
hidden width, while K/V consume a different encoder width.  The encoder-side
K/V projections may therefore be refreshed once at the encoder cadence and
read from static buffers inside a repeated denoise loop.  Self attention has
equal Q/K/V input widths and is deliberately excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from .buffers import StaticOutput, bind_cadence_static


@dataclass(frozen=True)
class CrossKvCandidate:
    path: str
    module: torch.nn.Module


def discover_cross_attention_kv(
    model: torch.nn.Module,
) -> tuple[CrossKvCandidate, ...]:
    """Find encoder-side K/V projections without host or model names."""
    found = []
    for path, attention in model.named_modules():
        q_proj = getattr(attention, "to_q", None)
        k_proj = getattr(attention, "to_k", None)
        v_proj = getattr(attention, "to_v", None)
        if not all(
            isinstance(module, torch.nn.Linear)
            for module in (q_proj, k_proj, v_proj)
        ):
            continue
        if q_proj.in_features == k_proj.in_features:
            continue
        if (
            k_proj.in_features != v_proj.in_features
            or k_proj.out_features != v_proj.out_features
        ):
            continue
        found.extend(
            (
                CrossKvCandidate(f"{path}.to_k", k_proj),
                CrossKvCandidate(f"{path}.to_v", v_proj),
            )
        )
    return tuple(found)


def capture_cross_attention_kv(
    candidates: Sequence[CrossKvCandidate],
    forward: Callable[[], object],
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Capture each candidate across one complete repeated-loop forward."""
    rows = [[] for _ in candidates]
    hooks = []
    for candidate, outputs in zip(candidates, rows):
        hooks.append(
            candidate.module.register_forward_hook(
                lambda _module, _args, output, outputs=outputs:
                outputs.append(output.detach().clone())
            )
        )
    try:
        with torch.no_grad():
            forward()
    finally:
        for hook in hooks:
            hook.remove()
    return tuple(tuple(outputs) for outputs in rows)


def bind_cross_attention_kv(
    candidates: Sequence[CrossKvCandidate],
    captures: Sequence[Sequence[torch.Tensor]],
    *,
    replacements: Mapping[str, torch.nn.Module] | None = None,
) -> tuple[dict[str, StaticOutput], tuple[StaticOutput, ...]]:
    """Bind K/V buffers, optionally consuming already-bound projections."""
    replacements = replacements or {}
    modules = [
        replacements.get(candidate.path, candidate.module)
        for candidate in candidates
    ]
    statics, _ = bind_cadence_static(modules, captures)
    return (
        {
            candidate.path: static
            for candidate, static in zip(candidates, statics)
        },
        tuple(statics),
    )


def refresh_cross_attention_kv(
    statics: Sequence[StaticOutput],
    encoder_hidden_states: torch.Tensor,
) -> None:
    """Refresh all K/V buffers once before the repeated attention loop."""
    with torch.no_grad():
        for static in statics:
            static.buffer.copy_(
                static.host_module(encoder_hidden_states)
            )
            static.refreshed()

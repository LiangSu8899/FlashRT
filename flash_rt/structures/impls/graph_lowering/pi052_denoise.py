"""Pi0.5 flow-matching family: the timestep schedule becomes resident.

The host builds its denoise schedule inside ``sample_actions`` as
``torch.tensor([...python floats...], device=cuda)`` — one
host-to-device copy per call. Whether that line survives capture has
depended on the compiler's mood: a dynamo that covers the whole method
bakes it into the graph, a dependency upgrade that adds a graph break
in front of it drops the copy into the capture stream and the capture
refuses (measured: the same host line passed on 2026-07-25 and
refused after a transformers upgrade landed the next day). A pin must
not gamble on coverage.

The pin scopes one rule around the host's own method: a
``torch.tensor`` call that constructs a *constant float list* on a
device resolves to a cached resident tensor — same values, same
device, same dtype, allocated once outside capture. Everything else
passes straight through, the schedule stays value-identical by
construction, and the undo restores the host method bit-for-bit.
"""

from __future__ import annotations

import types

import torch

from .protocol import GraphLowering


def _looks_like_pi05_flow(module) -> bool:
    return (callable(getattr(module, "sample_actions", None))
            and callable(getattr(module, "denoise_step", None))
            and callable(getattr(module, "embed_suffix", None))
            and hasattr(module, "paligemma_with_expert"))


class Pi05DenoiseGraphLoweringAdapter:
    """Family: pi05_denoise — one pin, the resident step schedule."""

    def lower(self, model, forward) -> GraphLowering | None:
        target = None
        if _looks_like_pi05_flow(model):
            target = model
        else:
            for _name, mod in getattr(
                    model, "named_modules", lambda: ())():
                if _looks_like_pi05_flow(mod):
                    target = mod
                    break
            if target is None and _looks_like_pi05_flow(
                    getattr(model, "model", None)):
                target = model.model
        if target is None:
            return None

        cache: dict[tuple, torch.Tensor] = {}
        real_tensor = torch.tensor
        host_fn = target.sample_actions
        had_instance = "sample_actions" in target.__dict__

        def caching_tensor(data, *args, **kwargs):
            device = kwargs.get("device")
            if (device is not None and isinstance(data, (list, tuple))
                    and data
                    and all(isinstance(x, float) for x in data)):
                key = (tuple(data), str(device),
                       str(kwargs.get("dtype")))
                hit = cache.get(key)
                if hit is None:
                    hit = real_tensor(data, *args, **kwargs)
                    cache[key] = hit
                return hit
            return real_tensor(data, *args, **kwargs)

        def pinned(self, *args, **kwargs):
            torch.tensor = caching_tensor
            try:
                return host_fn(*args, **kwargs)
            finally:
                torch.tensor = real_tensor

        target.sample_actions = types.MethodType(pinned, target)

        def undo() -> None:
            torch.tensor = real_tensor
            if had_instance:
                target.sample_actions = host_fn
            elif "sample_actions" in target.__dict__:
                del target.sample_actions

        return GraphLowering(
            undo=undo, family="pi05_denoise",
            pins=("resident_step_schedule",),
            details={"host": type(target).__name__})

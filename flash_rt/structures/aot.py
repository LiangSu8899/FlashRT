"""Whole-graph ahead-of-time packaging for swapped modules.

Graph integrity is a first-class property of the structures runtime:
a swapped module's hot path carries no Python bookkeeping (guards
qualify at bind time and step aside under compilation), so the whole
forward exports as one graph with the Hub kernels riding along as
``torch.library`` ops. This module turns that property into an
artifact: ``aot_package`` exports the module and compiles it with
AOTInductor into a self-contained package on disk; ``aot_load`` brings
it back as a callable that replays the compiled graph with no dynamo
in the loop and no JIT cost at first call.

The scoring suite treats an AoT arm like any other treated form:
stepwise parity, repeat chains, detach, and dual-baseline timing —
the package is a faster body for the same declared plan, never a
change of plan.
"""

from __future__ import annotations

import pathlib

import torch

__all__ = ["aot_package", "aot_load", "AotModule"]


def aot_package(module: torch.nn.Module, args=(), kwargs=None,
                package_path="module_aot.pt2") -> str:
    """Export ``module`` on example inputs and AOT-compile the graph.

    Returns the package path. Raises on graph breaks or export
    failure — a partial graph is a defect to fix at the seam, not a
    fallback to hide.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "refused: AOT packaging compiles for the present GPU; "
            "no CUDA device is visible")
    kwargs = dict(kwargs or {})
    with torch.no_grad():
        exported = torch.export.export(module, args=tuple(args),
                                       kwargs=kwargs)
    out = torch._inductor.aoti_compile_and_package(
        exported, package_path=str(pathlib.Path(package_path)))
    return str(out)


def aot_load(package_path: str):
    """Load an AOT package back as a callable graph."""
    return torch._inductor.aoti_load_package(str(package_path))


class AotModule(torch.nn.Module):
    """Drop-in stand-in that replays the packaged graph.

    Attribute lookups fall through to the host module, so pipeline
    glue that introspects config/dtype keeps working; ``host`` gives
    the original back for detach.
    """

    def __init__(self, compiled, host: torch.nn.Module):
        super().__init__()
        object.__setattr__(self, "_compiled", compiled)
        object.__setattr__(self, "host", host)

    def forward(self, *args, **kwargs):
        return self._compiled(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "host"), name)

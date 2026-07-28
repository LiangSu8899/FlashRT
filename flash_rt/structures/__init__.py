"""FlashRT structures — verified, composable model sub-blocks.

A structure is a versioned specification of one model region: boundary
tensors, framework-neutral weight slots, a plain reference implementation
used as ground truth, and qualification gates. This package hosts the
structure catalog and its registry. Implementations, host adapters, and
the qualification harness build on top of these specifications.
"""

from flash_rt.structures.binding import (
    BindingSpec,
    CoverageSegment,
    list_bindings,
    load_binding,
)
from flash_rt.structures.registry import StructureSpec, list_structures, load


def get(name):
    """Explicit door: pull one structure and bind it yourself.

    Mirrors ``kernels.get_kernel``: ``get("decoder_ffn").bind(module,
    calibration=[...])`` returns a gated drop-in replacement you plug in
    where you choose. See :mod:`flash_rt.structures.handle`.
    """
    from flash_rt.structures.handle import get as _get

    return _get(name)


def capture(fn, **kwargs):
    """Capture door: graph a hot stage with declared swap windows.

    See :func:`flash_rt.structures.stages.capture`.
    """
    from flash_rt.structures.stages import capture as _capture

    return _capture(fn, **kwargs)


def auto_swaps(model, forward, **kwargs):
    """Distribution layer: discover, calibrate, bind — one pass, no
    per-seam scaffolding. Returns an :class:`AutoPlan` of swaps.

    See :func:`flash_rt.structures.autobuild.auto_swaps`.
    """
    from flash_rt.structures.autobuild import auto_swaps as _auto

    return _auto(model, forward, **kwargs)


def run_recipe(recipe, model, ctx=None, **kwargs):
    """Recipe door: assemble declared levers, audit same-process on the
    graph, certify or refuse — one call, one receipt.

    See :mod:`flash_rt.structures.recipe` for ``Recipe``/``Lever``/
    ``Gates`` and the switch lifecycle.
    """
    from flash_rt.structures.recipe import run_recipe as _run

    return _run(recipe, model, ctx, **kwargs)


def attach(model, forward, **kwargs):
    """One-call front door: discover, calibrate, gate, activate.

    See :func:`flash_rt.structures.frontdoor.attach`. Imported lazily so
    that spec-only consumers do not pay for torch-side machinery.
    """
    from flash_rt.structures.frontdoor import attach as _attach

    return _attach(model, forward, **kwargs)


from . import schemes  # noqa: E402  (registry: quantisation schemes)

__all__ = [
    "BindingSpec",
    "CoverageSegment",
    "StructureSpec",
    "attach",
    "capture",
    "get",
    "list_bindings",
    "list_structures",
    "load",
    "load_binding",
    "run_recipe",
    "schemes",
]

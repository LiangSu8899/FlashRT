"""FlashRT structures — verified, composable model sub-blocks.

A structure is a versioned specification of one model region: boundary
tensors, framework-neutral weight slots, a plain reference implementation
used as ground truth, and qualification gates. This package hosts the
structure catalog and its registry. Implementations, host adapters, and
the qualification harness build on top of these specifications.
"""

from flash_rt.structures.registry import StructureSpec, list_structures, load


def attach(model, forward, **kwargs):
    """One-call front door: discover, calibrate, gate, activate.

    See :func:`flash_rt.structures.frontdoor.attach`. Imported lazily so
    that spec-only consumers do not pay for torch-side machinery.
    """
    from flash_rt.structures.frontdoor import attach as _attach

    return _attach(model, forward, **kwargs)


__all__ = ["StructureSpec", "attach", "list_structures", "load"]

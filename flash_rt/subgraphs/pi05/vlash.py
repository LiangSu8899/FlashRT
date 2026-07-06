"""Pi0.5 VLASh context/action cut.

VLASh does not modify the decoder graph.  Before context replay, the upper
runtime projects the robot state forward with the actions expected to execute
during inference.  Pi0.5 renders that projected state into its prompt, so the
normal ``context -> decode_only`` graphs consume the updated prompt embedding.

Call :func:`enable` before graph capture.  For live state updates, the frontend
must use a state-prompt mode whose shapes have been prewarmed (or fixed mode);
this module cannot make variable token lengths graph-safe by itself.
"""

from __future__ import annotations


def enable(target: object) -> None:
    """Enable the graphs required by ``stage_plan="vlash"``.

    The actual state projection and asynchronous scheduling live in
    :mod:`flash_rt.runtime.vlash`; capture only needs the existing separate
    context and decoder graphs.
    """
    from flash_rt.subgraphs.pi05.context_action import enable as enable_context
    from flash_rt.subgraphs.pi05 import stage_plans as _stage_plans  # noqa: F401

    enable_context(target)


__all__ = ["enable"]

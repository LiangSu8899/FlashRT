"""Shared FP8 layout selection for frontends."""

from __future__ import annotations

from typing import Optional


def select_fp8_layout(hardware: Optional[str], fp8_layout: Optional[str]) -> str:
    """Choose the FP8 weight layout used by RTX frontend kernels.

    ``kn`` is the existing SM120 path: weights are stored as [K,N] and use
    ``fp8_nn_dev``. ``nk`` is the SM89-compatible path: weights are stored
    as [N,K] and use ``fp8_nt_dev``.
    """
    if fp8_layout is not None:
        if fp8_layout not in ("kn", "nk"):
            raise ValueError(f"fp8_layout must be 'kn' or 'nk', got {fp8_layout!r}")
        return fp8_layout
    if hardware == "rtx_sm89":
        return "nk"
    if hardware == "rtx_sm120":
        return "kn"
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major == 8 and minor == 9:
                return "nk"
    except Exception:
        pass
    return "kn"

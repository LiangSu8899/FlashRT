"""FlashVLA — RTX consumer discrete GPU attention backends.

Hardware-specific attention backends for RTX GPUs (SM80/86/89/120).
Model pipelines live in ``flash_vla.models.*``; frontends live in
``flash_vla.frontends.*``.

Remaining here:
    attn_backend.py       — Pi0 / Pi0.5 attention backend
                             (:class:`RtxFlashAttnBackend`)
    attn_backend_groot.py — GROOT N1.6 attention backend
                             (:class:`RtxFlashAttnBackendGroot`)

Both backends wrap the vendored Flash-Attention 2 kernels
(:mod:`flash_vla.flash_vla_fa2`) and are framework-neutral — used
by both the torch and jax frontends. The ``Torch``-prefixed names
``TorchFlashAttnBackend`` / ``TorchFlashAttnBackendGroot`` are kept
as deprecated aliases and will be removed in a future major version.
"""

from .attn_backend import (
    AttnBackend,
    RtxFlashAttnBackend,
    TorchFlashAttnBackend,   # deprecated alias for RtxFlashAttnBackend
)
from .attn_backend_groot import (
    RtxFlashAttnBackendGroot,
    TorchFlashAttnBackendGroot,  # deprecated alias
)

__all__ = [
    "AttnBackend",
    "RtxFlashAttnBackend",
    "RtxFlashAttnBackendGroot",
    # Deprecated — remove in the next major version:
    "TorchFlashAttnBackend",
    "TorchFlashAttnBackendGroot",
]

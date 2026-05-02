"""FlashVLA — Blackwell (SM120+) hardware-specific backend.

Reserved for future Blackwell-only code paths (e.g. NVFP4/W4A8 block-scaled
GEMMs, DSMEM, SM120a-specific CUTLASS templates).

The Pi0.5 consumer-GPU path (5090 / 4090) now lives in
``flash_rt.hardware.rtx`` — that module targets SM89+ discrete GPUs with
the same kernel + frontend structure as ``flash_rt.hardware.thor``, and
the rtx pipeline auto-enables FP8 / NVFP4 where the hardware supports it.
"""

__all__: list[str] = []

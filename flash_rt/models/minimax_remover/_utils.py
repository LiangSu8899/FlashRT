"""Shared utilities for MiniMax-Remover pipelines."""

import logging

logger = logging.getLogger(__name__)

_REQUIRED_NVFP4_SYMBOLS = (
    "nvfp4_sf_swizzled_bytes",
    "bf16_weight_to_nvfp4_swizzled",
    "quantize_bf16_to_nvfp4_swizzled",
    "fp4_w4a16_gemm_sm120_bf16out_pingpong",
    "add_bias_bf16",
    "fp4_w4a16_gemm_bias_gelu_fp4out_sm120",
)

_REQUIRED_FP8_SYMBOLS = (
    "quantize_fp8_static_fp16",
    "fp8_gemm_descale_fp16",
    "add_bias_fp16",
)

# Generic elementwise block-fusion kernels used by both pipelines on the
# default hot path (gelu_mode="inplace"). They are part of the default
# flash_rt_kernels build; a build missing either variant must fail fast here
# instead of crashing mid-forward with a bare AttributeError. ``gelu_inplace``
# is the bf16 path (NVFP4 default transformer dtype), ``gelu_inplace_fp16``
# the fp16 path (FP8 default).
_REQUIRED_BLOCK_SYMBOLS = (
    "gelu_inplace",
    "gelu_inplace_fp16",
)


def _load_kernels(required_symbols=None):
    """Import flash_rt_kernels and validate required symbols.

    Args:
        required_symbols: tuple of symbol names to check (default: NVFP4
            precision surface plus the shared block-fusion surface)

    Returns the flash_rt_kernels module. Raises RuntimeError if any required
    symbol is missing, naming every missing symbol so a non-matching build
    fails fast instead of crashing mid-run.
    """
    if required_symbols is None:
        required_symbols = _REQUIRED_NVFP4_SYMBOLS + _REQUIRED_BLOCK_SYMBOLS

    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError:
        try:
            import flash_rt_kernels as fvk  # type: ignore
        except ImportError:
            raise RuntimeError(
                "flash_rt_kernels is not available. Build FlashRT with:\n"
                "    cmake -S . -B build -DGPU_ARCH=120 -DCMAKE_BUILD_TYPE=Release\n"
                "    cmake --build build -j --target flash_rt_kernels\n"
                "After building, the .so file is placed in flash_rt/ directly "
                "— no make install step is needed.") from None

    missing = [s for s in required_symbols if not hasattr(fvk, s)]

    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"flash_rt_kernels is missing required symbols: {names}\n"
            "Rebuild FlashRT for your target hardware. For the Blackwell NVFP4 "
            "kernels use GPU_ARCH=120/121 (auto-enables NVFP4):\n"
            "    cmake -S . -B build -DGPU_ARCH=120 -DCMAKE_BUILD_TYPE=Release\n"
            "    cmake --build build -j --target flash_rt_kernels\n"
            "FP8 kernels are part of the default build. After building, the .so "
            "lands in flash_rt/ (pip install -e . makes it importable).")

    return fvk


def load_nvfp4_kernels():
    """Load kernels required for the NVFP4 pipeline.

    Validates the NVFP4 precision surface plus the shared block-fusion
    surface (gelu), so a build that can run the NVFP4 GEMMs but not the
    fused block path still fails fast.
    """
    return _load_kernels(_REQUIRED_NVFP4_SYMBOLS + _REQUIRED_BLOCK_SYMBOLS)


def load_fp8_kernels():
    """Load kernels required for the FP8 pipeline.

    Validates the FP8 precision surface plus the shared block-fusion surface
    (gelu), so a build that can run the FP8 GEMMs but not the fused block
    path still fails fast.
    """
    return _load_kernels(_REQUIRED_FP8_SYMBOLS + _REQUIRED_BLOCK_SYMBOLS)
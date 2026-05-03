"""FlashRT — FvkContext: per-instance runtime resources.

Wraps the C++ FvkContext (cuBLAS handle) from flash_rt_kernels.so.
Created by frontend, passed to all kernel calls via hardware pipeline.

Usage:
    from flash_rt.core.context import FvkContext
    ctx = FvkContext()
    fvk.attention_qkv_fp16(ctx.cpp, Q, K, V, ...)
    fvk.gmm_fp16(ctx.cpp, A, B, C, ...)
"""

import logging

logger = logging.getLogger(__name__)


class FvkContext:
    """Per-instance runtime resources for FlashRT kernels.

    Owns:
        - cuBLAS handle (via C++ FvkContext)
        - GemmRunner (cuBLASLt, for FP8/BF16/FP16 GEMMs)

    All hardware pipeline kernel calls go through this context.
    """

    def __init__(self):
        from flash_rt import flash_rt_kernels as fvk
        self._cpp = fvk.FvkContext()
        self._gemm = fvk.GemmRunner()
        logger.debug("FvkContext created (cuBLAS handle + GemmRunner)")

    @property
    def cpp(self):
        """C++ FvkContext object (pass to pybind11 kernel calls)."""
        return self._cpp

    @property
    def gemm(self):
        """GemmRunner instance (cuBLASLt GEMMs)."""
        return self._gemm

    @property
    def handle_ptr(self):
        """Raw cuBLAS handle pointer (for debugging / interop)."""
        return self._cpp.handle_ptr

"""FP8 forward GEMM op for JAX, backed by FlashRT's cuBLASLt FP8 kernel."""

from .fp8_jax import (
    FP8_E4M3_MAX,
    FP8_SCALE_FLOOR,
    fp8_gemm_bf16_out,
    quantize_fp8_static,
    quantize_weight_to_fp8_bytes,
)
from .lora_patch import (
    disable as disable_fp8_patch,
    enable as enable_fp8_patch,
    get_stats as get_fp8_stats,
    is_installed as fp8_patch_installed,
    reset_stats as reset_fp8_stats,
)

__all__ = [
    "FP8_E4M3_MAX",
    "FP8_SCALE_FLOOR",
    "disable_fp8_patch",
    "enable_fp8_patch",
    "fp8_gemm_bf16_out",
    "fp8_patch_installed",
    "get_fp8_stats",
    "quantize_fp8_static",
    "quantize_weight_to_fp8_bytes",
    "reset_fp8_stats",
]

// ================================================================
// FlashRT training/jax_ffi: XLA FFI handler for FP8 static quantize.
//
// Delegates to quantize_fp8_static (csrc/kernels/quantize.cu), the
// same kernel the PyTorch training path drives via pybind. The
// handler is a thin wrapper:
//
//     y_fp8(*) = clamp(x_bf16(*) / scale, [-448, 448]).cast(e4m3)
//
// The scale must be 1-element float32 and pre-computed (e.g. by the
// LoRA injection / calibration pass) — this entry point is CUDA-Graph
// compatible because no host-side amax sync happens. JAX wrappers
// pass the scale as a `jnp.ndarray` whose lifetime is graph-managed.
//
// Stream discipline matches fp8_gemm_ffi.cu — caller's PlatformStream
// is forwarded to the kernel, never stream 0.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#include "xla/ffi/api/ffi.h"

#include "kernels/quantize.cuh"

namespace ffi = xla::ffi;

// ----------------------------------------------------------------
// QuantizeFp8StaticImpl
//   Inputs:
//     x_bf16  : bfloat16 tensor, any rank
//     scale   : float32, shape (1,) — per-tensor scale (s = amax / 448)
//   Output:
//     y_fp8   : uint8 tensor, same element-count as x_bf16
//             (FP8 E4M3 bits viewed as uint8)
// ----------------------------------------------------------------
static ffi::Error QuantizeFp8StaticImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::BF16> x_bf16,
    ffi::Buffer<ffi::DataType::F32>  scale,
    ffi::Result<ffi::AnyBuffer>      y_fp8
) {
    if (scale.element_count() != 1) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "quantize_fp8_static: scale must be 1-element float32");
    }
    int64_t n_in  = x_bf16.element_count();
    int64_t n_out = y_fp8->element_count();
    if (n_in != n_out) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "quantize_fp8_static: x and y element-count mismatch");
    }
    if (n_in > std::numeric_limits<int>::max()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "quantize_fp8_static: element count exceeds int32");
    }

    const __nv_bfloat16* x_ptr =
        reinterpret_cast<const __nv_bfloat16*>(x_bf16.typed_data());
    __nv_fp8_e4m3* y_ptr =
        reinterpret_cast<__nv_fp8_e4m3*>(y_fp8->untyped_data());
    const float* scale_ptr = scale.typed_data();

    quantize_fp8_static(x_ptr, y_ptr, scale_ptr,
                        static_cast<int>(n_in), stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    flashvla_quantize_fp8_static, QuantizeFp8StaticImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::BF16>>()    // x_bf16
        .Arg<ffi::Buffer<ffi::DataType::F32>>()     // scale
        .Ret<ffi::AnyBuffer>()                       // y_fp8 (uint8)
);

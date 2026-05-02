// ================================================================
// FlashRT training/jax_ffi: XLA FFI handler for FP8 GEMM.
//
// Delegates to GemmRunner::fp8_nn_dev (csrc/gemm/gemm_runner.cu),
// the same kernel the PyTorch training path drives via pybind. The
// handler is a thin wrapper:
//
//     y_bf16(M, N) = scale_a * scale_b * x_fp8(M, K) @ w_fp8(K, N)
//
// All buffers are XLA device pointers; the FP8 ones are passed as
// uint8 byte tensors so JAX's float8_e4m3fn → bf16 implicit promotion
// rules cannot trip the type-promotion checker. Scale tensors are
// 1-element float32 device buffers (CUDA-Graph compatible).
//
// Stream discipline: the handler reads the caller's PlatformStream
// from the FFI binding and forwards it to fp8_nn_dev. Defaulting to
// stream 0 here would escape XLA's GraphTrees capture and produce
// NaN-at-step-0, mirroring the PyTorch fix
// (training/lora/fp8_autograd.py:133).
// ================================================================

#include <cuda_runtime.h>
#include <cstdint>

#include "xla/ffi/api/ffi.h"

#include "gemm/gemm_runner.h"

namespace ffi = xla::ffi;

// Process-local GemmRunner. C++11 magic statics make this thread-safe.
// One handle, one workspace, one descriptor cache for the JAX path —
// kept independent of the PyTorch path's GemmRunner so a JAX run
// cannot evict descriptors a PyTorch run cached, and vice versa.
static GemmRunner& runner() {
    static GemmRunner r;
    return r;
}

// ----------------------------------------------------------------
// Fp8GemmBf16OutImpl
//   Inputs:
//     x_fp8     : uint8 byte tensor, rank >= 2, shape (..., M_outer, K)
//     w_fp8     : uint8 byte tensor, shape (K, N)
//     act_scale : float32, shape (1,) — per-tensor activation scale
//     w_scale   : float32, shape (1,) — per-tensor weight scale
//   Output:
//     y_bf16    : bfloat16 tensor, shape matches x_fp8 with last dim → N
//   Layout assumption:
//     x_fp8 last dim = K  (contiguous; XLA always emits row-major)
//     w_fp8 shape (K, N)  (no transpose; matches GemmRunner::fp8_nn_dev)
// ----------------------------------------------------------------
static ffi::Error Fp8GemmBf16OutImpl(
    cudaStream_t stream,
    ffi::AnyBuffer x_fp8,
    ffi::AnyBuffer w_fp8,
    ffi::Buffer<ffi::DataType::F32> act_scale,
    ffi::Buffer<ffi::DataType::F32> w_scale,
    ffi::Result<ffi::AnyBuffer> y_bf16
) {
    auto x_dims = x_fp8.dimensions();
    auto w_dims = w_fp8.dimensions();
    auto y_dims = y_bf16->dimensions();

    if (x_dims.size() < 2) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fp8_gemm: x_fp8 must have rank >= 2");
    }
    if (w_dims.size() != 2) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fp8_gemm: w_fp8 must have rank 2 (K, N)");
    }
    if (y_dims.size() != x_dims.size()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fp8_gemm: y rank must match x rank");
    }

    // Collapse leading axes of x into a single M.
    int64_t M = 1;
    for (size_t i = 0; i + 1 < x_dims.size(); ++i) {
        M *= x_dims[i];
    }
    int64_t K_x = x_dims.back();
    int64_t K_w = w_dims[0];
    int64_t N   = w_dims[1];

    if (K_x != K_w) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fp8_gemm: K mismatch between x and w");
    }
    // y last dim must equal N; leading axes must equal x's leading axes.
    if (y_dims.back() != N) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fp8_gemm: y last-dim must equal N");
    }
    int64_t M_y = 1;
    for (size_t i = 0; i + 1 < y_dims.size(); ++i) {
        M_y *= y_dims[i];
    }
    if (M_y != M) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fp8_gemm: y leading-axes product must equal x's M");
    }

    // Scales must be 1-element float32.
    if (act_scale.element_count() != 1 || w_scale.element_count() != 1) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fp8_gemm: scales must be 1-element float32");
    }

    // cuBLASLt is happy with int — guard against int32 overflow (M*K ≤ 2^31).
    if (M > std::numeric_limits<int>::max() ||
        K_x > std::numeric_limits<int>::max() ||
        N > std::numeric_limits<int>::max()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fp8_gemm: dimensions exceed int32 (cuBLASLt limit)");
    }

    void* x_ptr = const_cast<void*>(x_fp8.untyped_data());
    void* w_ptr = const_cast<void*>(w_fp8.untyped_data());
    void* y_ptr = y_bf16->untyped_data();
    float* sa = const_cast<float*>(act_scale.typed_data());
    float* sw = const_cast<float*>(w_scale.typed_data());

    try {
        runner().fp8_nn_dev(x_ptr, w_ptr, y_ptr,
                            static_cast<int>(M),
                            static_cast<int>(N),
                            static_cast<int>(K_x),
                            sa, sw, stream);
    } catch (const std::exception& e) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          std::string("fp8_gemm: ") + e.what());
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    flashvla_fp8_gemm_bf16_out, Fp8GemmBf16OutImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()                            // x_fp8 (uint8)
        .Arg<ffi::AnyBuffer>()                            // w_fp8 (uint8)
        .Arg<ffi::Buffer<ffi::DataType::F32>>()           // act_scale
        .Arg<ffi::Buffer<ffi::DataType::F32>>()           // w_scale
        .Ret<ffi::AnyBuffer>()                            // y_bf16
);

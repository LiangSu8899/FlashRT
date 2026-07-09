#include "flashrt/cpp/models/pi05/native_kernel_driver.h"

#include "gemm_runner.h"

#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include <exception>

void add_bias_bf16(__nv_bfloat16* x, const __nv_bfloat16* b,
                   int rows, int columns, cudaStream_t stream);

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

__global__ void native_silu_bf16_kernel(__nv_bfloat16* values,
                                        std::size_t elements) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float value = __bfloat162float(values[index]);
        values[index] =
            __float2bfloat16(value / (1.0f + expf(-value)));
    }
}

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

}  // namespace

struct NativeKernelDriver::Impl {
    GemmRunner gemm;
};

NativeKernelDriver::NativeKernelDriver() noexcept {
    try {
        impl_.reset(new Impl());
    } catch (const std::exception& e) {
        error_ = e.what();
    } catch (...) {
        error_ = "native kernel driver initialization failed";
    }
}

NativeKernelDriver::~NativeKernelDriver() = default;

modalities::Status NativeKernelDriver::status() const {
    return impl_ ? modalities::Status::ok() : backend(error_);
}

modalities::Status NativeKernelDriver::bf16_nn(
    void* a,
    void* b,
    void* output,
    int m,
    int n,
    int k,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!a || !b || !output || m <= 0 || n <= 0 || k <= 0) {
        return invalid("native BF16 GEMM arguments are invalid");
    }
    try {
        impl_->gemm.bf16_nn(a, b, output, m, n, k,
                            reinterpret_cast<cudaStream_t>(stream));
        return modalities::Status::ok();
    } catch (const std::exception& e) {
        return backend(e.what());
    } catch (...) {
        return backend("native BF16 GEMM launch failed");
    }
}

modalities::Status NativeKernelDriver::add_bias_bf16(
    void* values,
    const void* bias,
    int rows,
    int columns,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!values || !bias || rows <= 0 || columns <= 0) {
        return invalid("native BF16 bias arguments are invalid");
    }
    ::add_bias_bf16(static_cast<__nv_bfloat16*>(values),
                    static_cast<const __nv_bfloat16*>(bias), rows, columns,
                    reinterpret_cast<cudaStream_t>(stream));
    const cudaError_t rc = cudaGetLastError();
    return rc == cudaSuccess
               ? modalities::Status::ok()
               : backend(cudaGetErrorString(rc));
}

modalities::Status NativeKernelDriver::silu_bf16(
    void* values,
    std::size_t elements,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!values || !elements) {
        return invalid("native BF16 SiLU arguments are invalid");
    }
    native_silu_bf16_kernel<<<(elements + 255) / 256, 256, 0,
                              reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(values), elements);
    const cudaError_t rc = cudaGetLastError();
    return rc == cudaSuccess
               ? modalities::Status::ok()
               : backend(cudaGetErrorString(rc));
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

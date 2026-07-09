#include "flashrt/cpp/models/pi05/native_kernel_driver.h"

#include "gemm_runner.h"

#include <cuda_runtime_api.h>

#include <exception>
#include <utility>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

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

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_KERNEL_DRIVER_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_KERNEL_DRIVER_H

#include "flashrt/cpp/modalities/types.h"

#include <cstdint>
#include <memory>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {

class NativeKernelDriver {
public:
    NativeKernelDriver() noexcept;
    ~NativeKernelDriver();

    NativeKernelDriver(const NativeKernelDriver&) = delete;
    NativeKernelDriver& operator=(const NativeKernelDriver&) = delete;

    modalities::Status status() const;
    modalities::Status bf16_nn(void* a, void* b, void* output,
                               int m, int n, int k,
                               std::uintptr_t stream) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string error_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_KERNEL_DRIVER_H

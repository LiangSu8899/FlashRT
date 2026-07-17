#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_STYLE_PRECOMPUTE_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_STYLE_PRECOMPUTE_H

#include "flashrt/cpp/models/pi05/plans/sm120/native_kernel_driver.h"
#include "flashrt/cpp/models/pi05/support/native_workspace.h"

namespace flashrt {
namespace models {
namespace pi05 {

class NativeStylePrecomputer {
public:
    explicit NativeStylePrecomputer(const NativeKernelDriver* driver)
        : driver_(driver) {}

    modalities::Status run(const NativeDeviceWeightStore& weights,
                           NativeWorkspace* workspace,
                           std::uintptr_t stream) const;

private:
    const NativeKernelDriver* driver_ = nullptr;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_STYLE_PRECOMPUTE_H

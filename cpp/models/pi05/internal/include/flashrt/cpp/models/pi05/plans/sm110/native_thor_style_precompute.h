#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_STYLE_PRECOMPUTE_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_STYLE_PRECOMPUTE_H

#include "flashrt/cpp/models/pi05/support/native_device_weights.h"
#include "flashrt/cpp/models/pi05/plans/sm110/native_thor_kernel_driver.h"
#include "flashrt/cpp/models/pi05/support/native_workspace.h"

#include <cstdint>

namespace flashrt {
namespace models {
namespace pi05 {

class NativeThorStylePrecomputer {
public:
    explicit NativeThorStylePrecomputer(const NativeThorKernelDriver* driver)
        : driver_(driver) {}

    modalities::Status run(const NativeDeviceWeightStore& weights,
                           NativeWorkspace* workspace,
                           std::uintptr_t stream) const;

private:
    const NativeThorKernelDriver* driver_ = nullptr;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_STYLE_PRECOMPUTE_H

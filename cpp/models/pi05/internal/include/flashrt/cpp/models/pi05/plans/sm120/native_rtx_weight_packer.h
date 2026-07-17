#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_WEIGHT_PACKER_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_WEIGHT_PACKER_H

#include "flashrt/cpp/models/pi05/support/native_device_weights.h"
#include "flashrt/cpp/models/pi05/plans/sm120/native_kernel_driver.h"

#include <string>

namespace flashrt {
namespace models {
namespace pi05 {

class NativeRtxWeightPacker {
public:
    NativeRtxWeightPacker(NativeDeviceWeightStore* weights,
                          const NativeKernelDriver* driver)
        : weights_(weights), driver_(driver) {}

    modalities::Status pack_weight(
        const std::string& source_name,
        const std::string& packed_name = "");
    modalities::Status pack_all();

private:
    modalities::Status merge_bf16_columns(
        const std::string& left_name,
        const std::string& right_name,
        const std::string& output_name);

    NativeDeviceWeightStore* weights_ = nullptr;
    const NativeKernelDriver* driver_ = nullptr;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_WEIGHT_PACKER_H

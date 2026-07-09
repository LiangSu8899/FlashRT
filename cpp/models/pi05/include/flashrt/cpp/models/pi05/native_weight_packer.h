#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHT_PACKER_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHT_PACKER_H

#include "flashrt/cpp/models/pi05/native_device_weights.h"
#include "flashrt/cpp/models/pi05/native_quantization.h"

namespace flashrt {
namespace models {
namespace pi05 {

class NativeWeightPacker {
public:
    explicit NativeWeightPacker(NativeDeviceWeightStore* weights)
        : weights_(weights) {}

    modalities::Status pack_fp8(const std::string& name, bool transpose);
    modalities::Status pack_int8(const std::string& name);

private:
    modalities::Status load_bf16(const std::string& name,
                                 NativeFloatTensor* out) const;

    NativeDeviceWeightStore* weights_ = nullptr;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHT_PACKER_H

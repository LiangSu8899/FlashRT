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
    modalities::Status pack_fp8_as(const std::string& source_name,
                                   const std::string& packed_name,
                                   bool transpose);
    modalities::Status pack_int8(const std::string& name);
    modalities::Status merge_bf16_columns(const std::string& left_name,
                                           const std::string& right_name,
                                           const std::string& output_name);
    modalities::Status pack_all_fp8(bool transpose);
    modalities::Status pack_vision_int8();
    modalities::Status pack_encoder_int8();
    modalities::Status pack_decoder_int8();

private:
    modalities::Status load_bf16(const std::string& name,
                                 NativeFloatTensor* out) const;

    NativeDeviceWeightStore* weights_ = nullptr;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHT_PACKER_H

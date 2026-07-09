#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHT_MATERIALIZER_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHT_MATERIALIZER_H

#include "flashrt/cpp/loader/safetensors.h"
#include "flashrt/cpp/models/pi05/native_device_weights.h"

namespace flashrt {
namespace models {
namespace pi05 {

class NativeWeightMaterializer {
public:
    NativeWeightMaterializer(const loader::SafetensorsFile& source,
                             NativeDeviceWeightStore* destination)
        : source_(source), destination_(destination) {}

    modalities::Status materialize_encoder_layer(int layer);

private:
    modalities::Status load(const std::string& key, NativeFloatTensor* out);
    modalities::Status upload(const std::string& name,
                              const NativeFloatTensor& tensor);
    modalities::Status upload_rounded_transpose(
        const std::string& source_key,
        const std::string& destination_name);
    modalities::Status upload_folded_transpose(
        const std::string& source_key,
        const NativeFloatTensor& norm,
        const std::string& destination_name);

    const loader::SafetensorsFile& source_;
    NativeDeviceWeightStore* destination_ = nullptr;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHT_MATERIALIZER_H

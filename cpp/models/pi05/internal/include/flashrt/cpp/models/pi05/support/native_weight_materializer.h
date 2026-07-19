#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHT_MATERIALIZER_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHT_MATERIALIZER_H

#include "flashrt/cpp/loader/safetensors.h"
#include "flashrt/cpp/models/pi05/support/native_device_weights.h"
#include "flashrt/cpp/models/pi05/support/native_weight_ops.h"

namespace flashrt {
namespace models {
namespace pi05 {

struct NativeMaterializationOptions {
    int num_steps = 10;
    bool include_embedding = true;
};

class NativeWeightMaterializer {
public:
    NativeWeightMaterializer(const loader::SafetensorsFile& source,
                             NativeDeviceWeightStore* destination,
                             modalities::DType logical_scalar =
                                 modalities::DType::kBFloat16)
        : source_(source),
          destination_(destination),
          logical_scalar_(logical_scalar) {}

    modalities::Status materialize_encoder_layer(int layer);
    modalities::Status materialize_decoder_layer(int layer);
    modalities::Status materialize_vision_layer(int layer);
    modalities::Status materialize_vision_globals();
    modalities::Status materialize_decoder_globals(int num_steps);
    modalities::Status materialize_embedding();
    modalities::Status materialize_all(
        const NativeMaterializationOptions& options);

private:
    modalities::Status load(const std::string& key, NativeFloatTensor* out);
    modalities::Status upload(const std::string& name,
                              const NativeFloatTensor& tensor);
    modalities::Status upload_source(
        const std::string& source_key,
        const std::string& destination_name,
        bool transpose);
    modalities::Status upload_folded_transpose(
        const std::string& source_key,
        const NativeFloatTensor& norm,
        const std::string& destination_name);

    const loader::SafetensorsFile& source_;
    NativeDeviceWeightStore* destination_ = nullptr;
    modalities::DType logical_scalar_ = modalities::DType::kBFloat16;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHT_MATERIALIZER_H

#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_WEIGHT_MATERIALIZER_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_WEIGHT_MATERIALIZER_H

#include "flashrt/cpp/loader/safetensors.h"
#include "flashrt/cpp/models/pi05/support/native_device_weights.h"
#include "flashrt/cpp/models/pi05/support/native_quantization.h"

#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

struct NativeThorMaterializationOptions {
    int num_steps = 10;
    bool include_embedding = true;
};

struct NativeThorWeightScales {
    std::vector<float> vision;
    std::vector<float> encoder;
    std::vector<float> decoder;
};

class NativeThorWeightMaterializer {
public:
    NativeThorWeightMaterializer(const loader::SafetensorsFile& source,
                                 NativeDeviceWeightStore* destination)
        : source_(source), destination_(destination) {}

    modalities::Status materialize_all(
        const NativeThorMaterializationOptions& options,
        NativeThorWeightScales* scales);

private:
    modalities::Status upload_f16(const std::string& source_key,
                                  const std::string& destination_name,
                                  bool transpose);
    modalities::Status upload_f16(const std::string& destination_name,
                                  const NativeF16Tensor& tensor);
    modalities::Status upload_fp8(const std::string& destination_name,
                                  const NativeF16Tensor& tensor,
                                  std::vector<float>* scales);
    modalities::Status materialize_vision_globals();
    modalities::Status materialize_vision_layer(int layer,
                                                std::vector<float>* scales);
    modalities::Status materialize_encoder_layer(int layer,
                                                 std::vector<float>* scales);
    modalities::Status materialize_decoder_layer(int layer,
                                                 std::vector<float>* scales);
    modalities::Status materialize_decoder_globals(int num_steps);
    modalities::Status materialize_embedding();
    modalities::Status upload_scale_vector(const std::string& name,
                                           const std::vector<float>& values);

    const loader::SafetensorsFile& source_;
    NativeDeviceWeightStore* destination_ = nullptr;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_WEIGHT_MATERIALIZER_H

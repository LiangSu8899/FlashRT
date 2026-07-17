#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_QUANTIZATION_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_QUANTIZATION_H

#include "flashrt/cpp/models/pi05/support/native_weight_ops.h"

#include <cstdint>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

struct NativeFp8Tensor {
    std::vector<std::uint64_t> shape;
    std::vector<std::uint8_t> values;
    float scale = 0.0f;
};

struct NativeInt8Tensor {
    std::vector<std::uint64_t> shape;
    std::vector<std::int8_t> values;
    std::vector<float> scales;
};

modalities::Status native_quantize_fp8_e4m3(
    const NativeFloatTensor& bf16_weight,
    bool transpose,
    NativeFp8Tensor* out);

modalities::Status native_quantize_fp8_e4m3(
    const NativeF16Tensor& fp16_weight,
    bool transpose,
    NativeFp8Tensor* out);

modalities::Status native_quantize_int8_per_output(
    const NativeFloatTensor& bf16_weight,
    NativeInt8Tensor* out);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_QUANTIZATION_H

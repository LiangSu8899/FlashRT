#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHT_OPS_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHT_OPS_H

#include "flashrt/cpp/loader/safetensors.h"
#include "flashrt/cpp/modalities/types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

struct NativeFloatTensor {
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

struct NativeBf16Tensor {
    std::vector<std::uint64_t> shape;
    std::vector<std::uint16_t> values;
};

modalities::Status load_native_float_tensor(
    const loader::SafetensorsFile& file,
    const std::string& key,
    NativeFloatTensor* out);

modalities::Status native_to_bf16(const NativeFloatTensor& input,
                                  NativeBf16Tensor* out);

modalities::Status native_round_to_bf16_float(
    const NativeFloatTensor& input,
    NativeFloatTensor* out);

modalities::Status native_transpose_2d(const NativeFloatTensor& input,
                                       NativeFloatTensor* out);

modalities::Status native_patch_oihw_to_hwio(
    const NativeFloatTensor& input,
    NativeFloatTensor* out);

modalities::Status native_interleave_qk_rows(
    const NativeFloatTensor& input,
    std::uint64_t num_heads,
    NativeFloatTensor* out);

modalities::Status native_fold_rms_columns(
    const NativeFloatTensor& weight,
    const NativeFloatTensor& norm,
    NativeFloatTensor* out);

modalities::Status native_concat_rows_transpose(
    const std::vector<const NativeFloatTensor*>& inputs,
    NativeFloatTensor* out);

modalities::Status native_concat_columns(
    const NativeFloatTensor& left,
    const NativeFloatTensor& right,
    NativeFloatTensor* out);

modalities::Status native_scale(const NativeFloatTensor& input,
                                float scale,
                                NativeFloatTensor* out);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHT_OPS_H

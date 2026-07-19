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

struct NativeF16Tensor {
    std::vector<std::uint64_t> shape;
    std::vector<std::uint16_t> values;
};

enum class NativeSourceDType {
    kF32,
    kBf16,
    kF16,
};

struct NativeSourceTensorView {
    const void* data = nullptr;
    std::vector<std::uint64_t> shape;
    NativeSourceDType dtype = NativeSourceDType::kF32;
};

modalities::Status load_native_source_tensor(
    const loader::SafetensorsFile& file,
    const std::string& key,
    NativeSourceTensorView* out);

modalities::Status native_source_to_bf16(
    const NativeSourceTensorView& input,
    bool transpose,
    NativeBf16Tensor* out);

modalities::Status native_source_to_f16(
    const NativeSourceTensorView& input,
    bool transpose,
    NativeF16Tensor* out);

modalities::Status native_source_qkv_to_f16(
    const NativeSourceTensorView& q,
    const NativeSourceTensorView& k,
    const NativeSourceTensorView& v,
    std::uint64_t q_heads,
    std::uint64_t k_heads,
    const NativeFloatTensor* norm,
    bool transpose,
    NativeF16Tensor* out);

modalities::Status native_source_pair_to_f16(
    const NativeSourceTensorView& left,
    const NativeSourceTensorView& right,
    const NativeFloatTensor* norm,
    bool transpose,
    NativeF16Tensor* out);

modalities::Status native_source_concat_vectors_to_f16(
    const std::vector<const NativeSourceTensorView*>& inputs,
    NativeF16Tensor* out);

modalities::Status native_source_patch_oihw_to_hwio_f16(
    const NativeSourceTensorView& input,
    NativeF16Tensor* out);

modalities::Status native_source_fold_rms_columns_to_f16(
    const NativeSourceTensorView& weight,
    const NativeFloatTensor& norm,
    bool transpose,
    NativeF16Tensor* out);

modalities::Status native_source_fold_rms_columns_transpose(
    const NativeSourceTensorView& weight,
    const NativeFloatTensor& norm,
    NativeBf16Tensor* out);

modalities::Status native_source_round_scale_to_bf16(
    const NativeSourceTensorView& input,
    float scale,
    bool transpose,
    NativeBf16Tensor* out);

modalities::Status native_source_qkv_to_bf16(
    const NativeSourceTensorView& q,
    const NativeSourceTensorView& k,
    const NativeSourceTensorView& v,
    std::uint64_t q_heads,
    std::uint64_t k_heads,
    const NativeFloatTensor* norm,
    NativeBf16Tensor* out);

modalities::Status native_source_concat_vectors_to_bf16(
    const std::vector<const NativeSourceTensorView*>& inputs,
    NativeBf16Tensor* out);

modalities::Status native_source_patch_oihw_to_hwio_bf16(
    const NativeSourceTensorView& input,
    NativeBf16Tensor* out);

modalities::Status native_source_pair_transpose_concat_bf16(
    const NativeSourceTensorView& left,
    const NativeSourceTensorView& right,
    NativeBf16Tensor* out);

modalities::Status load_native_float_tensor(
    const loader::SafetensorsFile& file,
    const std::string& key,
    NativeFloatTensor* out);

modalities::Status native_to_bf16(const NativeFloatTensor& input,
                                  NativeBf16Tensor* out);

modalities::Status native_to_f16(const NativeFloatTensor& input,
                                 NativeF16Tensor* out);

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

modalities::Status native_concat_vectors(
    const std::vector<const NativeFloatTensor*>& inputs,
    NativeFloatTensor* out);

modalities::Status native_scale(const NativeFloatTensor& input,
                                float scale,
                                NativeFloatTensor* out);

modalities::Status native_pi05_time_embeddings(
    int num_steps,
    std::uint64_t embedding_dim,
    NativeFloatTensor* out);

modalities::Status native_pi05_time_embeddings_f16(
    int num_steps,
    std::uint64_t embedding_dim,
    NativeF16Tensor* out);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHT_OPS_H

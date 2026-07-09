#include "flashrt/cpp/models/pi05/native_weight_packer.h"

#include <utility>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

}  // namespace

modalities::Status NativeWeightPacker::load_bf16(
    const std::string& name,
    NativeFloatTensor* out) const {
    if (!weights_ || !out) return invalid("native weight packer is invalid");
    NativeBf16Tensor source;
    modalities::Status st = weights_->download_bf16(name, &source);
    if (!st.ok_status()) return st;
    NativeFloatTensor result;
    result.shape = source.shape;
    result.values.resize(source.values.size());
    for (std::size_t i = 0; i < source.values.size(); ++i) {
        result.values[i] =
            modalities::bfloat16_to_float(source.values[i]);
    }
    *out = std::move(result);
    return modalities::Status::ok();
}

modalities::Status NativeWeightPacker::pack_fp8(
    const std::string& name,
    bool transpose) {
    NativeFloatTensor source;
    modalities::Status st = load_bf16(name, &source);
    if (!st.ok_status()) return st;
    NativeFp8Tensor packed;
    st = native_quantize_fp8_e4m3(source, transpose, &packed);
    if (!st.ok_status()) return st;
    const std::string prefix = "fp8." + name;
    st = weights_->upload_bytes(prefix, packed.shape,
                                NativeWeightDType::kFp8E4M3,
                                packed.values.data(), packed.values.size());
    if (!st.ok_status()) return st;
    return weights_->upload_bytes(prefix + ".scale", {1},
                                  NativeWeightDType::kFloat32,
                                  &packed.scale, sizeof(packed.scale));
}

modalities::Status NativeWeightPacker::pack_int8(const std::string& name) {
    NativeFloatTensor source;
    modalities::Status st = load_bf16(name, &source);
    if (!st.ok_status()) return st;
    NativeInt8Tensor packed;
    st = native_quantize_int8_per_output(source, &packed);
    if (!st.ok_status()) return st;
    const std::string prefix = "int8." + name;
    st = weights_->upload_bytes(prefix, packed.shape,
                                NativeWeightDType::kInt8,
                                packed.values.data(), packed.values.size());
    if (!st.ok_status()) return st;
    return weights_->upload_bytes(
        prefix + ".scale", {static_cast<std::uint64_t>(packed.scales.size())},
        NativeWeightDType::kFloat32, packed.scales.data(),
        packed.scales.size() * sizeof(float));
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

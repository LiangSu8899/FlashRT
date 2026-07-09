#include "flashrt/cpp/models/pi05/native_weight_packer.h"

#include <string>
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
    return pack_fp8_as(name, name, transpose);
}

modalities::Status NativeWeightPacker::pack_fp8_as(
    const std::string& source_name,
    const std::string& packed_name,
    bool transpose) {
    NativeFloatTensor source;
    modalities::Status st = load_bf16(source_name, &source);
    if (!st.ok_status()) return st;
    NativeFp8Tensor packed;
    st = native_quantize_fp8_e4m3(source, transpose, &packed);
    if (!st.ok_status()) return st;
    const std::string prefix = "fp8." + packed_name;
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

modalities::Status NativeWeightPacker::merge_bf16_columns(
    const std::string& left_name,
    const std::string& right_name,
    const std::string& output_name) {
    NativeFloatTensor left;
    NativeFloatTensor right;
    NativeFloatTensor merged;
    modalities::Status st = load_bf16(left_name, &left);
    if (!st.ok_status()) return st;
    st = load_bf16(right_name, &right);
    if (!st.ok_status()) return st;
    st = native_concat_columns(left, right, &merged);
    if (!st.ok_status()) return st;
    NativeBf16Tensor bf16;
    st = native_to_bf16(merged, &bf16);
    if (!st.ok_status()) return st;
    return weights_->upload(output_name, bf16);
}

modalities::Status NativeWeightPacker::pack_all_fp8(bool transpose) {
    if (!weights_) return invalid("native weight packer is invalid");
    modalities::Status st;
    for (int layer = 0; layer < 27; ++layer) {
        for (const char* stem : {"vision_attn_qkv_w_", "vision_attn_o_w_",
                                 "vision_ffn_up_w_",
                                 "vision_ffn_down_w_"}) {
            st = pack_fp8(std::string(stem) + std::to_string(layer),
                          transpose);
            if (!st.ok_status()) return st;
        }
    }
    st = pack_fp8_as("encoder_multi_modal_projector_w",
                     "vision_projector_w", transpose);
    if (!st.ok_status()) return st;

    for (int layer = 0; layer < 18; ++layer) {
        const std::string suffix = std::to_string(layer);
        const std::string gate_up = "encoder_ffn_gate_up_w_" + suffix;
        st = merge_bf16_columns("encoder_ffn_gate_w_" + suffix,
                                "encoder_ffn_up_w_" + suffix, gate_up);
        if (!st.ok_status()) return st;
        for (const std::string& name : {
                 "encoder_attn_qkv_w_" + suffix,
                 "encoder_attn_o_w_" + suffix,
                 gate_up,
                 "encoder_ffn_down_w_" + suffix}) {
            st = pack_fp8(name, transpose);
            if (!st.ok_status()) return st;
        }
    }
    for (int layer = 0; layer < 18; ++layer) {
        const std::string suffix = std::to_string(layer);
        for (const std::string& name : {
                 "decoder_attn_qkv_w_" + suffix,
                 "decoder_attn_o_w_" + suffix,
                 "decoder_ffn_gate_up_w_" + suffix,
                 "decoder_ffn_down_w_" + suffix}) {
            st = pack_fp8(name, transpose);
            if (!st.ok_status()) return st;
        }
    }
    return modalities::Status::ok();
}

modalities::Status NativeWeightPacker::pack_vision_int8() {
    if (!weights_) return invalid("native weight packer is invalid");
    for (int layer = 0; layer < 27; ++layer) {
        for (const char* stem : {"vision_attn_qkv_w_", "vision_attn_o_w_",
                                 "vision_ffn_up_w_",
                                 "vision_ffn_down_w_"}) {
            const modalities::Status st =
                pack_int8(std::string(stem) + std::to_string(layer));
            if (!st.ok_status()) return st;
        }
    }
    return modalities::Status::ok();
}

modalities::Status NativeWeightPacker::pack_encoder_int8() {
    if (!weights_) return invalid("native weight packer is invalid");
    for (int layer = 0; layer < 18; ++layer) {
        const std::string suffix = std::to_string(layer);
        for (const std::string& name : {
                 "encoder_attn_qkv_w_" + suffix,
                 "encoder_attn_o_w_" + suffix,
                 "encoder_ffn_gate_w_" + suffix,
                 "encoder_ffn_up_w_" + suffix,
                 "encoder_ffn_down_w_" + suffix}) {
            const modalities::Status st = pack_int8(name);
            if (!st.ok_status()) return st;
        }
    }
    return modalities::Status::ok();
}

modalities::Status NativeWeightPacker::pack_decoder_int8() {
    if (!weights_) return invalid("native weight packer is invalid");
    for (int layer = 0; layer < 18; ++layer) {
        const std::string suffix = std::to_string(layer);
        for (const std::string& name : {
                 "decoder_attn_qkv_w_" + suffix,
                 "decoder_attn_o_w_" + suffix,
                 "decoder_ffn_gate_w_" + suffix,
                 "decoder_ffn_up_w_" + suffix,
                 "decoder_ffn_down_w_" + suffix}) {
            const modalities::Status st = pack_int8(name);
            if (!st.ok_status()) return st;
        }
    }
    return modalities::Status::ok();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

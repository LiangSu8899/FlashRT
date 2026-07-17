#include "flashrt/cpp/models/pi05/plans/sm120/native_rtx_weight_packer.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

void* dptr(const NativeDeviceWeight* weight) {
    return weight ? frt_buffer_dptr(weight->buffer) : nullptr;
}

}  // namespace

modalities::Status NativeRtxWeightPacker::pack_weight(
    const std::string& source_name,
    const std::string& packed_name) {
    if (!weights_ || !driver_ || source_name.empty()) {
        return invalid("native RTX weight packer is invalid");
    }
    const NativeDeviceWeight* source = weights_->find(source_name);
    if (!source || source->dtype != NativeWeightDType::kBf16 ||
        source->shape.size() != 2 || !source->shape[0] || !source->shape[1] ||
        source->shape[0] > std::numeric_limits<std::size_t>::max() /
                               source->shape[1]) {
        return invalid("native RTX FP8 source weight is invalid");
    }
    const std::string name = packed_name.empty() ? source_name : packed_name;
    const std::string prefix = "fp8." + name;
    modalities::Status st = weights_->allocate(
        prefix, source->shape, NativeWeightDType::kFp8E4M3);
    if (!st.ok_status()) return st;
    st = weights_->allocate(
        prefix + ".scale", {1}, NativeWeightDType::kFloat32);
    if (!st.ok_status()) return st;
    const NativeDeviceWeight* output = weights_->find(prefix);
    const NativeDeviceWeight* scale = weights_->find(prefix + ".scale");
    const std::size_t elements = static_cast<std::size_t>(source->shape[0]) *
                                 static_cast<std::size_t>(source->shape[1]);
    return driver_->quantize_fp8_weight_bf16(
        dptr(source), dptr(output), static_cast<float*>(dptr(scale)),
        elements, 0);
}

modalities::Status NativeRtxWeightPacker::merge_bf16_columns(
    const std::string& left_name,
    const std::string& right_name,
    const std::string& output_name) {
    if (!weights_ || output_name.empty()) {
        return invalid("native RTX merged weight arguments are invalid");
    }
    const NativeDeviceWeight* left = weights_->find(left_name);
    const NativeDeviceWeight* right = weights_->find(right_name);
    if (!left || !right || left->dtype != NativeWeightDType::kBf16 ||
        right->dtype != NativeWeightDType::kBf16 ||
        left->shape.size() != 2 || right->shape != left->shape ||
        left->shape[1] > std::numeric_limits<std::uint64_t>::max() / 2) {
        return invalid("native RTX merged BF16 weights are invalid");
    }
    const std::vector<std::uint64_t> shape = {
        left->shape[0], left->shape[1] * 2};
    modalities::Status st = weights_->allocate(
        output_name, shape, NativeWeightDType::kBf16);
    if (!st.ok_status()) return st;
    const NativeDeviceWeight* output = weights_->find(output_name);
    const std::size_t rows = static_cast<std::size_t>(left->shape[0]);
    const std::size_t columns = static_cast<std::size_t>(left->shape[1]);
    const std::size_t source_pitch = columns * sizeof(std::uint16_t);
    const std::size_t output_pitch = source_pitch * 2;
    auto* destination = static_cast<unsigned char*>(dptr(output));
    if (cudaMemcpy2DAsync(
            destination, output_pitch, dptr(left), source_pitch,
            source_pitch, rows, cudaMemcpyDeviceToDevice, nullptr) !=
            cudaSuccess ||
        cudaMemcpy2DAsync(
            destination + source_pitch, output_pitch, dptr(right),
            source_pitch, source_pitch, rows, cudaMemcpyDeviceToDevice,
            nullptr) != cudaSuccess) {
        return backend("native RTX merged BF16 copy failed");
    }
    return modalities::Status::ok();
}

modalities::Status NativeRtxWeightPacker::pack_all() {
    if (!weights_ || !driver_) {
        return invalid("native RTX weight packer is invalid");
    }
    modalities::Status st;
    for (int layer = 0; layer < 27; ++layer) {
        for (const char* stem : {"vision_attn_qkv_w_", "vision_attn_o_w_",
                                 "vision_ffn_up_w_",
                                 "vision_ffn_down_w_"}) {
            st = pack_weight(std::string(stem) + std::to_string(layer));
            if (!st.ok_status()) return st;
        }
    }
    st = pack_weight(
        "encoder_multi_modal_projector_w", "vision_projector_w");
    if (!st.ok_status()) return st;

    for (int layer = 0; layer < 18; ++layer) {
        const std::string suffix = std::to_string(layer);
        const std::string gate_up = "encoder_ffn_gate_up_w_" + suffix;
        st = merge_bf16_columns(
            "encoder_ffn_gate_w_" + suffix,
            "encoder_ffn_up_w_" + suffix, gate_up);
        if (!st.ok_status()) return st;
        for (const std::string& name : {
                 "encoder_attn_qkv_w_" + suffix,
                 "encoder_attn_o_w_" + suffix,
                 gate_up,
                 "encoder_ffn_down_w_" + suffix}) {
            st = pack_weight(name);
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
            st = pack_weight(name);
            if (!st.ok_status()) return st;
        }
    }
    return cudaDeviceSynchronize() == cudaSuccess
               ? modalities::Status::ok()
               : backend("native RTX FP8 weight packing failed");
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#include "flashrt/cpp/models/pi05/native_bf16_forward.h"

#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

bool shape_is(const NativeWorkspaceBuffer* buffer,
              std::initializer_list<std::uint64_t> shape) {
    return buffer && buffer->dtype == modalities::DType::kBFloat16 &&
           buffer->shape == std::vector<std::uint64_t>(shape);
}

bool shape_is(const NativeAttentionBuffer* buffer,
              std::initializer_list<std::uint64_t> shape) {
    return buffer && buffer->dtype == NativeAttentionDType::kBf16 &&
           buffer->shape == std::vector<std::uint64_t>(shape);
}

}  // namespace

modalities::Status NativeBf16Forward::encoder_qkv(
    int layer,
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    NativeRtxAttentionWorkspace* attention,
    std::uintptr_t stream) const {
    if (!driver_ || !workspace || !attention || layer < 0 || layer >= 18) {
        return invalid("native encoder QKV owner is invalid");
    }
    const int sequence = workspace->encoder_sequence();
    if (sequence <= 0) {
        return invalid("native encoder sequence is invalid");
    }
    const NativeWorkspaceBuffer* x = workspace->find("encoder_x");
    const NativeWorkspaceBuffer* x_norm = workspace->find("encoder_x_norm");
    const NativeWorkspaceBuffer* qkv = workspace->find("encoder_QKV");
    const NativeWorkspaceBuffer* rms = workspace->find("encoder_rms_ones");
    const NativeWorkspaceBuffer* rope =
        workspace->find("encoder_rope_weights");
    const NativeAttentionBuffer* query = attention->find("attn_enc_Q");
    const NativeAttentionBuffer* cache = attention->find("attn_enc_K");
    const NativeAttentionBuffer* value_cache = attention->find("attn_enc_V");
    const NativeDeviceWeight* qkv_weight =
        weights.find("encoder_attn_qkv_w_" + std::to_string(layer));
    if (!shape_is(x, {static_cast<std::uint64_t>(sequence), 2048}) ||
        !shape_is(x_norm, {static_cast<std::uint64_t>(sequence), 2048}) ||
        !shape_is(qkv, {static_cast<std::uint64_t>(sequence), 2560}) ||
        !shape_is(rms, {2048}) ||
        !shape_is(rope, {static_cast<std::uint64_t>(sequence), 256}) ||
        !shape_is(query, {static_cast<std::uint64_t>(sequence), 8, 256}) ||
        !cache || cache->dtype != NativeAttentionDType::kBf16 ||
        cache->shape.size() != 4 || cache->shape[0] != 18 ||
        cache->shape[1] < static_cast<std::uint64_t>(sequence) ||
        cache->shape[2] != 1 || cache->shape[3] != 256 || !qkv_weight ||
        !value_cache || value_cache->dtype != NativeAttentionDType::kBf16 ||
        value_cache->shape != cache->shape ||
        qkv_weight->dtype != NativeWeightDType::kBf16 ||
        qkv_weight->shape != std::vector<std::uint64_t>({2048, 2560})) {
        return invalid("native encoder QKV buffers or weight are invalid");
    }
    void* key = attention->encoder_k_layer_dptr(layer);
    void* value = attention->encoder_v_layer_dptr(layer);
    if (!key || !value) {
        return invalid("native encoder QKV cache layer is invalid");
    }
    modalities::Status st = driver_->rms_norm_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(rms->buffer),
        frt_buffer_dptr(x_norm->buffer), sequence, 2048, 1e-6f, stream);
    if (!st.ok_status()) return st;
    st = driver_->bf16_nn(
        frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(qkv_weight->buffer),
        frt_buffer_dptr(qkv->buffer), sequence, 2560, 2048, stream);
    if (!st.ok_status()) return st;
    return driver_->qkv_split_rope_bf16(
        frt_buffer_dptr(qkv->buffer), frt_buffer_dptr(rope->buffer),
        frt_buffer_dptr(query->buffer), key, value, sequence, 2048, 256, 256,
        256, stream);
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

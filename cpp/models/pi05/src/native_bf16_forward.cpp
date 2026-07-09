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

#ifdef FLASHRT_CPP_WITH_FA2
bool shape_is(const NativeDeviceWeight* weight,
              std::initializer_list<std::uint64_t> shape) {
    return weight && weight->dtype == NativeWeightDType::kBf16 &&
           weight->shape == std::vector<std::uint64_t>(shape);
}
#endif

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

#ifdef FLASHRT_CPP_WITH_FA2
modalities::Status NativeBf16Forward::encoder_layer(
    int layer,
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    NativeRtxAttentionWorkspace* attention,
    const NativeRtxAttentionDriver* attention_driver,
    std::uintptr_t stream) const {
    modalities::Status st =
        encoder_qkv(layer, weights, workspace, attention, stream);
    if (!st.ok_status() || layer == 17) return st;
    if (!attention_driver || !attention_driver->status().ok_status()) {
        return invalid("native encoder attention driver is invalid");
    }
    const int sequence = workspace->encoder_sequence();
    const NativeWorkspaceBuffer* x = workspace->find("encoder_x");
    const NativeWorkspaceBuffer* x_norm = workspace->find("encoder_x_norm");
    const NativeWorkspaceBuffer* gate =
        workspace->find("encoder_gate_merged");
    const NativeWorkspaceBuffer* hidden = workspace->find("encoder_hidden");
    const NativeWorkspaceBuffer* rms = workspace->find("encoder_rms_ones");
    const NativeDeviceWeight* output_weight =
        weights.find("encoder_attn_o_w_" + std::to_string(layer));
    const NativeDeviceWeight* gate_weight =
        weights.find("encoder_ffn_gate_w_" + std::to_string(layer));
    const NativeDeviceWeight* up_weight =
        weights.find("encoder_ffn_up_w_" + std::to_string(layer));
    const NativeDeviceWeight* down_weight =
        weights.find("encoder_ffn_down_w_" + std::to_string(layer));
    if (!shape_is(x, {static_cast<std::uint64_t>(sequence), 2048}) ||
        !shape_is(x_norm, {static_cast<std::uint64_t>(sequence), 2048}) ||
        !shape_is(gate, {static_cast<std::uint64_t>(sequence), 32768}) ||
        !shape_is(hidden, {static_cast<std::uint64_t>(sequence), 16384}) ||
        !shape_is(rms, {2048}) ||
        !shape_is(output_weight, {2048, 2048}) ||
        !shape_is(gate_weight, {2048, 16384}) ||
        !shape_is(up_weight, {2048, 16384}) ||
        !shape_is(down_weight, {16384, 2048})) {
        return invalid("native encoder layer buffers or weights are invalid");
    }
    st = attention_driver->encoder(layer, stream);
    if (!st.ok_status()) return st;
    st = driver_->bf16_nn(
        attention_driver->encoder_output(),
        frt_buffer_dptr(output_weight->buffer), frt_buffer_dptr(x_norm->buffer),
        sequence, 2048, 2048, stream);
    if (!st.ok_status()) return st;
    st = driver_->residual_add_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(x_norm->buffer),
        static_cast<std::size_t>(sequence) * 2048, stream);
    if (!st.ok_status()) return st;
    st = driver_->rms_norm_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(rms->buffer),
        frt_buffer_dptr(x_norm->buffer), sequence, 2048, 1e-6f, stream);
    if (!st.ok_status()) return st;
    st = driver_->bf16_nn(
        frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(gate_weight->buffer),
        frt_buffer_dptr(gate->buffer), sequence, 16384, 2048, stream);
    if (!st.ok_status()) return st;
    st = driver_->bf16_nn(
        frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(up_weight->buffer),
        frt_buffer_dptr(hidden->buffer), sequence, 16384, 2048, stream);
    if (!st.ok_status()) return st;
    st = driver_->gate_gelu_bf16(
        frt_buffer_dptr(gate->buffer), frt_buffer_dptr(hidden->buffer),
        frt_buffer_dptr(hidden->buffer),
        static_cast<std::size_t>(sequence) * 16384, stream);
    if (!st.ok_status()) return st;
    st = driver_->bf16_nn(
        frt_buffer_dptr(hidden->buffer), frt_buffer_dptr(down_weight->buffer),
        frt_buffer_dptr(x_norm->buffer), sequence, 2048, 16384, stream);
    if (!st.ok_status()) return st;
    return driver_->residual_add_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(x_norm->buffer),
        static_cast<std::size_t>(sequence) * 2048, stream);
}
#endif

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

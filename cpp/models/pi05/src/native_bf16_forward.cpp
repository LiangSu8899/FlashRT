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
modalities::Status NativeBf16Forward::vision_layer(
    int layer,
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    NativeRtxAttentionWorkspace* attention,
    const NativeRtxAttentionDriver* attention_driver,
    std::uintptr_t stream) const {
    if (!driver_ || !workspace || !attention || !attention_driver ||
        !attention_driver->status().ok_status() || layer < 0 || layer >= 27) {
        return invalid("native vision layer owner is invalid");
    }
    const int sequence = workspace->vision_sequence();
    const int num_views = sequence / 256;
    const NativeWorkspaceBuffer* x = workspace->find("vision_x");
    const NativeWorkspaceBuffer* x_norm = workspace->find("vision_x_norm");
    const NativeWorkspaceBuffer* qkv = workspace->find("vision_QKV");
    const NativeWorkspaceBuffer* hidden = workspace->find("vision_hidden");
    const NativeAttentionBuffer* query = attention->find("attn_vis_Q");
    const NativeAttentionBuffer* key = attention->find("attn_vis_K");
    const NativeAttentionBuffer* value = attention->find("attn_vis_V");
    const std::string suffix = std::to_string(layer);
    const NativeDeviceWeight* qkv_weight =
        weights.find("vision_attn_qkv_w_" + suffix);
    const NativeDeviceWeight* qkv_bias =
        weights.find("vision_attn_qkv_b_" + suffix);
    const NativeDeviceWeight* output_weight =
        weights.find("vision_attn_o_w_" + suffix);
    const NativeDeviceWeight* output_bias =
        weights.find("vision_attn_o_b_" + suffix);
    const NativeDeviceWeight* up_weight =
        weights.find("vision_ffn_up_w_" + suffix);
    const NativeDeviceWeight* up_bias =
        weights.find("vision_ffn_up_b_" + suffix);
    const NativeDeviceWeight* down_weight =
        weights.find("vision_ffn_down_w_" + suffix);
    const NativeDeviceWeight* down_bias =
        weights.find("vision_ffn_down_b_" + suffix);
    const NativeDeviceWeight* ffn_norm_weight =
        weights.find("vision_pre_ffn_norm_w_" + suffix);
    const NativeDeviceWeight* ffn_norm_bias =
        weights.find("vision_pre_ffn_norm_b_" + suffix);
    if (sequence <= 0 || sequence % 256 || num_views < 1 || num_views > 3 ||
        !shape_is(x, {static_cast<std::uint64_t>(sequence), 1152}) ||
        !shape_is(x_norm, {static_cast<std::uint64_t>(sequence), 1152}) ||
        !shape_is(qkv, {static_cast<std::uint64_t>(sequence), 3456}) ||
        !shape_is(hidden, {static_cast<std::uint64_t>(sequence), 4304}) ||
        !shape_is(query, {static_cast<std::uint64_t>(num_views), 256, 16,
                          72}) ||
        !shape_is(key, {static_cast<std::uint64_t>(num_views), 256, 16, 72}) ||
        !shape_is(value,
                  {static_cast<std::uint64_t>(num_views), 256, 16, 72}) ||
        !shape_is(qkv_weight, {1152, 3456}) ||
        !shape_is(qkv_bias, {3456}) ||
        !shape_is(output_weight, {1152, 1152}) ||
        !shape_is(output_bias, {1152}) ||
        !shape_is(up_weight, {1152, 4304}) ||
        !shape_is(up_bias, {4304}) ||
        !shape_is(down_weight, {4304, 1152}) ||
        !shape_is(down_bias, {1152}) ||
        !shape_is(ffn_norm_weight, {1152}) ||
        !shape_is(ffn_norm_bias, {1152})) {
        return invalid("native vision layer buffers or weights are invalid");
    }
    modalities::Status st = driver_->bf16_nn(
        frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(qkv_weight->buffer),
        frt_buffer_dptr(qkv->buffer), sequence, 3456, 1152, stream);
    if (!st.ok_status()) return st;
    st = driver_->add_bias_bf16(
        frt_buffer_dptr(qkv->buffer), frt_buffer_dptr(qkv_bias->buffer),
        sequence, 3456, stream);
    if (!st.ok_status()) return st;
    st = driver_->qkv_split_bf16(
        frt_buffer_dptr(qkv->buffer), frt_buffer_dptr(query->buffer),
        frt_buffer_dptr(key->buffer), frt_buffer_dptr(value->buffer), sequence,
        1152, 1152, 1152, stream);
    if (!st.ok_status()) return st;
    st = attention_driver->vision(stream);
    if (!st.ok_status()) return st;
    st = driver_->bf16_nn(
        attention_driver->vision_output(),
        frt_buffer_dptr(output_weight->buffer), frt_buffer_dptr(x_norm->buffer),
        sequence, 1152, 1152, stream);
    if (!st.ok_status()) return st;
    st = driver_->bias_residual_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(x_norm->buffer),
        frt_buffer_dptr(output_bias->buffer), sequence, 1152, stream);
    if (!st.ok_status()) return st;
    st = driver_->layer_norm_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(ffn_norm_weight->buffer),
        frt_buffer_dptr(ffn_norm_bias->buffer), frt_buffer_dptr(x_norm->buffer),
        sequence, 1152, 1e-5f, stream);
    if (!st.ok_status()) return st;
    st = driver_->bf16_nn(
        frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(up_weight->buffer),
        frt_buffer_dptr(hidden->buffer), sequence, 4304, 1152, stream);
    if (!st.ok_status()) return st;
    st = driver_->add_bias_bf16(
        frt_buffer_dptr(hidden->buffer), frt_buffer_dptr(up_bias->buffer),
        sequence, 4304, stream);
    if (!st.ok_status()) return st;
    st = driver_->gelu_bf16(
        frt_buffer_dptr(hidden->buffer),
        static_cast<std::size_t>(sequence) * 4304, stream);
    if (!st.ok_status()) return st;
    st = driver_->bf16_nn(
        frt_buffer_dptr(hidden->buffer), frt_buffer_dptr(down_weight->buffer),
        frt_buffer_dptr(x_norm->buffer), sequence, 1152, 4304, stream);
    if (!st.ok_status()) return st;
    st = driver_->bias_residual_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(x_norm->buffer),
        frt_buffer_dptr(down_bias->buffer), sequence, 1152, stream);
    if (!st.ok_status() || layer == 26) return st;
    const NativeDeviceWeight* next_norm_weight = weights.find(
        "vision_pre_attn_norm_w_" + std::to_string(layer + 1));
    const NativeDeviceWeight* next_norm_bias = weights.find(
        "vision_pre_attn_norm_b_" + std::to_string(layer + 1));
    if (!shape_is(next_norm_weight, {1152}) ||
        !shape_is(next_norm_bias, {1152})) {
        return invalid("native next vision norm weights are invalid");
    }
    return driver_->layer_norm_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(next_norm_weight->buffer),
        frt_buffer_dptr(next_norm_bias->buffer), frt_buffer_dptr(x_norm->buffer),
        sequence, 1152, 1e-5f, stream);
}

modalities::Status NativeBf16Forward::vision(
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    NativeRtxAttentionWorkspace* attention,
    const NativeRtxAttentionDriver* attention_driver,
    std::uintptr_t stream) const {
    if (!driver_ || !workspace || !attention || !attention_driver) {
        return invalid("native vision owner is invalid");
    }
    const int sequence = workspace->vision_sequence();
    const int encoder_sequence = workspace->encoder_vision_sequence();
    const int num_views = sequence / 256;
    const int pool_area = encoder_sequence > 0 ? sequence / encoder_sequence : 0;
    const int pool_factor = pool_area == 1 ? 1 : pool_area == 4 ? 2 :
                            pool_area == 16 ? 4 : 0;
    const NativeWorkspaceBuffer* images =
        workspace->find("observation_images_normalized");
    const NativeWorkspaceBuffer* patches = workspace->find("vision_patches");
    const NativeWorkspaceBuffer* position =
        workspace->find("vision_pos_embed_expanded");
    const NativeWorkspaceBuffer* x = workspace->find("vision_x");
    const NativeWorkspaceBuffer* x_norm = workspace->find("vision_x_norm");
    const NativeWorkspaceBuffer* pooled = workspace->find("vision_x_pooled");
    const NativeWorkspaceBuffer* encoder_x = workspace->find("encoder_x");
    const NativeDeviceWeight* patch_weight =
        weights.find("vision_patch_embedding_w");
    const NativeDeviceWeight* patch_bias =
        weights.find("vision_patch_embedding_b");
    const NativeDeviceWeight* first_norm_weight =
        weights.find("vision_pre_attn_norm_w_0");
    const NativeDeviceWeight* first_norm_bias =
        weights.find("vision_pre_attn_norm_b_0");
    const NativeDeviceWeight* final_norm_weight =
        weights.find("vision_final_norm_w");
    const NativeDeviceWeight* final_norm_bias =
        weights.find("vision_final_norm_b");
    const NativeDeviceWeight* projector_weight =
        weights.find("encoder_multi_modal_projector_w");
    const NativeDeviceWeight* projector_bias =
        weights.find("encoder_multi_modal_projector_b");
    if (sequence <= 0 || sequence % 256 || encoder_sequence <= 0 ||
        sequence % encoder_sequence || num_views < 1 || num_views > 3 ||
        !pool_factor ||
        !shape_is(images, {static_cast<std::uint64_t>(num_views), 224, 224,
                           3}) ||
        !shape_is(patches, {static_cast<std::uint64_t>(sequence), 588}) ||
        !shape_is(position, {static_cast<std::uint64_t>(sequence), 1152}) ||
        !shape_is(x, {static_cast<std::uint64_t>(sequence), 1152}) ||
        !shape_is(x_norm, {static_cast<std::uint64_t>(sequence), 1152}) ||
        !shape_is(pooled,
                  {static_cast<std::uint64_t>(encoder_sequence), 1152}) ||
        !encoder_x || encoder_x->dtype != modalities::DType::kBFloat16 ||
        encoder_x->shape.size() != 2 ||
        encoder_x->shape[0] < static_cast<std::uint64_t>(encoder_sequence) ||
        encoder_x->shape[1] != 2048 ||
        !shape_is(patch_weight, {14, 14, 3, 1152}) ||
        !shape_is(patch_bias, {1152}) ||
        !shape_is(first_norm_weight, {1152}) ||
        !shape_is(first_norm_bias, {1152}) ||
        !shape_is(final_norm_weight, {1152}) ||
        !shape_is(final_norm_bias, {1152}) ||
        !shape_is(projector_weight, {1152, 2048}) ||
        !shape_is(projector_bias, {2048})) {
        return invalid("native vision buffers or weights are invalid");
    }
    modalities::Status st = driver_->patch_im2col_16bit(
        frt_buffer_dptr(images->buffer), frt_buffer_dptr(patches->buffer),
        num_views, stream);
    if (!st.ok_status()) return st;
    st = driver_->bf16_nn(
        frt_buffer_dptr(patches->buffer), frt_buffer_dptr(patch_weight->buffer),
        frt_buffer_dptr(x->buffer), sequence, 1152, 588, stream);
    if (!st.ok_status()) return st;
    st = driver_->bias_residual_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(position->buffer),
        frt_buffer_dptr(patch_bias->buffer), sequence, 1152, stream);
    if (!st.ok_status()) return st;
    st = driver_->layer_norm_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(first_norm_weight->buffer),
        frt_buffer_dptr(first_norm_bias->buffer), frt_buffer_dptr(x_norm->buffer),
        sequence, 1152, 1e-5f, stream);
    if (!st.ok_status()) return st;
    for (int layer = 0; layer < 27; ++layer) {
        st = vision_layer(layer, weights, workspace, attention,
                          attention_driver, stream);
        if (!st.ok_status()) return st;
    }
    if (pool_factor > 1) {
        st = driver_->avg_pool_vision_bf16(
            frt_buffer_dptr(x->buffer), frt_buffer_dptr(pooled->buffer),
            num_views, 16, 16, 1152, pool_factor, stream);
        if (!st.ok_status()) return st;
    }
    st = driver_->layer_norm_bf16(
        frt_buffer_dptr(pooled->buffer),
        frt_buffer_dptr(final_norm_weight->buffer),
        frt_buffer_dptr(final_norm_bias->buffer), frt_buffer_dptr(x_norm->buffer),
        encoder_sequence, 1152, 1e-5f, stream);
    if (!st.ok_status()) return st;
    st = driver_->bf16_nn(
        frt_buffer_dptr(x_norm->buffer),
        frt_buffer_dptr(projector_weight->buffer),
        frt_buffer_dptr(encoder_x->buffer), encoder_sequence, 2048, 1152,
        stream);
    if (!st.ok_status()) return st;
    return driver_->add_bias_bf16(
        frt_buffer_dptr(encoder_x->buffer),
        frt_buffer_dptr(projector_bias->buffer), encoder_sequence, 2048,
        stream);
}

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

modalities::Status NativeBf16Forward::encoder(
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    NativeRtxAttentionWorkspace* attention,
    const NativeRtxAttentionDriver* attention_driver,
    std::uintptr_t stream) const {
    for (int layer = 0; layer < 18; ++layer) {
        modalities::Status st = encoder_layer(
            layer, weights, workspace, attention, attention_driver, stream);
        if (!st.ok_status()) return st;
    }
    return modalities::Status::ok();
}
#endif

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

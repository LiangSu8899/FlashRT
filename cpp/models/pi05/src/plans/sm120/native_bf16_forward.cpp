#include "flashrt/cpp/models/pi05/plans/sm120/native_bf16_forward.h"

#include <climits>
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

NativeRtxScaleSite vision_site(int layer, int slot) {
    return {NativeRtxScaleDomain::kVision, layer * 4 + slot};
}

NativeRtxScaleSite encoder_site(int layer, int slot) {
    return {NativeRtxScaleDomain::kEncoder, layer * 4 + slot};
}

NativeRtxScaleSite decoder_site(int step, int layer, int slot) {
    return {NativeRtxScaleDomain::kDecoder,
            (step * 18 + layer) * 4 + slot};
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
    const std::string qkv_name =
        "encoder_attn_qkv_w_" + std::to_string(layer);
    if (!shape_is(x, {static_cast<std::uint64_t>(sequence), 2048}) ||
        !shape_is(x_norm, {static_cast<std::uint64_t>(sequence), 2048}) ||
        !shape_is(qkv, {static_cast<std::uint64_t>(sequence), 2560}) ||
        !shape_is(rms, {2048}) ||
        !shape_is(rope, {static_cast<std::uint64_t>(sequence), 256}) ||
        !shape_is(query, {static_cast<std::uint64_t>(sequence), 8, 256}) ||
        !cache || cache->dtype != NativeAttentionDType::kBf16 ||
        cache->shape.size() != 4 || cache->shape[0] != 18 ||
        cache->shape[1] < static_cast<std::uint64_t>(sequence) ||
        cache->shape[2] != 1 || cache->shape[3] != 256 ||
        !value_cache || value_cache->dtype != NativeAttentionDType::kBf16 ||
        value_cache->shape != cache->shape ||
        !linear_->weight_shape_is(weights, qkv_name, {2048, 2560})) {
        return invalid("native encoder QKV buffers or weight are invalid");
    }
    void* key = attention->encoder_k_layer_dptr(layer);
    void* value = attention->encoder_v_layer_dptr(layer);
    if (!key || !value) {
        return invalid("native encoder QKV cache layer is invalid");
    }
    modalities::Status st;
    if (linear_->static_fp8()) {
        const NativeWorkspaceBuffer* scratch =
            workspace->find("rtx_fp8_scratch");
        const float* scale = linear_->scale(
            *workspace, encoder_site(layer, 0));
        if (!scratch || !scale) {
            return invalid("native encoder FP8 QKV storage is invalid");
        }
        st = layer == 0
                 ? driver_->rms_norm_fp8_bf16(
                       frt_buffer_dptr(x->buffer),
                       frt_buffer_dptr(rms->buffer),
                       frt_buffer_dptr(scratch->buffer), sequence, 2048,
                       1e-6f, scale, stream)
                 : driver_->residual_add_rms_norm_fp8_bf16(
                       frt_buffer_dptr(x->buffer),
                       frt_buffer_dptr(x_norm->buffer),
                       frt_buffer_dptr(rms->buffer),
                       frt_buffer_dptr(scratch->buffer), sequence, 2048,
                       1e-6f, scale, stream);
        if (!st.ok_status()) return st;
        st = linear_->run_prequantized(
            weights, qkv_name, encoder_site(layer, 0), *workspace,
            frt_buffer_dptr(scratch->buffer), frt_buffer_dptr(qkv->buffer),
            sequence, 2560, 2048, stream);
    } else {
        st = driver_->rms_norm_bf16(
            frt_buffer_dptr(x->buffer), frt_buffer_dptr(rms->buffer),
            frt_buffer_dptr(x_norm->buffer), sequence, 2048, 1e-6f, stream);
        if (!st.ok_status()) return st;
        st = linear_->run(
            weights, workspace, qkv_name, encoder_site(layer, 0),
            frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(qkv->buffer),
            sequence, 2560, 2048, stream);
    }
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
    const std::string qkv_name = "vision_attn_qkv_w_" + suffix;
    const std::string output_name = "vision_attn_o_w_" + suffix;
    const std::string up_name = "vision_ffn_up_w_" + suffix;
    const std::string down_name = "vision_ffn_down_w_" + suffix;
    const NativeDeviceWeight* qkv_bias =
        weights.find("vision_attn_qkv_b_" + suffix);
    const NativeDeviceWeight* output_bias =
        weights.find("vision_attn_o_b_" + suffix);
    const NativeDeviceWeight* up_bias =
        weights.find("vision_ffn_up_b_" + suffix);
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
        !linear_->weight_shape_is(weights, qkv_name, {1152, 3456}) ||
        !shape_is(qkv_bias, {3456}) ||
        !linear_->weight_shape_is(weights, output_name, {1152, 1152}) ||
        !shape_is(output_bias, {1152}) ||
        !linear_->weight_shape_is(weights, up_name, {1152, 4304}) ||
        !shape_is(up_bias, {4304}) ||
        !linear_->weight_shape_is(weights, down_name, {4304, 1152}) ||
        !shape_is(down_bias, {1152}) ||
        !shape_is(ffn_norm_weight, {1152}) ||
        !shape_is(ffn_norm_bias, {1152})) {
        return invalid("native vision layer buffers or weights are invalid");
    }
    modalities::Status st = linear_->run(
        weights, workspace, qkv_name, vision_site(layer, 0),
        frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(qkv->buffer),
        sequence, 3456, 1152, stream);
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
    st = linear_->run(
        weights, workspace, output_name, vision_site(layer, 1),
        attention_driver->vision_output(), frt_buffer_dptr(x_norm->buffer),
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
    st = linear_->run(
        weights, workspace, up_name, vision_site(layer, 2),
        frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(hidden->buffer),
        sequence, 4304, 1152, stream);
    if (!st.ok_status()) return st;
    st = driver_->add_bias_bf16(
        frt_buffer_dptr(hidden->buffer), frt_buffer_dptr(up_bias->buffer),
        sequence, 4304, stream);
    if (!st.ok_status()) return st;
    st = driver_->gelu_bf16(
        frt_buffer_dptr(hidden->buffer),
        static_cast<std::size_t>(sequence) * 4304, stream);
    if (!st.ok_status()) return st;
    st = linear_->run(
        weights, workspace, down_name, vision_site(layer, 3),
        frt_buffer_dptr(hidden->buffer), frt_buffer_dptr(x_norm->buffer),
        sequence, 1152, 4304, stream);
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
    const std::string projector_name = "encoder_multi_modal_projector_w";
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
        !linear_->weight_shape_is(weights, projector_name, {1152, 2048}) ||
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
    st = linear_->run(
        weights, workspace, projector_name,
        {NativeRtxScaleDomain::kVision, 108},
        frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(encoder_x->buffer),
        encoder_sequence, 2048, 1152, stream);
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
    const std::string suffix = std::to_string(layer);
    const std::string output_name = "encoder_attn_o_w_" + suffix;
    const std::string gate_name = "encoder_ffn_gate_w_" + suffix;
    const std::string up_name = "encoder_ffn_up_w_" + suffix;
    const std::string gate_up_name = "encoder_ffn_gate_up_w_" + suffix;
    const std::string down_name = "encoder_ffn_down_w_" + suffix;
    const bool ffn_weights_valid =
        linear_->fp8()
            ? linear_->weight_shape_is(
                  weights, gate_up_name, {2048, 32768})
            : linear_->weight_shape_is(weights, gate_name, {2048, 16384}) &&
                  linear_->weight_shape_is(
                      weights, up_name, {2048, 16384});
    if (!shape_is(x, {static_cast<std::uint64_t>(sequence), 2048}) ||
        !shape_is(x_norm, {static_cast<std::uint64_t>(sequence), 2048}) ||
        !shape_is(gate, {static_cast<std::uint64_t>(sequence), 32768}) ||
        !shape_is(hidden, {static_cast<std::uint64_t>(sequence), 16384}) ||
        !shape_is(rms, {2048}) || !ffn_weights_valid ||
        !linear_->weight_shape_is(weights, output_name, {2048, 2048}) ||
        !linear_->weight_shape_is(weights, down_name, {16384, 2048})) {
        return invalid("native encoder layer buffers or weights are invalid");
    }
    st = attention_driver->encoder(layer, stream);
    if (!st.ok_status()) return st;
    st = linear_->run(
        weights, workspace, output_name, encoder_site(layer, 1),
        attention_driver->encoder_output(), frt_buffer_dptr(x_norm->buffer),
        sequence, 2048, 2048, stream);
    if (!st.ok_status()) return st;
    if (linear_->static_fp8()) {
        const NativeWorkspaceBuffer* scratch =
            workspace->find("rtx_fp8_scratch");
        const float* gate_up_scale = linear_->scale(
            *workspace, encoder_site(layer, 2));
        const float* down_scale = linear_->scale(
            *workspace, encoder_site(layer, 3));
        if (!scratch || !gate_up_scale || !down_scale) {
            return invalid("native encoder fused FP8 storage is invalid");
        }
        st = driver_->residual_add_rms_norm_fp8_bf16(
            frt_buffer_dptr(x->buffer), frt_buffer_dptr(x_norm->buffer),
            frt_buffer_dptr(rms->buffer), frt_buffer_dptr(scratch->buffer),
            sequence, 2048, 1e-6f, gate_up_scale, stream);
        if (!st.ok_status()) return st;
        st = linear_->run_prequantized(
            weights, gate_up_name, encoder_site(layer, 2), *workspace,
            frt_buffer_dptr(scratch->buffer), frt_buffer_dptr(gate->buffer),
            sequence, 32768, 2048, stream);
        if (!st.ok_status()) return st;
        st = driver_->gate_gelu_merged_fp8_bf16(
            frt_buffer_dptr(gate->buffer),
            frt_buffer_dptr(scratch->buffer), sequence, 16384, down_scale,
            stream);
        if (!st.ok_status()) return st;
        return linear_->run_prequantized(
            weights, down_name, encoder_site(layer, 3), *workspace,
            frt_buffer_dptr(scratch->buffer), frt_buffer_dptr(x_norm->buffer),
            sequence, 2048, 16384, stream);
    }
    st = driver_->residual_add_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(x_norm->buffer),
        static_cast<std::size_t>(sequence) * 2048, stream);
    if (!st.ok_status()) return st;
    st = driver_->rms_norm_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(rms->buffer),
        frt_buffer_dptr(x_norm->buffer), sequence, 2048, 1e-6f, stream);
    if (!st.ok_status()) return st;
    if (linear_->fp8()) {
        st = linear_->run(
            weights, workspace, gate_up_name, encoder_site(layer, 2),
            frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(gate->buffer),
            sequence, 32768, 2048, stream);
        if (!st.ok_status()) return st;
        st = driver_->gate_gelu_merged_bf16(
            frt_buffer_dptr(gate->buffer), frt_buffer_dptr(hidden->buffer),
            sequence, 16384, stream);
    } else {
        st = linear_->run(
            weights, workspace, gate_name, encoder_site(layer, 2),
            frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(gate->buffer),
            sequence, 16384, 2048, stream);
        if (!st.ok_status()) return st;
        st = linear_->run(
            weights, workspace, up_name, encoder_site(layer, 2),
            frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(hidden->buffer),
            sequence, 16384, 2048, stream);
        if (!st.ok_status()) return st;
        st = driver_->gate_gelu_bf16(
            frt_buffer_dptr(gate->buffer), frt_buffer_dptr(hidden->buffer),
            frt_buffer_dptr(hidden->buffer),
            static_cast<std::size_t>(sequence) * 16384, stream);
    }
    if (!st.ok_status()) return st;
    st = linear_->run(
        weights, workspace, down_name, encoder_site(layer, 3),
        frt_buffer_dptr(hidden->buffer), frt_buffer_dptr(x_norm->buffer),
        sequence, 2048, 16384, stream);
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

modalities::Status NativeBf16Forward::decoder_layer(
    int layer,
    int step,
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    NativeRtxAttentionWorkspace* attention,
    const NativeRtxAttentionDriver* attention_driver,
    std::uintptr_t stream) const {
    if (!driver_ || !workspace || !attention || !attention_driver ||
        !attention_driver->status().ok_status() || layer < 0 || layer >= 18) {
        return invalid("native decoder layer owner is invalid");
    }
    const NativeWorkspaceBuffer* x = workspace->find("decoder_x");
    const NativeWorkspaceBuffer* x_norm = workspace->find("x_normed_buf");
    const NativeWorkspaceBuffer* gate = workspace->find("gate_buf");
    const NativeWorkspaceBuffer* qkv = workspace->find("decoder_QKV");
    const NativeWorkspaceBuffer* hidden = workspace->find("decoder_hidden");
    const NativeWorkspaceBuffer* gate_projection =
        workspace->find("decoder_gate_merged");
    const NativeWorkspaceBuffer* rms = workspace->find("decoder_rms_ones");
    const NativeWorkspaceBuffer* rope =
        workspace->find("decoder_rope_weights");
    const NativeWorkspaceBuffer* style_attn =
        workspace->find("decoder_style_attn");
    const NativeWorkspaceBuffer* style_ffn =
        workspace->find("decoder_style_ffn");
    if (!x || x->shape.size() != 2) {
        return invalid("native decoder workspace is invalid");
    }
    const int sequence = static_cast<int>(x->shape[0]);
    const NativeAttentionBuffer* query = attention->find("attn_dec_Q");
    const NativeAttentionBuffer* devpos = attention->find("attn_dec_devpos");
    const std::string suffix = std::to_string(layer);
    const std::string qkv_name = "decoder_attn_qkv_w_" + suffix;
    const std::string output_name = "decoder_attn_o_w_" + suffix;
    const std::string gate_name = "decoder_ffn_gate_w_" + suffix;
    const std::string up_name = "decoder_ffn_up_w_" + suffix;
    const std::string gate_up_name = "decoder_ffn_gate_up_w_" + suffix;
    const std::string down_name = "decoder_ffn_down_w_" + suffix;
    const bool ffn_weights_valid =
        linear_->fp8()
            ? linear_->weight_shape_is(
                  weights, gate_up_name, {1024, 8192})
            : linear_->weight_shape_is(weights, gate_name, {1024, 4096}) &&
                  linear_->weight_shape_is(weights, up_name, {1024, 4096});
    if (sequence <= 0 || step < 0 ||
        !shape_is(x, {static_cast<std::uint64_t>(sequence), 1024}) ||
        !shape_is(x_norm, {static_cast<std::uint64_t>(sequence), 1024}) ||
        !shape_is(gate, {static_cast<std::uint64_t>(sequence), 1024}) ||
        !shape_is(qkv, {static_cast<std::uint64_t>(sequence), 2560}) ||
        !shape_is(hidden, {static_cast<std::uint64_t>(sequence), 4096}) ||
        !shape_is(gate_projection,
                  {static_cast<std::uint64_t>(sequence), 8192}) ||
        !shape_is(rms, {1024}) ||
        !shape_is(rope, {static_cast<std::uint64_t>(sequence), 256}) ||
        !style_attn || style_attn->dtype != modalities::DType::kBFloat16 ||
        style_attn->shape.size() != 4 ||
        style_attn->shape[0] <= static_cast<std::uint64_t>(step) ||
        style_attn->shape[1] != 18 ||
        style_attn->shape[2] != static_cast<std::uint64_t>(sequence) ||
        style_attn->shape[3] != 3072 || !style_ffn ||
        style_ffn->dtype != modalities::DType::kBFloat16 ||
        style_ffn->shape != style_attn->shape ||
        !shape_is(query, {static_cast<std::uint64_t>(sequence), 8, 256}) ||
        !devpos || devpos->dtype != NativeAttentionDType::kInt32 ||
        devpos->shape != std::vector<std::uint64_t>({1}) ||
        !ffn_weights_valid ||
        !linear_->weight_shape_is(weights, qkv_name, {1024, 2560}) ||
        !linear_->weight_shape_is(weights, output_name, {2048, 1024}) ||
        !linear_->weight_shape_is(weights, down_name, {4096, 1024})) {
        return invalid("native decoder layer buffers or weights are invalid");
    }
    const std::size_t style_offset =
        (static_cast<std::size_t>(step) * 18 + layer) * sequence * 3072 *
        sizeof(std::uint16_t);
    const auto* attn_style =
        static_cast<const unsigned char*>(frt_buffer_dptr(style_attn->buffer)) +
        style_offset;
    const auto* ffn_style =
        static_cast<const unsigned char*>(frt_buffer_dptr(style_ffn->buffer)) +
        style_offset;
    modalities::Status st;
    const NativeWorkspaceBuffer* fp8_scratch =
        linear_->static_fp8() ? workspace->find("rtx_fp8_scratch") : nullptr;
    if (linear_->static_fp8()) {
        const float* qkv_scale = linear_->scale(
            *workspace, decoder_site(step, layer, 0));
        if (!fp8_scratch || !qkv_scale) {
            return invalid("native decoder FP8 QKV storage is invalid");
        }
        if (layer == 0) {
            st = driver_->ada_rms_norm_style_fp8_bf16(
                frt_buffer_dptr(x->buffer), frt_buffer_dptr(rms->buffer),
                attn_style, frt_buffer_dptr(fp8_scratch->buffer),
                frt_buffer_dptr(gate->buffer), sequence, 1024, 1e-6f,
                qkv_scale, stream);
            if (!st.ok_status()) return st;
        }
        st = linear_->run_prequantized(
            weights, qkv_name, decoder_site(step, layer, 0), *workspace,
            frt_buffer_dptr(fp8_scratch->buffer),
            frt_buffer_dptr(qkv->buffer), sequence, 2560, 1024, stream);
    } else {
        st = driver_->ada_rms_norm_style_bf16(
            frt_buffer_dptr(x->buffer), frt_buffer_dptr(rms->buffer),
            attn_style, frt_buffer_dptr(x_norm->buffer),
            frt_buffer_dptr(gate->buffer), sequence, 1024, 1e-6f, stream);
        if (!st.ok_status()) return st;
        st = linear_->run(
            weights, workspace, qkv_name, decoder_site(step, layer, 0),
            frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(qkv->buffer),
            sequence, 2560, 1024, stream);
    }
    if (!st.ok_status()) return st;
    st = driver_->qkv_split_rope_devpos_bf16(
        frt_buffer_dptr(qkv->buffer), frt_buffer_dptr(rope->buffer),
        frt_buffer_dptr(query->buffer), attention->encoder_k_layer_dptr(layer),
        attention->encoder_v_layer_dptr(layer),
        frt_buffer_dptr(devpos->buffer), sequence, 2048, 256, 256, 256,
        stream);
    if (!st.ok_status()) return st;
    st = attention_driver->decoder(layer, stream);
    if (!st.ok_status()) return st;
    st = linear_->run(
        weights, workspace, output_name, decoder_site(step, layer, 1),
        attention_driver->decoder_output(), frt_buffer_dptr(x_norm->buffer),
        sequence, 1024, 2048, stream);
    if (!st.ok_status()) return st;
    if (linear_->static_fp8()) {
        const float* gate_up_scale = linear_->scale(
            *workspace, decoder_site(step, layer, 2));
        const float* down_scale = linear_->scale(
            *workspace, decoder_site(step, layer, 3));
        if (!fp8_scratch || !gate_up_scale || !down_scale) {
            return invalid("native decoder fused FP8 storage is invalid");
        }
        st = driver_->gate_residual_ada_norm_fp8_bf16(
            frt_buffer_dptr(x->buffer), frt_buffer_dptr(x_norm->buffer),
            frt_buffer_dptr(gate->buffer), frt_buffer_dptr(rms->buffer),
            ffn_style, frt_buffer_dptr(fp8_scratch->buffer),
            frt_buffer_dptr(gate->buffer), sequence, 1024, 1e-6f,
            gate_up_scale, stream);
        if (!st.ok_status()) return st;
        st = linear_->run_prequantized(
            weights, gate_up_name, decoder_site(step, layer, 2), *workspace,
            frt_buffer_dptr(fp8_scratch->buffer),
            frt_buffer_dptr(gate_projection->buffer), sequence, 8192, 1024,
            stream);
        if (!st.ok_status()) return st;
        st = driver_->gate_gelu_merged_fp8_bf16(
            frt_buffer_dptr(gate_projection->buffer),
            frt_buffer_dptr(fp8_scratch->buffer), sequence, 4096,
            down_scale, stream);
        if (!st.ok_status()) return st;
        st = linear_->run_prequantized(
            weights, down_name, decoder_site(step, layer, 3), *workspace,
            frt_buffer_dptr(fp8_scratch->buffer),
            frt_buffer_dptr(x_norm->buffer), sequence, 1024, 4096, stream);
        if (!st.ok_status()) return st;
        if (layer == 17) {
            return driver_->gate_mul_residual_bf16(
                frt_buffer_dptr(x->buffer), frt_buffer_dptr(x_norm->buffer),
                frt_buffer_dptr(gate->buffer),
                static_cast<std::size_t>(sequence) * 1024, stream);
        }
        const float* next_qkv_scale = linear_->scale(
            *workspace, decoder_site(step, layer + 1, 0));
        if (!next_qkv_scale) {
            return invalid("native decoder next-layer FP8 scale is invalid");
        }
        const auto* next_attn_style =
            attn_style + static_cast<std::size_t>(sequence) * 3072 *
                             sizeof(std::uint16_t);
        return driver_->gate_residual_ada_norm_fp8_bf16(
            frt_buffer_dptr(x->buffer), frt_buffer_dptr(x_norm->buffer),
            frt_buffer_dptr(gate->buffer), frt_buffer_dptr(rms->buffer),
            next_attn_style, frt_buffer_dptr(fp8_scratch->buffer),
            frt_buffer_dptr(gate->buffer), sequence, 1024, 1e-6f,
            next_qkv_scale, stream);
    }
    st = driver_->gate_mul_residual_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(x_norm->buffer),
        frt_buffer_dptr(gate->buffer),
        static_cast<std::size_t>(sequence) * 1024, stream);
    if (!st.ok_status()) return st;
    st = driver_->ada_rms_norm_style_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(rms->buffer), ffn_style,
        frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(gate->buffer),
        sequence, 1024, 1e-6f, stream);
    if (!st.ok_status()) return st;
    if (linear_->fp8()) {
        st = linear_->run(
            weights, workspace, gate_up_name, decoder_site(step, layer, 2),
            frt_buffer_dptr(x_norm->buffer),
            frt_buffer_dptr(gate_projection->buffer), sequence, 8192, 1024,
            stream);
        if (!st.ok_status()) return st;
        st = driver_->gate_gelu_merged_bf16(
            frt_buffer_dptr(gate_projection->buffer),
            frt_buffer_dptr(hidden->buffer), sequence, 4096, stream);
    } else {
        st = linear_->run(
            weights, workspace, gate_name, decoder_site(step, layer, 2),
            frt_buffer_dptr(x_norm->buffer),
            frt_buffer_dptr(gate_projection->buffer), sequence, 4096, 1024,
            stream);
        if (!st.ok_status()) return st;
        st = linear_->run(
            weights, workspace, up_name, decoder_site(step, layer, 2),
            frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(hidden->buffer),
            sequence, 4096, 1024, stream);
        if (!st.ok_status()) return st;
        st = driver_->gate_gelu_bf16(
            frt_buffer_dptr(gate_projection->buffer),
            frt_buffer_dptr(hidden->buffer), frt_buffer_dptr(hidden->buffer),
            static_cast<std::size_t>(sequence) * 4096, stream);
    }
    if (!st.ok_status()) return st;
    st = linear_->run(
        weights, workspace, down_name, decoder_site(step, layer, 3),
        frt_buffer_dptr(hidden->buffer), frt_buffer_dptr(x_norm->buffer),
        sequence, 1024, 4096, stream);
    if (!st.ok_status()) return st;
    return driver_->gate_mul_residual_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(x_norm->buffer),
        frt_buffer_dptr(gate->buffer),
        static_cast<std::size_t>(sequence) * 1024, stream);
}

modalities::Status NativeBf16Forward::diffusion_step(
    int step,
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    NativeRtxAttentionWorkspace* attention,
    const NativeRtxAttentionDriver* attention_driver,
    std::uintptr_t stream) const {
    if (!driver_ || !workspace || !attention || !attention_driver || step < 0) {
        return invalid("native diffusion step owner is invalid");
    }
    const NativeWorkspaceBuffer* noise = workspace->find("diffusion_noise");
    const NativeWorkspaceBuffer* x = workspace->find("decoder_x");
    const NativeWorkspaceBuffer* action =
        workspace->find("decoder_action_buf");
    const NativeWorkspaceBuffer* x_norm = workspace->find("x_normed_buf");
    const NativeWorkspaceBuffer* gate = workspace->find("gate_buf");
    const NativeWorkspaceBuffer* rms = workspace->find("decoder_rms_ones");
    const NativeWorkspaceBuffer* style =
        workspace->find("decoder_style_final");
    if (!noise || noise->shape.size() != 2) {
        return invalid("native diffusion workspace is invalid");
    }
    const int sequence = static_cast<int>(noise->shape[0]);
    const NativeDeviceWeight* input_weight =
        weights.find("decoder_action_in_proj_w");
    const NativeDeviceWeight* input_bias =
        weights.find("decoder_action_in_proj_b");
    const NativeDeviceWeight* output_weight =
        weights.find("decoder_action_out_proj_w");
    const NativeDeviceWeight* output_bias =
        weights.find("decoder_action_out_proj_b");
    if (sequence <= 0 ||
        !shape_is(noise, {static_cast<std::uint64_t>(sequence), 32}) ||
        !shape_is(x, {static_cast<std::uint64_t>(sequence), 1024}) ||
        !shape_is(action, {static_cast<std::uint64_t>(sequence), 32}) ||
        !shape_is(x_norm, {static_cast<std::uint64_t>(sequence), 1024}) ||
        !shape_is(gate, {static_cast<std::uint64_t>(sequence), 1024}) ||
        !shape_is(rms, {1024}) || !style ||
        style->dtype != modalities::DType::kBFloat16 ||
        style->shape.size() != 3 ||
        style->shape[0] <= static_cast<std::uint64_t>(step) ||
        style->shape[1] != static_cast<std::uint64_t>(sequence) ||
        style->shape[2] != 3072 ||
        !shape_is(input_weight, {32, 1024}) ||
        !shape_is(input_bias, {1024}) ||
        !shape_is(output_weight, {1024, 32}) ||
        !shape_is(output_bias, {32})) {
        return invalid("native diffusion buffers or weights are invalid");
    }
    modalities::Status st = driver_->bf16_nn(
        frt_buffer_dptr(noise->buffer), frt_buffer_dptr(input_weight->buffer),
        frt_buffer_dptr(x->buffer), sequence, 1024, 32, stream);
    if (!st.ok_status()) return st;
    st = driver_->add_bias_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(input_bias->buffer),
        sequence, 1024, stream);
    if (!st.ok_status()) return st;
    for (int layer = 0; layer < 18; ++layer) {
        st = decoder_layer(layer, step, weights, workspace, attention,
                           attention_driver, stream);
        if (!st.ok_status()) return st;
    }
    const std::size_t style_offset =
        static_cast<std::size_t>(step) * sequence * 3072 *
        sizeof(std::uint16_t);
    const auto* final_style =
        static_cast<const unsigned char*>(frt_buffer_dptr(style->buffer)) +
        style_offset;
    st = driver_->ada_rms_norm_style_bf16(
        frt_buffer_dptr(x->buffer), frt_buffer_dptr(rms->buffer), final_style,
        frt_buffer_dptr(x_norm->buffer), frt_buffer_dptr(gate->buffer),
        sequence, 1024, 1e-6f, stream);
    if (!st.ok_status()) return st;
    st = driver_->bf16_nn(
        frt_buffer_dptr(x_norm->buffer),
        frt_buffer_dptr(output_weight->buffer), frt_buffer_dptr(action->buffer),
        sequence, 32, 1024, stream);
    if (!st.ok_status()) return st;
    st = driver_->add_bias_bf16(
        frt_buffer_dptr(action->buffer), frt_buffer_dptr(output_bias->buffer),
        sequence, 32, stream);
    if (!st.ok_status()) return st;
    return driver_->residual_add_bf16(
        frt_buffer_dptr(noise->buffer), frt_buffer_dptr(action->buffer),
        static_cast<std::size_t>(sequence) * 32, stream);
}

#endif

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

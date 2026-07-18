#include "flashrt/cpp/models/pi05/support/native_resource_resolver.h"

#include <initializer_list>
#include <limits>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status missing(const std::string& kind,
                           const std::string& name) {
    return modalities::Status::error(
        modalities::StatusCode::kNotFound,
        "PI0.5 materialized " + kind + " is missing: " + name);
}

bool copy_shape(const std::vector<std::uint64_t>& source,
                modalities::Shape* destination) {
    if (!destination || source.empty() ||
        source.size() > modalities::Shape::kMaxRank) {
        return false;
    }
    modalities::Shape result;
    result.rank = static_cast<std::uint32_t>(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
        if (!source[i]) return false;
        result.dims[i] = source[i];
    }
    *destination = result;
    return true;
}

bool physical_buffer_is(const Pi05ResolvedBuffer& buffer,
                        modalities::DType dtype,
                        std::initializer_list<std::uint64_t> dimensions) {
    if (!buffer.buffer || !buffer.storage_identity ||
        buffer.physical_dtype != dtype ||
        buffer.physical_shape.rank != dimensions.size() ||
        buffer.physical_bytes != frt_buffer_bytes(buffer.buffer) ||
        buffer.storage_bytes != buffer.physical_bytes ||
        buffer.storage_offset != 0 ||
        buffer.storage_identity != frt_buffer_dptr(buffer.buffer)) {
        return false;
    }
    std::uint64_t elements = 1;
    std::size_t index = 0;
    for (const std::uint64_t dimension : dimensions) {
        if (!dimension ||
            buffer.physical_shape.dims[index++] != dimension ||
            elements > std::numeric_limits<std::uint64_t>::max() / dimension) {
            return false;
        }
        elements *= dimension;
    }
    const std::size_t width = modalities::dtype_size(dtype);
    return width &&
           elements <= std::numeric_limits<std::uint64_t>::max() / width &&
           buffer.physical_bytes == elements * width;
}

bool weight_storage(NativeWeightDType source,
                    Pi05WeightStorage* destination) {
    if (!destination) return false;
    switch (source) {
        case NativeWeightDType::kBf16:
            *destination = Pi05WeightStorage::kBFloat16;
            return true;
        case NativeWeightDType::kFloat16:
            *destination = Pi05WeightStorage::kFloat16;
            return true;
        case NativeWeightDType::kFp8E4M3:
            *destination = Pi05WeightStorage::kFp8E4M3;
            return true;
        case NativeWeightDType::kInt8:
        case NativeWeightDType::kFloat32:
            return false;
    }
    return false;
}

modalities::Status resolve_weight(
    const NativeDeviceWeightStore& store,
    const std::string& name,
    Pi05ResolvedWeight* out) {
    if (!out) return invalid("PI0.5 resolved weight destination is null");
    const NativeDeviceWeight* source = store.find(name);
    if (!source) return missing("weight", name);
    Pi05ResolvedWeight result;
    if (!source->buffer || !frt_buffer_dptr(source->buffer) ||
        !frt_buffer_bytes(source->buffer) ||
        !copy_shape(source->shape, &result.shape) ||
        !weight_storage(source->dtype, &result.storage)) {
        return invalid("PI0.5 materialized weight metadata is invalid");
    }
    result.device_data = frt_buffer_dptr(source->buffer);
    result.bytes = frt_buffer_bytes(source->buffer);
    *out = result;
    return modalities::Status::ok();
}

modalities::Status resolve_workspace_buffer(
    const NativeWorkspace& workspace,
    const char* name,
    Pi05ValueId value,
    const Pi05ResolvedShape& shape,
    Pi05ResolvedBuffer* out) {
    if (!name || !out) {
        return invalid("PI0.5 workspace binding request is invalid");
    }
    const NativeWorkspaceBuffer* source = workspace.find(name);
    if (!source) return missing("workspace buffer", name);
    Pi05ResolvedBuffer result;
    if (!source->buffer || !frt_buffer_dptr(source->buffer) ||
        !frt_buffer_bytes(source->buffer) ||
        !copy_shape(source->shape, &result.physical_shape)) {
        return invalid("PI0.5 workspace buffer metadata is invalid");
    }
    modalities::Status status = pi05_value_spec(value, shape,
                                                &result.logical_spec);
    if (!status.ok_status()) return status;
    result.buffer = source->buffer;
    result.physical_dtype = source->dtype;
    result.physical_bytes = frt_buffer_bytes(source->buffer);
    result.storage_identity = frt_buffer_dptr(source->buffer);
    result.storage_bytes = result.physical_bytes;
    *out = result;
    return modalities::Status::ok();
}

modalities::Status resolve_workspace_control(
    const NativeWorkspace& workspace,
    const char* name,
    Pi05ResolvedBuffer* out) {
    if (!name || !out) {
        return invalid("PI0.5 workspace control request is invalid");
    }
    const NativeWorkspaceBuffer* source = workspace.find(name);
    if (!source) return missing("workspace control", name);
    Pi05ResolvedBuffer result;
    if (!source->buffer || !frt_buffer_dptr(source->buffer) ||
        !frt_buffer_bytes(source->buffer) ||
        !copy_shape(source->shape, &result.physical_shape)) {
        return invalid("PI0.5 workspace control metadata is invalid");
    }
    result.buffer = source->buffer;
    result.physical_dtype = source->dtype;
    result.physical_bytes = frt_buffer_bytes(source->buffer);
    result.storage_identity = frt_buffer_dptr(source->buffer);
    result.storage_bytes = result.physical_bytes;
    *out = result;
    return modalities::Status::ok();
}

std::string layer_name(const char* stem, std::size_t layer) {
    return std::string(stem) + std::to_string(layer);
}

std::string feed_forward_name(const char* prefix,
                              const char* component,
                              std::size_t layer) {
    return std::string(prefix) + component + std::to_string(layer);
}

modalities::Status resolve_feed_forward(
    const NativeDeviceWeightStore& store,
    const char* prefix,
    std::size_t layer,
    NativeFeedForwardLayout layout,
    Pi05FeedForwardWeights* out) {
    if (!prefix || !out) {
        return invalid("PI0.5 feed-forward resolve request is invalid");
    }
    Pi05FeedForwardWeights result;
    modalities::Status status;
    if (layout == NativeFeedForwardLayout::kSeparateGateUp) {
        status = resolve_weight(
            store, feed_forward_name(prefix, "gate_w_", layer),
            &result.gate_weight);
        if (!status.ok_status()) return status;
        status = resolve_weight(
            store, feed_forward_name(prefix, "up_w_", layer),
            &result.up_weight);
        if (!status.ok_status()) return status;
    } else if (layout == NativeFeedForwardLayout::kFusedGateUp) {
        status = resolve_weight(
            store, feed_forward_name(prefix, "gate_up_w_", layer),
            &result.gate_up_weight);
        if (!status.ok_status()) return status;
    } else {
        return invalid("PI0.5 feed-forward layout is invalid");
    }
    status = resolve_weight(
        store, feed_forward_name(prefix, "down_w_", layer),
        &result.down_weight);
    if (!status.ok_status()) return status;
    *out = result;
    return modalities::Status::ok();
}

}  // namespace

modalities::Status resolve_pi05_native_buffers(
    const NativeWorkspace& workspace,
    const Pi05TargetBufferBindings& target,
    const Pi05ResolvedShape& shape,
    Pi05ResolvedBuffers* out) {
    if (!out) return invalid("PI0.5 resolved buffer destination is null");
    modalities::Status status = validate_pi05_resolved_shape(shape);
    if (!status.ok_status()) return status;
    if (workspace.num_views() != shape.num_views ||
        workspace.vision_sequence() != shape.vision_sequence ||
        workspace.encoder_vision_sequence() !=
            shape.encoder_vision_sequence ||
        workspace.encoder_sequence() != shape.encoder_sequence ||
        workspace.total_keys() != shape.total_attention_keys ||
        workspace.chunk_size() != shape.chunk ||
        workspace.num_steps() != shape.num_steps) {
        return invalid("PI0.5 workspace shape identity is invalid");
    }

    Pi05ResolvedBuffers result;
    const struct {
        const char* name;
        Pi05ValueId value;
        Pi05ResolvedBuffer* destination;
    } values[] = {
        {"observation_images_normalized", Pi05ValueId::kImages,
         &result.images},
        {"prompt_embedding", Pi05ValueId::kPromptEmbedding,
         &result.prompt_embedding},
        {"vision_x", Pi05ValueId::kVisionState, &result.vision_state},
        {"encoder_x", Pi05ValueId::kEncoderState, &result.encoder_state},
        {"diffusion_noise", Pi05ValueId::kNoise, &result.noise},
        {"decoder_x", Pi05ValueId::kDecoderState, &result.decoder_state},
        {"decoder_action_buf", Pi05ValueId::kActionDelta,
         &result.action_delta},
        {"decoder_time_emb", Pi05ValueId::kTimeState, &result.time_state},
        {"decoder_style_attn", Pi05ValueId::kAttentionStyle,
         &result.attention_style},
        {"decoder_style_ffn", Pi05ValueId::kMlpStyle, &result.mlp_style},
        {"decoder_style_final", Pi05ValueId::kFinalStyle,
         &result.final_style},
    };
    for (const auto& entry : values) {
        status = resolve_workspace_buffer(
            workspace, entry.name, entry.value, shape, entry.destination);
        if (!status.ok_status()) return status;
    }

    result.key_cache = target.key_cache;
    result.value_cache = target.value_cache;
    result.encoder_valid_tokens = target.encoder_valid_tokens;
    result.decoder_valid_tokens = target.decoder_valid_tokens;
    result.decoder_position = target.decoder_position;

    const struct {
        const char* name;
        Pi05ResolvedBuffer* destination;
    } controls[] = {
        {"encoder_rope_weights", &result.encoder_rope},
        {"decoder_rope_weights", &result.decoder_rope},
        {"rtc_prev_action_chunk", &result.previous_actions},
        {"rtc_prefix_weights", &result.prefix_weights},
        {"rtc_guidance_weight", &result.guidance_weight},
    };
    for (const auto& entry : controls) {
        status = resolve_workspace_control(
            workspace, entry.name, entry.destination);
        if (!status.ok_status()) return status;
    }
    status = validate_pi05_resolved_buffers(result, shape);
    if (!status.ok_status()) return status;
    *out = result;
    return modalities::Status::ok();
}

modalities::Status resolve_pi05_native_support_buffers(
    const NativeWorkspace& workspace,
    const Pi05ResolvedShape& shape,
    Pi05NativeSupportBuffers* out) {
    if (!out) {
        return invalid("PI0.5 native support destination is null");
    }
    modalities::Status status = validate_pi05_resolved_shape(shape);
    if (!status.ok_status()) return status;
    const modalities::DType activation = workspace.activation_dtype();
    if (workspace.vision_sequence() != shape.vision_sequence ||
        workspace.encoder_vision_sequence() !=
            shape.encoder_vision_sequence ||
        (activation != modalities::DType::kBFloat16 &&
         activation != modalities::DType::kFloat16)) {
        return invalid("PI0.5 native support shape identity is invalid");
    }

    Pi05NativeSupportBuffers result;
    const struct {
        const char* name;
        Pi05ResolvedBuffer* destination;
    } entries[] = {
        {"vision_patches", &result.vision_patches},
        {"vision_x_pooled", &result.pooled_vision_state},
        {"vision_pos_embed_expanded", &result.expanded_vision_position},
        {"encoder_rms_ones", &result.encoder_rms_weight},
        {"decoder_rms_ones", &result.decoder_rms_weight},
    };
    for (const auto& entry : entries) {
        status = resolve_workspace_control(
            workspace, entry.name, entry.destination);
        if (!status.ok_status()) return status;
    }

    const std::uint64_t vision_sequence =
        static_cast<std::uint64_t>(shape.vision_sequence);
    const std::uint64_t encoder_vision_sequence =
        static_cast<std::uint64_t>(shape.encoder_vision_sequence);
    const std::uint64_t patch_width =
        static_cast<std::uint64_t>(kPi05ModelDims.vision_patch) *
        static_cast<std::uint64_t>(kPi05ModelDims.vision_patch) *
        static_cast<std::uint64_t>(kPi05ModelDims.image_channels);
    const std::uint64_t vision_width =
        static_cast<std::uint64_t>(kPi05ModelDims.vision_width);
    if (!physical_buffer_is(
            result.vision_patches, activation,
            {vision_sequence, patch_width}) ||
        !physical_buffer_is(
            result.pooled_vision_state, activation,
            {encoder_vision_sequence, vision_width}) ||
        !physical_buffer_is(
            result.expanded_vision_position, activation,
            {vision_sequence, vision_width}) ||
        !physical_buffer_is(
            result.encoder_rms_weight, activation,
            {static_cast<std::uint64_t>(kPi05ModelDims.encoder_width)}) ||
        !physical_buffer_is(
            result.decoder_rms_weight, activation,
            {static_cast<std::uint64_t>(kPi05ModelDims.decoder_width)})) {
        return invalid("PI0.5 native support metadata is invalid");
    }
    *out = result;
    return modalities::Status::ok();
}

modalities::Status resolve_pi05_materialized_weights(
    const NativeDeviceWeightStore& store,
    const Pi05ResolvedShape& shape,
    modalities::DType activation_dtype,
    Pi05NativeWeightLayout layout,
    Pi05ResolvedWeights* out) {
    if (!out) return invalid("PI0.5 resolved weight destination is null");
    modalities::Status status = validate_pi05_resolved_shape(shape);
    if (!status.ok_status()) return status;

    Pi05ResolvedWeights result;
    const struct {
        const char* name;
        Pi05ResolvedWeight* destination;
    } globals[] = {
        {"embedding_weight", &result.embedding_table},
        {"vision_patch_embedding_w", &result.vision.patch_weight},
        {"vision_patch_embedding_b", &result.vision.patch_bias},
        {"vision_position_embedding", &result.vision.position_embedding},
        {"vision_final_norm_w", &result.vision.final_norm_weight},
        {"vision_final_norm_b", &result.vision.final_norm_bias},
        {"encoder_multi_modal_projector_w", &result.vision.projector_weight},
        {"encoder_multi_modal_projector_b", &result.vision.projector_bias},
        {"decoder_time_embeds", &result.decoder.time_embeddings},
        {"decoder_time_mlp_in_w", &result.decoder.time_mlp_in_weight},
        {"decoder_time_mlp_in_b", &result.decoder.time_mlp_in_bias},
        {"decoder_time_mlp_out_w", &result.decoder.time_mlp_out_weight},
        {"decoder_time_mlp_out_b", &result.decoder.time_mlp_out_bias},
        {"decoder_final_norm_mod_w", &result.decoder.final_norm_mod_weight},
        {"decoder_final_norm_mod_b", &result.decoder.final_norm_mod_bias},
        {"decoder_action_in_proj_w", &result.decoder.action_in_weight},
        {"decoder_action_in_proj_b", &result.decoder.action_in_bias},
        {"decoder_action_out_proj_w", &result.decoder.action_out_weight},
        {"decoder_action_out_proj_b", &result.decoder.action_out_bias},
    };
    for (const auto& entry : globals) {
        status = resolve_weight(store, entry.name, entry.destination);
        if (!status.ok_status()) return status;
    }

    for (std::size_t i = 0; i < result.vision_layers.size(); ++i) {
        Pi05VisionLayerWeights& layer = result.vision_layers[i];
        const struct {
            const char* stem;
            Pi05ResolvedWeight* destination;
        } entries[] = {
            {"vision_pre_attn_norm_w_", &layer.pre_attention_norm_weight},
            {"vision_pre_attn_norm_b_", &layer.pre_attention_norm_bias},
            {"vision_attn_qkv_w_", &layer.attention_qkv_weight},
            {"vision_attn_qkv_b_", &layer.attention_qkv_bias},
            {"vision_attn_o_w_", &layer.attention_output_weight},
            {"vision_attn_o_b_", &layer.attention_output_bias},
            {"vision_pre_ffn_norm_w_", &layer.pre_mlp_norm_weight},
            {"vision_pre_ffn_norm_b_", &layer.pre_mlp_norm_bias},
            {"vision_ffn_up_w_", &layer.mlp_up_weight},
            {"vision_ffn_up_b_", &layer.mlp_up_bias},
            {"vision_ffn_down_w_", &layer.mlp_down_weight},
            {"vision_ffn_down_b_", &layer.mlp_down_bias},
        };
        for (const auto& entry : entries) {
            status = resolve_weight(
                store, layer_name(entry.stem, i), entry.destination);
            if (!status.ok_status()) return status;
        }
    }

    for (std::size_t i = 0; i < result.encoder_layers.size(); ++i) {
        Pi05EncoderLayerWeights& layer = result.encoder_layers[i];
        status = resolve_weight(
            store, layer_name("encoder_attn_qkv_w_", i),
            &layer.attention_qkv_weight);
        if (!status.ok_status()) return status;
        status = resolve_weight(
            store, layer_name("encoder_attn_o_w_", i),
            &layer.attention_output_weight);
        if (!status.ok_status()) return status;
        status = resolve_feed_forward(
            store, "encoder_ffn_", i, layout.encoder, &layer.mlp);
        if (!status.ok_status()) return status;
    }

    for (std::size_t i = 0; i < result.decoder_layers.size(); ++i) {
        Pi05DecoderLayerWeights& layer = result.decoder_layers[i];
        status = resolve_weight(
            store, layer_name("decoder_attn_qkv_w_", i),
            &layer.attention_qkv_weight);
        if (!status.ok_status()) return status;
        status = resolve_weight(
            store, layer_name("decoder_attn_o_w_", i),
            &layer.attention_output_weight);
        if (!status.ok_status()) return status;
        status = resolve_feed_forward(
            store, "decoder_ffn_", i, layout.decoder, &layer.mlp);
        if (!status.ok_status()) return status;
        const struct {
            const char* stem;
            Pi05ResolvedWeight* destination;
        } entries[] = {
            {"decoder_pre_attn_norm_mod_w_", &layer.attention_mod_weight},
            {"decoder_pre_attn_norm_mod_b_", &layer.attention_mod_bias},
            {"decoder_pre_ffn_norm_mod_w_", &layer.mlp_mod_weight},
            {"decoder_pre_ffn_norm_mod_b_", &layer.mlp_mod_bias},
        };
        for (const auto& entry : entries) {
            status = resolve_weight(
                store, layer_name(entry.stem, i), entry.destination);
            if (!status.ok_status()) return status;
        }
    }

    status = validate_pi05_resolved_weights(
        result, shape, activation_dtype);
    if (!status.ok_status()) return status;
    *out = result;
    return modalities::Status::ok();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

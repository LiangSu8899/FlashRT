#include "flashrt/cpp/models/pi05/model/resolved_resources.h"

#include "flashrt/cpp/models/pi05/model/execution_plan.h"

#include <limits>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

bool same_shape(const modalities::Shape& lhs,
                const modalities::Shape& rhs) {
    if (lhs.rank != rhs.rank) return false;
    for (std::size_t i = 0; i < lhs.rank; ++i) {
        if (lhs.dims[i] != rhs.dims[i]) return false;
    }
    return true;
}

bool same_spec(const Pi05TensorSpec& lhs, const Pi05TensorSpec& rhs) {
    return lhs.scalar == rhs.scalar && lhs.rank == rhs.rank &&
           lhs.dimensions == rhs.dimensions;
}

bool byte_count(const modalities::Shape& shape,
                std::size_t width,
                std::uint64_t* out) {
    if (!width || !shape.rank || shape.rank > modalities::Shape::kMaxRank) {
        return false;
    }
    std::uint64_t elements = 1;
    for (std::size_t i = 0; i < shape.rank; ++i) {
        const std::uint64_t dimension = shape.dims[i];
        if (!dimension ||
            elements > std::numeric_limits<std::uint64_t>::max() /
                           dimension) {
            return false;
        }
        elements *= dimension;
    }
    if (elements > std::numeric_limits<std::uint64_t>::max() / width) {
        return false;
    }
    if (out) *out = elements * width;
    return true;
}

bool tensor_elements(const Pi05TensorSpec& spec, std::uint64_t* out) {
    if (!spec.rank || spec.rank > spec.dimensions.size()) return false;
    std::uint64_t elements = 1;
    for (std::size_t i = 0; i < spec.rank; ++i) {
        const std::uint64_t dimension = spec.dimensions[i];
        if (!dimension ||
            elements > std::numeric_limits<std::uint64_t>::max() /
                           dimension) {
            return false;
        }
        elements *= dimension;
    }
    if (out) *out = elements;
    return true;
}

bool valid_storage_range(const Pi05ResolvedBuffer& buffer) {
    return buffer.buffer && buffer.storage_identity && buffer.physical_bytes &&
           buffer.storage_bytes &&
           buffer.storage_offset <= buffer.storage_bytes &&
           buffer.physical_bytes <=
               buffer.storage_bytes - buffer.storage_offset;
}

modalities::Status validate_physical_buffer(
    const Pi05ResolvedBuffer& buffer,
    modalities::DType dtype,
    const modalities::Shape& shape) {
    std::uint64_t bytes = 0;
    if (!valid_storage_range(buffer) || buffer.physical_dtype != dtype ||
        !same_shape(buffer.physical_shape, shape) ||
        !byte_count(shape, modalities::dtype_size(dtype), &bytes) ||
        buffer.physical_bytes != bytes) {
        return invalid("PI0.5 physical buffer metadata is invalid");
    }
    return modalities::Status::ok();
}

modalities::Status validate_semantic_buffer(
    const Pi05ResolvedBuffer& buffer,
    Pi05ValueId id,
    const Pi05ResolvedShape& shape,
    modalities::DType activation_dtype) {
    Pi05TensorSpec expected;
    modalities::Status status = pi05_value_spec(id, shape, &expected);
    if (!status.ok_status()) return status;
    std::uint64_t logical_elements = 0;
    std::uint64_t physical_bytes = 0;
    if (!same_spec(buffer.logical_spec, expected) ||
        !tensor_elements(expected, &logical_elements) ||
        !byte_count(buffer.physical_shape,
                    modalities::dtype_size(buffer.physical_dtype),
                    &physical_bytes) ||
        physical_bytes != buffer.physical_bytes ||
        !valid_storage_range(buffer)) {
        return invalid("PI0.5 semantic buffer metadata is invalid");
    }
    std::uint64_t physical_elements = 0;
    if (!byte_count(buffer.physical_shape, 1, &physical_elements) ||
        physical_elements != logical_elements) {
        return invalid("PI0.5 semantic buffer physical layout is invalid");
    }
    const bool action_delta = id == Pi05ValueId::kActionDelta;
    if (buffer.physical_dtype != activation_dtype &&
        !(action_delta &&
          buffer.physical_dtype == modalities::DType::kFloat32)) {
        return invalid("PI0.5 semantic buffer dtype is invalid");
    }
    return modalities::Status::ok();
}

std::size_t weight_width(Pi05WeightStorage storage) {
    switch (storage) {
        case Pi05WeightStorage::kBFloat16:
        case Pi05WeightStorage::kFloat16:
            return sizeof(std::uint16_t);
        case Pi05WeightStorage::kFp8E4M3:
            return sizeof(std::uint8_t);
    }
    return 0;
}

Pi05WeightStorage activation_storage(modalities::DType dtype) {
    return dtype == modalities::DType::kFloat16
               ? Pi05WeightStorage::kFloat16
               : Pi05WeightStorage::kBFloat16;
}

bool empty_weight(const Pi05ResolvedWeight& weight) {
    return !weight.device_data && !weight.bytes && !weight.shape.rank;
}

modalities::Status validate_weight(
    const Pi05ResolvedWeight& weight,
    const modalities::Shape& shape,
    Pi05WeightStorage activation,
    bool allow_fp8 = false) {
    std::uint64_t bytes = 0;
    if (!weight.device_data || !same_shape(weight.shape, shape) ||
        (weight.storage != activation &&
         !(allow_fp8 && weight.storage == Pi05WeightStorage::kFp8E4M3)) ||
        !byte_count(shape, weight_width(weight.storage), &bytes) ||
        weight.bytes != bytes) {
        return invalid("PI0.5 resolved weight metadata is invalid");
    }
    return modalities::Status::ok();
}

modalities::Status validate_feed_forward(
    const Pi05FeedForwardWeights& mlp,
    int input_width,
    int hidden_width,
    Pi05WeightStorage activation) {
    const bool separate = !empty_weight(mlp.gate_weight) &&
                          !empty_weight(mlp.up_weight);
    const bool fused = !empty_weight(mlp.gate_up_weight);
    if (separate == fused ||
        (separate && !empty_weight(mlp.gate_up_weight)) ||
        (fused && (!empty_weight(mlp.gate_weight) ||
                   !empty_weight(mlp.up_weight)))) {
        return invalid("PI0.5 feed-forward weight layout is invalid");
    }
    modalities::Status status;
    if (separate) {
        status = validate_weight(
            mlp.gate_weight,
            modalities::Shape({static_cast<std::uint64_t>(input_width),
                               static_cast<std::uint64_t>(hidden_width)}),
            activation, true);
        if (!status.ok_status()) return status;
        status = validate_weight(
            mlp.up_weight,
            modalities::Shape({static_cast<std::uint64_t>(input_width),
                               static_cast<std::uint64_t>(hidden_width)}),
            activation, true);
        if (!status.ok_status()) return status;
    } else {
        status = validate_weight(
            mlp.gate_up_weight,
            modalities::Shape({
                static_cast<std::uint64_t>(input_width),
                static_cast<std::uint64_t>(2 * hidden_width)}),
            activation, true);
        if (!status.ok_status()) return status;
    }
    return validate_weight(
        mlp.down_weight,
        modalities::Shape({static_cast<std::uint64_t>(hidden_width),
                           static_cast<std::uint64_t>(input_width)}),
        activation, true);
}

}  // namespace

Pi05ResolvedBuffer* pi05_resolved_buffer(Pi05ResolvedBuffers* buffers,
                                         Pi05ValueId id) {
    if (!buffers) return nullptr;
    switch (id) {
        case Pi05ValueId::kImages: return &buffers->images;
        case Pi05ValueId::kPromptEmbedding:
            return &buffers->prompt_embedding;
        case Pi05ValueId::kVisionState: return &buffers->vision_state;
        case Pi05ValueId::kEncoderState: return &buffers->encoder_state;
        case Pi05ValueId::kKeyCache: return &buffers->key_cache;
        case Pi05ValueId::kValueCache: return &buffers->value_cache;
        case Pi05ValueId::kNoise: return &buffers->noise;
        case Pi05ValueId::kDecoderState: return &buffers->decoder_state;
        case Pi05ValueId::kActionDelta: return &buffers->action_delta;
        case Pi05ValueId::kTimeState: return &buffers->time_state;
        case Pi05ValueId::kAttentionStyle: return &buffers->attention_style;
        case Pi05ValueId::kMlpStyle: return &buffers->mlp_style;
        case Pi05ValueId::kFinalStyle: return &buffers->final_style;
        case Pi05ValueId::kCount: return nullptr;
    }
    return nullptr;
}

const Pi05ResolvedBuffer* pi05_resolved_buffer(
    const Pi05ResolvedBuffers& buffers,
    Pi05ValueId id) {
    switch (id) {
        case Pi05ValueId::kImages: return &buffers.images;
        case Pi05ValueId::kPromptEmbedding:
            return &buffers.prompt_embedding;
        case Pi05ValueId::kVisionState: return &buffers.vision_state;
        case Pi05ValueId::kEncoderState: return &buffers.encoder_state;
        case Pi05ValueId::kKeyCache: return &buffers.key_cache;
        case Pi05ValueId::kValueCache: return &buffers.value_cache;
        case Pi05ValueId::kNoise: return &buffers.noise;
        case Pi05ValueId::kDecoderState: return &buffers.decoder_state;
        case Pi05ValueId::kActionDelta: return &buffers.action_delta;
        case Pi05ValueId::kTimeState: return &buffers.time_state;
        case Pi05ValueId::kAttentionStyle: return &buffers.attention_style;
        case Pi05ValueId::kMlpStyle: return &buffers.mlp_style;
        case Pi05ValueId::kFinalStyle: return &buffers.final_style;
        case Pi05ValueId::kCount: return nullptr;
    }
    return nullptr;
}

modalities::Status validate_pi05_resolved_buffers(
    const Pi05ResolvedBuffers& buffers,
    const Pi05ResolvedShape& shape) {
    modalities::Status status = validate_pi05_resolved_shape(shape);
    if (!status.ok_status()) return status;
    const modalities::DType activation = buffers.images.physical_dtype;
    if (activation != modalities::DType::kBFloat16 &&
        activation != modalities::DType::kFloat16) {
        return invalid("PI0.5 activation buffer dtype is invalid");
    }
    for (std::size_t i = 0;
         i < static_cast<std::size_t>(Pi05ValueId::kCount); ++i) {
        const Pi05ValueId id = static_cast<Pi05ValueId>(i);
        const Pi05ResolvedBuffer* buffer = pi05_resolved_buffer(buffers, id);
        if (!buffer) return invalid("PI0.5 semantic buffer is missing");
        status = validate_semantic_buffer(*buffer, id, shape, activation);
        if (!status.ok_status()) return status;
    }

    status = validate_physical_buffer(
        buffers.encoder_rope, activation,
        modalities::Shape({
            static_cast<std::uint64_t>(shape.encoder_sequence),
            static_cast<std::uint64_t>(kPi05ModelDims.encoder_head_dim)}));
    if (!status.ok_status()) return status;
    status = validate_physical_buffer(
        buffers.decoder_rope, activation,
        modalities::Shape({
            static_cast<std::uint64_t>(shape.chunk),
            static_cast<std::uint64_t>(kPi05ModelDims.decoder_head_dim)}));
    if (!status.ok_status()) return status;
    const modalities::Shape control_shape({sizeof(std::int32_t)});
    for (const Pi05ResolvedBuffer* control : {
             &buffers.encoder_valid_tokens,
             &buffers.decoder_valid_tokens,
             &buffers.decoder_position}) {
        status = validate_physical_buffer(
            *control, modalities::DType::kUInt8, control_shape);
        if (!status.ok_status()) return status;
    }
    status = validate_physical_buffer(
        buffers.previous_actions, activation,
        modalities::Shape({
            static_cast<std::uint64_t>(shape.chunk),
            static_cast<std::uint64_t>(kPi05ModelDims.action_width)}));
    if (!status.ok_status()) return status;
    status = validate_physical_buffer(
        buffers.prefix_weights, modalities::DType::kFloat32,
        modalities::Shape({static_cast<std::uint64_t>(shape.chunk)}));
    if (!status.ok_status()) return status;
    return validate_physical_buffer(
        buffers.guidance_weight, modalities::DType::kFloat32,
        modalities::Shape({1}));
}

modalities::Status validate_pi05_resolved_weights(
    const Pi05ResolvedWeights& weights,
    const Pi05ResolvedShape& shape,
    modalities::DType activation_dtype) {
    modalities::Status status = validate_pi05_resolved_shape(shape);
    if (!status.ok_status()) return status;
    if (activation_dtype != modalities::DType::kBFloat16 &&
        activation_dtype != modalities::DType::kFloat16) {
        return invalid("PI0.5 resolved weight activation dtype is invalid");
    }
    const Pi05WeightStorage activation = activation_storage(activation_dtype);
    const auto dim = [](int value) {
        return static_cast<std::uint64_t>(value);
    };
    const int vision = kPi05ModelDims.vision_width;
    const int encoder = kPi05ModelDims.encoder_width;
    const int decoder = kPi05ModelDims.decoder_width;
    const int vision_hidden = kPi05ModelDims.vision_hidden;
    const int encoder_hidden = kPi05ModelDims.encoder_hidden;
    const int decoder_hidden = kPi05ModelDims.decoder_hidden;
    const int encoder_qkv =
        encoder + 2 * kPi05ModelDims.encoder_head_dim;
    const int decoder_qkv =
        encoder + 2 * kPi05ModelDims.decoder_head_dim;

    const auto check = [&](const Pi05ResolvedWeight& weight,
                           modalities::Shape expected,
                           bool allow_fp8 = false) {
        return validate_weight(weight, expected, activation, allow_fp8);
    };

    status = check(
        weights.embedding_table,
        modalities::Shape({dim(kPi05ModelDims.embedding_vocab),
                           dim(kPi05ModelDims.embedding_width)}));
    if (!status.ok_status()) return status;
    const Pi05VisionGlobalWeights& vg = weights.vision;
    status = check(vg.patch_weight,
                   modalities::Shape({dim(kPi05ModelDims.vision_patch),
                                      dim(kPi05ModelDims.vision_patch),
                                      dim(kPi05ModelDims.image_channels),
                                      dim(vision)}));
    if (!status.ok_status()) return status;
    status = check(vg.patch_bias, modalities::Shape({dim(vision)}));
    if (!status.ok_status()) return status;
    status = check(vg.final_norm_weight,
                   modalities::Shape({dim(vision)}));
    if (!status.ok_status()) return status;
    status = check(vg.final_norm_bias,
                   modalities::Shape({dim(vision)}));
    if (!status.ok_status()) return status;
    status = check(vg.projector_bias,
                   modalities::Shape({dim(encoder)}));
    if (!status.ok_status()) return status;
    status = check(
        vg.position_embedding,
        modalities::Shape({dim(kPi05ModelDims.vision_tokens_per_view),
                           dim(vision)}));
    if (!status.ok_status()) return status;
    status = check(vg.projector_weight,
                   modalities::Shape({dim(vision), dim(encoder)}), true);
    if (!status.ok_status()) return status;

    const Pi05DecoderGlobalWeights& dg = weights.decoder;
    status = check(dg.time_mlp_in_weight,
                   modalities::Shape({dim(decoder), dim(decoder)}));
    if (!status.ok_status()) return status;
    status = check(dg.time_mlp_out_weight,
                   modalities::Shape({dim(decoder), dim(decoder)}));
    if (!status.ok_status()) return status;
    status = check(
        dg.final_norm_mod_weight,
        modalities::Shape({dim(decoder), dim(3 * decoder)}));
    if (!status.ok_status()) return status;
    status = check(
        dg.action_in_weight,
        modalities::Shape(
            {dim(kPi05ModelDims.action_width), dim(decoder)}));
    if (!status.ok_status()) return status;
    status = check(
        dg.action_out_weight,
        modalities::Shape(
            {dim(decoder), dim(kPi05ModelDims.action_width)}));
    if (!status.ok_status()) return status;
    status = check(
        dg.time_embeddings,
        modalities::Shape({dim(shape.num_steps), dim(decoder)}));
    if (!status.ok_status()) return status;
    status = check(dg.time_mlp_in_bias,
                   modalities::Shape({dim(decoder)}));
    if (!status.ok_status()) return status;
    status = check(dg.time_mlp_out_bias,
                   modalities::Shape({dim(decoder)}));
    if (!status.ok_status()) return status;
    status = check(dg.final_norm_mod_bias,
                   modalities::Shape({dim(3 * decoder)}));
    if (!status.ok_status()) return status;
    status = check(dg.action_in_bias,
                   modalities::Shape({dim(decoder)}));
    if (!status.ok_status()) return status;
    status = check(
        dg.action_out_bias,
        modalities::Shape({dim(kPi05ModelDims.action_width)}));
    if (!status.ok_status()) return status;

    for (const Pi05VisionLayerWeights& layer : weights.vision_layers) {
        status = check(layer.pre_attention_norm_weight,
                       modalities::Shape({dim(vision)}));
        if (!status.ok_status()) return status;
        status = check(layer.pre_attention_norm_bias,
                       modalities::Shape({dim(vision)}));
        if (!status.ok_status()) return status;
        status = check(layer.attention_qkv_bias,
                       modalities::Shape({dim(3 * vision)}));
        if (!status.ok_status()) return status;
        status = check(layer.attention_output_bias,
                       modalities::Shape({dim(vision)}));
        if (!status.ok_status()) return status;
        status = check(layer.pre_mlp_norm_weight,
                       modalities::Shape({dim(vision)}));
        if (!status.ok_status()) return status;
        status = check(layer.pre_mlp_norm_bias,
                       modalities::Shape({dim(vision)}));
        if (!status.ok_status()) return status;
        status = check(layer.mlp_up_bias,
                       modalities::Shape({dim(vision_hidden)}));
        if (!status.ok_status()) return status;
        status = check(layer.mlp_down_bias,
                       modalities::Shape({dim(vision)}));
        if (!status.ok_status()) return status;
        status = check(
            layer.attention_qkv_weight,
            modalities::Shape({dim(vision), dim(3 * vision)}), true);
        if (!status.ok_status()) return status;
        status = check(
            layer.attention_output_weight,
            modalities::Shape({dim(vision), dim(vision)}), true);
        if (!status.ok_status()) return status;
        status = check(
            layer.mlp_up_weight,
            modalities::Shape({dim(vision), dim(vision_hidden)}), true);
        if (!status.ok_status()) return status;
        status = check(
            layer.mlp_down_weight,
            modalities::Shape({dim(vision_hidden), dim(vision)}), true);
        if (!status.ok_status()) return status;
    }

    for (const Pi05EncoderLayerWeights& layer : weights.encoder_layers) {
        status = check(
            layer.attention_qkv_weight,
            modalities::Shape({dim(encoder), dim(encoder_qkv)}), true);
        if (!status.ok_status()) return status;
        status = check(
            layer.attention_output_weight,
            modalities::Shape({dim(encoder), dim(encoder)}), true);
        if (!status.ok_status()) return status;
        status = validate_feed_forward(
            layer.mlp, encoder, encoder_hidden, activation);
        if (!status.ok_status()) return status;
    }

    for (const Pi05DecoderLayerWeights& layer : weights.decoder_layers) {
        status = check(
            layer.attention_qkv_weight,
            modalities::Shape({dim(decoder), dim(decoder_qkv)}), true);
        if (!status.ok_status()) return status;
        status = check(
            layer.attention_output_weight,
            modalities::Shape({dim(encoder), dim(decoder)}), true);
        if (!status.ok_status()) return status;
        status = validate_feed_forward(
            layer.mlp, decoder, decoder_hidden, activation);
        if (!status.ok_status()) return status;
        status = check(
            layer.attention_mod_weight,
            modalities::Shape({dim(decoder), dim(3 * decoder)}));
        if (!status.ok_status()) return status;
        status = check(layer.attention_mod_bias,
                       modalities::Shape({dim(3 * decoder)}));
        if (!status.ok_status()) return status;
        status = check(
            layer.mlp_mod_weight,
            modalities::Shape({dim(decoder), dim(3 * decoder)}));
        if (!status.ok_status()) return status;
        status = check(layer.mlp_mod_bias,
                       modalities::Shape({dim(3 * decoder)}));
        if (!status.ok_status()) return status;
    }
    return modalities::Status::ok();
}

modalities::Status validate_pi05_resolved_resources(
    const Pi05ResolvedResources& resources,
    const Pi05ResolvedShape& shape) {
    modalities::Status status =
        validate_pi05_resolved_buffers(resources.buffers, shape);
    return status.ok_status()
               ? validate_pi05_resolved_weights(
                     resources.weights, shape,
                     resources.buffers.images.physical_dtype)
               : status;
}

modalities::Status make_pi05_graph_bindings(
    const Pi05ResolvedBuffers& buffers,
    Pi05ResolvedGraphBindings* out) {
    if (!out) return invalid("PI0.5 graph binding destination is null");
    Pi05ResolvedGraphBindings resolved;
    const struct {
        Pi05GraphBindingId id;
        const Pi05ResolvedBuffer* buffer;
    } entries[] = {
        {Pi05GraphBindingId::kImages, &buffers.images},
        {Pi05GraphBindingId::kPromptEmbedding, &buffers.prompt_embedding},
        {Pi05GraphBindingId::kEncoderState, &buffers.encoder_state},
        {Pi05GraphBindingId::kNoise, &buffers.noise},
        {Pi05GraphBindingId::kPreviousActions, &buffers.previous_actions},
        {Pi05GraphBindingId::kPrefixWeights, &buffers.prefix_weights},
        {Pi05GraphBindingId::kGuidanceWeight, &buffers.guidance_weight},
    };
    for (const auto& entry : entries) {
        modalities::Status status =
            resolved.bind(entry.id, entry.buffer->buffer);
        if (!status.ok_status()) return status;
    }
    *out = resolved;
    return modalities::Status::ok();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

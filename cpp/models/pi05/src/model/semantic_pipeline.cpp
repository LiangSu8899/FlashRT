#include "flashrt/cpp/models/pi05/model/semantic_pipeline.h"

#include "flashrt/cpp/models/pi05/model/frontend_ops.h"

#include <cstddef>
#include <limits>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

constexpr Pi05ValueId kUnusedValue = Pi05ValueId::kCount;

constexpr std::array<std::uint8_t, kPi05MaxOperationValues> aliases(
    std::uint8_t first = kPi05NoAlias,
    std::uint8_t second = kPi05NoAlias,
    std::uint8_t third = kPi05NoAlias,
    std::uint8_t fourth = kPi05NoAlias) {
    return {first, second, third, fourth};
}

constexpr std::array<Pi05ValueId, kPi05MaxOperationValues> values(
    Pi05ValueId first = kUnusedValue,
    Pi05ValueId second = kUnusedValue,
    Pi05ValueId third = kUnusedValue,
    Pi05ValueId fourth = kUnusedValue) {
    return {first, second, third, fourth};
}

constexpr Pi05OperationContract kContracts[] = {
    {Pi05OperationId::kTimeMlp, "time_mlp",
     Pi05IndexDomain::kDiffusionStep,
     values(), 0,
     values(Pi05ValueId::kTimeState), 1, aliases()},
    {Pi05OperationId::kAttentionStyle, "attention_style",
     Pi05IndexDomain::kDecoderLayer,
     values(Pi05ValueId::kTimeState), 1,
     values(Pi05ValueId::kAttentionStyle), 1, aliases()},
    {Pi05OperationId::kMlpStyle, "mlp_style",
     Pi05IndexDomain::kDecoderLayer,
     values(Pi05ValueId::kTimeState), 1,
     values(Pi05ValueId::kMlpStyle), 1, aliases()},
    {Pi05OperationId::kFinalStyle, "final_style",
     Pi05IndexDomain::kDiffusionStep,
     values(Pi05ValueId::kTimeState), 1,
     values(Pi05ValueId::kFinalStyle), 1, aliases()},
    {Pi05OperationId::kComposePrompt, "compose_prompt",
     Pi05IndexDomain::kNone,
     values(Pi05ValueId::kPromptEmbedding), 1,
     values(Pi05ValueId::kEncoderState), 1, aliases()},
    {Pi05OperationId::kVisionEmbed, "vision_embed",
     Pi05IndexDomain::kNone,
     values(Pi05ValueId::kImages), 1,
     values(Pi05ValueId::kVisionState), 1, aliases()},
    {Pi05OperationId::kVisionAttention, "vision_attention",
     Pi05IndexDomain::kVisionLayer,
     values(Pi05ValueId::kVisionState), 1,
     values(Pi05ValueId::kVisionState), 1, aliases(0)},
    {Pi05OperationId::kVisionMlp, "vision_mlp",
     Pi05IndexDomain::kVisionLayer,
     values(Pi05ValueId::kVisionState), 1,
     values(Pi05ValueId::kVisionState), 1, aliases(0)},
    {Pi05OperationId::kVisionProject, "vision_project",
     Pi05IndexDomain::kNone,
     values(Pi05ValueId::kVisionState, Pi05ValueId::kEncoderState), 2,
     values(Pi05ValueId::kEncoderState), 1, aliases(1)},
    {Pi05OperationId::kEncoderAttention, "encoder_attention",
     Pi05IndexDomain::kEncoderLayer,
     values(Pi05ValueId::kEncoderState, Pi05ValueId::kKeyCache,
            Pi05ValueId::kValueCache), 3,
     values(Pi05ValueId::kEncoderState, Pi05ValueId::kKeyCache,
            Pi05ValueId::kValueCache), 3,
     aliases(0, 1, 2)},
    {Pi05OperationId::kEncoderMlp, "encoder_mlp",
     Pi05IndexDomain::kEncoderLayer,
     values(Pi05ValueId::kEncoderState), 1,
     values(Pi05ValueId::kEncoderState), 1, aliases(0)},
    {Pi05OperationId::kEncoderCacheFinalize, "encoder_cache_finalize",
     Pi05IndexDomain::kEncoderFinalLayer,
     values(Pi05ValueId::kEncoderState, Pi05ValueId::kKeyCache,
            Pi05ValueId::kValueCache), 3,
     values(Pi05ValueId::kKeyCache, Pi05ValueId::kValueCache), 2,
     aliases(1, 2)},
    {Pi05OperationId::kDiffusionInputProject, "diffusion_input_project",
     Pi05IndexDomain::kDiffusionStep,
     values(Pi05ValueId::kNoise), 1,
     values(Pi05ValueId::kDecoderState), 1, aliases()},
    {Pi05OperationId::kDecoderAttention, "decoder_attention",
     Pi05IndexDomain::kDecoderLayer,
     values(Pi05ValueId::kDecoderState, Pi05ValueId::kKeyCache,
            Pi05ValueId::kValueCache, Pi05ValueId::kAttentionStyle), 4,
     values(Pi05ValueId::kDecoderState, Pi05ValueId::kKeyCache,
            Pi05ValueId::kValueCache), 3,
     aliases(0, 1, 2)},
    {Pi05OperationId::kDecoderMlp, "decoder_mlp",
     Pi05IndexDomain::kDecoderLayer,
     values(Pi05ValueId::kDecoderState, Pi05ValueId::kMlpStyle), 2,
     values(Pi05ValueId::kDecoderState), 1, aliases(0)},
    {Pi05OperationId::kActionProject, "action_project",
     Pi05IndexDomain::kDiffusionStep,
     values(Pi05ValueId::kDecoderState, Pi05ValueId::kFinalStyle), 2,
     values(Pi05ValueId::kActionDelta), 1, aliases()},
    {Pi05OperationId::kDiffusionUpdate, "diffusion_update",
     Pi05IndexDomain::kDiffusionStep,
     values(Pi05ValueId::kNoise, Pi05ValueId::kActionDelta), 2,
     values(Pi05ValueId::kNoise), 1, aliases(0)},
};

static_assert(sizeof(kContracts) / sizeof(kContracts[0]) ==
                  static_cast<std::size_t>(Pi05OperationId::kCount),
              "PI0.5 operation catalog must be complete");

modalities::Status emit(
    Pi05OperationSink& sink,
    const Pi05ResolvedShape& shape,
    Pi05OperationId id,
    int layer,
    int step,
    std::array<std::uint64_t, kPi05MaxOperationValues> inputs,
    std::array<std::uint64_t, kPi05MaxOperationValues> outputs,
    Pi05Stream stream) {
    Pi05OperationCall call;
    call.id = id;
    call.layer = layer;
    call.step = step;
    call.input_generation = inputs;
    call.output_generation = outputs;
    modalities::Status status = validate_pi05_operation_call(call, shape);
    return status.ok_status() ? sink.record(call, shape, stream) : status;
}

std::array<std::uint64_t, kPi05MaxOperationValues> generations(
    std::uint64_t first = 0,
    std::uint64_t second = 0,
    std::uint64_t third = 0,
    std::uint64_t fourth = 0) {
    return {first, second, third, fourth};
}

template <typename Visitor>
modalities::Status traverse_prepare(const Pi05ResolvedShape& shape,
                                    Visitor&& visitor) {
    std::uint64_t time_state = 0;
    std::uint64_t attention_style = 0;
    std::uint64_t mlp_style = 0;
    std::uint64_t final_style = 0;
    auto visit = [&](Pi05OperationId id,
                     int layer,
                     int step,
                     std::array<std::uint64_t, kPi05MaxOperationValues> inputs,
                     std::array<std::uint64_t, kPi05MaxOperationValues> outputs) {
        Pi05OperationCall call;
        call.id = id;
        call.layer = layer;
        call.step = step;
        call.input_generation = inputs;
        call.output_generation = outputs;
        modalities::Status status = validate_pi05_operation_call(call, shape);
        return status.ok_status() ? visitor(call) : status;
    };

    for (int step = 0; step < shape.num_steps; ++step) {
        modalities::Status status = visit(
            Pi05OperationId::kTimeMlp, -1, step, generations(),
            generations(time_state + 1));
        if (!status.ok_status()) return status;
        ++time_state;
        for (int layer = 0; layer < kPi05ModelDims.decoder_layers; ++layer) {
            status = visit(Pi05OperationId::kAttentionStyle, layer, step,
                           generations(time_state),
                           generations(attention_style + 1));
            if (!status.ok_status()) return status;
            ++attention_style;
            status = visit(Pi05OperationId::kMlpStyle, layer, step,
                           generations(time_state),
                           generations(mlp_style + 1));
            if (!status.ok_status()) return status;
            ++mlp_style;
        }
        status = visit(Pi05OperationId::kFinalStyle, -1, step,
                       generations(time_state),
                       generations(final_style + 1));
        if (!status.ok_status()) return status;
        ++final_style;
    }
    return modalities::Status::ok();
}

void* buffer_data(const Pi05ResolvedBuffer& buffer) {
    return buffer.buffer ? frt_buffer_dptr(buffer.buffer) : nullptr;
}

const Pi05PrimitiveSet* active_primitives(const Pi05FrontendOps& ops) {
    switch (ops.profile.activation_dtype) {
        case modalities::DType::kBFloat16: return &ops.bf16;
        case modalities::DType::kFloat16: return &ops.f16;
        default: return nullptr;
    }
}

bool complete(const Pi05PrimitiveSet* ops) {
    return ops && ops->linear && ops->add_bias && ops->silu && ops->copy;
}

void* byte_offset(void* base, std::size_t bytes) {
    return static_cast<unsigned char*>(base) + bytes;
}

const void* byte_offset(const void* base, std::size_t bytes) {
    return static_cast<const unsigned char*>(base) + bytes;
}

modalities::Status execute_time_mlp(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    Pi05PrepareExecution& execution,
    Pi05Stream stream) {
    const int width = kPi05ModelDims.decoder_width;
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    const std::size_t scalar_bytes =
        modalities::dtype_size(execution.ops->profile.activation_dtype);
    const std::size_t row_bytes = static_cast<std::size_t>(width) *
                                  scalar_bytes;
    if (!scalar_bytes || execution.scratch.bytes < row_bytes) {
        return invalid("PI0.5 prepare scratch is too small");
    }
    const Pi05DecoderGlobalWeights& weights =
        execution.resources->weights.decoder;
    void* output = buffer_data(execution.resources->buffers.time_state);
    const void* source = byte_offset(
        weights.time_embeddings.device_data,
        static_cast<std::size_t>(call.step) * row_bytes);

    modalities::Status status = ops.linear(
        ops.state, weights.time_mlp_in_weight, source,
        execution.scratch.mlp_hidden,
        1, width, width, stream);
    if (!status.ok_status()) return status;
    status = ops.add_bias(ops.state, execution.scratch.mlp_hidden,
                          weights.time_mlp_in_bias, 1, width, stream);
    if (!status.ok_status()) return status;
    status = ops.silu(ops.state, execution.scratch.mlp_hidden, width, stream);
    if (!status.ok_status()) return status;
    status = ops.linear(ops.state, weights.time_mlp_out_weight,
                        execution.scratch.mlp_hidden,
                        execution.scratch.mlp_output, 1, width, width, stream);
    if (!status.ok_status()) return status;
    status = ops.add_bias(ops.state, execution.scratch.mlp_output,
                          weights.time_mlp_out_bias, 1, width, stream);
    if (!status.ok_status()) return status;
    status = ops.silu(ops.state, execution.scratch.mlp_output, width, stream);
    if (!status.ok_status()) return status;

    void* step_output = byte_offset(
        output, static_cast<std::size_t>(call.step) * shape.chunk * row_bytes);
    for (int row = 0; row < shape.chunk; ++row) {
        status = ops.copy(
            ops.state,
            byte_offset(step_output, static_cast<std::size_t>(row) * row_bytes),
            execution.scratch.mlp_output, row_bytes, stream);
        if (!status.ok_status()) return status;
    }
    return modalities::Status::ok();
}

modalities::Status execute_style(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    const Pi05ResolvedWeight& weight,
    const Pi05ResolvedWeight& bias,
    const Pi05ResolvedBuffer& destination,
    Pi05PrepareExecution& execution,
    Pi05Stream stream) {
    const int input_width = kPi05ModelDims.decoder_width;
    const int output_width = 3 * input_width;
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    const std::size_t scalar_bytes =
        modalities::dtype_size(execution.ops->profile.activation_dtype);
    const std::size_t input_offset =
        static_cast<std::size_t>(call.step) * shape.chunk * input_width *
        scalar_bytes;
    std::size_t output_index = static_cast<std::size_t>(call.step);
    if (call.layer >= 0) {
        output_index = output_index * kPi05ModelDims.decoder_layers +
                       static_cast<std::size_t>(call.layer);
    }
    void* output = byte_offset(
        buffer_data(destination),
        output_index * shape.chunk * output_width * scalar_bytes);
    const void* input = byte_offset(
        buffer_data(execution.resources->buffers.time_state), input_offset);
    modalities::Status status = ops.linear(
        ops.state, weight, input, output, shape.chunk, input_width,
        output_width, stream);
    return status.ok_status()
               ? ops.add_bias(ops.state, output, bias, shape.chunk,
                              output_width, stream)
               : status;
}

modalities::Status execute_prepare_call(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    Pi05PrepareExecution& execution,
    Pi05Stream stream) {
    const Pi05ResolvedResources& resources = *execution.resources;
    switch (call.id) {
        case Pi05OperationId::kTimeMlp:
            return execute_time_mlp(call, shape, execution, stream);
        case Pi05OperationId::kAttentionStyle: {
            const Pi05DecoderLayerWeights& layer =
                resources.weights.decoder_layers[
                    static_cast<std::size_t>(call.layer)];
            return execute_style(call, shape, layer.attention_mod_weight,
                                 layer.attention_mod_bias,
                                 resources.buffers.attention_style, execution,
                                 stream);
        }
        case Pi05OperationId::kMlpStyle: {
            const Pi05DecoderLayerWeights& layer =
                resources.weights.decoder_layers[
                    static_cast<std::size_t>(call.layer)];
            return execute_style(call, shape, layer.mlp_mod_weight,
                                 layer.mlp_mod_bias,
                                 resources.buffers.mlp_style, execution,
                                 stream);
        }
        case Pi05OperationId::kFinalStyle:
            return execute_style(
                call, shape,
                resources.weights.decoder.final_norm_mod_weight,
                resources.weights.decoder.final_norm_mod_bias,
                resources.buffers.final_style, execution, stream);
        default:
            return invalid("PI0.5 prepare operation is invalid");
    }
}

bool complete_vision(const Pi05PrimitiveSet* ops) {
    return ops && ops->linear && ops->projected_linear && ops->add_bias &&
           ops->patchify && ops->bias_residual && ops->layer_norm &&
           ops->qkv_split && ops->attention && ops->gelu &&
           ops->vision_pool;
}

bool complete_vision(const Pi05VisionExecution& vision) {
    return vision.patches && vision.expanded_position && vision.pooled &&
           vision.normalized && vision.qkv && vision.hidden && vision.query &&
           vision.key && vision.value && vision.attention_output;
}

bool complete_encoder(const Pi05PrimitiveSet* ops) {
    return ops && ops->projected_linear && ops->normalize_for_linear &&
           ops->qkv_rope && ops->attention && ops->gate_up &&
           ops->gated_activation;
}

bool complete_encoder(const Pi05EncoderExecution& encoder) {
    return buffer_data(encoder.rms_weight) && encoder.normalized &&
           encoder.qkv && encoder.gate &&
           encoder.hidden && encoder.query && encoder.attention_output;
}

bool complete_decoder(const Pi05PrimitiveSet* ops) {
    return ops && ops->adaptive_normalize && ops->diffusion_update;
}

bool complete_decoder(const Pi05DecoderExecution& decoder) {
    return buffer_data(decoder.rms_weight) && decoder.normalized &&
           decoder.gate && decoder.qkv && decoder.gate_projection &&
           decoder.hidden && decoder.query && decoder.attention_output;
}

const void* style_slice(const Pi05ResolvedBuffer& styles,
                        const Pi05ResolvedShape& shape,
                        modalities::DType dtype,
                        int step,
                        int layer) {
    const std::size_t scalar_bytes = modalities::dtype_size(dtype);
    const std::size_t width = kPi05ModelDims.decoder_width;
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (!scalar_bytes || shape.chunk <= 0 || step < 0 ||
        static_cast<std::size_t>(shape.chunk) > maximum / 3 / width /
                                                   scalar_bytes) {
        return nullptr;
    }
    const std::size_t style_bytes =
        static_cast<std::size_t>(shape.chunk) * 3 * width * scalar_bytes;
    std::size_t index = static_cast<std::size_t>(step);
    if (layer >= 0) {
        index = index * kPi05ModelDims.decoder_layers +
                static_cast<std::size_t>(layer);
    }
    if (index > maximum / style_bytes) return nullptr;
    const std::size_t offset = index * style_bytes;
    return buffer_data(styles) &&
                   offset <= styles.physical_bytes &&
                   style_bytes <= styles.physical_bytes - offset
               ? byte_offset(buffer_data(styles), offset)
               : nullptr;
}

void* cache_layer_data(const Pi05ResolvedBuffer& cache,
                       const Pi05ResolvedShape& shape,
                       modalities::DType dtype,
                       int layer) {
    const std::size_t scalar_bytes = modalities::dtype_size(dtype);
    const std::size_t layer_bytes =
        static_cast<std::size_t>(shape.total_attention_keys) *
        kPi05ModelDims.encoder_kv_heads * kPi05ModelDims.encoder_head_dim *
        scalar_bytes;
    const std::size_t offset = static_cast<std::size_t>(layer) * layer_bytes;
    return scalar_bytes && layer >= 0 && buffer_data(cache) &&
                   offset <= cache.physical_bytes &&
                   layer_bytes <= cache.physical_bytes - offset
               ? byte_offset(buffer_data(cache), offset)
               : nullptr;
}

modalities::Status execute_vision_embed(
    const Pi05ResolvedShape& shape,
    Pi05ForwardExecution& execution,
    Pi05Stream stream) {
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    const Pi05VisionGlobalWeights& weights =
        execution.resources->weights.vision;
    const int rows = shape.vision_sequence;
    const int width = kPi05ModelDims.vision_width;
    const int patch_width = kPi05ModelDims.vision_patch *
                            kPi05ModelDims.vision_patch *
                            kPi05ModelDims.image_channels;
    void* state = buffer_data(execution.resources->buffers.vision_state);
    modalities::Status status = ops.patchify(
        ops.state, buffer_data(execution.resources->buffers.images),
        execution.vision.patches, shape.num_views, stream);
    if (!status.ok_status()) return status;
    status = ops.linear(ops.state, weights.patch_weight,
                        execution.vision.patches, state, rows, patch_width,
                        width, stream);
    if (!status.ok_status()) return status;
    status = ops.bias_residual(
        ops.state, state, execution.vision.expanded_position,
        weights.patch_bias, rows, width, stream);
    if (!status.ok_status()) return status;
    const Pi05VisionLayerWeights& first =
        execution.resources->weights.vision_layers[0];
    return ops.layer_norm(
        ops.state, state, first.pre_attention_norm_weight,
        first.pre_attention_norm_bias, execution.vision.normalized, rows,
        width, kPi05ModelNumerics.vision_layer_norm_epsilon, true,
        &execution.vision.normalized_input, stream);
}

modalities::Status execute_compose_prompt(
    const Pi05ResolvedShape& shape,
    Pi05ForwardExecution& execution,
    Pi05Stream stream) {
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    const Pi05ResolvedBuffer& prompt =
        execution.resources->buffers.prompt_embedding;
    const Pi05ResolvedBuffer& encoder =
        execution.resources->buffers.encoder_state;
    const std::size_t scalar_bytes =
        modalities::dtype_size(execution.ops->profile.activation_dtype);
    const std::size_t offset =
        static_cast<std::size_t>(shape.encoder_vision_sequence) *
        kPi05ModelDims.encoder_width * scalar_bytes;
    if (!scalar_bytes || !buffer_data(prompt) || !buffer_data(encoder) ||
        offset > encoder.physical_bytes ||
        prompt.physical_bytes > encoder.physical_bytes - offset) {
        return invalid("PI0.5 prompt composition buffers are invalid");
    }
    return ops.copy(ops.state, byte_offset(buffer_data(encoder), offset),
                    buffer_data(prompt), prompt.physical_bytes, stream);
}

modalities::Status execute_vision_attention(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    Pi05ForwardExecution& execution,
    Pi05Stream stream) {
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    const Pi05VisionLayerWeights& weights =
        execution.resources->weights.vision_layers[
            static_cast<std::size_t>(call.layer)];
    const int rows = shape.vision_sequence;
    const int width = kPi05ModelDims.vision_width;
    void* state = buffer_data(execution.resources->buffers.vision_state);
    modalities::Status status = ops.projected_linear(
        ops.state, weights.attention_qkv_weight,
        {Pi05LinearDomain::kVision, Pi05LinearRole::kAttentionQkv,
         call.layer},
        -1, execution.vision.normalized_input.data, execution.vision.qkv,
        rows, width, 3 * width,
        execution.vision.normalized_input.prequantized,
        {Pi05LinearEpilogueKind::kBias,
         &weights.attention_qkv_bias, nullptr},
        stream);
    if (!status.ok_status()) return status;
    status = ops.qkv_split(
        ops.state, execution.vision.qkv, execution.vision.query,
        execution.vision.key, execution.vision.value, rows, width, width,
        width, stream);
    if (!status.ok_status()) return status;
    status = ops.attention(
        ops.state, Pi05LinearDomain::kVision, -1, execution.vision.query,
        execution.vision.key, execution.vision.value,
        execution.vision.attention_output, shape.num_views,
        rows / shape.num_views, rows / shape.num_views,
        kPi05ModelDims.vision_heads, kPi05ModelDims.vision_heads,
        kPi05ModelDims.vision_head_dim, stream);
    if (!status.ok_status()) return status;
    status = ops.projected_linear(
        ops.state, weights.attention_output_weight,
        {Pi05LinearDomain::kVision, Pi05LinearRole::kAttentionOutput,
         call.layer},
        -1, execution.vision.attention_output, execution.vision.normalized,
        rows, width, width, false,
        {Pi05LinearEpilogueKind::kBiasResidual,
         &weights.attention_output_bias, state},
        stream);
    return status.ok_status()
               ? ops.layer_norm(
                     ops.state, state, weights.pre_mlp_norm_weight,
                     weights.pre_mlp_norm_bias, execution.vision.normalized,
                     rows, width,
                     kPi05ModelNumerics.vision_layer_norm_epsilon, true,
                     &execution.vision.normalized_input, stream)
               : status;
}

modalities::Status execute_vision_mlp(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    Pi05ForwardExecution& execution,
    Pi05Stream stream) {
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    const Pi05VisionLayerWeights& weights =
        execution.resources->weights.vision_layers[
            static_cast<std::size_t>(call.layer)];
    const int rows = shape.vision_sequence;
    const int width = kPi05ModelDims.vision_width;
    const int hidden_width = kPi05ModelDims.vision_hidden;
    void* state = buffer_data(execution.resources->buffers.vision_state);
    modalities::Status status = ops.projected_linear(
        ops.state, weights.mlp_up_weight,
        {Pi05LinearDomain::kVision, Pi05LinearRole::kMlpUp, call.layer}, -1,
        execution.vision.normalized_input.data, execution.vision.hidden,
        rows, width, hidden_width,
        execution.vision.normalized_input.prequantized,
        {Pi05LinearEpilogueKind::kBiasGelu, &weights.mlp_up_bias, nullptr},
        stream);
    if (!status.ok_status()) return status;
    status = ops.projected_linear(
        ops.state, weights.mlp_down_weight,
        {Pi05LinearDomain::kVision, Pi05LinearRole::kMlpDown, call.layer}, -1,
        execution.vision.hidden, execution.vision.normalized, rows,
        hidden_width, width, false,
        {Pi05LinearEpilogueKind::kBiasResidual,
         &weights.mlp_down_bias, state},
        stream);
    if (!status.ok_status() ||
        call.layer + 1 == kPi05ModelDims.vision_layers) {
        return status;
    }
    const Pi05VisionLayerWeights& next =
        execution.resources->weights.vision_layers[
            static_cast<std::size_t>(call.layer + 1)];
    return ops.layer_norm(
        ops.state, state, next.pre_attention_norm_weight,
        next.pre_attention_norm_bias, execution.vision.normalized, rows,
        width, kPi05ModelNumerics.vision_layer_norm_epsilon, true,
        &execution.vision.normalized_input, stream);
}

modalities::Status execute_vision_project(
    const Pi05ResolvedShape& shape,
    Pi05ForwardExecution& execution,
    Pi05Stream stream) {
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    const Pi05VisionGlobalWeights& weights =
        execution.resources->weights.vision;
    const int rows = shape.encoder_vision_sequence;
    const int width = kPi05ModelDims.vision_width;
    void* state = buffer_data(execution.resources->buffers.vision_state);
    modalities::Status status = modalities::Status::ok();
    if (shape.vision_pool_factor > 1) {
        status = ops.vision_pool(
            ops.state, state, execution.vision.pooled, shape.num_views,
            kPi05ModelDims.image_height / kPi05ModelDims.vision_patch,
            kPi05ModelDims.image_width / kPi05ModelDims.vision_patch, width,
            shape.vision_pool_factor, stream);
        if (!status.ok_status()) return status;
    }
    Pi05LinearInput projector_input;
    status = ops.layer_norm(
        ops.state, execution.vision.pooled, weights.final_norm_weight,
        weights.final_norm_bias, execution.vision.normalized, rows, width,
        kPi05ModelNumerics.vision_layer_norm_epsilon, false,
        &projector_input, stream);
    if (!status.ok_status() || !projector_input.data ||
        projector_input.prequantized) {
        return status.ok_status()
                   ? invalid("PI0.5 vision projector input is quantized")
                   : status;
    }
    void* encoder = buffer_data(execution.resources->buffers.encoder_state);
    return ops.projected_linear(
        ops.state, weights.projector_weight,
        {Pi05LinearDomain::kVision, Pi05LinearRole::kProjector, -1}, -1,
        projector_input.data, encoder, rows, width,
        kPi05ModelDims.encoder_width, false,
        {Pi05LinearEpilogueKind::kBias, &weights.projector_bias, nullptr},
        stream);
}

modalities::Status execute_encoder_attention(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    Pi05ForwardExecution& execution,
    Pi05Stream stream) {
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    const Pi05EncoderLayerWeights& weights =
        execution.resources->weights.encoder_layers[
            static_cast<std::size_t>(call.layer)];
    const int rows = shape.encoder_sequence;
    const int width = kPi05ModelDims.encoder_width;
    const int key_width =
        kPi05ModelDims.encoder_kv_heads * kPi05ModelDims.encoder_head_dim;
    void* state = buffer_data(execution.resources->buffers.encoder_state);
    void* key = cache_layer_data(
        execution.resources->buffers.key_cache, shape,
        execution.ops->profile.activation_dtype, call.layer);
    void* value = cache_layer_data(
        execution.resources->buffers.value_cache, shape,
        execution.ops->profile.activation_dtype, call.layer);
    const Pi05LinearWeightKey qkv_key{
        Pi05LinearDomain::kEncoder, Pi05LinearRole::kAttentionQkv,
        call.layer};
    const Pi05LinearWeightKey output_key{
        Pi05LinearDomain::kEncoder, Pi05LinearRole::kAttentionOutput,
        call.layer};
    const Pi05LinearWeightKey gate_up_key{
        Pi05LinearDomain::kEncoder, Pi05LinearRole::kMlpGateUpGroup,
        call.layer};
    if (call.layer == 0) {
        execution.encoder.pending_update = nullptr;
    }
    const void* qkv_input = nullptr;
    bool qkv_prequantized = false;
    const void* pending = execution.encoder.pending_update;
    modalities::Status status = ops.normalize_for_linear(
        ops.state, state, pending, execution.encoder.rms_weight,
        execution.encoder.normalized, qkv_key, -1, rows, width,
        kPi05ModelNumerics.encoder_rms_norm_epsilon, &qkv_input,
        &qkv_prequantized, stream);
    if (!status.ok_status()) return status;
    execution.encoder.pending_update = nullptr;
    status = ops.projected_linear(
        ops.state, weights.attention_qkv_weight, qkv_key, -1, qkv_input,
        execution.encoder.qkv, rows, width, width + 2 * key_width,
        qkv_prequantized, {}, stream);
    if (!status.ok_status()) return status;
    status = ops.qkv_rope(
        ops.state, execution.encoder.qkv,
        execution.resources->buffers.encoder_rope, nullptr,
        execution.encoder.query, key, value, rows, width, key_width, key_width,
        kPi05ModelDims.encoder_head_dim, stream);
    if (!status.ok_status()) return status;
    status = ops.attention(
        ops.state, Pi05LinearDomain::kEncoder, call.layer,
        execution.encoder.query, key, value,
        execution.encoder.attention_output, 1, rows, rows,
        kPi05ModelDims.encoder_heads, kPi05ModelDims.encoder_kv_heads,
        kPi05ModelDims.encoder_head_dim, stream);
    if (!status.ok_status()) return status;
    status = ops.projected_linear(
        ops.state, weights.attention_output_weight, output_key, -1,
        execution.encoder.attention_output, execution.encoder.normalized,
        rows, width, width, false, {}, stream);
    if (!status.ok_status()) return status;
    return ops.normalize_for_linear(
        ops.state, state, execution.encoder.normalized,
        execution.encoder.rms_weight,
        execution.encoder.normalized, gate_up_key, -1, rows, width,
        kPi05ModelNumerics.encoder_rms_norm_epsilon,
        &execution.encoder.mlp_input,
        &execution.encoder.mlp_input_prequantized, stream);
}

modalities::Status execute_encoder_mlp(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    Pi05ForwardExecution& execution,
    Pi05Stream stream) {
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    const Pi05FeedForwardWeights& weights =
        execution.resources->weights.encoder_layers[
            static_cast<std::size_t>(call.layer)].mlp;
    const Pi05LinearWeightKey gate_up_key{
        Pi05LinearDomain::kEncoder, Pi05LinearRole::kMlpGateUpGroup,
        call.layer};
    const Pi05LinearWeightKey down_key{
        Pi05LinearDomain::kEncoder, Pi05LinearRole::kMlpDown, call.layer};
    bool merged = false;
    modalities::Status status = ops.gate_up(
        ops.state, weights,
        gate_up_key, -1, execution.encoder.mlp_input,
        execution.encoder.mlp_input_prequantized, execution.encoder.gate,
        execution.encoder.hidden, shape.encoder_sequence,
        kPi05ModelDims.encoder_width, kPi05ModelDims.encoder_hidden,
        &merged, stream);
    if (!status.ok_status()) return status;
    const void* down_input = nullptr;
    bool down_prequantized = false;
    status = ops.gated_activation(
        ops.state, execution.encoder.gate, execution.encoder.hidden, merged,
        execution.encoder.hidden, shape.encoder_sequence,
        kPi05ModelDims.encoder_hidden, down_key, -1, &down_input,
        &down_prequantized, stream);
    if (!status.ok_status()) return status;
    status = ops.projected_linear(
        ops.state, weights.down_weight, down_key, -1, down_input,
        execution.encoder.normalized, shape.encoder_sequence,
        kPi05ModelDims.encoder_hidden, kPi05ModelDims.encoder_width,
        down_prequantized, {}, stream);
    if (status.ok_status()) {
        execution.encoder.pending_update = execution.encoder.normalized;
    }
    return status;
}

modalities::Status execute_encoder_cache_finalize(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    Pi05ForwardExecution& execution,
    Pi05Stream stream) {
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    const int rows = shape.encoder_sequence;
    const int width = kPi05ModelDims.encoder_width;
    const int key_width =
        kPi05ModelDims.encoder_kv_heads * kPi05ModelDims.encoder_head_dim;
    const Pi05LinearWeightKey qkv_key{
        Pi05LinearDomain::kEncoder, Pi05LinearRole::kAttentionQkv,
        call.layer};
    const void* qkv_input = nullptr;
    bool prequantized = false;
    const void* pending = execution.encoder.pending_update;
    modalities::Status status = ops.normalize_for_linear(
        ops.state, buffer_data(execution.resources->buffers.encoder_state),
        pending, execution.encoder.rms_weight,
        execution.encoder.normalized, qkv_key, -1, rows, width,
        kPi05ModelNumerics.encoder_rms_norm_epsilon, &qkv_input,
        &prequantized, stream);
    if (!status.ok_status()) return status;
    execution.encoder.pending_update = nullptr;
    const Pi05ResolvedWeight& weight =
        execution.resources->weights.encoder_layers[
            static_cast<std::size_t>(call.layer)].attention_qkv_weight;
    status = ops.projected_linear(
        ops.state, weight, qkv_key, -1, qkv_input, execution.encoder.qkv,
        rows, width, width + 2 * key_width, prequantized, {}, stream);
    if (!status.ok_status()) return status;
    void* key = cache_layer_data(
        execution.resources->buffers.key_cache, shape,
        execution.ops->profile.activation_dtype, call.layer);
    void* value = cache_layer_data(
        execution.resources->buffers.value_cache, shape,
        execution.ops->profile.activation_dtype, call.layer);
    return ops.qkv_rope(
        ops.state, execution.encoder.qkv,
        execution.resources->buffers.encoder_rope, nullptr,
        execution.encoder.query, key, value, rows, width, key_width, key_width,
        kPi05ModelDims.encoder_head_dim, stream);
}

modalities::Status execute_diffusion_input_project(
    const Pi05ResolvedShape& shape,
    Pi05ForwardExecution& execution,
    Pi05Stream stream) {
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    const Pi05DecoderGlobalWeights& weights =
        execution.resources->weights.decoder;
    void* decoder =
        buffer_data(execution.resources->buffers.decoder_state);
    modalities::Status status = ops.linear(
        ops.state, weights.action_in_weight,
        buffer_data(execution.resources->buffers.noise), decoder,
        shape.chunk, kPi05ModelDims.action_width,
        kPi05ModelDims.decoder_width, stream);
    if (!status.ok_status()) return status;
    execution.decoder.pending_update = nullptr;
    execution.decoder.pending_gate = nullptr;
    return ops.add_bias(
        ops.state, decoder, weights.action_in_bias, shape.chunk,
        kPi05ModelDims.decoder_width, stream);
}

modalities::Status execute_decoder_attention(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    Pi05ForwardExecution& execution,
    Pi05Stream stream) {
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    const Pi05DecoderLayerWeights& weights =
        execution.resources->weights.decoder_layers[
            static_cast<std::size_t>(call.layer)];
    const int rows = shape.chunk;
    const int width = kPi05ModelDims.decoder_width;
    const int query_width =
        kPi05ModelDims.decoder_heads * kPi05ModelDims.decoder_head_dim;
    const int key_width =
        kPi05ModelDims.decoder_kv_heads * kPi05ModelDims.decoder_head_dim;
    const Pi05LinearWeightKey qkv_key{
        Pi05LinearDomain::kDecoder, Pi05LinearRole::kAttentionQkv,
        call.layer};
    const Pi05LinearWeightKey output_key{
        Pi05LinearDomain::kDecoder, Pi05LinearRole::kAttentionOutput,
        call.layer};
    const void* style = style_slice(
        execution.resources->buffers.attention_style, shape,
        execution.ops->profile.activation_dtype, call.step, call.layer);
    const void* qkv_input = nullptr;
    bool qkv_prequantized = false;
    modalities::Status status = ops.adaptive_normalize(
        ops.state,
        buffer_data(execution.resources->buffers.decoder_state),
        execution.decoder.pending_update, execution.decoder.pending_gate,
        execution.decoder.rms_weight, style, execution.decoder.normalized,
        execution.decoder.gate, qkv_key, call.step, rows, width,
        kPi05ModelNumerics.decoder_rms_norm_epsilon, true, &qkv_input,
        &qkv_prequantized, stream);
    if (!status.ok_status()) return status;
    execution.decoder.pending_update = nullptr;
    execution.decoder.pending_gate = nullptr;
    status = ops.projected_linear(
        ops.state, weights.attention_qkv_weight, qkv_key, call.step,
        qkv_input, execution.decoder.qkv, rows, width,
        query_width + 2 * key_width, qkv_prequantized, {}, stream);
    if (!status.ok_status()) return status;
    void* key = cache_layer_data(
        execution.resources->buffers.key_cache, shape,
        execution.ops->profile.activation_dtype, call.layer);
    void* value = cache_layer_data(
        execution.resources->buffers.value_cache, shape,
        execution.ops->profile.activation_dtype, call.layer);
    status = ops.qkv_rope(
        ops.state, execution.decoder.qkv,
        execution.resources->buffers.decoder_rope,
        &execution.resources->buffers.decoder_position,
        execution.decoder.query, key, value, rows, query_width, key_width,
        key_width, kPi05ModelDims.decoder_head_dim, stream);
    if (!status.ok_status()) return status;
    status = ops.attention(
        ops.state, Pi05LinearDomain::kDecoder, call.layer,
        execution.decoder.query, key, value,
        execution.decoder.attention_output, 1, rows,
        shape.total_attention_keys, kPi05ModelDims.decoder_heads,
        kPi05ModelDims.decoder_kv_heads, kPi05ModelDims.decoder_head_dim,
        stream);
    if (!status.ok_status()) return status;
    status = ops.projected_linear(
        ops.state, weights.attention_output_weight, output_key, call.step,
        execution.decoder.attention_output, execution.decoder.normalized,
        rows, query_width, width, false, {}, stream);
    if (status.ok_status()) {
        execution.decoder.pending_update = execution.decoder.normalized;
        execution.decoder.pending_gate = execution.decoder.gate;
    }
    return status;
}

modalities::Status execute_decoder_mlp(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    Pi05ForwardExecution& execution,
    Pi05Stream stream) {
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    const Pi05FeedForwardWeights& weights =
        execution.resources->weights.decoder_layers[
            static_cast<std::size_t>(call.layer)].mlp;
    const int rows = shape.chunk;
    const int width = kPi05ModelDims.decoder_width;
    const int hidden_width = kPi05ModelDims.decoder_hidden;
    const Pi05LinearWeightKey gate_up_key{
        Pi05LinearDomain::kDecoder, Pi05LinearRole::kMlpGateUpGroup,
        call.layer};
    const Pi05LinearWeightKey down_key{
        Pi05LinearDomain::kDecoder, Pi05LinearRole::kMlpDown, call.layer};
    const void* style = style_slice(
        execution.resources->buffers.mlp_style, shape,
        execution.ops->profile.activation_dtype, call.step, call.layer);
    const void* gate_up_input = nullptr;
    bool gate_up_prequantized = false;
    modalities::Status status = ops.adaptive_normalize(
        ops.state,
        buffer_data(execution.resources->buffers.decoder_state),
        execution.decoder.pending_update, execution.decoder.pending_gate,
        execution.decoder.rms_weight, style, execution.decoder.normalized,
        execution.decoder.gate, gate_up_key, call.step, rows, width,
        kPi05ModelNumerics.decoder_rms_norm_epsilon, true, &gate_up_input,
        &gate_up_prequantized, stream);
    if (!status.ok_status()) return status;
    execution.decoder.pending_update = nullptr;
    execution.decoder.pending_gate = nullptr;
    bool merged = false;
    status = ops.gate_up(
        ops.state, weights, gate_up_key, call.step, gate_up_input,
        gate_up_prequantized, execution.decoder.gate_projection,
        execution.decoder.hidden, rows, width, hidden_width, &merged,
        stream);
    if (!status.ok_status()) return status;
    const void* down_input = nullptr;
    bool down_prequantized = false;
    status = ops.gated_activation(
        ops.state, execution.decoder.gate_projection,
        execution.decoder.hidden, merged, execution.decoder.hidden, rows,
        hidden_width, down_key, call.step, &down_input,
        &down_prequantized, stream);
    if (!status.ok_status()) return status;
    status = ops.projected_linear(
        ops.state, weights.down_weight, down_key, call.step, down_input,
        execution.decoder.normalized, rows, hidden_width, width,
        down_prequantized, {}, stream);
    if (status.ok_status()) {
        execution.decoder.pending_update = execution.decoder.normalized;
        execution.decoder.pending_gate = execution.decoder.gate;
    }
    return status;
}

modalities::Status execute_action_project(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    Pi05ForwardExecution& execution,
    Pi05Stream stream) {
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    const Pi05DecoderGlobalWeights& weights =
        execution.resources->weights.decoder;
    const void* style = style_slice(
        execution.resources->buffers.final_style, shape,
        execution.ops->profile.activation_dtype, call.step, -1);
    const void* normalized = nullptr;
    bool prequantized = false;
    modalities::Status status = ops.adaptive_normalize(
        ops.state,
        buffer_data(execution.resources->buffers.decoder_state),
        execution.decoder.pending_update, execution.decoder.pending_gate,
        execution.decoder.rms_weight, style, execution.decoder.normalized,
        execution.decoder.gate, {}, call.step, shape.chunk,
        kPi05ModelDims.decoder_width,
        kPi05ModelNumerics.decoder_rms_norm_epsilon, false, &normalized,
        &prequantized, stream);
    if (!status.ok_status() || prequantized) {
        return status.ok_status()
                   ? invalid("PI0.5 action projection input is quantized")
                   : status;
    }
    execution.decoder.pending_update = nullptr;
    execution.decoder.pending_gate = nullptr;
    void* action =
        buffer_data(execution.resources->buffers.action_delta);
    return ops.linear(
        ops.state, weights.action_out_weight, normalized, action,
        shape.chunk, kPi05ModelDims.decoder_width,
        kPi05ModelDims.action_width, stream);
}

modalities::Status execute_diffusion_update(
    const Pi05ResolvedShape& shape,
    Pi05ForwardExecution& execution,
    Pi05Stream stream) {
    const Pi05PrimitiveSet& ops = *active_primitives(*execution.ops);
    return ops.diffusion_update(
        ops.state, buffer_data(execution.resources->buffers.noise),
        buffer_data(execution.resources->buffers.action_delta),
        execution.resources->weights.decoder.action_out_bias, shape.chunk,
        kPi05ModelDims.action_width,
        shape.num_steps,
        stream);
}

}  // namespace

modalities::Status Pi05ForwardExecution::record(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    Pi05Stream stream) {
    modalities::Status status = validate_pi05_operation_call(call, shape);
    if (!status.ok_status()) return status;
    if (!resources || !ops || !complete_vision(vision) ||
        !complete_encoder(encoder) || !complete_decoder(decoder) ||
        !complete_vision(active_primitives(*ops)) ||
        !complete_encoder(active_primitives(*ops)) ||
        !complete_decoder(active_primitives(*ops))) {
        return invalid("PI0.5 forward execution is incomplete");
    }
    switch (call.id) {
        case Pi05OperationId::kComposePrompt:
            return execute_compose_prompt(shape, *this, stream);
        case Pi05OperationId::kVisionEmbed:
            return execute_vision_embed(shape, *this, stream);
        case Pi05OperationId::kVisionAttention:
            return execute_vision_attention(call, shape, *this, stream);
        case Pi05OperationId::kVisionMlp:
            return execute_vision_mlp(call, shape, *this, stream);
        case Pi05OperationId::kVisionProject:
            return execute_vision_project(shape, *this, stream);
        case Pi05OperationId::kEncoderAttention:
            return execute_encoder_attention(call, shape, *this, stream);
        case Pi05OperationId::kEncoderMlp:
            return execute_encoder_mlp(call, shape, *this, stream);
        case Pi05OperationId::kEncoderCacheFinalize:
            return execute_encoder_cache_finalize(call, shape, *this, stream);
        case Pi05OperationId::kDiffusionInputProject:
            return execute_diffusion_input_project(shape, *this, stream);
        case Pi05OperationId::kDecoderAttention:
            return execute_decoder_attention(call, shape, *this, stream);
        case Pi05OperationId::kDecoderMlp:
            return execute_decoder_mlp(call, shape, *this, stream);
        case Pi05OperationId::kActionProject:
            return execute_action_project(call, shape, *this, stream);
        case Pi05OperationId::kDiffusionUpdate:
            return execute_diffusion_update(shape, *this, stream);
        default:
            return invalid("PI0.5 forward operation is unavailable");
    }
}

const char* pi05_value_name(Pi05ValueId id) {
    static constexpr const char* kNames[] = {
        "images", "prompt_embedding", "vision_state", "encoder_state",
        "key_cache", "value_cache", "noise", "decoder_state",
        "action_delta", "time_state", "attention_style", "mlp_style",
        "final_style",
    };
    static_assert(sizeof(kNames) / sizeof(kNames[0]) ==
                      static_cast<std::size_t>(Pi05ValueId::kCount),
                  "PI0.5 value name catalog must be complete");
    const std::size_t index = static_cast<std::size_t>(id);
    return index < static_cast<std::size_t>(Pi05ValueId::kCount)
               ? kNames[index]
               : nullptr;
}

const char* pi05_operation_name(Pi05OperationId id) {
    const Pi05OperationContract* contract = pi05_operation_contract(id);
    return contract ? contract->name : nullptr;
}

const Pi05OperationContract* pi05_operation_contract(Pi05OperationId id) {
    const std::size_t index = static_cast<std::size_t>(id);
    if (index >= static_cast<std::size_t>(Pi05OperationId::kCount) ||
        kContracts[index].id != id) {
        return nullptr;
    }
    return &kContracts[index];
}

modalities::Status pi05_value_spec(Pi05ValueId id,
                                   const Pi05ResolvedShape& shape,
                                   Pi05TensorSpec* out) {
    if (!out) return invalid("PI0.5 value spec destination is null");
    modalities::Status status = validate_pi05_resolved_shape(shape);
    if (!status.ok_status()) return status;

    Pi05TensorSpec spec;
    switch (id) {
        case Pi05ValueId::kImages:
            spec.rank = 4;
            spec.dimensions = {
                static_cast<std::uint64_t>(shape.num_views),
                static_cast<std::uint64_t>(kPi05ModelDims.image_height),
                static_cast<std::uint64_t>(kPi05ModelDims.image_width),
                static_cast<std::uint64_t>(kPi05ModelDims.image_channels)};
            break;
        case Pi05ValueId::kPromptEmbedding:
            spec.rank = 2;
            spec.dimensions = {
                static_cast<std::uint64_t>(shape.max_prompt_tokens),
                static_cast<std::uint64_t>(kPi05ModelDims.embedding_width),
                0, 0};
            break;
        case Pi05ValueId::kVisionState:
            spec.rank = 2;
            spec.dimensions = {
                static_cast<std::uint64_t>(shape.vision_sequence),
                static_cast<std::uint64_t>(kPi05ModelDims.vision_width),
                0, 0};
            break;
        case Pi05ValueId::kEncoderState:
            spec.rank = 2;
            spec.dimensions = {
                static_cast<std::uint64_t>(shape.encoder_sequence),
                static_cast<std::uint64_t>(kPi05ModelDims.encoder_width),
                0, 0};
            break;
        case Pi05ValueId::kKeyCache:
        case Pi05ValueId::kValueCache:
            spec.rank = 3;
            spec.dimensions = {
                static_cast<std::uint64_t>(kPi05ModelDims.encoder_layers),
                static_cast<std::uint64_t>(shape.total_attention_keys),
                static_cast<std::uint64_t>(kPi05ModelDims.encoder_head_dim),
                0};
            break;
        case Pi05ValueId::kNoise:
            spec.rank = 2;
            spec.dimensions = {
                static_cast<std::uint64_t>(shape.chunk),
                static_cast<std::uint64_t>(kPi05ModelDims.action_width),
                0, 0};
            break;
        case Pi05ValueId::kDecoderState:
            spec.rank = 2;
            spec.dimensions = {
                static_cast<std::uint64_t>(shape.chunk),
                static_cast<std::uint64_t>(kPi05ModelDims.decoder_width),
                0, 0};
            break;
        case Pi05ValueId::kActionDelta:
            spec.scalar = Pi05ScalarKind::kActionUpdate;
            spec.rank = 2;
            spec.dimensions = {
                static_cast<std::uint64_t>(shape.chunk),
                static_cast<std::uint64_t>(kPi05ModelDims.action_width),
                0, 0};
            break;
        case Pi05ValueId::kTimeState:
            spec.rank = 3;
            spec.dimensions = {
                static_cast<std::uint64_t>(shape.num_steps),
                static_cast<std::uint64_t>(shape.chunk),
                static_cast<std::uint64_t>(kPi05ModelDims.decoder_width),
                0};
            break;
        case Pi05ValueId::kAttentionStyle:
        case Pi05ValueId::kMlpStyle:
            spec.rank = 4;
            spec.dimensions = {
                static_cast<std::uint64_t>(shape.num_steps),
                static_cast<std::uint64_t>(kPi05ModelDims.decoder_layers),
                static_cast<std::uint64_t>(shape.chunk),
                static_cast<std::uint64_t>(
                    3 * kPi05ModelDims.decoder_width)};
            break;
        case Pi05ValueId::kFinalStyle:
            spec.rank = 3;
            spec.dimensions = {
                static_cast<std::uint64_t>(shape.num_steps),
                static_cast<std::uint64_t>(shape.chunk),
                static_cast<std::uint64_t>(
                    3 * kPi05ModelDims.decoder_width),
                0};
            break;
        case Pi05ValueId::kCount:
            return invalid("PI0.5 logical value id is invalid");
    }
    *out = spec;
    return modalities::Status::ok();
}

modalities::Status validate_pi05_operation_call(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape) {
    modalities::Status status = validate_pi05_resolved_shape(shape);
    if (!status.ok_status()) return status;
    const Pi05OperationContract* contract = pi05_operation_contract(call.id);
    if (!contract) return invalid("PI0.5 operation id is invalid");

    bool index_valid = false;
    switch (contract->index_domain) {
        case Pi05IndexDomain::kNone:
            index_valid = call.layer == -1 && call.step == -1;
            break;
        case Pi05IndexDomain::kVisionLayer:
            index_valid = call.step == -1 && call.layer >= 0 &&
                          call.layer < kPi05ModelDims.vision_layers;
            break;
        case Pi05IndexDomain::kEncoderLayer:
            index_valid = call.step == -1 && call.layer >= 0 &&
                          call.layer < kPi05ModelDims.encoder_layers - 1;
            break;
        case Pi05IndexDomain::kEncoderFinalLayer:
            index_valid = call.step == -1 &&
                          call.layer == kPi05ModelDims.encoder_layers - 1;
            break;
        case Pi05IndexDomain::kDiffusionStep:
            index_valid = call.layer == -1 && call.step >= 0 &&
                          call.step < shape.num_steps;
            break;
        case Pi05IndexDomain::kDecoderLayer:
            index_valid = call.step >= 0 && call.step < shape.num_steps &&
                          call.layer >= 0 &&
                          call.layer < kPi05ModelDims.decoder_layers;
            break;
    }
    if (!index_valid) return invalid("PI0.5 operation index is out of range");

    for (std::size_t i = 0; i < contract->input_count; ++i) {
        Pi05TensorSpec spec;
        status = pi05_value_spec(contract->inputs[i], shape, &spec);
        if (!status.ok_status()) return status;
    }
    for (std::size_t i = 0; i < contract->output_count; ++i) {
        Pi05TensorSpec spec;
        status = pi05_value_spec(contract->outputs[i], shape, &spec);
        if (!status.ok_status()) return status;
        const std::uint8_t alias = contract->output_alias_input[i];
        if (alias == kPi05NoAlias) {
            if (call.output_generation[i] == 0) {
                return invalid("PI0.5 new value generation is invalid");
            }
        } else {
            if (alias >= contract->input_count ||
                contract->outputs[i] != contract->inputs[alias] ||
                call.input_generation[alias] ==
                    std::numeric_limits<std::uint64_t>::max() ||
                call.output_generation[i] !=
                    call.input_generation[alias] + 1) {
                return invalid("PI0.5 aliased value generation is invalid");
            }
        }
    }
    for (std::size_t i = contract->input_count;
         i < kPi05MaxOperationValues; ++i) {
        if (call.input_generation[i] != 0) {
            return invalid("PI0.5 operation has an unused input generation");
        }
    }
    for (std::size_t i = contract->output_count;
         i < kPi05MaxOperationValues; ++i) {
        if (call.output_generation[i] != 0) {
            return invalid("PI0.5 operation has an unused output generation");
        }
    }
    return modalities::Status::ok();
}

modalities::Status Pi05SemanticPipeline::record_prepare(
    Pi05OperationSink& sink,
    Pi05Stream stream) const {
    modalities::Status status = validate_pi05_resolved_shape(shape_);
    if (!status.ok_status()) return status;
    return traverse_prepare(shape_, [&](const Pi05OperationCall& call) {
        return sink.record(call, shape_, stream);
    });
}

modalities::Status Pi05SemanticPipeline::record_prepare(
    Pi05PrepareExecution& execution,
    Pi05Stream stream) const {
    modalities::Status status = validate_pi05_resolved_shape(shape_);
    if (!status.ok_status()) return status;
    if (!execution.resources || !execution.ops ||
        !execution.scratch.mlp_hidden || !execution.scratch.mlp_output) {
        return invalid("PI0.5 prepare execution is invalid");
    }
    status = validate_pi05_resolved_resources(*execution.resources, shape_);
    if (!status.ok_status()) return status;
    if (!complete(active_primitives(*execution.ops))) {
        return invalid("PI0.5 prepare primitive profile is invalid");
    }
    return traverse_prepare(shape_, [&](const Pi05OperationCall& call) {
        return execute_prepare_call(call, shape_, execution, stream);
    });
}

modalities::Status Pi05SemanticPipeline::record_context(
    Pi05OperationSink& sink,
    Pi05Stream stream) const {
    modalities::Status status = validate_pi05_resolved_shape(shape_);
    if (!status.ok_status()) return status;

    std::uint64_t vision = 0;
    std::uint64_t encoder = 0;
    std::uint64_t key_cache = 0;
    std::uint64_t value_cache = 0;

    status = emit(sink, shape_, Pi05OperationId::kComposePrompt, -1, -1,
                  generations(0), generations(++encoder), stream);
    if (!status.ok_status()) return status;
    status = emit(sink, shape_, Pi05OperationId::kVisionEmbed, -1, -1,
                  generations(0), generations(++vision), stream);
    if (!status.ok_status()) return status;
    for (int layer = 0; layer < kPi05ModelDims.vision_layers; ++layer) {
        status = emit(sink, shape_, Pi05OperationId::kVisionAttention,
                      layer, -1, generations(vision),
                      generations(vision + 1), stream);
        if (!status.ok_status()) return status;
        ++vision;
        status = emit(sink, shape_, Pi05OperationId::kVisionMlp, layer, -1,
                      generations(vision), generations(vision + 1), stream);
        if (!status.ok_status()) return status;
        ++vision;
    }
    status = emit(sink, shape_, Pi05OperationId::kVisionProject, -1, -1,
                  generations(vision, encoder),
                  generations(encoder + 1), stream);
    if (!status.ok_status()) return status;
    ++encoder;

    for (int layer = 0; layer < kPi05ModelDims.encoder_layers - 1; ++layer) {
        status = emit(
            sink, shape_, Pi05OperationId::kEncoderAttention, layer, -1,
            generations(encoder, key_cache, value_cache),
            generations(encoder + 1, key_cache + 1, value_cache + 1),
            stream);
        if (!status.ok_status()) return status;
        ++encoder;
        ++key_cache;
        ++value_cache;
        status = emit(sink, shape_, Pi05OperationId::kEncoderMlp, layer, -1,
                      generations(encoder), generations(encoder + 1), stream);
        if (!status.ok_status()) return status;
        ++encoder;
    }
    return emit(
        sink, shape_, Pi05OperationId::kEncoderCacheFinalize,
        kPi05ModelDims.encoder_layers - 1, -1,
        generations(encoder, key_cache, value_cache),
        generations(key_cache + 1, value_cache + 1), stream);
}

modalities::Status Pi05SemanticPipeline::record_decode(
    Pi05OperationSink& sink,
    Pi05Stream stream) const {
    modalities::Status status = validate_pi05_resolved_shape(shape_);
    if (!status.ok_status()) return status;

    std::uint64_t noise = 0;
    std::uint64_t decoder = 0;
    std::uint64_t action_delta = 0;
    std::uint64_t key_cache =
        static_cast<std::uint64_t>(kPi05ModelDims.encoder_layers);
    std::uint64_t value_cache = key_cache;
    const std::uint64_t attention_style =
        static_cast<std::uint64_t>(shape_.num_steps) *
        static_cast<std::uint64_t>(kPi05ModelDims.decoder_layers);
    const std::uint64_t mlp_style = attention_style;
    const std::uint64_t final_style =
        static_cast<std::uint64_t>(shape_.num_steps);

    for (int step = 0; step < shape_.num_steps; ++step) {
        status = emit(sink, shape_, Pi05OperationId::kDiffusionInputProject,
                      -1, step, generations(noise),
                      generations(decoder + 1), stream);
        if (!status.ok_status()) return status;
        ++decoder;
        for (int layer = 0; layer < kPi05ModelDims.decoder_layers; ++layer) {
            status = emit(
                sink, shape_, Pi05OperationId::kDecoderAttention, layer, step,
                generations(decoder, key_cache, value_cache, attention_style),
                generations(decoder + 1, key_cache + 1, value_cache + 1),
                stream);
            if (!status.ok_status()) return status;
            ++decoder;
            ++key_cache;
            ++value_cache;
            status = emit(sink, shape_, Pi05OperationId::kDecoderMlp,
                          layer, step, generations(decoder, mlp_style),
                          generations(decoder + 1), stream);
            if (!status.ok_status()) return status;
            ++decoder;
        }
        status = emit(sink, shape_, Pi05OperationId::kActionProject, -1,
                      step, generations(decoder, final_style),
                      generations(action_delta + 1), stream);
        if (!status.ok_status()) return status;
        ++action_delta;
        status = emit(sink, shape_, Pi05OperationId::kDiffusionUpdate, -1,
                      step, generations(noise, action_delta),
                      generations(noise + 1), stream);
        if (!status.ok_status()) return status;
        ++noise;
    }
    return modalities::Status::ok();
}

modalities::Status Pi05SemanticPipeline::record_full(
    Pi05OperationSink& sink,
    Pi05Stream stream) const {
    modalities::Status status = record_context(sink, stream);
    return status.ok_status() ? record_decode(sink, stream) : status;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

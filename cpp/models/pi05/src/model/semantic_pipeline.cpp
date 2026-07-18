#include "flashrt/cpp/models/pi05/model/semantic_pipeline.h"

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

}  // namespace

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

    std::uint64_t time_state = 0;
    std::uint64_t attention_style = 0;
    std::uint64_t mlp_style = 0;
    std::uint64_t final_style = 0;
    for (int step = 0; step < shape_.num_steps; ++step) {
        status = emit(sink, shape_, Pi05OperationId::kTimeMlp, -1, step,
                      generations(), generations(time_state + 1), stream);
        if (!status.ok_status()) return status;
        ++time_state;
        for (int layer = 0; layer < kPi05ModelDims.decoder_layers; ++layer) {
            status = emit(sink, shape_, Pi05OperationId::kAttentionStyle,
                          layer, step, generations(time_state),
                          generations(attention_style + 1), stream);
            if (!status.ok_status()) return status;
            ++attention_style;
            status = emit(sink, shape_, Pi05OperationId::kMlpStyle,
                          layer, step, generations(time_state),
                          generations(mlp_style + 1), stream);
            if (!status.ok_status()) return status;
            ++mlp_style;
        }
        status = emit(sink, shape_, Pi05OperationId::kFinalStyle, -1, step,
                      generations(time_state), generations(final_style + 1),
                      stream);
        if (!status.ok_status()) return status;
        ++final_style;
    }
    return modalities::Status::ok();
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

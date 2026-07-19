#include "flashrt/cpp/models/pi05/targets/sm110/physical_resources.h"

#include "flashrt/cpp/models/pi05/model/linear_weight_groups.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm110 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

std::uint64_t dim(int value) {
    return static_cast<std::uint64_t>(value);
}

bool byte_count(const modalities::Shape& shape,
                modalities::DType dtype,
                std::size_t* out) {
    if (!shape.rank || shape.rank > modalities::Shape::kMaxRank) return false;
    std::size_t elements = 1;
    for (std::size_t i = 0; i < shape.rank; ++i) {
        const std::uint64_t dimension = shape.dims[i];
        if (!dimension || dimension > std::numeric_limits<std::size_t>::max() ||
            elements > std::numeric_limits<std::size_t>::max() /
                           static_cast<std::size_t>(dimension)) {
            return false;
        }
        elements *= static_cast<std::size_t>(dimension);
    }
    const std::size_t width = modalities::dtype_size(dtype);
    if (!width || elements > std::numeric_limits<std::size_t>::max() / width) {
        return false;
    }
    if (out) *out = elements * width;
    return true;
}

modalities::Status resolve_value(const Sm110Buffer& source,
                                 Pi05ValueId value,
                                 const Pi05ResolvedShape& shape,
                                 Pi05ResolvedBuffer* out) {
    if (!out || !source.buffer || !source.device_data() || !source.bytes()) {
        return invalid("SM110 target value backing is invalid");
    }
    Pi05ResolvedBuffer result;
    modalities::Status status = pi05_value_spec(value, shape,
                                                &result.logical_spec);
    if (!status.ok_status()) return status;
    result.buffer = source.buffer;
    result.physical_dtype = source.dtype;
    result.physical_shape = source.shape;
    result.physical_bytes = source.bytes();
    result.storage_identity = source.device_data();
    result.storage_bytes = source.bytes();
    *out = result;
    return modalities::Status::ok();
}

Pi05ResolvedBuffer resolve_control(const Sm110Buffer& source) {
    Pi05ResolvedBuffer result;
    result.buffer = source.buffer;
    result.physical_dtype = source.dtype;
    result.physical_shape = source.shape;
    result.physical_bytes = source.bytes();
    result.storage_identity = source.device_data();
    result.storage_bytes = source.bytes();
    return result;
}

bool calibration_matches(const NativeCalibrationArtifact& calibration,
                         const Pi05ResolvedShape& shape) {
    return calibration.activation_dtype == "float16" &&
           calibration.hardware == "sm110" &&
           calibration.num_views == shape.num_views &&
           calibration.max_prompt_tokens == shape.max_prompt_tokens &&
           calibration.chunk_size == shape.chunk &&
           calibration.num_steps == shape.num_steps &&
           calibration.vision_pool_factor == shape.vision_pool_factor &&
           calibration.state_dim == shape.state_dim;
}

}  // namespace

modalities::Status Sm110PhysicalResources::add(
    const char* name,
    const modalities::Shape& shape,
    modalities::DType dtype,
    Sm110Buffer* out) {
    if (!context_ || !name || !name[0] || !out || out->buffer) {
        return invalid("SM110 private buffer definition is invalid");
    }
    std::size_t bytes = 0;
    if (!byte_count(shape, dtype, &bytes)) {
        return invalid("SM110 private buffer shape is invalid");
    }
    frt_buffer buffer = frt_buffer_alloc(context_, name, bytes);
    if (!buffer || !frt_buffer_dptr(buffer)) {
        return backend("SM110 private buffer allocation failed");
    }
    out->buffer = buffer;
    out->dtype = dtype;
    out->shape = shape;
    ++allocation_count_;
    allocated_bytes_ += bytes;
    return modalities::Status::ok();
}

modalities::Status Sm110PhysicalResources::allocate(
    const Pi05ResolvedShape& shape) {
    if (!context_ || allocation_started_) {
        return invalid("SM110 physical resources cannot be allocated");
    }
    allocation_started_ = true;
    modalities::Status status = validate_pi05_resolved_shape(shape);
    if (!status.ok_status()) return status;
    shape_ = shape;
    padded_key_stride_ = shape.total_attention_keys +
                         (shape.total_attention_keys & 1);

    const std::uint64_t vision_sequence = dim(shape.vision_sequence);
    const std::uint64_t encoder_sequence = dim(shape.encoder_sequence);
    const std::uint64_t decoder_sequence = dim(shape.chunk);
    const std::uint64_t steps = dim(shape.num_steps);
    const std::uint64_t keys = dim(shape.total_attention_keys);
    const std::uint64_t padded_keys = dim(padded_key_stride_);
    const std::uint64_t encoder_layers =
        dim(kPi05ModelDims.encoder_layers);
    const std::uint64_t decoder_layers =
        dim(kPi05ModelDims.decoder_layers);
    const std::uint64_t vision_width = dim(kPi05ModelDims.vision_width);
    const std::uint64_t encoder_width = dim(kPi05ModelDims.encoder_width);
    const std::uint64_t decoder_width = dim(kPi05ModelDims.decoder_width);
    const std::uint64_t encoder_qkv =
        encoder_width + 2 * dim(kPi05ModelDims.encoder_kv_heads) *
                            dim(kPi05ModelDims.encoder_head_dim);
    const std::uint64_t decoder_qkv =
        dim(kPi05ModelDims.decoder_heads) *
            dim(kPi05ModelDims.decoder_head_dim) +
        2 * dim(kPi05ModelDims.decoder_kv_heads) *
            dim(kPi05ModelDims.decoder_head_dim);

#define FRT_ADD(member, name, shape_value, dtype_value)              \
    do {                                                              \
        status = add(name, shape_value, dtype_value, &member);        \
        if (!status.ok_status()) return status;                        \
    } while (false)

    FRT_ADD(vision_.state_fp8, "pi05_sm110_vision_state_fp8",
            modalities::Shape({vision_sequence, vision_width}),
            modalities::DType::kUInt8);
    FRT_ADD(vision_.qkv, "pi05_sm110_vision_qkv",
            modalities::Shape({vision_sequence, 3 * vision_width}),
            modalities::DType::kFloat16);
    FRT_ADD(vision_.attention, "pi05_sm110_vision_attention",
            modalities::Shape({vision_sequence, vision_width}),
            modalities::DType::kFloat16);
    vision_.post_norm = vision_.attention;
    vision_.post_norm.alias = true;
    FRT_ADD(vision_.hidden, "pi05_sm110_vision_hidden",
            modalities::Shape(
                {vision_sequence, dim(kPi05ModelDims.vision_hidden)}),
            modalities::DType::kFloat16);
    FRT_ADD(vision_.hidden_fp8, "pi05_sm110_vision_hidden_fp8",
            modalities::Shape(
                {vision_sequence, dim(kPi05ModelDims.vision_hidden)}),
            modalities::DType::kUInt8);
    FRT_ADD(vision_.unit_scale, "pi05_sm110_vision_unit_scale",
            modalities::Shape({1}), modalities::DType::kFloat32);

    FRT_ADD(encoder_.state_fp8, "pi05_sm110_encoder_state_fp8",
            modalities::Shape({encoder_sequence, encoder_width}),
            modalities::DType::kUInt8);
    FRT_ADD(encoder_.qkv, "pi05_sm110_encoder_qkv",
            modalities::Shape({encoder_sequence, encoder_qkv}),
            modalities::DType::kFloat16);
    FRT_ADD(encoder_.logits, "pi05_sm110_encoder_logits",
            modalities::Shape({encoder_sequence *
                                   dim(kPi05ModelDims.encoder_heads),
                               padded_keys}),
            modalities::DType::kFloat16);
    FRT_ADD(encoder_.attention, "pi05_sm110_encoder_attention",
            modalities::Shape({encoder_sequence, encoder_width}),
            modalities::DType::kFloat16);
    FRT_ADD(encoder_.output_fp8, "pi05_sm110_encoder_output_fp8",
            modalities::Shape({encoder_sequence, encoder_width}),
            modalities::DType::kUInt8);
    FRT_ADD(encoder_.gate_up, "pi05_sm110_encoder_gate_up",
            modalities::Shape(
                {encoder_sequence, 2 * dim(kPi05ModelDims.encoder_hidden)}),
            modalities::DType::kFloat16);
    FRT_ADD(encoder_.hidden_fp8, "pi05_sm110_encoder_hidden_fp8",
            modalities::Shape(
                {encoder_sequence, dim(kPi05ModelDims.encoder_hidden)}),
            modalities::DType::kUInt8);
    FRT_ADD(encoder_.residual_output, "pi05_sm110_encoder_residual_output",
            modalities::Shape({encoder_sequence, encoder_width}),
            modalities::DType::kFloat16);
    FRT_ADD(encoder_.activation_scales,
            "pi05_sm110_encoder_activation_scales",
            modalities::Shape(
                {encoder_layers, kPi05LinearScalesPerLayer}),
            modalities::DType::kFloat32);
    FRT_ADD(encoder_.key_cache, "pi05_sm110_key_cache",
            modalities::Shape(
                {encoder_layers, keys,
                 dim(kPi05ModelDims.encoder_head_dim)}),
            modalities::DType::kFloat16);
    FRT_ADD(encoder_.value_cache, "pi05_sm110_value_cache",
            modalities::Shape(
                {encoder_layers, keys,
                 dim(kPi05ModelDims.encoder_head_dim)}),
            modalities::DType::kFloat16);

    FRT_ADD(decoder_.normalized, "pi05_sm110_decoder_normalized",
            modalities::Shape({decoder_sequence, decoder_width}),
            modalities::DType::kFloat16);
    FRT_ADD(decoder_.gate, "pi05_sm110_decoder_gate",
            modalities::Shape({decoder_sequence, decoder_width}),
            modalities::DType::kFloat16);
    FRT_ADD(decoder_.qkv, "pi05_sm110_decoder_qkv",
            modalities::Shape({decoder_sequence, decoder_qkv}),
            modalities::DType::kFloat16);
    FRT_ADD(decoder_.logits, "pi05_sm110_decoder_logits",
            modalities::Shape({decoder_sequence *
                                   dim(kPi05ModelDims.decoder_heads),
                               padded_keys}),
            modalities::DType::kFloat16);
    FRT_ADD(decoder_.attention, "pi05_sm110_decoder_attention",
            modalities::Shape({decoder_sequence, encoder_width}),
            modalities::DType::kFloat16);
    FRT_ADD(decoder_.projection, "pi05_sm110_decoder_projection",
            modalities::Shape(
                {decoder_sequence, 2 * dim(kPi05ModelDims.decoder_hidden)}),
            modalities::DType::kFloat16);
    FRT_ADD(decoder_.action_delta, "pi05_sm110_action_delta",
            modalities::Shape(
                {decoder_sequence, dim(kPi05ModelDims.action_width)}),
            modalities::DType::kFloat32);
    FRT_ADD(decoder_.state_fp8, "pi05_sm110_decoder_state_fp8",
            modalities::Shape({decoder_sequence, decoder_width}),
            modalities::DType::kUInt8);
    FRT_ADD(decoder_.hidden_fp8, "pi05_sm110_decoder_hidden_fp8",
            modalities::Shape(
                {decoder_sequence, dim(kPi05ModelDims.decoder_hidden)}),
            modalities::DType::kUInt8);
    FRT_ADD(decoder_.context_fp8, "pi05_sm110_decoder_context_fp8",
            modalities::Shape({decoder_sequence, encoder_width}),
            modalities::DType::kUInt8);
    FRT_ADD(decoder_.activation_scales,
            "pi05_sm110_decoder_activation_scales",
            modalities::Shape(
                {steps, decoder_layers, kPi05LinearScalesPerLayer}),
            modalities::DType::kFloat32);

    const modalities::Shape control_shape({sizeof(std::int32_t)});
    FRT_ADD(controls_.encoder_valid_tokens,
            "pi05_sm110_encoder_valid_tokens", control_shape,
            modalities::DType::kUInt8);
    FRT_ADD(controls_.decoder_valid_tokens,
            "pi05_sm110_decoder_valid_tokens", control_shape,
            modalities::DType::kUInt8);
    FRT_ADD(controls_.decoder_position,
            "pi05_sm110_decoder_position", control_shape,
            modalities::DType::kUInt8);
#undef FRT_ADD

    allocated_ = true;
    return modalities::Status::ok();
}

modalities::Status Sm110PhysicalResources::initialize_static(
    const NativeCalibrationArtifact& calibration) {
    modalities::Status status = validate_native_calibration_artifact(
        calibration);
    if (!allocated_ || initialized_ || !status.ok_status() ||
        !calibration_matches(calibration, shape_)) {
        return invalid("SM110 static resource calibration is invalid");
    }
    const float unit = 1.0f;
    const std::size_t encoder_bytes =
        calibration.encoder_scales.size() * sizeof(float);
    const std::size_t decoder_bytes =
        calibration.decoder_scales.size() * sizeof(float);
    if (encoder_bytes != encoder_.activation_scales.bytes() ||
        decoder_bytes != decoder_.activation_scales.bytes() ||
        cudaMemcpy(vision_.unit_scale.device_data(), &unit, sizeof(unit),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(encoder_.activation_scales.device_data(),
                   calibration.encoder_scales.data(), encoder_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(decoder_.activation_scales.device_data(),
                   calibration.decoder_scales.data(), decoder_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return backend("SM110 static resource upload failed");
    }
    status = set_prompt_length(0);
    if (!status.ok_status()) return status;
    initialized_ = true;
    return modalities::Status::ok();
}

modalities::Status Sm110PhysicalResources::set_prompt_length(
    int prompt_tokens) {
    if (!allocated_ || prompt_tokens < 0 ||
        prompt_tokens > shape_.max_prompt_tokens) {
        return invalid("SM110 prompt length is invalid");
    }
    const std::int32_t valid =
        shape_.encoder_vision_sequence + prompt_tokens;
    const std::int32_t decoder_valid = valid + shape_.chunk;
    if (cudaMemcpy(controls_.encoder_valid_tokens.device_data(), &valid,
                   sizeof(valid), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(controls_.decoder_valid_tokens.device_data(),
                   &decoder_valid, sizeof(decoder_valid),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(controls_.decoder_position.device_data(), &valid,
                   sizeof(valid), cudaMemcpyHostToDevice) != cudaSuccess) {
        return backend("SM110 prompt control upload failed");
    }
    return modalities::Status::ok();
}

modalities::Status Sm110PhysicalResources::make_target_bindings(
    Pi05TargetBufferBindings* out) const {
    if (!out || !initialized_) {
        return invalid("SM110 target binding destination is invalid");
    }
    Pi05TargetBufferBindings result;
    modalities::Status status = resolve_value(
        encoder_.key_cache, Pi05ValueId::kKeyCache, shape_,
        &result.key_cache);
    if (!status.ok_status()) return status;
    status = resolve_value(
        encoder_.value_cache, Pi05ValueId::kValueCache, shape_,
        &result.value_cache);
    if (!status.ok_status()) return status;
    status = resolve_value(
        decoder_.action_delta, Pi05ValueId::kActionDelta, shape_,
        &result.action_delta);
    if (!status.ok_status()) return status;
    result.encoder_valid_tokens =
        resolve_control(controls_.encoder_valid_tokens);
    result.decoder_valid_tokens =
        resolve_control(controls_.decoder_valid_tokens);
    result.decoder_position = resolve_control(controls_.decoder_position);
    *out = result;
    return modalities::Status::ok();
}

}  // namespace sm110
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

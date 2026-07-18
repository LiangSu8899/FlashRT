#include "flashrt/cpp/models/pi05/targets/sm120/attention.h"

#include "flashrt/cpp/models/pi05/model/dims.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {
namespace {

constexpr std::uint64_t kVendorAccumulatedHeadWidth = 96;

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

std::uint64_t dim(int value) {
    return static_cast<std::uint64_t>(value);
}

std::uint64_t round_up_128(std::uint64_t value) {
    return ((value + 127) / 128) * 128;
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

modalities::Status resolve_target_buffer(
    const Sm120AttentionBuffer& source,
    Pi05ValueId value,
    const Pi05ResolvedShape& shape,
    Pi05ResolvedBuffer* out) {
    if (!out || !source.buffer || !source.device_data() || !source.bytes()) {
        return invalid("SM120 attention target buffer is invalid");
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

Pi05ResolvedBuffer resolve_control(const Sm120AttentionBuffer& source) {
    Pi05ResolvedBuffer result;
    result.buffer = source.buffer;
    result.physical_dtype = source.dtype;
    result.physical_shape = source.shape;
    result.physical_bytes = source.bytes();
    result.storage_identity = source.device_data();
    result.storage_bytes = source.bytes();
    return result;
}

}  // namespace

modalities::Status Sm120AttentionBacking::add(
    const char* name,
    const modalities::Shape& shape,
    modalities::DType dtype,
    Sm120AttentionBuffer* out) {
    if (!context_ || !name || !name[0] || !out || out->buffer) {
        return invalid("SM120 attention buffer definition is invalid");
    }
    std::size_t bytes = 0;
    if (!byte_count(shape, dtype, &bytes)) {
        return invalid("SM120 attention buffer shape is invalid");
    }
    frt_buffer buffer = frt_buffer_alloc(context_, name, bytes);
    if (!buffer) return backend("SM120 attention allocation failed");
    out->buffer = buffer;
    out->dtype = dtype;
    out->shape = shape;
    ++allocation_count_;
    allocated_bytes_ += bytes;
    return modalities::Status::ok();
}

modalities::Status Sm120AttentionBacking::allocate(
    const Pi05ResolvedShape& shape) {
    if (!context_ || allocation_started_) {
        return invalid("SM120 attention backing cannot be allocated");
    }
    allocation_started_ = true;
    modalities::Status status = validate_pi05_resolved_shape(shape);
    if (!status.ok_status()) return status;
    shape_ = shape;

    const std::uint64_t views = dim(shape.num_views);
    const std::uint64_t encoder_sequence = dim(shape.encoder_sequence);
    const std::uint64_t chunk = dim(shape.chunk);
    const std::uint64_t total_keys = dim(shape.total_attention_keys);
    const std::uint64_t encoder_layers = dim(kPi05ModelDims.encoder_layers);
    decoder_splits_ =
        std::min(128, (shape.total_attention_keys + 63) / 64);
    cache_layer_stride_bytes_ =
        static_cast<std::size_t>(shape.total_attention_keys) *
        kPi05ModelDims.encoder_head_dim * sizeof(std::uint16_t);

#define FRT_ADD(member, name, shape_value, dtype_value)  \
    do {                                                  \
        status = add(name, shape_value, dtype_value, &member); \
        if (!status.ok_status()) return status;            \
    } while (false)

    FRT_ADD(vision_.query, "sm120_attention_vision_query",
            modalities::Shape({views,
                               dim(kPi05ModelDims.vision_tokens_per_view),
                               dim(kPi05ModelDims.vision_heads),
                               dim(kPi05ModelDims.vision_head_dim)}),
            modalities::DType::kBFloat16);
    FRT_ADD(vision_.key, "sm120_attention_vision_key",
            modalities::Shape({views,
                               dim(kPi05ModelDims.vision_tokens_per_view),
                               dim(kPi05ModelDims.vision_heads),
                               dim(kPi05ModelDims.vision_head_dim)}),
            modalities::DType::kBFloat16);
    FRT_ADD(vision_.value, "sm120_attention_vision_value",
            modalities::Shape({views,
                               dim(kPi05ModelDims.vision_tokens_per_view),
                               dim(kPi05ModelDims.vision_heads),
                               dim(kPi05ModelDims.vision_head_dim)}),
            modalities::DType::kBFloat16);
    FRT_ADD(vision_.output, "sm120_attention_vision_output",
            modalities::Shape({views,
                               dim(kPi05ModelDims.vision_tokens_per_view),
                               dim(kPi05ModelDims.vision_heads),
                               dim(kPi05ModelDims.vision_head_dim)}),
            modalities::DType::kBFloat16);
    FRT_ADD(vision_.logsumexp, "sm120_attention_vision_logsumexp",
            modalities::Shape({views, dim(kPi05ModelDims.vision_heads),
                               dim(kPi05ModelDims.vision_tokens_per_view)}),
            modalities::DType::kFloat32);
    FRT_ADD(vision_.logsumexp_accumulator,
            "sm120_attention_vision_logsumexp_accumulator",
            modalities::Shape({2, views, dim(kPi05ModelDims.vision_heads),
                               dim(kPi05ModelDims.vision_tokens_per_view)}),
            modalities::DType::kFloat32);
    FRT_ADD(vision_.output_accumulator,
            "sm120_attention_vision_output_accumulator",
            modalities::Shape({2, views, dim(kPi05ModelDims.vision_heads),
                               dim(kPi05ModelDims.vision_tokens_per_view),
                               kVendorAccumulatedHeadWidth}),
            modalities::DType::kFloat32);

    FRT_ADD(encoder_.query, "sm120_attention_encoder_query",
            modalities::Shape({encoder_sequence,
                               dim(kPi05ModelDims.encoder_heads),
                               dim(kPi05ModelDims.encoder_head_dim)}),
            modalities::DType::kBFloat16);
    FRT_ADD(encoder_.output, "sm120_attention_encoder_output",
            modalities::Shape({1, encoder_sequence,
                               dim(kPi05ModelDims.encoder_heads),
                               dim(kPi05ModelDims.encoder_head_dim)}),
            modalities::DType::kBFloat16);
    FRT_ADD(encoder_.logsumexp, "sm120_attention_encoder_logsumexp",
            modalities::Shape({1, dim(kPi05ModelDims.encoder_heads),
                               round_up_128(encoder_sequence)}),
            modalities::DType::kFloat32);
    FRT_ADD(decoder_.query, "sm120_attention_decoder_query",
            modalities::Shape({chunk, dim(kPi05ModelDims.decoder_heads),
                               dim(kPi05ModelDims.decoder_head_dim)}),
            modalities::DType::kBFloat16);
    FRT_ADD(decoder_.output, "sm120_attention_decoder_output",
            modalities::Shape({1, chunk,
                               dim(kPi05ModelDims.decoder_heads),
                               dim(kPi05ModelDims.decoder_head_dim)}),
            modalities::DType::kBFloat16);
    FRT_ADD(decoder_.logsumexp, "sm120_attention_decoder_logsumexp",
            modalities::Shape({1, dim(kPi05ModelDims.decoder_heads),
                               round_up_128(chunk)}),
            modalities::DType::kFloat32);
    FRT_ADD(decoder_.logsumexp_accumulator,
            "sm120_attention_decoder_logsumexp_accumulator",
            modalities::Shape({dim(decoder_splits_), 1,
                               dim(kPi05ModelDims.decoder_heads), chunk}),
            modalities::DType::kFloat32);
    FRT_ADD(decoder_.output_accumulator,
            "sm120_attention_decoder_output_accumulator",
            modalities::Shape({dim(decoder_splits_), 1,
                               dim(kPi05ModelDims.decoder_heads), chunk,
                               dim(kPi05ModelDims.decoder_head_dim)}),
            modalities::DType::kFloat32);

    FRT_ADD(cache_.key, "sm120_attention_key_cache",
            modalities::Shape({encoder_layers, total_keys,
                               dim(kPi05ModelDims.encoder_kv_heads),
                               dim(kPi05ModelDims.encoder_head_dim)}),
            modalities::DType::kBFloat16);
    FRT_ADD(cache_.value, "sm120_attention_value_cache",
            modalities::Shape({encoder_layers, total_keys,
                               dim(kPi05ModelDims.encoder_kv_heads),
                               dim(kPi05ModelDims.encoder_head_dim)}),
            modalities::DType::kBFloat16);

    const modalities::Shape control_shape({sizeof(std::int32_t)});
    FRT_ADD(controls_.encoder_valid_tokens,
            "sm120_attention_encoder_valid_tokens", control_shape,
            modalities::DType::kUInt8);
    FRT_ADD(controls_.decoder_valid_tokens,
            "sm120_attention_decoder_valid_tokens", control_shape,
            modalities::DType::kUInt8);
    FRT_ADD(controls_.decoder_position,
            "sm120_attention_decoder_position", control_shape,
            modalities::DType::kUInt8);
#undef FRT_ADD

    status = upload_controls(0);
    if (!status.ok_status()) return status;
    allocated_ = true;
    return modalities::Status::ok();
}

modalities::Status Sm120AttentionBacking::upload_controls(
    int prompt_tokens) {
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
        return backend("SM120 attention control upload failed");
    }
    return modalities::Status::ok();
}

modalities::Status Sm120AttentionBacking::set_prompt_length(
    int prompt_tokens) {
    if (!allocated_ || prompt_tokens < 0 ||
        prompt_tokens > shape_.max_prompt_tokens) {
        return invalid("SM120 attention prompt length is invalid");
    }
    return upload_controls(prompt_tokens);
}

modalities::Status Sm120AttentionBacking::make_target_bindings(
    Pi05TargetBufferBindings* out) const {
    if (!out || !allocated_) {
        return invalid("SM120 attention binding destination is invalid");
    }
    Pi05TargetBufferBindings result;
    modalities::Status status = resolve_target_buffer(
        cache_.key, Pi05ValueId::kKeyCache, shape_, &result.key_cache);
    if (!status.ok_status()) return status;
    status = resolve_target_buffer(
        cache_.value, Pi05ValueId::kValueCache, shape_, &result.value_cache);
    if (!status.ok_status()) return status;
    result.encoder_valid_tokens =
        resolve_control(controls_.encoder_valid_tokens);
    result.decoder_valid_tokens =
        resolve_control(controls_.decoder_valid_tokens);
    result.decoder_position = resolve_control(controls_.decoder_position);
    *out = result;
    return modalities::Status::ok();
}

void* Sm120AttentionBacking::key_layer_data(int layer) const {
    if (!allocated_ || layer < 0 || layer >= kPi05ModelDims.encoder_layers) {
        return nullptr;
    }
    return static_cast<unsigned char*>(cache_.key.device_data()) +
           static_cast<std::size_t>(layer) * cache_layer_stride_bytes_;
}

void* Sm120AttentionBacking::value_layer_data(int layer) const {
    if (!allocated_ || layer < 0 || layer >= kPi05ModelDims.encoder_layers) {
        return nullptr;
    }
    return static_cast<unsigned char*>(cache_.value.device_data()) +
           static_cast<std::size_t>(layer) * cache_layer_stride_bytes_;
}

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

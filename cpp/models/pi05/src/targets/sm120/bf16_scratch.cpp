#include "flashrt/cpp/models/pi05/targets/sm120/bf16_scratch.h"

#include <limits>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {
namespace {

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

bool byte_count(const modalities::Shape& shape, std::size_t* out) {
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
    if (elements > std::numeric_limits<std::size_t>::max() /
                       sizeof(std::uint16_t)) {
        return false;
    }
    if (out) *out = elements * sizeof(std::uint16_t);
    return true;
}

}  // namespace

modalities::Status Sm120Bf16ScratchBacking::add(
    const char* name,
    const modalities::Shape& shape,
    Sm120DeviceBuffer* out) {
    if (!context_ || !name || !name[0] || !out || out->buffer) {
        return invalid("SM120 BF16 scratch definition is invalid");
    }
    std::size_t bytes = 0;
    if (!byte_count(shape, &bytes)) {
        return invalid("SM120 BF16 scratch shape is invalid");
    }
    frt_buffer buffer = frt_buffer_alloc(context_, name, bytes);
    if (!buffer) return backend("SM120 BF16 scratch allocation failed");
    out->buffer = buffer;
    out->dtype = modalities::DType::kBFloat16;
    out->shape = shape;
    ++allocation_count_;
    allocated_bytes_ += bytes;
    return modalities::Status::ok();
}

modalities::Status Sm120Bf16ScratchBacking::allocate(
    const Pi05ResolvedShape& shape) {
    if (!context_ || allocation_started_) {
        return invalid("SM120 BF16 scratch cannot be allocated");
    }
    modalities::Status status = validate_pi05_resolved_shape(shape);
    if (!status.ok_status()) return status;
    allocation_started_ = true;
    shape_ = shape;

    const std::uint64_t vision_sequence = dim(shape.vision_sequence);
    const std::uint64_t encoder_sequence = dim(shape.encoder_sequence);
    const std::uint64_t decoder_sequence = dim(shape.chunk);
    const std::uint64_t vision_width = dim(kPi05ModelDims.vision_width);
    const std::uint64_t encoder_width = dim(kPi05ModelDims.encoder_width);
    const std::uint64_t decoder_width = dim(kPi05ModelDims.decoder_width);
    const std::uint64_t encoder_qkv =
        encoder_width +
        2 * dim(kPi05ModelDims.encoder_kv_heads) *
            dim(kPi05ModelDims.encoder_head_dim);
    const std::uint64_t decoder_qkv =
        dim(kPi05ModelDims.decoder_heads) *
            dim(kPi05ModelDims.decoder_head_dim) +
        2 * dim(kPi05ModelDims.decoder_kv_heads) *
            dim(kPi05ModelDims.decoder_head_dim);

#define FRT_ADD(member, name, shape_value)                 \
    do {                                                    \
        status = add(name, shape_value, &member);          \
        if (!status.ok_status()) return status;            \
    } while (false)

    FRT_ADD(vision_.normalized, "sm120_bf16_vision_normalized",
            modalities::Shape({vision_sequence, vision_width}));
    FRT_ADD(vision_.qkv, "sm120_bf16_vision_qkv",
            modalities::Shape({vision_sequence, 3 * vision_width}));
    FRT_ADD(vision_.hidden, "sm120_bf16_vision_hidden",
            modalities::Shape(
                {vision_sequence, dim(kPi05ModelDims.vision_hidden)}));

    FRT_ADD(encoder_.normalized, "sm120_bf16_encoder_normalized",
            modalities::Shape({encoder_sequence, encoder_width}));
    FRT_ADD(encoder_.qkv, "sm120_bf16_encoder_qkv",
            modalities::Shape({encoder_sequence, encoder_qkv}));
    FRT_ADD(encoder_.gate, "sm120_bf16_encoder_gate",
            modalities::Shape(
                {encoder_sequence, dim(kPi05ModelDims.encoder_hidden)}));
    FRT_ADD(encoder_.hidden, "sm120_bf16_encoder_hidden",
            modalities::Shape(
                {encoder_sequence, dim(kPi05ModelDims.encoder_hidden)}));

    FRT_ADD(decoder_.normalized, "sm120_bf16_decoder_normalized",
            modalities::Shape({decoder_sequence, decoder_width}));
    FRT_ADD(decoder_.gate, "sm120_bf16_decoder_gate",
            modalities::Shape({decoder_sequence, decoder_width}));
    FRT_ADD(decoder_.qkv, "sm120_bf16_decoder_qkv",
            modalities::Shape({decoder_sequence, decoder_qkv}));
    FRT_ADD(decoder_.gate_projection, "sm120_bf16_decoder_gate_projection",
            modalities::Shape(
                {decoder_sequence, dim(kPi05ModelDims.decoder_hidden)}));
    FRT_ADD(decoder_.hidden, "sm120_bf16_decoder_hidden",
            modalities::Shape(
                {decoder_sequence, dim(kPi05ModelDims.decoder_hidden)}));
#undef FRT_ADD

    allocated_ = true;
    return modalities::Status::ok();
}

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#ifndef FLASHRT_CPP_TESTS_PI05_RESOLVED_FIXTURE_H
#define FLASHRT_CPP_TESTS_PI05_RESOLVED_FIXTURE_H

#include "flashrt/cpp/models/pi05/model/resolved_resources.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace flashrt {
namespace tests {
namespace pi05_fixture {

inline void require(bool condition) {
    if (!condition) std::abort();
}

inline models::pi05::Pi05ResolvedShape canonical_shape() {
    models::pi05::Pi05ShapeConfig config;
    config.num_views = 3;
    config.max_prompt_tokens = 64;
    config.chunk = 10;
    config.num_steps = 10;
    config.vision_pool_factor = 1;
    config.state_dim = 8;
    config.robot_action_dim = 7;
    models::pi05::Pi05ResolvedShape shape;
    require(models::pi05::resolve_pi05_shape(config, &shape).ok_status());
    return shape;
}

inline modalities::Shape physical_shape(
    const models::pi05::Pi05TensorSpec& spec) {
    modalities::Shape shape;
    shape.rank = spec.rank;
    for (std::size_t i = 0; i < spec.rank; ++i) {
        shape.dims[i] = spec.dimensions[i];
    }
    return shape;
}

inline std::uint64_t buffer_bytes(const modalities::Shape& shape,
                                  modalities::DType dtype) {
    const std::uint64_t elements = shape.elements();
    const std::size_t width = modalities::dtype_size(dtype);
    require(elements != 0 && width != 0);
    return elements * width;
}

inline models::pi05::Pi05ResolvedBuffer make_buffer(
    frt_ctx context,
    void* storage,
    const modalities::Shape& shape,
    modalities::DType dtype,
    models::pi05::Pi05TensorSpec logical_spec = {}) {
    const std::uint64_t bytes = buffer_bytes(shape, dtype);
    frt_buffer handle = frt_buffer_wrap(
        context, "pi05_resolved_fixture", storage,
        static_cast<std::size_t>(bytes));
    require(handle != nullptr);
    models::pi05::Pi05ResolvedBuffer buffer;
    buffer.buffer = handle;
    buffer.physical_dtype = dtype;
    buffer.physical_shape = shape;
    buffer.physical_bytes = bytes;
    buffer.logical_spec = logical_spec;
    buffer.storage_identity = storage;
    buffer.storage_bytes = bytes;
    return buffer;
}

inline models::pi05::Pi05WeightStorage activation_storage(
    modalities::DType dtype) {
    return dtype == modalities::DType::kFloat16
               ? models::pi05::Pi05WeightStorage::kFloat16
               : models::pi05::Pi05WeightStorage::kBFloat16;
}

inline std::size_t weight_width(
    models::pi05::Pi05WeightStorage storage) {
    switch (storage) {
        case models::pi05::Pi05WeightStorage::kBFloat16:
        case models::pi05::Pi05WeightStorage::kFloat16: return 2;
        case models::pi05::Pi05WeightStorage::kFp8E4M3: return 1;
    }
    return 0;
}

inline models::pi05::Pi05ResolvedWeight make_weight(
    void* storage,
    const modalities::Shape& shape,
    models::pi05::Pi05WeightStorage type) {
    models::pi05::Pi05ResolvedWeight weight;
    weight.device_data = storage;
    if (type == models::pi05::Pi05WeightStorage::kFp8E4M3) {
        weight.scale_data = static_cast<const float*>(storage);
    }
    weight.shape = shape;
    weight.storage = type;
    weight.bytes = shape.elements() * weight_width(type);
    return weight;
}

inline void fill_buffers(
    models::pi05::Pi05ResolvedBuffers* buffers,
    frt_ctx context,
    void* storage,
    const models::pi05::Pi05ResolvedShape& shape,
    modalities::DType activation,
    bool rank4_cache) {
    require(buffers != nullptr);
    for (std::size_t i = 0;
         i < static_cast<std::size_t>(models::pi05::Pi05ValueId::kCount);
         ++i) {
        const auto id = static_cast<models::pi05::Pi05ValueId>(i);
        models::pi05::Pi05TensorSpec logical;
        require(models::pi05::pi05_value_spec(id, shape, &logical)
                    .ok_status());
        modalities::Shape physical = physical_shape(logical);
        if (rank4_cache &&
            (id == models::pi05::Pi05ValueId::kKeyCache ||
             id == models::pi05::Pi05ValueId::kValueCache)) {
            physical = modalities::Shape({
                static_cast<std::uint64_t>(
                    models::pi05::kPi05ModelDims.encoder_layers),
                static_cast<std::uint64_t>(shape.total_attention_keys),
                1,
                static_cast<std::uint64_t>(
                    models::pi05::kPi05ModelDims.encoder_head_dim)});
        }
        models::pi05::Pi05ResolvedBuffer* destination =
            models::pi05::pi05_resolved_buffer(buffers, id);
        require(destination != nullptr);
        *destination = make_buffer(
            context, storage, physical, activation, logical);
    }

    buffers->encoder_rope = make_buffer(
        context, storage,
        modalities::Shape({
            static_cast<std::uint64_t>(shape.encoder_sequence),
            static_cast<std::uint64_t>(
                models::pi05::kPi05ModelDims.encoder_head_dim)}),
        activation);
    buffers->decoder_rope = make_buffer(
        context, storage,
        modalities::Shape({
            static_cast<std::uint64_t>(shape.chunk),
            static_cast<std::uint64_t>(
                models::pi05::kPi05ModelDims.decoder_head_dim)}),
        activation);
    const modalities::Shape raw_int32({sizeof(std::int32_t)});
    buffers->encoder_valid_tokens = make_buffer(
        context, storage, raw_int32, modalities::DType::kUInt8);
    buffers->decoder_valid_tokens = make_buffer(
        context, storage, raw_int32, modalities::DType::kUInt8);
    buffers->decoder_position = make_buffer(
        context, storage, raw_int32, modalities::DType::kUInt8);
    buffers->previous_actions = make_buffer(
        context, storage,
        modalities::Shape({
            static_cast<std::uint64_t>(shape.chunk),
            static_cast<std::uint64_t>(
                models::pi05::kPi05ModelDims.action_width)}),
        activation);
    buffers->prefix_weights = make_buffer(
        context, storage,
        modalities::Shape({static_cast<std::uint64_t>(shape.chunk)}),
        modalities::DType::kFloat32);
    buffers->guidance_weight = make_buffer(
        context, storage, modalities::Shape({1}),
        modalities::DType::kFloat32);
}

inline void fill_feed_forward(
    models::pi05::Pi05FeedForwardWeights* weights,
    void* storage,
    int input_width,
    int hidden_width,
    models::pi05::Pi05WeightStorage type,
    bool fused) {
    require(weights != nullptr);
    const auto dim = [](int value) {
        return static_cast<std::uint64_t>(value);
    };
    if (fused) {
        weights->gate_up_weight = make_weight(
            storage,
            modalities::Shape({dim(input_width), dim(2 * hidden_width)}),
            type);
    } else {
        weights->gate_weight = make_weight(
            storage,
            modalities::Shape({dim(input_width), dim(hidden_width)}), type);
        weights->up_weight = make_weight(
            storage,
            modalities::Shape({dim(input_width), dim(hidden_width)}), type);
    }
    weights->down_weight = make_weight(
        storage,
        modalities::Shape({dim(hidden_width), dim(input_width)}), type);
}

inline void fill_weights(
    models::pi05::Pi05ResolvedWeights* weights,
    void* storage,
    const models::pi05::Pi05ResolvedShape& shape,
    modalities::DType activation,
    bool fp8) {
    require(weights != nullptr);
    const auto dim = [](int value) {
        return static_cast<std::uint64_t>(value);
    };
    const models::pi05::Pi05WeightStorage normal =
        activation_storage(activation);
    const models::pi05::Pi05WeightStorage matrix =
        fp8 ? models::pi05::Pi05WeightStorage::kFp8E4M3 : normal;
    const int vision = models::pi05::kPi05ModelDims.vision_width;
    const int encoder = models::pi05::kPi05ModelDims.encoder_width;
    const int decoder = models::pi05::kPi05ModelDims.decoder_width;

    weights->embedding_table = make_weight(
        storage,
        modalities::Shape({
            dim(models::pi05::kPi05ModelDims.embedding_vocab),
            dim(models::pi05::kPi05ModelDims.embedding_width)}),
        normal);
    models::pi05::Pi05VisionGlobalWeights& vg = weights->vision;
    vg.patch_weight = make_weight(
        storage,
        modalities::Shape({
            dim(models::pi05::kPi05ModelDims.vision_patch),
            dim(models::pi05::kPi05ModelDims.vision_patch),
            dim(models::pi05::kPi05ModelDims.image_channels), dim(vision)}),
        normal);
    vg.patch_bias = make_weight(
        storage, modalities::Shape({dim(vision)}), normal);
    vg.position_embedding = make_weight(
        storage,
        modalities::Shape({
            dim(models::pi05::kPi05ModelDims.vision_tokens_per_view),
            dim(vision)}),
        normal);
    vg.final_norm_weight = make_weight(
        storage, modalities::Shape({dim(vision)}), normal);
    vg.final_norm_bias = make_weight(
        storage, modalities::Shape({dim(vision)}), normal);
    vg.projector_weight = make_weight(
        storage, modalities::Shape({dim(vision), dim(encoder)}), matrix);
    vg.projector_bias = make_weight(
        storage, modalities::Shape({dim(encoder)}), normal);

    models::pi05::Pi05DecoderGlobalWeights& dg = weights->decoder;
    dg.time_embeddings = make_weight(
        storage,
        modalities::Shape({dim(shape.num_steps), dim(decoder)}), normal);
    dg.time_mlp_in_weight = make_weight(
        storage, modalities::Shape({dim(decoder), dim(decoder)}), normal);
    dg.time_mlp_in_bias = make_weight(
        storage, modalities::Shape({dim(decoder)}), normal);
    dg.time_mlp_out_weight = make_weight(
        storage, modalities::Shape({dim(decoder), dim(decoder)}), normal);
    dg.time_mlp_out_bias = make_weight(
        storage, modalities::Shape({dim(decoder)}), normal);
    dg.final_norm_mod_weight = make_weight(
        storage,
        modalities::Shape({dim(decoder), dim(3 * decoder)}), normal);
    dg.final_norm_mod_bias = make_weight(
        storage, modalities::Shape({dim(3 * decoder)}), normal);
    dg.action_in_weight = make_weight(
        storage,
        modalities::Shape({
            dim(models::pi05::kPi05ModelDims.action_width), dim(decoder)}),
        normal);
    dg.action_in_bias = make_weight(
        storage, modalities::Shape({dim(decoder)}), normal);
    dg.action_out_weight = make_weight(
        storage,
        modalities::Shape({
            dim(decoder), dim(models::pi05::kPi05ModelDims.action_width)}),
        normal);
    dg.action_out_bias = make_weight(
        storage,
        modalities::Shape({
            dim(models::pi05::kPi05ModelDims.action_width)}),
        normal);

    for (models::pi05::Pi05VisionLayerWeights& layer :
         weights->vision_layers) {
        layer.pre_attention_norm_weight = make_weight(
            storage, modalities::Shape({dim(vision)}), normal);
        layer.pre_attention_norm_bias = make_weight(
            storage, modalities::Shape({dim(vision)}), normal);
        layer.attention_qkv_weight = make_weight(
            storage,
            modalities::Shape({dim(vision), dim(3 * vision)}), matrix);
        layer.attention_qkv_bias = make_weight(
            storage, modalities::Shape({dim(3 * vision)}), normal);
        layer.attention_output_weight = make_weight(
            storage, modalities::Shape({dim(vision), dim(vision)}), matrix);
        layer.attention_output_bias = make_weight(
            storage, modalities::Shape({dim(vision)}), normal);
        layer.pre_mlp_norm_weight = make_weight(
            storage, modalities::Shape({dim(vision)}), normal);
        layer.pre_mlp_norm_bias = make_weight(
            storage, modalities::Shape({dim(vision)}), normal);
        layer.mlp_up_weight = make_weight(
            storage,
            modalities::Shape({
                dim(vision),
                dim(models::pi05::kPi05ModelDims.vision_hidden)}),
            matrix);
        layer.mlp_up_bias = make_weight(
            storage,
            modalities::Shape({
                dim(models::pi05::kPi05ModelDims.vision_hidden)}),
            normal);
        layer.mlp_down_weight = make_weight(
            storage,
            modalities::Shape({
                dim(models::pi05::kPi05ModelDims.vision_hidden),
                dim(vision)}),
            matrix);
        layer.mlp_down_bias = make_weight(
            storage, modalities::Shape({dim(vision)}), normal);
    }

    for (models::pi05::Pi05EncoderLayerWeights& layer :
         weights->encoder_layers) {
        layer.attention_qkv_weight = make_weight(
            storage,
            modalities::Shape({
                dim(encoder),
                dim(encoder +
                    2 * models::pi05::kPi05ModelDims.encoder_head_dim)}),
            matrix);
        layer.attention_output_weight = make_weight(
            storage, modalities::Shape({dim(encoder), dim(encoder)}),
            matrix);
        fill_feed_forward(
            &layer.mlp, storage, encoder,
            models::pi05::kPi05ModelDims.encoder_hidden, matrix, fp8);
    }

    for (models::pi05::Pi05DecoderLayerWeights& layer :
         weights->decoder_layers) {
        layer.attention_qkv_weight = make_weight(
            storage,
            modalities::Shape({
                dim(decoder),
                dim(encoder +
                    2 * models::pi05::kPi05ModelDims.decoder_head_dim)}),
            matrix);
        layer.attention_output_weight = make_weight(
            storage, modalities::Shape({dim(encoder), dim(decoder)}),
            matrix);
        fill_feed_forward(
            &layer.mlp, storage, decoder,
            models::pi05::kPi05ModelDims.decoder_hidden, matrix, fp8);
        layer.attention_mod_weight = make_weight(
            storage,
            modalities::Shape({dim(decoder), dim(3 * decoder)}), normal);
        layer.attention_mod_bias = make_weight(
            storage, modalities::Shape({dim(3 * decoder)}), normal);
        layer.mlp_mod_weight = make_weight(
            storage,
            modalities::Shape({dim(decoder), dim(3 * decoder)}), normal);
        layer.mlp_mod_bias = make_weight(
            storage, modalities::Shape({dim(3 * decoder)}), normal);
    }
}

inline models::pi05::Pi05ResolvedResources make_resources(
    frt_ctx context,
    frt_buffer anchor,
    modalities::DType activation,
    bool fp8,
    bool rank4_cache) {
    require(context != nullptr && anchor != nullptr);
    void* storage = frt_buffer_dptr(anchor);
    require(storage != nullptr);
    const models::pi05::Pi05ResolvedShape shape = canonical_shape();
    models::pi05::Pi05ResolvedResources resources;
    fill_buffers(&resources.buffers, context, storage, shape, activation,
                 rank4_cache);
    fill_weights(&resources.weights, storage, shape, activation, fp8);
    return resources;
}

class ResourceOwner final {
public:
    ResourceOwner() : context_(frt_ctx_create()) {
        require(context_ != nullptr);
        anchor_ = frt_buffer_alloc(context_, "pi05_fixture_anchor", 1);
        require(anchor_ != nullptr);
    }

    ~ResourceOwner() { frt_ctx_destroy(context_); }

    ResourceOwner(const ResourceOwner&) = delete;
    ResourceOwner& operator=(const ResourceOwner&) = delete;

    frt_ctx context() const { return context_; }
    frt_buffer anchor() const { return anchor_; }

private:
    frt_ctx context_ = nullptr;
    frt_buffer anchor_ = nullptr;
};

}  // namespace pi05_fixture
}  // namespace tests
}  // namespace flashrt

#endif  // FLASHRT_CPP_TESTS_PI05_RESOLVED_FIXTURE_H

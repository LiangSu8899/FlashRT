#include "flashrt/cpp/models/pi05/model/execution_plan.h"
#include "flashrt/cpp/models/pi05/model/resolved_resources.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace modalities = flashrt::modalities;
namespace pi05 = flashrt::models::pi05;

namespace {

[[noreturn]] void fail(const char* expression, int line) {
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    std::abort();
}

#define CHECK(expression)                              \
    do {                                               \
        if (!(expression)) fail(#expression, __LINE__); \
    } while (false)

std::array<unsigned char, 32> g_buffer_tokens{};
unsigned char g_weight_token = 0;

pi05::Pi05ResolvedShape canonical_shape() {
    pi05::Pi05ShapeConfig config;
    config.num_views = 3;
    config.max_prompt_tokens = 64;
    config.chunk = 10;
    config.num_steps = 10;
    config.vision_pool_factor = 1;
    config.state_dim = 8;
    config.robot_action_dim = 7;
    pi05::Pi05ResolvedShape shape;
    CHECK(pi05::resolve_pi05_shape(config, &shape).ok_status());
    return shape;
}

modalities::Shape physical_shape(const pi05::Pi05TensorSpec& spec) {
    modalities::Shape shape;
    shape.rank = spec.rank;
    for (std::size_t i = 0; i < spec.rank; ++i) {
        shape.dims[i] = spec.dimensions[i];
    }
    return shape;
}

std::uint64_t buffer_bytes(const modalities::Shape& shape,
                           modalities::DType dtype) {
    const std::uint64_t elements = shape.elements();
    const std::size_t width = modalities::dtype_size(dtype);
    CHECK(elements != 0);
    CHECK(width != 0);
    return elements * width;
}

pi05::Pi05ResolvedBuffer make_buffer(
    const modalities::Shape& shape,
    modalities::DType dtype,
    std::size_t token,
    pi05::Pi05TensorSpec logical_spec = {}) {
    CHECK(token < g_buffer_tokens.size());
    pi05::Pi05ResolvedBuffer buffer;
    buffer.buffer = reinterpret_cast<frt_buffer>(&g_buffer_tokens[token]);
    buffer.physical_dtype = dtype;
    buffer.physical_shape = shape;
    buffer.physical_bytes = buffer_bytes(shape, dtype);
    buffer.logical_spec = logical_spec;
    buffer.storage_identity = &g_buffer_tokens[token];
    buffer.storage_offset = 8;
    buffer.storage_bytes = buffer.physical_bytes + buffer.storage_offset;
    return buffer;
}

pi05::Pi05WeightStorage activation_storage(modalities::DType dtype) {
    return dtype == modalities::DType::kFloat16
               ? pi05::Pi05WeightStorage::kFloat16
               : pi05::Pi05WeightStorage::kBFloat16;
}

std::size_t weight_width(pi05::Pi05WeightStorage storage) {
    switch (storage) {
        case pi05::Pi05WeightStorage::kBFloat16:
        case pi05::Pi05WeightStorage::kFloat16: return 2;
        case pi05::Pi05WeightStorage::kFp8E4M3: return 1;
    }
    return 0;
}

pi05::Pi05ResolvedWeight make_weight(
    const modalities::Shape& shape,
    pi05::Pi05WeightStorage storage) {
    pi05::Pi05ResolvedWeight weight;
    weight.device_data = &g_weight_token;
    weight.shape = shape;
    weight.storage = storage;
    weight.bytes = shape.elements() * weight_width(storage);
    return weight;
}

void fill_buffers(pi05::Pi05ResolvedBuffers* buffers,
                  const pi05::Pi05ResolvedShape& shape,
                  modalities::DType activation,
                  bool rank4_cache) {
    CHECK(buffers != nullptr);
    std::size_t token = 0;
    for (std::size_t i = 0;
         i < static_cast<std::size_t>(pi05::Pi05ValueId::kCount); ++i) {
        const auto id = static_cast<pi05::Pi05ValueId>(i);
        pi05::Pi05TensorSpec logical;
        CHECK(pi05::pi05_value_spec(id, shape, &logical).ok_status());
        modalities::Shape physical = physical_shape(logical);
        if (rank4_cache &&
            (id == pi05::Pi05ValueId::kKeyCache ||
             id == pi05::Pi05ValueId::kValueCache)) {
            physical = modalities::Shape({
                static_cast<std::uint64_t>(pi05::kPi05ModelDims.encoder_layers),
                static_cast<std::uint64_t>(shape.total_attention_keys),
                1,
                static_cast<std::uint64_t>(
                    pi05::kPi05ModelDims.encoder_head_dim)});
        }
        pi05::Pi05ResolvedBuffer* destination =
            pi05::pi05_resolved_buffer(buffers, id);
        CHECK(destination != nullptr);
        *destination = make_buffer(physical, activation, token++, logical);
    }

    buffers->encoder_rope = make_buffer(
        modalities::Shape({
            static_cast<std::uint64_t>(shape.encoder_sequence),
            static_cast<std::uint64_t>(
                pi05::kPi05ModelDims.encoder_head_dim)}),
        activation, token++);
    buffers->decoder_rope = make_buffer(
        modalities::Shape({
            static_cast<std::uint64_t>(shape.chunk),
            static_cast<std::uint64_t>(
                pi05::kPi05ModelDims.decoder_head_dim)}),
        activation, token++);
    const modalities::Shape raw_int32({sizeof(std::int32_t)});
    buffers->encoder_valid_tokens = make_buffer(
        raw_int32, modalities::DType::kUInt8, token++);
    buffers->decoder_valid_tokens = make_buffer(
        raw_int32, modalities::DType::kUInt8, token++);
    buffers->decoder_position = make_buffer(
        raw_int32, modalities::DType::kUInt8, token++);
    buffers->previous_actions = make_buffer(
        modalities::Shape({
            static_cast<std::uint64_t>(shape.chunk),
            static_cast<std::uint64_t>(pi05::kPi05ModelDims.action_width)}),
        activation, token++);
    buffers->prefix_weights = make_buffer(
        modalities::Shape({static_cast<std::uint64_t>(shape.chunk)}),
        modalities::DType::kFloat32, token++);
    buffers->guidance_weight = make_buffer(
        modalities::Shape({1}), modalities::DType::kFloat32, token++);
}

void fill_feed_forward(pi05::Pi05FeedForwardWeights* weights,
                       int input_width,
                       int hidden_width,
                       pi05::Pi05WeightStorage storage,
                       bool fused) {
    CHECK(weights != nullptr);
    const auto dim = [](int value) {
        return static_cast<std::uint64_t>(value);
    };
    if (fused) {
        weights->gate_up_weight = make_weight(
            modalities::Shape({dim(input_width), dim(2 * hidden_width)}),
            storage);
    } else {
        weights->gate_weight = make_weight(
            modalities::Shape({dim(input_width), dim(hidden_width)}), storage);
        weights->up_weight = make_weight(
            modalities::Shape({dim(input_width), dim(hidden_width)}), storage);
    }
    weights->down_weight = make_weight(
        modalities::Shape({dim(hidden_width), dim(input_width)}), storage);
}

void fill_weights(pi05::Pi05ResolvedWeights* weights,
                  const pi05::Pi05ResolvedShape& shape,
                  modalities::DType activation,
                  bool fp8) {
    CHECK(weights != nullptr);
    const auto dim = [](int value) {
        return static_cast<std::uint64_t>(value);
    };
    const pi05::Pi05WeightStorage normal = activation_storage(activation);
    const pi05::Pi05WeightStorage matrix =
        fp8 ? pi05::Pi05WeightStorage::kFp8E4M3 : normal;
    const int vision = pi05::kPi05ModelDims.vision_width;
    const int encoder = pi05::kPi05ModelDims.encoder_width;
    const int decoder = pi05::kPi05ModelDims.decoder_width;

    weights->embedding_table = make_weight(
        modalities::Shape({dim(pi05::kPi05ModelDims.embedding_vocab),
                           dim(pi05::kPi05ModelDims.embedding_width)}),
        normal);
    pi05::Pi05VisionGlobalWeights& vg = weights->vision;
    vg.patch_weight = make_weight(
        modalities::Shape({dim(pi05::kPi05ModelDims.vision_patch),
                           dim(pi05::kPi05ModelDims.vision_patch),
                           dim(pi05::kPi05ModelDims.image_channels),
                           dim(vision)}),
        normal);
    vg.patch_bias = make_weight(modalities::Shape({dim(vision)}), normal);
    vg.position_embedding = make_weight(
        modalities::Shape({dim(pi05::kPi05ModelDims.vision_tokens_per_view),
                           dim(vision)}),
        normal);
    vg.final_norm_weight = make_weight(
        modalities::Shape({dim(vision)}), normal);
    vg.final_norm_bias = make_weight(
        modalities::Shape({dim(vision)}), normal);
    vg.projector_weight = make_weight(
        modalities::Shape({dim(vision), dim(encoder)}), matrix);
    vg.projector_bias = make_weight(
        modalities::Shape({dim(encoder)}), normal);

    pi05::Pi05DecoderGlobalWeights& dg = weights->decoder;
    dg.time_embeddings = make_weight(
        modalities::Shape({dim(shape.num_steps), dim(decoder)}), normal);
    dg.time_mlp_in_weight = make_weight(
        modalities::Shape({dim(decoder), dim(decoder)}), normal);
    dg.time_mlp_in_bias = make_weight(
        modalities::Shape({dim(decoder)}), normal);
    dg.time_mlp_out_weight = make_weight(
        modalities::Shape({dim(decoder), dim(decoder)}), normal);
    dg.time_mlp_out_bias = make_weight(
        modalities::Shape({dim(decoder)}), normal);
    dg.final_norm_mod_weight = make_weight(
        modalities::Shape({dim(decoder), dim(3 * decoder)}), normal);
    dg.final_norm_mod_bias = make_weight(
        modalities::Shape({dim(3 * decoder)}), normal);
    dg.action_in_weight = make_weight(
        modalities::Shape({dim(pi05::kPi05ModelDims.action_width),
                           dim(decoder)}),
        normal);
    dg.action_in_bias = make_weight(
        modalities::Shape({dim(decoder)}), normal);
    dg.action_out_weight = make_weight(
        modalities::Shape({dim(decoder),
                           dim(pi05::kPi05ModelDims.action_width)}),
        normal);
    dg.action_out_bias = make_weight(
        modalities::Shape({dim(pi05::kPi05ModelDims.action_width)}), normal);

    for (pi05::Pi05VisionLayerWeights& layer : weights->vision_layers) {
        layer.pre_attention_norm_weight = make_weight(
            modalities::Shape({dim(vision)}), normal);
        layer.pre_attention_norm_bias = make_weight(
            modalities::Shape({dim(vision)}), normal);
        layer.attention_qkv_weight = make_weight(
            modalities::Shape({dim(vision), dim(3 * vision)}), matrix);
        layer.attention_qkv_bias = make_weight(
            modalities::Shape({dim(3 * vision)}), normal);
        layer.attention_output_weight = make_weight(
            modalities::Shape({dim(vision), dim(vision)}), matrix);
        layer.attention_output_bias = make_weight(
            modalities::Shape({dim(vision)}), normal);
        layer.pre_mlp_norm_weight = make_weight(
            modalities::Shape({dim(vision)}), normal);
        layer.pre_mlp_norm_bias = make_weight(
            modalities::Shape({dim(vision)}), normal);
        layer.mlp_up_weight = make_weight(
            modalities::Shape({dim(vision),
                               dim(pi05::kPi05ModelDims.vision_hidden)}),
            matrix);
        layer.mlp_up_bias = make_weight(
            modalities::Shape({dim(pi05::kPi05ModelDims.vision_hidden)}),
            normal);
        layer.mlp_down_weight = make_weight(
            modalities::Shape({dim(pi05::kPi05ModelDims.vision_hidden),
                               dim(vision)}),
            matrix);
        layer.mlp_down_bias = make_weight(
            modalities::Shape({dim(vision)}), normal);
    }

    for (pi05::Pi05EncoderLayerWeights& layer : weights->encoder_layers) {
        layer.attention_qkv_weight = make_weight(
            modalities::Shape({
                dim(encoder),
                dim(encoder + 2 * pi05::kPi05ModelDims.encoder_head_dim)}),
            matrix);
        layer.attention_output_weight = make_weight(
            modalities::Shape({dim(encoder), dim(encoder)}), matrix);
        fill_feed_forward(&layer.mlp, encoder,
                          pi05::kPi05ModelDims.encoder_hidden, matrix, fp8);
    }

    for (pi05::Pi05DecoderLayerWeights& layer : weights->decoder_layers) {
        layer.attention_qkv_weight = make_weight(
            modalities::Shape({
                dim(decoder),
                dim(encoder + 2 * pi05::kPi05ModelDims.decoder_head_dim)}),
            matrix);
        layer.attention_output_weight = make_weight(
            modalities::Shape({dim(encoder), dim(decoder)}), matrix);
        fill_feed_forward(&layer.mlp, decoder,
                          pi05::kPi05ModelDims.decoder_hidden, matrix, fp8);
        layer.attention_mod_weight = make_weight(
            modalities::Shape({dim(decoder), dim(3 * decoder)}), normal);
        layer.attention_mod_bias = make_weight(
            modalities::Shape({dim(3 * decoder)}), normal);
        layer.mlp_mod_weight = make_weight(
            modalities::Shape({dim(decoder), dim(3 * decoder)}), normal);
        layer.mlp_mod_bias = make_weight(
            modalities::Shape({dim(3 * decoder)}), normal);
    }
}

pi05::Pi05ResolvedResources make_resources(modalities::DType activation,
                                           bool fp8,
                                           bool rank4_cache) {
    const pi05::Pi05ResolvedShape shape = canonical_shape();
    pi05::Pi05ResolvedResources resources;
    fill_buffers(&resources.buffers, shape, activation, rank4_cache);
    fill_weights(&resources.weights, shape, activation, fp8);
    return resources;
}

void test_valid_resource_variants() {
    const pi05::Pi05ResolvedShape shape = canonical_shape();
    pi05::Pi05ResolvedResources bf16 =
        make_resources(modalities::DType::kBFloat16, false, false);
    CHECK(pi05::validate_pi05_resolved_resources(bf16, shape).ok_status());

    pi05::Pi05ResolvedResources sm120_fp8 =
        make_resources(modalities::DType::kBFloat16, true, true);
    CHECK(pi05::validate_pi05_resolved_resources(sm120_fp8, shape)
              .ok_status());
    CHECK(sm120_fp8.buffers.key_cache.logical_spec.rank == 3);
    CHECK(sm120_fp8.buffers.key_cache.physical_shape.rank == 4);

    pi05::Pi05ResolvedResources sm110_fp8 =
        make_resources(modalities::DType::kFloat16, true, false);
    sm110_fp8.buffers.action_delta.physical_dtype =
        modalities::DType::kFloat32;
    sm110_fp8.buffers.action_delta.physical_bytes = buffer_bytes(
        sm110_fp8.buffers.action_delta.physical_shape,
        modalities::DType::kFloat32);
    sm110_fp8.buffers.action_delta.storage_bytes =
        sm110_fp8.buffers.action_delta.storage_offset +
        sm110_fp8.buffers.action_delta.physical_bytes;
    CHECK(pi05::validate_pi05_resolved_resources(sm110_fp8, shape)
              .ok_status());
}

void test_buffer_rejections() {
    const pi05::Pi05ResolvedShape shape = canonical_shape();

    pi05::Pi05ResolvedResources bad =
        make_resources(modalities::DType::kBFloat16, false, false);
    ++bad.buffers.vision_state.logical_spec.dimensions[0];
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(modalities::DType::kBFloat16, false, false);
    ++bad.buffers.key_cache.physical_shape.dims[1];
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(modalities::DType::kBFloat16, false, false);
    bad.buffers.decoder_state.physical_shape.dims[0] = 0;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(modalities::DType::kBFloat16, false, false);
    bad.buffers.decoder_state.physical_shape.rank = 2;
    bad.buffers.decoder_state.physical_shape.dims[0] =
        UINT64_MAX;
    bad.buffers.decoder_state.physical_shape.dims[1] = 2;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(modalities::DType::kBFloat16, false, false);
    bad.buffers.decoder_state.physical_dtype = modalities::DType::kFloat32;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(modalities::DType::kBFloat16, false, false);
    bad.buffers.noise.storage_offset = bad.buffers.noise.storage_bytes;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(modalities::DType::kBFloat16, false, false);
    bad.buffers.noise.storage_identity = nullptr;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(modalities::DType::kBFloat16, false, false);
    bad.buffers.encoder_valid_tokens.physical_dtype =
        modalities::DType::kFloat32;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(modalities::DType::kBFloat16, false, false);
    bad.buffers.images.buffer = nullptr;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());
}

void test_weight_rejections() {
    const pi05::Pi05ResolvedShape shape = canonical_shape();

    pi05::Pi05ResolvedResources bad =
        make_resources(modalities::DType::kBFloat16, false, false);
    bad.weights.embedding_table.device_data = nullptr;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(modalities::DType::kBFloat16, false, false);
    ++bad.weights.vision.patch_weight.shape.dims[0];
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(modalities::DType::kBFloat16, false, false);
    --bad.weights.decoder.action_out_weight.bytes;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(modalities::DType::kBFloat16, false, false);
    bad.weights.embedding_table.storage =
        pi05::Pi05WeightStorage::kFp8E4M3;
    bad.weights.embedding_table.bytes =
        bad.weights.embedding_table.shape.elements();
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(modalities::DType::kBFloat16, false, false);
    bad.weights.encoder_layers[0].mlp.gate_up_weight = make_weight(
        modalities::Shape({
            static_cast<std::uint64_t>(pi05::kPi05ModelDims.encoder_width),
            static_cast<std::uint64_t>(
                2 * pi05::kPi05ModelDims.encoder_hidden)}),
        pi05::Pi05WeightStorage::kBFloat16);
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(modalities::DType::kBFloat16, true, false);
    bad.weights.decoder_layers[0].mlp.gate_up_weight = {};
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());
}

void test_graph_bindings_are_typed_and_atomic() {
    pi05::Pi05ResolvedResources resources =
        make_resources(modalities::DType::kBFloat16, false, false);
    pi05::Pi05ResolvedGraphBindings bindings;
    CHECK(pi05::make_pi05_graph_bindings(resources.buffers, &bindings)
              .ok_status());
    CHECK(bindings.get(pi05::Pi05GraphBindingId::kImages) ==
          resources.buffers.images.buffer);
    CHECK(bindings.get(pi05::Pi05GraphBindingId::kPromptEmbedding) ==
          resources.buffers.prompt_embedding.buffer);
    CHECK(bindings.get(pi05::Pi05GraphBindingId::kEncoderState) ==
          resources.buffers.encoder_state.buffer);
    CHECK(bindings.get(pi05::Pi05GraphBindingId::kNoise) ==
          resources.buffers.noise.buffer);
    CHECK(bindings.get(pi05::Pi05GraphBindingId::kPreviousActions) ==
          resources.buffers.previous_actions.buffer);
    CHECK(bindings.get(pi05::Pi05GraphBindingId::kPrefixWeights) ==
          resources.buffers.prefix_weights.buffer);
    CHECK(bindings.get(pi05::Pi05GraphBindingId::kGuidanceWeight) ==
          resources.buffers.guidance_weight.buffer);

    pi05::Pi05ResolvedGraphBindings unchanged;
    const frt_buffer sentinel = reinterpret_cast<frt_buffer>(
        &g_buffer_tokens.back());
    CHECK(unchanged.bind(pi05::Pi05GraphBindingId::kNoise, sentinel)
              .ok_status());
    resources.buffers.guidance_weight.buffer = nullptr;
    CHECK(!pi05::make_pi05_graph_bindings(resources.buffers, &unchanged)
               .ok_status());
    CHECK(unchanged.get(pi05::Pi05GraphBindingId::kNoise) == sentinel);
    CHECK(unchanged.get(pi05::Pi05GraphBindingId::kImages) == nullptr);
    CHECK(!pi05::make_pi05_graph_bindings(resources.buffers, nullptr)
               .ok_status());
}

}  // namespace

int main() {
    test_valid_resource_variants();
    test_buffer_rejections();
    test_weight_rejections();
    test_graph_bindings_are_typed_and_atomic();
    std::cout << "PASS - PI0.5 resolved resource contract\n";
    return 0;
}

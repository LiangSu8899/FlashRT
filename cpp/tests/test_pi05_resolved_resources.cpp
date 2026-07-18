#include "flashrt/cpp/models/pi05/model/execution_plan.h"
#include "flashrt/cpp/models/pi05/model/resolved_resources.h"

#include "pi05_resolved_fixture.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace modalities = flashrt::modalities;
namespace pi05 = flashrt::models::pi05;
namespace fixture = flashrt::tests::pi05_fixture;

namespace {

[[noreturn]] void fail(const char* expression, int line) {
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    std::abort();
}

#define CHECK(expression)                              \
    do {                                               \
        if (!(expression)) fail(#expression, __LINE__); \
    } while (false)

pi05::Pi05ResolvedResources make_resources(
    fixture::ResourceOwner& owner,
    modalities::DType activation = modalities::DType::kBFloat16,
    bool fp8 = false,
    bool rank4_cache = false) {
    return fixture::make_resources(
        owner.context(), owner.anchor(), activation, fp8, rank4_cache);
}

void test_valid_resource_variants() {
    fixture::ResourceOwner owner;
    const pi05::Pi05ResolvedShape shape = fixture::canonical_shape();
    pi05::Pi05ResolvedResources bf16 = make_resources(owner);
    CHECK(pi05::validate_pi05_resolved_resources(bf16, shape).ok_status());

    pi05::Pi05ResolvedResources fp8_rank4 = make_resources(
        owner, modalities::DType::kBFloat16, true, true);
    CHECK(pi05::validate_pi05_resolved_resources(fp8_rank4, shape)
              .ok_status());
    CHECK(fp8_rank4.buffers.key_cache.logical_spec.rank == 3);
    CHECK(fp8_rank4.buffers.key_cache.physical_shape.rank == 4);

    pi05::Pi05ResolvedResources f16_fp8 = make_resources(
        owner, modalities::DType::kFloat16, true, false);
    f16_fp8.buffers.action_delta.physical_dtype =
        modalities::DType::kFloat32;
    f16_fp8.buffers.action_delta.physical_bytes = fixture::buffer_bytes(
        f16_fp8.buffers.action_delta.physical_shape,
        modalities::DType::kFloat32);
    f16_fp8.buffers.action_delta.storage_bytes =
        f16_fp8.buffers.action_delta.physical_bytes;
    f16_fp8.buffers.action_delta.buffer = frt_buffer_wrap(
        owner.context(), "pi05_f32_action", frt_buffer_dptr(owner.anchor()),
        static_cast<std::size_t>(
            f16_fp8.buffers.action_delta.physical_bytes));
    CHECK(f16_fp8.buffers.action_delta.buffer != nullptr);
    CHECK(pi05::validate_pi05_resolved_resources(f16_fp8, shape)
              .ok_status());
}

void test_buffer_rejections() {
    fixture::ResourceOwner owner;
    const pi05::Pi05ResolvedShape shape = fixture::canonical_shape();

    pi05::Pi05ResolvedResources bad = make_resources(owner);
    ++bad.buffers.vision_state.logical_spec.dimensions[0];
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner);
    ++bad.buffers.key_cache.physical_shape.dims[1];
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner);
    bad.buffers.decoder_state.physical_shape.dims[0] = 0;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner);
    bad.buffers.decoder_state.physical_shape.rank = 2;
    bad.buffers.decoder_state.physical_shape.dims[0] = UINT64_MAX;
    bad.buffers.decoder_state.physical_shape.dims[1] = 2;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner);
    bad.buffers.decoder_state.physical_dtype = modalities::DType::kFloat32;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner);
    bad.buffers.noise.storage_offset = bad.buffers.noise.storage_bytes;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner);
    bad.buffers.noise.storage_identity = nullptr;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner);
    bad.buffers.noise.storage_offset = 1;
    ++bad.buffers.noise.storage_bytes;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner);
    bad.buffers.noise.physical_bytes =
        frt_buffer_bytes(bad.buffers.noise.buffer) + 1;
    bad.buffers.noise.storage_bytes = bad.buffers.noise.physical_bytes;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner);
    bad.buffers.encoder_valid_tokens.physical_dtype =
        modalities::DType::kFloat32;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner);
    bad.buffers.images.buffer = nullptr;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());
}

void test_weight_rejections() {
    fixture::ResourceOwner owner;
    const pi05::Pi05ResolvedShape shape = fixture::canonical_shape();

    pi05::Pi05ResolvedResources bad = make_resources(owner);
    bad.weights.embedding_table.device_data = nullptr;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner);
    ++bad.weights.vision.patch_weight.shape.dims[0];
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner);
    --bad.weights.decoder.action_out_weight.bytes;
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner);
    bad.weights.embedding_table.storage = pi05::Pi05WeightStorage::kFp8E4M3;
    bad.weights.embedding_table.bytes =
        bad.weights.embedding_table.shape.elements();
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner);
    bad.weights.encoder_layers[0].mlp.gate_up_weight = fixture::make_weight(
        frt_buffer_dptr(owner.anchor()),
        modalities::Shape({
            static_cast<std::uint64_t>(pi05::kPi05ModelDims.encoder_width),
            static_cast<std::uint64_t>(
                2 * pi05::kPi05ModelDims.encoder_hidden)}),
        pi05::Pi05WeightStorage::kBFloat16);
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());

    bad = make_resources(owner, modalities::DType::kBFloat16, true);
    bad.weights.decoder_layers[0].mlp.gate_up_weight = {};
    CHECK(!pi05::validate_pi05_resolved_resources(bad, shape).ok_status());
}

void test_graph_bindings_are_typed_and_atomic() {
    fixture::ResourceOwner owner;
    pi05::Pi05ResolvedResources resources = make_resources(owner);
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
    CHECK(unchanged.bind(pi05::Pi05GraphBindingId::kNoise, owner.anchor())
              .ok_status());
    resources.buffers.guidance_weight.buffer = nullptr;
    CHECK(!pi05::make_pi05_graph_bindings(resources.buffers, &unchanged)
               .ok_status());
    CHECK(unchanged.get(pi05::Pi05GraphBindingId::kNoise) == owner.anchor());
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

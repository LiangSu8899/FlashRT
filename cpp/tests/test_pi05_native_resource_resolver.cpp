#include "flashrt/cpp/models/pi05/support/native_resource_resolver.h"

#include <cassert>
#include <cstdint>

namespace {

flashrt::models::pi05::Pi05ResolvedShape canonical_shape() {
    flashrt::models::pi05::Pi05ShapeConfig config;
    config.num_views = 3;
    config.max_prompt_tokens = 64;
    config.chunk = 10;
    config.num_steps = 10;
    config.vision_pool_factor = 1;
    config.state_dim = 8;
    config.robot_action_dim = 7;
    flashrt::models::pi05::Pi05ResolvedShape shape;
    assert(flashrt::models::pi05::resolve_pi05_shape(config, &shape)
               .ok_status());
    return shape;
}

}  // namespace

int main() {
    using namespace flashrt::models::pi05;
    const Pi05ResolvedShape shape = canonical_shape();

    NativeDeviceWeightStore empty_weights(nullptr);
    Pi05ResolvedWeights weights;
    std::uint8_t sentinel = 0;
    weights.embedding_table.device_data = &sentinel;
    assert(!resolve_pi05_materialized_weights(
                empty_weights, shape,
                flashrt::modalities::DType::kBFloat16,
                Pi05NativeWeightLayout{}, &weights)
                .ok_status());
    assert(weights.embedding_table.device_data == &sentinel);
    assert(!resolve_pi05_materialized_weights(
                empty_weights, shape,
                flashrt::modalities::DType::kBFloat16,
                Pi05NativeWeightLayout{}, nullptr)
                .ok_status());

    NativeWorkspace empty_workspace(nullptr);
    Pi05TargetBufferBindings target;
    Pi05ResolvedBuffers buffers;
    buffers.images.storage_identity = &sentinel;
    assert(!resolve_pi05_native_buffers(
                empty_workspace, target, shape, &buffers)
                .ok_status());
    assert(buffers.images.storage_identity == &sentinel);
    assert(!resolve_pi05_native_buffers(
                empty_workspace, target, shape, nullptr)
                .ok_status());
    return 0;
}

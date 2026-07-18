#include "flashrt/cpp/models/pi05/model/semantic_pipeline.h"
#include "flashrt/cpp/models/pi05/targets/sm120/target.h"
#include "flashrt/exec.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {

flashrt::models::pi05::Pi05ResolvedShape canonical_shape() {
    flashrt::models::pi05::Pi05ShapeConfig config;
    config.num_views = 3;
    config.max_prompt_tokens = 64;
    config.chunk = 10;
    config.num_steps = 10;
    config.vision_pool_factor = 2;
    config.state_dim = 8;
    config.robot_action_dim = 7;
    flashrt::models::pi05::Pi05ResolvedShape shape;
    assert(flashrt::models::pi05::resolve_pi05_shape(config, &shape)
               .ok_status());
    return shape;
}

std::int32_t read_control(
    const flashrt::models::pi05::Pi05ResolvedBuffer& buffer) {
    std::int32_t value = 0;
    assert(buffer.buffer);
    assert(frt_buffer_bytes(buffer.buffer) == sizeof(value));
    assert(cudaMemcpy(&value, frt_buffer_dptr(buffer.buffer), sizeof(value),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    return value;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || !device_count) {
        cudaGetLastError();
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    int device = 0;
    cudaDeviceProp properties{};
    assert(cudaGetDevice(&device) == cudaSuccess);
    assert(cudaGetDeviceProperties(&properties, device) == cudaSuccess);
    if (properties.major != 12 || properties.minor != 0) {
        std::printf("SKIP - SM120 target needs compute capability 12.0\n");
        return 0;
    }

    using namespace flashrt::models::pi05;
    using flashrt::models::pi05::targets::sm120::Sm120TargetBundle;

    const Pi05ResolvedShape shape = canonical_shape();
    flashrt::modalities::Status status;
    assert(!Sm120TargetBundle::create(nullptr, shape, "missing", &status));
    assert(!status.ok_status());

    frt_ctx context = frt_ctx_create();
    assert(context);
    Pi05ResolvedShape invalid_shape = shape;
    ++invalid_shape.encoder_sequence;
    assert(!Sm120TargetBundle::create(context, invalid_shape, "missing",
                                      &status));
    assert(!status.ok_status());
    assert(!Sm120TargetBundle::create(context, shape, "", &status));
    assert(!status.ok_status());

    std::unique_ptr<Sm120TargetBundle> missing =
        Sm120TargetBundle::create(context, shape, "missing", &status);
    assert(missing);
    Pi05ResolvedResources resources;
    assert(!missing->resolve_resources(&resources).ok_status());
    assert(!missing->finalize_setup().ok_status());
    assert(!missing->initialize_resources().ok_status());
    assert(!missing->initialize_resources().ok_status());
    assert(frt_ctx_stream(context, 0) >= 0);
    missing.reset();
    frt_ctx_destroy(context);

    const char* checkpoint = std::getenv("FLASHRT_PI05_CHECKPOINT");
    if (!checkpoint || !checkpoint[0]) {
        std::printf("PASS - PI0.5 SM120 target failure contract\n");
        return 0;
    }

    context = frt_ctx_create();
    assert(context);
    std::unique_ptr<Sm120TargetBundle> target =
        Sm120TargetBundle::create(context, shape, checkpoint, &status);
    assert(target && status.ok_status());
    assert(target->initialize_resources().ok_status());
    assert(!target->initialize_resources().ok_status());
    assert(target->materialized_weight_count() > 0);
    assert(target->resolve_resources(&resources).ok_status());
    assert(!target->resolve_resources(&resources).ok_status());
    assert(validate_pi05_resolved_resources(resources, shape).ok_status());
    assert(!target->finalize_setup().ok_status());

    Pi05SemanticPipeline pipeline(shape);
    assert(pipeline.record_prepare(*target).ok_status());
    assert(target->prepare_call_count() ==
           static_cast<std::size_t>(shape.num_steps) *
               (2 + 2 * kPi05ModelDims.decoder_layers));
    assert(target->finalize_setup().ok_status());
    assert(!target->finalize_setup().ok_status());

    assert(target->set_prompt_length(0).ok_status());
    assert(target->set_prompt_length(shape.max_prompt_tokens).ok_status());
    const std::int32_t committed =
        read_control(resources.buffers.encoder_valid_tokens);
    assert(committed == shape.encoder_vision_sequence +
                            shape.max_prompt_tokens);
    assert(!target->set_prompt_length(shape.max_prompt_tokens + 1)
                .ok_status());
    assert(read_control(resources.buffers.encoder_valid_tokens) == committed);

    assert(target->initialize_capture_inputs().ok_status());
    assert(!target->initialize_capture_inputs().ok_status());
    assert(target->ready_for_capture());
    assert(target->reset_after_warmup().ok_status());
    assert(pipeline.record_context(*target).ok_status());
    assert(pipeline.record_decode(*target).ok_status());
    assert(cudaDeviceSynchronize() == cudaSuccess);

    target.reset();
    frt_ctx_destroy(context);
    std::printf("PASS - PI0.5 SM120 target setup contract\n");
    return 0;
}

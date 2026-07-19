#include "flashrt/cpp/models/pi05/model/semantic_pipeline.h"
#include "flashrt/cpp/models/pi05/targets/sm120/fp8_linear.h"
#include "flashrt/cpp/models/pi05/targets/sm120/target.h"

#include "flashrt/cpp/models/pi05/model/frontend_ops.h"
#include "flashrt/exec.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

flashrt::models::pi05::targets::sm120::Sm120TargetConfig target_config(
    std::string checkpoint,
    const char* calibration_path = nullptr) {
    using namespace flashrt::models::pi05;
    using namespace flashrt::models::pi05::targets::sm120;
    Sm120TargetConfig config;
    config.checkpoint_path = std::move(checkpoint);
    if (calibration_path && calibration_path[0]) {
        NativeCalibrationArtifact artifact;
        assert(load_native_calibration_artifact(calibration_path, &artifact)
                   .ok_status());
        config.execution_mode = Sm120ExecutionMode::kStaticFp8E4M3;
        config.calibration = std::move(artifact);
    }
    return config;
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
    using flashrt::models::pi05::targets::sm120::Sm120Fp8Linear;
    using flashrt::models::pi05::targets::sm120::Sm120ExecutionMode;
    using flashrt::models::pi05::targets::sm120::Sm120TargetConfig;
    using flashrt::models::pi05::targets::sm120::Sm120TargetBundle;

    const Pi05ResolvedShape shape = canonical_shape();
    flashrt::modalities::Status status;
    assert(!Sm120TargetBundle::create(
        nullptr, shape, target_config("missing"), &status));
    assert(!status.ok_status());

    frt_ctx context = frt_ctx_create();
    assert(context);
    Pi05ResolvedShape invalid_shape = shape;
    ++invalid_shape.encoder_sequence;
    assert(!Sm120TargetBundle::create(
        context, invalid_shape, target_config("missing"), &status));
    assert(!status.ok_status());
    assert(!Sm120TargetBundle::create(
        context, shape, target_config(""), &status));
    assert(!status.ok_status());
    Sm120TargetConfig missing_artifact = target_config("missing");
    missing_artifact.execution_mode =
        Sm120ExecutionMode::kStaticFp8E4M3;
    assert(!Sm120TargetBundle::create(
        context, shape, std::move(missing_artifact), &status));
    assert(!status.ok_status());
    Sm120TargetConfig unexpected_artifact = target_config("missing");
    unexpected_artifact.calibration = NativeCalibrationArtifact{};
    assert(!Sm120TargetBundle::create(
        context, shape, std::move(unexpected_artifact), &status));
    assert(!status.ok_status());
    Sm120TargetConfig observed_with_artifact = target_config("missing");
    observed_with_artifact.execution_mode =
        Sm120ExecutionMode::kObservedFp8E4M3;
    observed_with_artifact.calibration = NativeCalibrationArtifact{};
    assert(!Sm120TargetBundle::create(
        context, shape, std::move(observed_with_artifact), &status));
    assert(!status.ok_status());

    std::unique_ptr<Sm120TargetBundle> missing =
        Sm120TargetBundle::create(
            context, shape, target_config("missing"), &status);
    assert(missing);
    Pi05ResolvedResources resources;
    assert(!missing->resolve_resources(&resources).ok_status());
    assert(!missing->finalize_setup().ok_status());
    assert(!missing->initialize_resources().ok_status());
    assert(!missing->initialize_resources().ok_status());
    assert(missing->execution_mode() == Sm120ExecutionMode::kBf16);
    std::vector<float> vision;
    std::vector<float> encoder;
    std::vector<float> decoder;
    assert(!missing->reset_observer_scales(0).ok_status());
    assert(!missing->download_observer_scales(
                       &vision, &encoder, &decoder)
                .ok_status());
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
    const char* calibration = std::getenv("FLASHRT_PI05_CALIBRATION");
    Sm120TargetConfig config = target_config(checkpoint, calibration);
    const bool fp8 =
        config.execution_mode == Sm120ExecutionMode::kStaticFp8E4M3;
    Pi05ResolvedShape runtime_shape = shape;
    if (config.calibration) {
        Pi05ShapeConfig runtime_config;
        runtime_config.num_views = config.calibration->num_views;
        runtime_config.max_prompt_tokens =
            config.calibration->max_prompt_tokens;
        runtime_config.chunk = config.calibration->chunk_size;
        runtime_config.num_steps = config.calibration->num_steps;
        runtime_config.vision_pool_factor =
            config.calibration->vision_pool_factor;
        runtime_config.state_dim = config.calibration->state_dim;
        runtime_config.robot_action_dim = shape.robot_action_dim;
        assert(resolve_pi05_shape(runtime_config, &runtime_shape).ok_status());
    }
    std::unique_ptr<Sm120TargetBundle> target =
        Sm120TargetBundle::create(
            context, runtime_shape, std::move(config), &status);
    assert(target && status.ok_status());
    status = target->initialize_resources();
    if (fp8 && !Sm120Fp8Linear::runtime_status().ok_status()) {
        assert(!status.ok_status());
        assert(status.code == flashrt::modalities::StatusCode::kUnsupported);
        target.reset();
        frt_ctx_destroy(context);
        std::printf("PASS - PI0.5 SM120 target FP8 runtime rejection\n");
        return 0;
    }
    assert(status.ok_status());
    assert(!target->initialize_resources().ok_status());
    assert(target->materialized_weight_count() > 0);
    status = target->resolve_resources(&resources);
    if (!status.ok_status()) {
        std::fprintf(stderr, "resolve_resources failed: %s\n",
                     status.message.c_str());
    }
    assert(status.ok_status());
    assert(!target->resolve_resources(&resources).ok_status());
    assert(validate_pi05_resolved_resources(resources, runtime_shape)
               .ok_status());
    assert(target->packed_weight_count() == (fp8 ? 253u : 0u));
    assert(!target->finalize_setup().ok_status());

    Pi05SemanticPipeline pipeline(runtime_shape);
    Pi05PrepareExecution prepare;
    assert(target->make_prepare_execution(&prepare).ok_status());
    assert(pipeline.record_prepare(prepare).ok_status());
    assert(target->complete_prepare().ok_status());
    assert(!target->complete_prepare().ok_status());
    assert(target->prepare_call_count() ==
           static_cast<std::size_t>(runtime_shape.num_steps) *
               (2 + 2 * kPi05ModelDims.decoder_layers));
    assert(target->finalize_setup().ok_status());
    assert(target->autotuned_shape_count() == (fp8 ? 13u : 0u));
    assert(!target->finalize_setup().ok_status());

    assert(target->set_prompt_length(0).ok_status());
    assert(target->set_prompt_length(runtime_shape.max_prompt_tokens)
               .ok_status());
    const std::int32_t committed =
        read_control(resources.buffers.encoder_valid_tokens);
    assert(committed == runtime_shape.encoder_vision_sequence +
                            runtime_shape.max_prompt_tokens);
    assert(!target->set_prompt_length(runtime_shape.max_prompt_tokens + 1)
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

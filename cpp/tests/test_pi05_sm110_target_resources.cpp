#include "flashrt/cpp/models/pi05/model/linear_weight_groups.h"
#include "flashrt/cpp/models/pi05/model/native_session.h"
#include "flashrt/cpp/models/pi05/support/native_calibration.h"
#include "flashrt/cpp/models/pi05/targets/sm110/fp8_weight_packer.h"
#include "flashrt/cpp/models/pi05/targets/sm110/physical_resources.h"
#include "flashrt/cpp/models/pi05/targets/sm110/target.h"

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

using flashrt::models::pi05::NativeCalibrationArtifact;
using flashrt::models::pi05::Pi05ResolvedShape;
using flashrt::models::pi05::Pi05ShapeConfig;

Pi05ResolvedShape shape_from(const NativeCalibrationArtifact& artifact) {
    Pi05ShapeConfig config;
    config.num_views = artifact.num_views;
    config.max_prompt_tokens = artifact.max_prompt_tokens;
    config.chunk = artifact.chunk_size;
    config.num_steps = artifact.num_steps;
    config.vision_pool_factor = artifact.vision_pool_factor;
    config.state_dim = artifact.state_dim;
    config.robot_action_dim = 7;
    Pi05ResolvedShape shape;
    assert(flashrt::models::pi05::resolve_pi05_shape(config, &shape)
               .ok_status());
    return shape;
}

NativeCalibrationArtifact test_artifact() {
    using flashrt::models::pi05::kPi05LinearScalesPerLayer;
    using flashrt::models::pi05::kPi05ModelDims;
    NativeCalibrationArtifact artifact;
    artifact.activation_dtype = "float16";
    artifact.hardware = "sm110";
    artifact.weights_sha256 = std::string(64, 'a');
    artifact.tokenizer_sha256 = std::string(64, 'b');
    artifact.num_views = 3;
    artifact.max_prompt_tokens = 64;
    artifact.state_dim = 8;
    artifact.chunk_size = 10;
    artifact.num_steps = 10;
    artifact.vision_pool_factor = 1;
    artifact.sample_count = 1;
    artifact.encoder_scales.assign(
        static_cast<std::size_t>(kPi05ModelDims.encoder_layers) *
            kPi05LinearScalesPerLayer,
        1.0f);
    artifact.decoder_scales.assign(
        static_cast<std::size_t>(artifact.num_steps) *
            kPi05ModelDims.decoder_layers *
            kPi05LinearScalesPerLayer,
        1.0f);
    return artifact;
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

bool sm110_device() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) {
        cudaGetLastError();
        return false;
    }
    int device = 0;
    cudaDeviceProp properties{};
    return cudaGetDevice(&device) == cudaSuccess &&
           cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
           properties.major == 11 && properties.minor == 0;
}

void run_real_resource_contract(const char* checkpoint,
                                const char* calibration) {
    using namespace flashrt::models::pi05;
    using namespace flashrt::models::pi05::targets::sm110;

    NativeCalibrationArtifact artifact;
    assert(load_native_calibration_artifact(calibration, &artifact)
               .ok_status());
    const Pi05ResolvedShape shape = shape_from(artifact);
    Sm110TargetConfig config;
    config.checkpoint_path = checkpoint;
    config.calibration = std::move(artifact);
    frt_ctx context = frt_ctx_create();
    assert(context);
    flashrt::modalities::Status status;
    std::unique_ptr<Sm110TargetBundle> target =
        Sm110TargetBundle::create(
            context, shape, std::move(config), &status);
    assert(target && status.ok_status());
    assert(target->initialize_resources().ok_status());
    assert(!target->initialize_resources().ok_status());
    assert(target->materialized_weight_count() > 0);
    assert(target->logical_workspace_count() > 0);
    Pi05ResolvedResources resources;
    assert(target->resolve_resources(&resources).ok_status());
    assert(!target->resolve_resources(&resources).ok_status());
    assert(target->resources_ready());
    assert(validate_pi05_resolved_resources(resources, shape).ok_status());

    const Sm110Fp8WeightPacker* packer = target->weight_packer();
    assert(packer && packer->finished());
    const std::size_t expected_packed =
        static_cast<std::size_t>(
            kPi05ModelDims.vision_layers +
            kPi05ModelDims.encoder_layers +
            kPi05ModelDims.decoder_layers) *
        kPi05LinearScalesPerLayer;
    assert(packer->size() == expected_packed);
    assert(!packer->packed_weight(expected_packed));
    const Sm110Fp8PackedWeight* first_packed = packer->packed_weight(0);
    assert(first_packed);
    assert(packer->packed_weight(first_packed->key) == first_packed);
    const Sm110PhysicalResources* physical =
        target->physical_resources();
    assert(physical && physical->initialized());
    assert(physical->padded_key_stride() ==
           shape.total_attention_keys + (shape.total_attention_keys & 1));
    assert(resources.buffers.action_delta.physical_dtype ==
           flashrt::modalities::DType::kFloat32);

    assert(cudaMemset(
               frt_buffer_dptr(resources.buffers.prompt_embedding.buffer), 0,
               resources.buffers.prompt_embedding.physical_bytes) ==
           cudaSuccess);
    assert(target->set_prompt_length(3).ok_status());
    assert(read_control(resources.buffers.encoder_valid_tokens) ==
           shape.encoder_vision_sequence + 4);
    const std::int32_t committed =
        read_control(resources.buffers.decoder_valid_tokens);
    assert(committed ==
           shape.encoder_vision_sequence + 4 + shape.chunk);
    assert(!target->set_prompt_length(shape.max_prompt_tokens + 1)
                .ok_status());
    assert(read_control(resources.buffers.decoder_valid_tokens) == committed);

    assert(resources.buffers.decoder_state.buffer);
    const std::size_t decoder_bytes =
        frt_buffer_bytes(resources.buffers.decoder_state.buffer);
    assert(cudaMemset(frt_buffer_dptr(resources.buffers.decoder_state.buffer),
                      0x5a, decoder_bytes) == cudaSuccess);
    Pi05SemanticPipeline pipeline(shape);
    Pi05PrepareExecution prepare;
    assert(target->make_prepare_execution(&prepare).ok_status());
    assert(pipeline.record_prepare(prepare).ok_status());
    assert(target->complete_prepare().ok_status());
    assert(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<unsigned char> decoder_state(decoder_bytes);
    assert(cudaMemcpy(decoder_state.data(),
                      frt_buffer_dptr(resources.buffers.decoder_state.buffer),
                      decoder_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (unsigned char value : decoder_state) assert(value == 0x5a);
    assert(target->resources_ready());
    assert(target->resolved_resources());
    assert(!target->make_prepare_execution(&prepare).ok_status());
    assert(target->finalize_setup().ok_status());
    Pi05ForwardExecution forward;
    assert(target->make_forward_execution(&forward).ok_status());
    assert(forward.resources == target->resolved_resources());
    assert(forward.ops && forward.vision.qkv && forward.encoder.qkv &&
           forward.decoder.qkv);
    assert(forward.ops->profile.vision_final_norm_epsilon == 1.0e-6f);
    assert(target->initialize_capture_inputs().ok_status());
    assert(target->resources_ready());
    assert(target->set_prompt_length(2).ok_status());
    assert(target->reset_after_warmup().ok_status());

    target.reset();
    frt_ctx_destroy(context);
}

void run_real_session_contract(const char* checkpoint,
                               const char* calibration) {
    using namespace flashrt::models::pi05;
    using namespace flashrt::models::pi05::targets::sm110;

    NativeCalibrationArtifact artifact;
    assert(load_native_calibration_artifact(calibration, &artifact)
               .ok_status());
    const Pi05ResolvedShape shape = shape_from(artifact);
    Sm110TargetConfig config;
    config.checkpoint_path = checkpoint;
    config.calibration = std::move(artifact);
    frt_ctx context = frt_ctx_create();
    assert(context);
    flashrt::modalities::Status status;
    std::unique_ptr<Sm110TargetBundle> concrete =
        Sm110TargetBundle::create(
            context, shape, std::move(config), &status);
    assert(concrete && status.ok_status());
    std::unique_ptr<Pi05TargetBundle> target(std::move(concrete));
    std::unique_ptr<Pi05NativeSession> session = Pi05NativeSession::create(
        context, shape, std::move(target), &status);
    if (!session || !status.ok_status()) {
        std::fprintf(stderr, "%s\n", status.message.c_str());
        assert(false);
    }
    assert(session->set_prompt_length(2).ok_status());
    const Pi05ResolvedBuffers& buffers = session->resources().buffers;
    const Pi05ResolvedBuffer* inputs[] = {
        &buffers.images, &buffers.prompt_embedding, &buffers.noise};
    for (const Pi05ResolvedBuffer* input : inputs) {
        assert(input->buffer);
        assert(cudaMemset(frt_buffer_dptr(input->buffer), 0,
                          input->physical_bytes) == cudaSuccess);
    }
    assert(session->replay(Pi05GraphId::kInfer) == FRT_OK);
    assert(session->synchronize().ok_status());
    std::vector<unsigned char> full(buffers.noise.physical_bytes);
    assert(cudaMemcpy(full.data(), frt_buffer_dptr(buffers.noise.buffer),
                      full.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemset(frt_buffer_dptr(buffers.noise.buffer), 0,
                      buffers.noise.physical_bytes) == cudaSuccess);
    assert(session->replay(Pi05GraphId::kContext) == FRT_OK);
    assert(session->replay(Pi05GraphId::kDecodeOnly) == FRT_OK);
    assert(session->synchronize().ok_status());
    std::vector<unsigned char> split(buffers.noise.physical_bytes);
    assert(cudaMemcpy(split.data(), frt_buffer_dptr(buffers.noise.buffer),
                      split.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(full == split);
}

}  // namespace

int main() {
    using namespace flashrt::models::pi05;
    using namespace flashrt::models::pi05::targets::sm110;

    NativeCalibrationArtifact artifact = test_artifact();
    const Pi05ResolvedShape shape = shape_from(artifact);
    Sm110TargetConfig config;
    config.checkpoint_path = "missing";
    config.calibration = artifact;
    flashrt::modalities::Status status;
    assert(!Sm110TargetBundle::create(
        nullptr, shape, config, &status));
    assert(!status.ok_status());

    frt_ctx context = frt_ctx_create();
    assert(context);
    Sm110Fp8WeightPacker failed_packer(context);
    Pi05LinearWeightGroup invalid_group;
    assert(!failed_packer.record(invalid_group).ok_status());
    assert(!failed_packer.finish().ok_status());
    Pi05ResolvedShape invalid_shape = shape;
    ++invalid_shape.encoder_sequence;
    assert(!Sm110TargetBundle::create(
        context, invalid_shape, config, &status));
    assert(!status.ok_status());
    Sm110TargetConfig wrong_hardware = config;
    wrong_hardware.calibration->hardware = "sm120";
    assert(!Sm110TargetBundle::create(
        context, shape, std::move(wrong_hardware), &status));
    assert(!status.ok_status());
    Sm110TargetConfig empty_path = config;
    empty_path.checkpoint_path.clear();
    assert(!Sm110TargetBundle::create(
        context, shape, std::move(empty_path), &status));
    assert(!status.ok_status());

    std::unique_ptr<Sm110TargetBundle> missing =
        Sm110TargetBundle::create(context, shape, config, &status);
    assert(missing && status.ok_status());
    Pi05ResolvedResources resources;
    assert(!missing->resolve_resources(&resources).ok_status());
    assert(!missing->set_prompt_length(0).ok_status());
    assert(!missing->initialize_resources().ok_status());
    assert(!missing->initialize_resources().ok_status());
    missing.reset();
    frt_ctx_destroy(context);

    const char* checkpoint = std::getenv("FLASHRT_PI05_CHECKPOINT");
    const char* calibration = std::getenv("FLASHRT_PI05_CALIBRATION");
    if (!checkpoint || !checkpoint[0] || !calibration || !calibration[0]) {
        std::printf("PASS - PI0.5 SM110 target failure contract\n");
        return 0;
    }
    if (!sm110_device()) {
        std::printf("SKIP - SM110 resource contract needs compute capability 11.0\n");
        return 0;
    }

    run_real_resource_contract(checkpoint, calibration);
    run_real_session_contract(checkpoint, calibration);
    std::printf("PASS - PI0.5 SM110 target resource contract\n");
    return 0;
}

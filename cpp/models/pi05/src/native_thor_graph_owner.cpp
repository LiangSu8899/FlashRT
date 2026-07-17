#include "flashrt/cpp/models/pi05/native_thor_graph_owner.h"

#include "flashrt/cpp/models/pi05/native_thor_style_precompute.h"
#include "flashrt/cpp/models/pi05/native_thor_weight_materializer.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

modalities::Status copy_scales(const NativeWorkspace& workspace,
                               const char* name,
                               const std::vector<float>& values) {
    const NativeWorkspaceBuffer* destination = workspace.find(name);
    if (!destination || destination->dtype != modalities::DType::kFloat32 ||
        frt_buffer_bytes(destination->buffer) !=
            values.size() * sizeof(float)) {
        return invalid("Thor activation scale workspace is invalid");
    }
    const cudaError_t rc = cudaMemcpy(
        frt_buffer_dptr(destination->buffer), values.data(),
        values.size() * sizeof(float), cudaMemcpyHostToDevice);
    return rc == cudaSuccess
               ? modalities::Status::ok()
               : backend("Thor activation scale upload failed");
}

bool calibration_matches(const NativeCalibrationArtifact& artifact,
                         const NativeGraphConfig& config) {
    return artifact.num_views == config.num_views &&
           artifact.max_prompt_tokens == config.max_prompt_tokens &&
           artifact.chunk_size == config.chunk_size &&
           artifact.num_steps == config.num_steps &&
           artifact.vision_pool_factor == config.vision_pool_factor;
}

}  // namespace

NativeThorGraphOwner::NativeThorGraphOwner(
    frt_ctx ctx,
    const NativeGraphConfig& config)
    : graphs_(ctx, static_cast<std::size_t>(NativeGraphKind::kCount)),
      config_(config),
      weights_(ctx),
      workspace_(ctx),
      forward_(&driver_) {}

NativeThorGraphOwner::~NativeThorGraphOwner() = default;

std::unique_ptr<NativeThorGraphOwner> NativeThorGraphOwner::create(
    const std::string& checkpoint_path,
    const NativeGraphConfig& config,
    const NativeCalibrationArtifact& calibration,
    modalities::Status* status) {
    if (config.num_views < 1 || config.num_views > 3 ||
        config.max_prompt_tokens < 1 || (config.max_prompt_tokens & 1) ||
        config.chunk_size < 1 || config.num_steps < 1 ||
        static_cast<std::uint64_t>(config.max_prompt_tokens) +
                static_cast<std::uint64_t>(config.chunk_size) +
                static_cast<std::uint64_t>(config.num_views) * 256 >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        config.vision_pool_factor != 1 ||
        !calibration_matches(calibration, config)) {
        if (status) {
            *status = invalid("Thor native graph configuration is invalid");
        }
        return nullptr;
    }
    modalities::Status st =
        validate_native_calibration_artifact(calibration);
    if (!st.ok_status()) {
        if (status) *status = st;
        return nullptr;
    }
    frt_ctx ctx = frt_ctx_create();
    if (!ctx) {
        if (status) *status = backend("Thor graph context creation failed");
        return nullptr;
    }
    std::unique_ptr<NativeThorGraphOwner> owner(
        new (std::nothrow) NativeThorGraphOwner(ctx, config));
    if (!owner) {
        frt_ctx_destroy(ctx);
        if (status) *status = backend("Thor graph owner allocation failed");
        return nullptr;
    }
    st = owner->initialize(checkpoint_path, calibration);
    if (!st.ok_status()) {
        if (status) *status = st;
        return nullptr;
    }
    if (status) *status = modalities::Status::ok();
    return owner;
}

modalities::Status NativeThorGraphOwner::initialize(
    const std::string& checkpoint_path,
    const NativeCalibrationArtifact& calibration) {
    const bool profile_setup = std::getenv("FLASHRT_PROFILE_NATIVE_SETUP");
    const auto setup_begin = std::chrono::steady_clock::now();
    auto checkpoint = setup_begin;
    const auto report = [&](const char* phase) {
        const auto now = std::chrono::steady_clock::now();
        if (profile_setup) {
            std::fprintf(stderr, "native_thor_setup %s_ms=%.3f\n", phase,
                         std::chrono::duration<double, std::milli>(
                             now - checkpoint).count());
        }
        checkpoint = now;
    };

    loader::SafetensorsFile source;
    if (!source.open(checkpoint_path + "/model.safetensors")) {
        return modalities::Status::error(modalities::StatusCode::kNotFound,
                                         source.error());
    }
    report("header");
    NativeThorWeightMaterializer materializer(source, &weights_);
    NativeThorMaterializationOptions options;
    options.num_steps = config_.num_steps;
    options.include_embedding = true;
    modalities::Status st =
        materializer.materialize_all(options, &weight_scales_);
    if (!st.ok_status()) return st;
    report("materialize");

    NativeWorkspaceConfig workspace_config;
    workspace_config.num_views = config_.num_views;
    workspace_config.max_prompt_tokens = config_.max_prompt_tokens;
    workspace_config.chunk_size = config_.chunk_size;
    workspace_config.num_steps = config_.num_steps;
    workspace_config.vision_pool_factor = config_.vision_pool_factor;
    workspace_config.flavor = NativeWorkspaceFlavor::kThorFp8;
    st = workspace_.allocate(workspace_config);
    if (!st.ok_status()) return st;
    st = workspace_.expand_vision_position_embedding(weights_);
    if (!st.ok_status()) return st;
    st = copy_scales(workspace_, "encoder_activation_scales",
                     calibration.encoder_scales);
    if (!st.ok_status()) return st;
    st = copy_scales(workspace_, "decoder_activation_scales",
                     calibration.decoder_scales);
    if (!st.ok_status()) return st;
    encoder_alphas_.resize(calibration.encoder_scales.size());
    if (encoder_alphas_.size() != weight_scales_.encoder.size()) {
        return invalid("Thor encoder scale count is invalid");
    }
    for (std::size_t i = 0; i < encoder_alphas_.size(); ++i) {
        encoder_alphas_[i] =
            calibration.encoder_scales[i] * weight_scales_.encoder[i];
        if (!std::isfinite(encoder_alphas_[i]) ||
            !(encoder_alphas_[i] > 0.0f)) {
            return invalid("Thor encoder alpha is invalid");
        }
    }
    st = set_prompt_length(0);
    if (!st.ok_status()) return st;
    NativeThorStylePrecomputer precomputer(&driver_);
    st = precomputer.run(weights_, &workspace_, 0);
    if (!st.ok_status()) return st;
    report("workspace_style");

    for (const char* name : {"observation_images_normalized",
                             "prompt_embedding", "diffusion_noise"}) {
        const NativeWorkspaceBuffer* input = workspace_.find(name);
        if (!input ||
            cudaMemset(frt_buffer_dptr(input->buffer), 0,
                       frt_buffer_bytes(input->buffer)) != cudaSuccess) {
            return backend("Thor graph input initialization failed");
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return backend("Thor graph setup synchronization failed");
    }

    // CUTLASS, cuBLAS, and FMHA initialize tactics outside CUDA capture.
    st = record(NativeGraphKind::kInfer, nullptr);
    if (!st.ok_status()) return st;
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return backend("Thor graph warmup synchronization failed");
    }
    const NativeWorkspaceBuffer* noise = workspace_.find("diffusion_noise");
    if (!noise ||
        cudaMemset(frt_buffer_dptr(noise->buffer), 0,
                   frt_buffer_bytes(noise->buffer)) != cudaSuccess) {
        return backend("Thor graph warmup reset failed");
    }
    report("warmup");

    st = capture_native_graph(
        &graphs_,
        NativeGraphKind::kInfer, workspace_,
        {"observation_images_normalized", "prompt_embedding", "encoder_x",
         "diffusion_noise", "rtc_prev_action_chunk", "rtc_prefix_weights",
         "rtc_guidance_weight"},
        record_graph, this);
    if (!st.ok_status()) return st;
    st = capture_native_graph(
        &graphs_,
        NativeGraphKind::kDecodeOnly, workspace_,
        {"encoder_x", "diffusion_noise", "rtc_prev_action_chunk",
         "rtc_prefix_weights", "rtc_guidance_weight"},
        record_graph, this);
    if (!st.ok_status()) return st;
    st = capture_native_graph(
        &graphs_,
        NativeGraphKind::kContext, workspace_,
        {"observation_images_normalized", "prompt_embedding", "encoder_x"},
        record_graph, this);
    if (!st.ok_status()) return st;
    report("capture");

    st = graphs_.create_replay_stream();
    if (!st.ok_status()) return st;
    report("stream");
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "native_thor_setup total_ms=%.3f\n",
                     std::chrono::duration<double, std::milli>(
                         now - setup_begin).count());
    }
    return modalities::Status::ok();
}

modalities::Status NativeThorGraphOwner::record_context(void* stream) {
    modalities::Status st = copy_prompt_to_encoder(&workspace_, stream);
    if (!st.ok_status()) return st;
    const std::uintptr_t stream_id = reinterpret_cast<std::uintptr_t>(stream);
    st = forward_.vision(weights_, &workspace_, weight_scales_, stream_id);
    if (!st.ok_status()) return st;
    return forward_.encoder(
        weights_, &workspace_, encoder_alphas_, stream_id);
}

modalities::Status NativeThorGraphOwner::record_action(void* stream) {
    return forward_.diffusion(
        weights_, &workspace_, reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status NativeThorGraphOwner::record(NativeGraphKind kind,
                                                 void* stream) {
    if (kind == NativeGraphKind::kContext) return record_context(stream);
    if (kind == NativeGraphKind::kDecodeOnly) return record_action(stream);
    if (kind != NativeGraphKind::kInfer) {
        return invalid("Thor graph kind is invalid");
    }
    modalities::Status st = record_context(stream);
    return st.ok_status() ? record_action(stream) : st;
}

modalities::Status NativeThorGraphOwner::record_graph(
    void* user, std::size_t slot, void* stream) {
    auto* owner = static_cast<NativeThorGraphOwner*>(user);
    return owner->record(static_cast<NativeGraphKind>(slot), stream);
}

modalities::Status NativeThorGraphOwner::set_prompt_length(int prompt_tokens) {
    return workspace_.set_fixed_prompt_length(prompt_tokens);
}

int NativeThorGraphOwner::replay(NativeGraphKind kind) const {
    return graphs_.replay(static_cast<std::size_t>(kind));
}

modalities::Status NativeThorGraphOwner::synchronize() const {
    return graphs_.synchronize();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#include "flashrt/cpp/models/pi05/backends/sm110/session.h"

#include "flashrt/cpp/models/pi05/backends/sm110/native_thor_style_precompute.h"
#include "flashrt/cpp/models/pi05/backends/sm110/native_thor_weight_materializer.h"

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
                         const BackendConfig& config) {
    return artifact.num_views == config.num_views &&
           artifact.max_prompt_tokens == config.max_prompt_tokens &&
           artifact.chunk_size == config.chunk_size &&
           artifact.num_steps == config.num_steps &&
           artifact.vision_pool_factor == config.vision_pool_factor;
}

}  // namespace

Sm110BackendSession::Sm110BackendSession(
    frt_ctx ctx,
    const BackendConfig& config)
    : Pi05Pipeline(ctx, config),
      weights_(ctx),
      workspace_(ctx),
      forward_(&driver_) {}

Sm110BackendSession::~Sm110BackendSession() = default;

std::unique_ptr<Sm110BackendSession> Sm110BackendSession::create(
    const std::string& checkpoint_path,
    const BackendConfig& config,
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
    std::unique_ptr<Sm110BackendSession> session(
        new (std::nothrow) Sm110BackendSession(ctx, config));
    if (!session) {
        frt_ctx_destroy(ctx);
        if (status) *status = backend("SM110 backend session allocation failed");
        return nullptr;
    }
    st = session->initialize(checkpoint_path, calibration);
    if (!st.ok_status()) {
        if (status) *status = st;
        return nullptr;
    }
    if (status) *status = modalities::Status::ok();
    return session;
}

modalities::Status Sm110BackendSession::initialize(
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
    options.num_steps = config().num_steps;
    options.include_embedding = true;
    modalities::Status st =
        materializer.materialize_all(options, &weight_scales_);
    if (!st.ok_status()) return st;
    report("materialize");

    NativeWorkspaceConfig workspace_config;
    workspace_config.num_views = config().num_views;
    workspace_config.max_prompt_tokens = config().max_prompt_tokens;
    workspace_config.chunk_size = config().chunk_size;
    workspace_config.num_steps = config().num_steps;
    workspace_config.vision_pool_factor = config().vision_pool_factor;
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

    st = resolve_backend_artifacts(
        workspace_, weights_, NativeWeightDType::kFloat16, &artifacts_);
    if (!st.ok_status()) return st;

    st = finish_prepare(true);
    if (!st.ok_status()) return st;
    report("graph_prepare");
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "native_thor_setup total_ms=%.3f\n",
                     std::chrono::duration<double, std::milli>(
                         now - setup_begin).count());
    }
    return modalities::Status::ok();
}

modalities::Status Sm110BackendSession::record_vision(void* stream) {
    return forward_.vision(
        weights_, &workspace_, weight_scales_,
        reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status Sm110BackendSession::record_encoder(void* stream) {
    return forward_.encoder(
        weights_, &workspace_, encoder_alphas_,
        reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status Sm110BackendSession::record_diffusion(void* stream) {
    return forward_.diffusion(
        weights_, &workspace_, reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status Sm110BackendSession::set_prompt_length(int prompt_tokens) {
    return workspace_.set_fixed_prompt_length(prompt_tokens);
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

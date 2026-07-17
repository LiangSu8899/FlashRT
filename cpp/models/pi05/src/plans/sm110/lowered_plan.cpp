#include "flashrt/cpp/models/pi05/plans/sm110/lowered_plan.h"

#include "flashrt/cpp/models/pi05/plans/sm110/native_thor_style_precompute.h"
#include "flashrt/cpp/models/pi05/plans/sm110/native_thor_weight_materializer.h"

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
                         const Pi05PipelineConfig& config) {
    return artifact.num_views == config.num_views &&
           artifact.max_prompt_tokens == config.max_prompt_tokens &&
           artifact.chunk_size == config.chunk_size &&
           artifact.num_steps == config.num_steps &&
           artifact.vision_pool_factor == config.vision_pool_factor;
}

}  // namespace

NativeWorkspaceRequirements make_sm110_workspace_requirements(
    const NativeWorkspaceConfig& config,
    bool enable_calibration) {
    NativeWorkspaceRequirements requirements;
    requirements.activation_dtype = modalities::DType::kFloat16;
    requirements.fixed_prompt_controls = true;

    const std::uint64_t vs =
        static_cast<std::uint64_t>(config.num_views) * 256;
    const std::uint64_t pool =
        static_cast<std::uint64_t>(config.vision_pool_factor) *
        static_cast<std::uint64_t>(config.vision_pool_factor);
    const std::uint64_t es =
        vs / pool + static_cast<std::uint64_t>(config.max_prompt_tokens);
    const std::uint64_t ds =
        static_cast<std::uint64_t>(config.chunk_size);
    const std::uint64_t steps =
        static_cast<std::uint64_t>(config.num_steps);
    const std::uint64_t keys = es + ds;
    const auto add_f16 =
        [&](const char* name,
            std::initializer_list<std::uint64_t> shape) {
            requirements.add_buffer(
                name, shape, modalities::DType::kFloat16);
        };

    requirements.add_buffer(
        "vision_x_fp8", {vs, 1152}, modalities::DType::kUInt8);
    add_f16("vision_QKV", {vs, 3456});
    add_f16("vision_attn", {vs, 1152});
    add_f16("vision_hidden", {vs, 4304});
    requirements.add_buffer(
        "vision_hidden_fp8", {vs, 4304}, modalities::DType::kUInt8);
    requirements.add_buffer(
        "vision_unit_scale", {1}, modalities::DType::kFloat32);
    requirements.add_alias(
        "vision_postln_scratch", "vision_attn", {vs, 1152});

    requirements.add_buffer(
        "encoder_x_fp8", {es, 2048}, modalities::DType::kUInt8);
    add_f16("encoder_QKV", {es, 2560});
    add_f16("encoder_logits", {es * 8, keys});
    add_f16("encoder_attn", {es, 2048});
    requirements.add_buffer(
        "encoder_o_fp8", {es, 2048}, modalities::DType::kUInt8);
    add_f16("encoder_gate_merged", {es, 32768});
    add_f16("encoder_hidden", {es, 16384});
    requirements.add_buffer(
        "encoder_hidden_fp8", {es, 16384}, modalities::DType::kUInt8);
    add_f16("encoder_fg", {es, 2048});
    requirements.add_buffer(
        "encoder_activation_scales", {18, 4},
        modalities::DType::kFloat32);
    add_f16("encoder_k_cache", {18, keys, 256});
    add_f16("encoder_v_cache", {18, keys, 256});
    requirements.add_buffer(
        "attn_enc_seqused", {sizeof(std::int32_t)},
        modalities::DType::kUInt8);
    requirements.add_buffer(
        "attn_dec_seqused", {sizeof(std::int32_t)},
        modalities::DType::kUInt8);
    requirements.add_buffer(
        "attn_dec_devpos", {sizeof(std::int32_t)},
        modalities::DType::kUInt8);

    add_f16("x_normed_buf", {ds, 1024});
    add_f16("gate_buf", {ds, 1024});
    add_f16("decoder_QKV", {ds, 2560});
    add_f16("decoder_logits", {ds * 8, keys});
    add_f16("decoder_attn", {ds, 2048});
    add_f16("decoder_hidden", {ds, 8192});
    add_f16("decoder_fg", {ds, 8192});
    requirements.add_buffer(
        "decoder_action_f32", {ds, 32}, modalities::DType::kFloat32);
    requirements.add_buffer(
        "decoder_x_fp8", {ds, 1024}, modalities::DType::kUInt8);
    requirements.add_buffer(
        "decoder_hidden_fp8", {ds, 4096}, modalities::DType::kUInt8);
    requirements.add_buffer(
        "decoder_context_fp8", {ds, 2048}, modalities::DType::kUInt8);
    requirements.add_buffer(
        "decoder_activation_scales", {steps, 18, 4},
        modalities::DType::kFloat32);

    if (enable_calibration) {
        add_f16("encoder_norm_scratch", {es, 2048});
        add_f16("encoder_x_scratch", {es, 2048});
        requirements.add_buffer(
            "encoder_fp8_scratch", {es, 16384},
            modalities::DType::kUInt8);
        requirements.add_buffer(
            "encoder_sample_scales", {18, 4},
            modalities::DType::kFloat32);
        requirements.add_buffer(
            "decoder_fp8_scratch", {ds, 4096},
            modalities::DType::kUInt8);
        requirements.add_buffer(
            "decoder_sample_scales", {steps, 18, 4},
            modalities::DType::kFloat32);
        requirements.add_buffer(
            "calibration_scale", {1}, modalities::DType::kFloat32);
    }
    return requirements;
}

modalities::Status initialize_sm110_workspace(NativeWorkspace* workspace) {
    const NativeWorkspaceBuffer* unit =
        workspace ? workspace->find("vision_unit_scale") : nullptr;
    if (!unit || unit->dtype != modalities::DType::kFloat32 ||
        unit->shape != std::vector<std::uint64_t>({1})) {
        return invalid("SM110 unit scale workspace is invalid");
    }
    const float value = 1.0f;
    const cudaError_t rc = cudaMemcpy(
        frt_buffer_dptr(unit->buffer), &value, sizeof(value),
        cudaMemcpyHostToDevice);
    return rc == cudaSuccess
               ? modalities::Status::ok()
               : backend("SM110 unit scale upload failed");
}

Sm110LoweredPlan::Sm110LoweredPlan(
    frt_ctx ctx,
    const Pi05PipelineConfig& config)
    : Pi05Pipeline(ctx, config),
      weights_(ctx),
      workspace_(ctx),
      forward_(&driver_) {}

Sm110LoweredPlan::~Sm110LoweredPlan() = default;

std::unique_ptr<Sm110LoweredPlan> Sm110LoweredPlan::create(
    const std::string& checkpoint_path,
    const Pi05PipelineConfig& config,
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
    std::unique_ptr<Sm110LoweredPlan> session(
        new (std::nothrow) Sm110LoweredPlan(ctx, config));
    if (!session) {
        frt_ctx_destroy(ctx);
        if (status) *status = backend("SM110 lowered plan allocation failed");
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

modalities::Status Sm110LoweredPlan::initialize(
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
    NativeWorkspaceRequirements workspace_requirements =
        make_sm110_workspace_requirements(workspace_config, false);
    st = workspace_.allocate(workspace_config, workspace_requirements);
    if (!st.ok_status()) return st;
    st = initialize_sm110_workspace(&workspace_);
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

    st = resolve_pipeline_artifacts(
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

modalities::Status Sm110LoweredPlan::record_vision_begin(void* stream) {
    return forward_.vision_begin(
        weights_, &workspace_, reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status Sm110LoweredPlan::record_vision_layer(
    int layer, void* stream) {
    return forward_.vision_layer(
        layer, weights_, &workspace_, weight_scales_,
        reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status Sm110LoweredPlan::record_vision_end(void* stream) {
    return forward_.vision_end(
        weights_, &workspace_, reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status Sm110LoweredPlan::record_encoder_layer(
    int layer, void* stream) {
    return forward_.encoder_layer(
        layer, weights_, &workspace_, encoder_alphas_,
        reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status Sm110LoweredPlan::record_diffusion_begin(
    int step, void* stream) {
    return forward_.diffusion_begin(
        step, weights_, &workspace_,
        reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status Sm110LoweredPlan::record_decoder_layer(
    int step, int layer, void* stream) {
    return forward_.decoder_layer(
        step, layer, weights_, &workspace_,
        reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status Sm110LoweredPlan::record_diffusion_end(
    int step, void* stream) {
    return forward_.diffusion_end(
        step, weights_, &workspace_,
        reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status Sm110LoweredPlan::set_prompt_length(int prompt_tokens) {
    return workspace_.set_fixed_prompt_length(prompt_tokens);
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

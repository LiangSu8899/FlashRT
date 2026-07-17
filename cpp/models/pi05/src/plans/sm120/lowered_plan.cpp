#include "flashrt/cpp/models/pi05/plans/sm120/lowered_plan.h"

#include "flashrt/cpp/models/pi05/plans/sm120/native_style_precompute.h"
#include "flashrt/cpp/models/pi05/support/native_weight_materializer.h"
#include "flashrt/cpp/models/pi05/plans/sm120/native_rtx_autotune.h"
#include "flashrt/cpp/models/pi05/plans/sm120/native_rtx_weight_packer.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
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

bool valid_sm120_config(const Pi05PipelineConfig& config) {
    const std::uint64_t total_sequence =
        static_cast<std::uint64_t>(config.max_prompt_tokens) +
        static_cast<std::uint64_t>(config.chunk_size) +
        static_cast<std::uint64_t>(config.num_views) * 256;
    return config.num_views >= 1 && config.num_views <= 3 &&
           config.max_prompt_tokens >= 1 && config.chunk_size >= 1 &&
           config.num_steps >= 1 &&
           total_sequence <=
               static_cast<std::uint64_t>(
                   std::numeric_limits<int>::max()) &&
           (config.vision_pool_factor == 1 ||
            config.vision_pool_factor == 2 ||
            config.vision_pool_factor == 4);
}

modalities::Status upload_scales(
    NativeWorkspace* workspace,
    const char* name,
    const std::vector<float>& values) {
    const NativeWorkspaceBuffer* destination =
        workspace ? workspace->find(name) : nullptr;
    if (!destination || destination->dtype != modalities::DType::kFloat32 ||
        destination->shape !=
            std::vector<std::uint64_t>({values.size()}) ||
        values.empty()) {
        return invalid("native RTX FP8 scale payload is invalid");
    }
    const cudaError_t rc = cudaMemcpy(
        frt_buffer_dptr(destination->buffer), values.data(),
        values.size() * sizeof(float), cudaMemcpyHostToDevice);
    return rc == cudaSuccess
               ? modalities::Status::ok()
               : backend("native RTX FP8 scale upload failed");
}

}  // namespace

NativeWorkspaceRequirements make_sm120_workspace_requirements(
    const NativeWorkspaceConfig& config,
    bool fp8) {
    NativeWorkspaceRequirements requirements;
    requirements.activation_dtype = modalities::DType::kBFloat16;

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
    const auto add_bf16 =
        [&](const char* name,
            std::initializer_list<std::uint64_t> shape) {
            requirements.add_buffer(
                name, shape, modalities::DType::kBFloat16);
        };

    add_bf16("vision_x_norm", {vs, 1152});
    add_bf16("vision_QKV", {vs, 3456});
    add_bf16("vision_hidden", {vs, 4304});
    add_bf16("encoder_x_norm", {es, 2048});
    add_bf16("encoder_QKV", {es, 2560});
    add_bf16("encoder_hidden", {es, 16384});
    add_bf16("encoder_gate_merged", {es, 32768});
    add_bf16("encoder_gate_buf", {es, 16384});
    add_bf16("decoder_action_buf", {ds, 32});
    add_bf16("decoder_QKV", {ds, 2560});
    add_bf16("decoder_hidden", {ds, 4096});
    add_bf16("decoder_gate_merged", {ds, 8192});
    add_bf16("decoder_gate_buf", {ds, 4096});
    add_bf16("x_normed_buf", {ds, 1024});
    add_bf16("gate_buf", {ds, 1024});

    if (fp8) {
        const std::uint64_t scratch_elements = std::max({
            vs * 4304, es * 16384, ds * 4096});
        requirements.add_buffer(
            "rtx_fp8_scratch", {scratch_elements},
            modalities::DType::kUInt8);
        requirements.add_buffer(
            "rtx_fp8_vision_scales", {109},
            modalities::DType::kFloat32);
        requirements.add_buffer(
            "rtx_fp8_encoder_scales", {18 * 4},
            modalities::DType::kFloat32);
        requirements.add_buffer(
            "rtx_fp8_decoder_scales", {steps * 18 * 4},
            modalities::DType::kFloat32);
    }
    return requirements;
}

Sm120LoweredPlan::Sm120LoweredPlan(
    frt_ctx ctx,
    const Pi05PipelineConfig& config,
    NativeRtxLinearMode linear_mode)
    : Pi05Pipeline(ctx, config),
      weights_(ctx),
      workspace_(ctx),
      attention_(ctx),
      linear_(&driver_, linear_mode),
      forward_(&driver_, &linear_) {}

Sm120LoweredPlan::~Sm120LoweredPlan() = default;

std::unique_ptr<Sm120LoweredPlan> Sm120LoweredPlan::create(
    const std::string& checkpoint_path,
    const Pi05PipelineConfig& config,
    modalities::Status* status) {
    if (config.precision != Pi05Precision::kBf16 ||
        !valid_sm120_config(config)) {
        if (status) *status = invalid("native graph configuration is invalid");
        return nullptr;
    }
    frt_ctx ctx = frt_ctx_create();
    if (!ctx) {
        if (status) *status = backend("native graph context creation failed");
        return nullptr;
    }
    std::unique_ptr<Sm120LoweredPlan> session(
        new (std::nothrow) Sm120LoweredPlan(
            ctx, config, NativeRtxLinearMode::kBf16));
    if (!session) {
        frt_ctx_destroy(ctx);
        if (status) *status = backend("SM120 lowered plan allocation failed");
        return nullptr;
    }
    modalities::Status st = session->initialize(checkpoint_path, nullptr);
    if (!st.ok_status()) {
        if (status) *status = st;
        return nullptr;
    }
    if (status) *status = modalities::Status::ok();
    return session;
}

std::unique_ptr<Sm120LoweredPlan> Sm120LoweredPlan::create(
    const std::string& checkpoint_path,
    const Pi05PipelineConfig& config,
    const NativeCalibrationArtifact& calibration,
    modalities::Status* status) {
    if (config.precision != Pi05Precision::kFp8E4M3 ||
        !valid_sm120_config(config)) {
        if (status) *status = invalid("native RTX FP8 graph precision is invalid");
        return nullptr;
    }
    frt_ctx ctx = frt_ctx_create();
    if (!ctx) {
        if (status) *status = backend("native graph context creation failed");
        return nullptr;
    }
    std::unique_ptr<Sm120LoweredPlan> session(
        new (std::nothrow) Sm120LoweredPlan(
            ctx, config, NativeRtxLinearMode::kFp8Static));
    if (!session) {
        frt_ctx_destroy(ctx);
        if (status) *status = backend("SM120 lowered plan allocation failed");
        return nullptr;
    }
    modalities::Status st = session->initialize(checkpoint_path, &calibration);
    if (!st.ok_status()) {
        if (status) *status = st;
        return nullptr;
    }
    if (status) *status = modalities::Status::ok();
    return session;
}

std::unique_ptr<Sm120LoweredPlan> Sm120LoweredPlan::create_calibration(
    const std::string& checkpoint_path,
    const Pi05PipelineConfig& config,
    modalities::Status* status) {
    if (config.precision != Pi05Precision::kFp8E4M3 ||
        !valid_sm120_config(config)) {
        if (status) {
            *status = invalid("native RTX calibration precision is invalid");
        }
        return nullptr;
    }
    frt_ctx ctx = frt_ctx_create();
    if (!ctx) {
        if (status) *status = backend("native graph context creation failed");
        return nullptr;
    }
    std::unique_ptr<Sm120LoweredPlan> session(
        new (std::nothrow) Sm120LoweredPlan(
            ctx, config, NativeRtxLinearMode::kFp8Dynamic));
    if (!session) {
        frt_ctx_destroy(ctx);
        if (status) *status = backend("SM120 lowered plan allocation failed");
        return nullptr;
    }
    modalities::Status st = session->initialize(checkpoint_path, nullptr);
    if (!st.ok_status()) {
        if (status) *status = st;
        return nullptr;
    }
    if (status) *status = modalities::Status::ok();
    return session;
}

modalities::Status Sm120LoweredPlan::initialize(
    const std::string& checkpoint_path,
    const NativeCalibrationArtifact* calibration) {
    const bool fp8 = linear_.fp8();
    if ((calibration != nullptr) != linear_.static_fp8() ||
        (calibration && calibration->activation_dtype != "bfloat16")) {
        return invalid("native RTX FP8 calibration is incompatible");
    }
    const bool profile_setup = std::getenv("FLASHRT_PROFILE_NATIVE_SETUP");
    const auto setup_begin = std::chrono::steady_clock::now();
    auto checkpoint = setup_begin;
    const auto report = [&](const char* phase) {
        const auto now = std::chrono::steady_clock::now();
        if (profile_setup) {
            std::fprintf(stderr, "native_setup %s_ms=%.3f\n", phase,
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
    NativeWeightMaterializer materializer(source, &weights_);
    NativeMaterializationOptions options;
    options.num_steps = config().num_steps;
    options.merge_decoder_gate_up = fp8;
    options.include_embedding = true;
    modalities::Status st = materializer.materialize_all(options);
    if (!st.ok_status()) return st;
    report("materialize");
    if (fp8) {
        NativeRtxWeightPacker packer(&weights_, &driver_);
        st = packer.pack_all();
        if (!st.ok_status()) return st;
        report("fp8_pack");
    }

    NativeWorkspaceConfig workspace_config;
    workspace_config.num_views = config().num_views;
    workspace_config.max_prompt_tokens = config().max_prompt_tokens;
    workspace_config.chunk_size = config().chunk_size;
    workspace_config.num_steps = config().num_steps;
    workspace_config.vision_pool_factor = config().vision_pool_factor;
    NativeWorkspaceRequirements workspace_requirements =
        make_sm120_workspace_requirements(workspace_config, fp8);
    st = workspace_.allocate(workspace_config, workspace_requirements);
    if (!st.ok_status()) return st;
    if (linear_.static_fp8()) {
        st = upload_scales(
            &workspace_, "rtx_fp8_vision_scales",
            calibration->vision_scales);
        if (!st.ok_status()) return st;
        st = upload_scales(
            &workspace_, "rtx_fp8_encoder_scales",
            calibration->encoder_scales);
        if (!st.ok_status()) return st;
        st = upload_scales(
            &workspace_, "rtx_fp8_decoder_scales",
            calibration->decoder_scales);
        if (!st.ok_status()) return st;
    } else if (linear_.dynamic_fp8()) {
        st = upload_scales(
            &workspace_, "rtx_fp8_vision_scales",
            std::vector<float>(109, 1.0f));
        if (!st.ok_status()) return st;
        st = upload_scales(
            &workspace_, "rtx_fp8_encoder_scales",
            std::vector<float>(18 * 4, 1.0f));
        if (!st.ok_status()) return st;
        st = upload_scales(
            &workspace_, "rtx_fp8_decoder_scales",
            std::vector<float>(
                static_cast<std::size_t>(config().num_steps) * 18 * 4,
                1.0f));
        if (!st.ok_status()) return st;
    }
    st = workspace_.expand_vision_position_embedding(weights_);
    if (!st.ok_status()) return st;

    NativeRtxAttentionConfig attention_config;
    attention_config.num_views = config().num_views;
    attention_config.encoder_sequence = workspace_.encoder_sequence();
    attention_config.encoder_vision_sequence =
        workspace_.encoder_vision_sequence();
    attention_config.chunk_size = config().chunk_size;
    st = attention_.allocate(attention_config);
    if (!st.ok_status()) return st;
    st = set_prompt_length(0);
    if (!st.ok_status()) return st;

    NativeStylePrecomputer precomputer(&driver_);
    st = precomputer.run(weights_, &workspace_, 0);
    if (!st.ok_status()) return st;
    attention_driver_.reset(new (std::nothrow)
                                NativeRtxAttentionDriver(&attention_));
    if (!attention_driver_) {
        return backend("native attention driver allocation failed");
    }
    st = attention_driver_->status();
    if (!st.ok_status()) return st;
    if (fp8) {
        st = autotune_native_rtx_fp8(
            weights_, &workspace_, linear_, config().num_views,
            config().chunk_size);
        if (!st.ok_status()) return st;
        report("fp8_autotune");
    }
    report("workspace_style");

    st = resolve_pipeline_artifacts(
        workspace_, weights_, NativeWeightDType::kBf16, &artifacts_);
    if (!st.ok_status()) return st;

    st = finish_prepare(false);
    if (!st.ok_status()) return st;
    report("graph_prepare");
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "native_setup total_ms=%.3f\n",
                     std::chrono::duration<double, std::milli>(
                         now - setup_begin).count());
    }
    return modalities::Status::ok();
}

modalities::Status Sm120LoweredPlan::record_vision(void* stream) {
    return forward_.vision(
        weights_, &workspace_, &attention_, attention_driver_.get(),
        reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status Sm120LoweredPlan::record_encoder(void* stream) {
    return forward_.encoder(weights_, &workspace_, &attention_,
                            attention_driver_.get(),
                            reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status Sm120LoweredPlan::record_diffusion_step(
    int step,
    void* stream) {
    return forward_.diffusion_step(
        step, weights_, &workspace_, &attention_, attention_driver_.get(),
        reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status Sm120LoweredPlan::set_prompt_length(int prompt_tokens) {
    modalities::Status st = attention_.set_fixed_prompt_length(prompt_tokens);
    if (!st.ok_status()) return st;
    return workspace_.update_decoder_rope(prompt_tokens);
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

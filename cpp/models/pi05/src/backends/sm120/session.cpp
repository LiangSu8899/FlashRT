#include "flashrt/cpp/models/pi05/backends/sm120/session.h"

#include "flashrt/cpp/models/pi05/backends/sm120/native_style_precompute.h"
#include "flashrt/cpp/models/pi05/support/native_weight_materializer.h"
#include "flashrt/cpp/models/pi05/backends/sm120/native_rtx_autotune.h"
#include "flashrt/cpp/models/pi05/backends/sm120/native_rtx_weight_packer.h"

#include <cuda_runtime_api.h>

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

Sm120BackendSession::Sm120BackendSession(
    frt_ctx ctx,
    const BackendConfig& config,
    NativeRtxLinearMode linear_mode)
    : graphs_(ctx, static_cast<std::size_t>(GraphKind::kCount)),
      config_(config),
      weights_(ctx),
      workspace_(ctx),
      attention_(ctx),
      linear_(&driver_, linear_mode),
      forward_(&driver_, &linear_) {}

Sm120BackendSession::~Sm120BackendSession() = default;

std::unique_ptr<Sm120BackendSession> Sm120BackendSession::create(
    const std::string& checkpoint_path,
    const BackendConfig& config,
    modalities::Status* status) {
    if (config.precision != BackendPrecision::kBf16 ||
        config.num_views < 1 || config.num_views > 3 ||
        config.max_prompt_tokens < 1 || config.chunk_size < 1 ||
        config.num_steps < 1 ||
        static_cast<std::uint64_t>(config.max_prompt_tokens) +
                static_cast<std::uint64_t>(config.chunk_size) +
                static_cast<std::uint64_t>(config.num_views) * 256 >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        (config.vision_pool_factor != 1 &&
         config.vision_pool_factor != 2 && config.vision_pool_factor != 4)) {
        if (status) *status = invalid("native graph configuration is invalid");
        return nullptr;
    }
    frt_ctx ctx = frt_ctx_create();
    if (!ctx) {
        if (status) *status = backend("native graph context creation failed");
        return nullptr;
    }
    std::unique_ptr<Sm120BackendSession> session(
        new (std::nothrow) Sm120BackendSession(
            ctx, config, NativeRtxLinearMode::kBf16));
    if (!session) {
        frt_ctx_destroy(ctx);
        if (status) *status = backend("SM120 backend session allocation failed");
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

std::unique_ptr<Sm120BackendSession> Sm120BackendSession::create(
    const std::string& checkpoint_path,
    const BackendConfig& config,
    const NativeCalibrationArtifact& calibration,
    modalities::Status* status) {
    if (config.precision != BackendPrecision::kFp8E4M3) {
        if (status) *status = invalid("native RTX FP8 graph precision is invalid");
        return nullptr;
    }
    frt_ctx ctx = frt_ctx_create();
    if (!ctx) {
        if (status) *status = backend("native graph context creation failed");
        return nullptr;
    }
    std::unique_ptr<Sm120BackendSession> session(
        new (std::nothrow) Sm120BackendSession(
            ctx, config, NativeRtxLinearMode::kFp8Static));
    if (!session) {
        frt_ctx_destroy(ctx);
        if (status) *status = backend("SM120 backend session allocation failed");
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

std::unique_ptr<Sm120BackendSession> Sm120BackendSession::create_calibration(
    const std::string& checkpoint_path,
    const BackendConfig& config,
    modalities::Status* status) {
    if (config.precision != BackendPrecision::kFp8E4M3) {
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
    std::unique_ptr<Sm120BackendSession> session(
        new (std::nothrow) Sm120BackendSession(
            ctx, config, NativeRtxLinearMode::kFp8Dynamic));
    if (!session) {
        frt_ctx_destroy(ctx);
        if (status) *status = backend("SM120 backend session allocation failed");
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

modalities::Status Sm120BackendSession::initialize(
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
    options.num_steps = config_.num_steps;
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
    workspace_config.num_views = config_.num_views;
    workspace_config.max_prompt_tokens = config_.max_prompt_tokens;
    workspace_config.chunk_size = config_.chunk_size;
    workspace_config.num_steps = config_.num_steps;
    workspace_config.vision_pool_factor = config_.vision_pool_factor;
    workspace_config.flavor = fp8 ? NativeWorkspaceFlavor::kRtxFp8
                                  : NativeWorkspaceFlavor::kBf16;
    st = workspace_.allocate(workspace_config);
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
                static_cast<std::size_t>(config_.num_steps) * 18 * 4,
                1.0f));
        if (!st.ok_status()) return st;
    }
    st = workspace_.expand_vision_position_embedding(weights_);
    if (!st.ok_status()) return st;

    NativeRtxAttentionConfig attention_config;
    attention_config.num_views = config_.num_views;
    attention_config.encoder_sequence = workspace_.encoder_sequence();
    attention_config.encoder_vision_sequence =
        workspace_.encoder_vision_sequence();
    attention_config.chunk_size = config_.chunk_size;
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
            weights_, &workspace_, linear_, config_.num_views,
            config_.chunk_size);
        if (!st.ok_status()) return st;
        report("fp8_autotune");
    }
    report("workspace_style");

    st = resolve_backend_artifacts(
        workspace_, weights_, NativeWeightDType::kBf16, &artifacts_);
    if (!st.ok_status()) return st;

    for (const char* name : {"observation_images_normalized",
                             "prompt_embedding", "encoder_x",
                             "diffusion_noise"}) {
        const NativeWorkspaceBuffer* buffer = workspace_.find(name);
        if (!buffer ||
            cudaMemset(frt_buffer_dptr(buffer->buffer), 0,
                       frt_buffer_bytes(buffer->buffer)) != cudaSuccess) {
            return backend("native graph input initialization failed");
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return backend("native graph setup synchronization failed");
    }
    report("input_init");

    st = capture_backend_graph(
        &graphs_,
        GraphKind::kInfer, workspace_,
        {"observation_images_normalized", "prompt_embedding", "encoder_x",
         "diffusion_noise", "rtc_prev_action_chunk", "rtc_prefix_weights",
         "rtc_guidance_weight"},
        record_graph, this);
    if (!st.ok_status()) return st;
    st = capture_backend_graph(
        &graphs_,
        GraphKind::kDecodeOnly, workspace_,
        {"encoder_x", "diffusion_noise", "rtc_prev_action_chunk",
         "rtc_prefix_weights", "rtc_guidance_weight"},
        record_graph, this);
    if (!st.ok_status()) return st;
    st = capture_backend_graph(
        &graphs_,
        GraphKind::kContext, workspace_,
        {"observation_images_normalized", "prompt_embedding", "encoder_x"},
        record_graph, this);
    if (!st.ok_status()) return st;
    report("capture");

    st = graphs_.create_replay_stream();
    if (!st.ok_status()) return st;
    report("stream");
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "native_setup total_ms=%.3f\n",
                     std::chrono::duration<double, std::milli>(
                         now - setup_begin).count());
    }
    return modalities::Status::ok();
}

modalities::Status Sm120BackendSession::record_context(void* stream) {
    modalities::Status st = copy_prompt_to_encoder(&workspace_, stream);
    if (!st.ok_status()) return st;
    st = forward_.vision(
        weights_, &workspace_, &attention_, attention_driver_.get(),
        reinterpret_cast<std::uintptr_t>(stream));
    if (!st.ok_status()) return st;
    st = forward_.encoder(weights_, &workspace_, &attention_,
                          attention_driver_.get(),
                          reinterpret_cast<std::uintptr_t>(stream));
    if (!st.ok_status()) return st;
    return st;
}

modalities::Status Sm120BackendSession::record_action(void* stream) {
    return forward_.diffusion(weights_, &workspace_, &attention_,
                              attention_driver_.get(),
                              reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status Sm120BackendSession::record(GraphKind kind,
                                             void* stream) {
    if (kind == GraphKind::kContext) return record_context(stream);
    if (kind == GraphKind::kDecodeOnly) return record_action(stream);
    if (kind != GraphKind::kInfer) {
        return invalid("native graph kind is invalid");
    }
    modalities::Status st = record_context(stream);
    return st.ok_status() ? record_action(stream) : st;
}

modalities::Status Sm120BackendSession::record_graph(
    void* user, std::size_t slot, void* stream) {
    auto* session = static_cast<Sm120BackendSession*>(user);
    return session->record(static_cast<GraphKind>(slot), stream);
}

modalities::Status Sm120BackendSession::set_prompt_length(int prompt_tokens) {
    modalities::Status st = attention_.set_fixed_prompt_length(prompt_tokens);
    if (!st.ok_status()) return st;
    return workspace_.update_decoder_rope(prompt_tokens);
}

int Sm120BackendSession::replay(GraphKind kind) const {
    return graphs_.replay(static_cast<std::size_t>(kind));
}

modalities::Status Sm120BackendSession::synchronize() const {
    return graphs_.synchronize();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

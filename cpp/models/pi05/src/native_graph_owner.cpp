#include "flashrt/cpp/models/pi05/native_graph_owner.h"

#include "flashrt/cpp/models/pi05/native_style_precompute.h"
#include "flashrt/cpp/models/pi05/native_weight_materializer.h"
#include "flashrt/cpp/models/pi05/native_rtx_weight_packer.h"

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

NativeGraphOwner::NativeGraphOwner(
    frt_ctx ctx,
    const NativeGraphConfig& config)
    : graphs_(ctx),
      config_(config),
      weights_(ctx),
      workspace_(ctx),
      attention_(ctx),
      linear_(&driver_, config.precision == NativeGraphPrecision::kFp8E4M3
                            ? NativeRtxLinearMode::kFp8Static
                            : NativeRtxLinearMode::kBf16),
      forward_(&driver_, &linear_) {}

NativeGraphOwner::~NativeGraphOwner() = default;

std::unique_ptr<NativeGraphOwner> NativeGraphOwner::create(
    const std::string& checkpoint_path,
    const NativeGraphConfig& config,
    modalities::Status* status) {
    if (config.num_views < 1 || config.num_views > 3 ||
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
    std::unique_ptr<NativeGraphOwner> owner(
        new (std::nothrow) NativeGraphOwner(ctx, config));
    if (!owner) {
        frt_ctx_destroy(ctx);
        if (status) *status = backend("native graph owner allocation failed");
        return nullptr;
    }
    modalities::Status st = owner->initialize(checkpoint_path, nullptr);
    if (!st.ok_status()) {
        if (status) *status = st;
        return nullptr;
    }
    if (status) *status = modalities::Status::ok();
    return owner;
}

std::unique_ptr<NativeGraphOwner> NativeGraphOwner::create(
    const std::string& checkpoint_path,
    const NativeGraphConfig& config,
    const NativeCalibrationArtifact& calibration,
    modalities::Status* status) {
    if (config.precision != NativeGraphPrecision::kFp8E4M3) {
        if (status) *status = invalid("native RTX FP8 graph precision is invalid");
        return nullptr;
    }
    frt_ctx ctx = frt_ctx_create();
    if (!ctx) {
        if (status) *status = backend("native graph context creation failed");
        return nullptr;
    }
    std::unique_ptr<NativeGraphOwner> owner(
        new (std::nothrow) NativeGraphOwner(ctx, config));
    if (!owner) {
        frt_ctx_destroy(ctx);
        if (status) *status = backend("native graph owner allocation failed");
        return nullptr;
    }
    modalities::Status st = owner->initialize(checkpoint_path, &calibration);
    if (!st.ok_status()) {
        if (status) *status = st;
        return nullptr;
    }
    if (status) *status = modalities::Status::ok();
    return owner;
}

modalities::Status NativeGraphOwner::initialize(
    const std::string& checkpoint_path,
    const NativeCalibrationArtifact* calibration) {
    const bool fp8 = config_.precision == NativeGraphPrecision::kFp8E4M3;
    if (fp8 != (calibration != nullptr) ||
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
    if (fp8) {
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
        st = autotune_fp8();
        if (!st.ok_status()) return st;
        report("fp8_autotune");
    }
    report("workspace_style");

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

    st = graphs_.capture(
        NativeGraphKind::kInfer, workspace_,
        {"observation_images_normalized", "prompt_embedding", "encoder_x",
         "diffusion_noise", "rtc_prev_action_chunk", "rtc_prefix_weights",
         "rtc_guidance_weight"},
        record_graph, this);
    if (!st.ok_status()) return st;
    st = graphs_.capture(
        NativeGraphKind::kDecodeOnly, workspace_,
        {"encoder_x", "diffusion_noise", "rtc_prev_action_chunk",
         "rtc_prefix_weights", "rtc_guidance_weight"},
        record_graph, this);
    if (!st.ok_status()) return st;
    st = graphs_.capture(
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
        std::fprintf(stderr, "native_setup total_ms=%.3f\n",
                     std::chrono::duration<double, std::milli>(
                         now - setup_begin).count());
    }
    return modalities::Status::ok();
}

modalities::Status NativeGraphOwner::autotune_fp8() {
    const int vision_sequence = config_.num_views * 256;
    const int encoder_vision_sequence = workspace_.encoder_vision_sequence();
    const int encoder_sequence = workspace_.encoder_sequence();
    const int decoder_sequence = config_.chunk_size;
    struct Shape {
        const char* weight;
        NativeRtxScaleSite site;
        const char* output;
        int m;
        int n;
        int k;
    };
    const Shape shapes[] = {
        {"vision_attn_qkv_w_0", {NativeRtxScaleDomain::kVision, 0},
         "vision_QKV",
         vision_sequence, 3456, 1152},
        {"vision_attn_o_w_0", {NativeRtxScaleDomain::kVision, 1},
         "vision_x_norm",
         vision_sequence, 1152, 1152},
        {"vision_ffn_up_w_0", {NativeRtxScaleDomain::kVision, 2},
         "vision_hidden",
         vision_sequence, 4304, 1152},
        {"vision_ffn_down_w_0", {NativeRtxScaleDomain::kVision, 3},
         "vision_x_norm",
         vision_sequence, 1152, 4304},
        {"encoder_multi_modal_projector_w",
         {NativeRtxScaleDomain::kVision, 108}, "encoder_x",
         encoder_vision_sequence, 2048, 1152},
        {"encoder_attn_qkv_w_0", {NativeRtxScaleDomain::kEncoder, 0},
         "encoder_QKV",
         encoder_sequence, 2560, 2048},
        {"encoder_attn_o_w_0", {NativeRtxScaleDomain::kEncoder, 1},
         "encoder_x_norm",
         encoder_sequence, 2048, 2048},
        {"encoder_ffn_gate_up_w_0", {NativeRtxScaleDomain::kEncoder, 2},
         "encoder_gate_merged", encoder_sequence, 32768, 2048},
        {"encoder_ffn_down_w_0", {NativeRtxScaleDomain::kEncoder, 3},
         "encoder_x_norm",
         encoder_sequence, 2048, 16384},
        {"decoder_attn_qkv_w_0", {NativeRtxScaleDomain::kDecoder, 0},
         "decoder_QKV",
         decoder_sequence, 2560, 1024},
        {"decoder_attn_o_w_0", {NativeRtxScaleDomain::kDecoder, 1},
         "x_normed_buf",
         decoder_sequence, 1024, 2048},
        {"decoder_ffn_gate_up_w_0", {NativeRtxScaleDomain::kDecoder, 2},
         "decoder_gate_merged", decoder_sequence, 8192, 1024},
        {"decoder_ffn_down_w_0", {NativeRtxScaleDomain::kDecoder, 3},
         "x_normed_buf",
         decoder_sequence, 1024, 4096},
    };
    for (const Shape& shape : shapes) {
        const NativeWorkspaceBuffer* output = workspace_.find(shape.output);
        if (!output) return invalid("native FP8 autotune output is missing");
        modalities::Status st = linear_.autotune(
            weights_, &workspace_, shape.weight, shape.site,
            frt_buffer_dptr(output->buffer), shape.m, shape.n, shape.k);
        if (!st.ok_status()) return st;
    }
    return modalities::Status::ok();
}

modalities::Status NativeGraphOwner::record_context(void* stream) {
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

modalities::Status NativeGraphOwner::record_action(void* stream) {
    return forward_.diffusion(weights_, &workspace_, &attention_,
                              attention_driver_.get(),
                              reinterpret_cast<std::uintptr_t>(stream));
}

modalities::Status NativeGraphOwner::record(NativeGraphKind kind,
                                             void* stream) {
    if (kind == NativeGraphKind::kContext) return record_context(stream);
    if (kind == NativeGraphKind::kDecodeOnly) return record_action(stream);
    if (kind != NativeGraphKind::kInfer) {
        return invalid("native graph kind is invalid");
    }
    modalities::Status st = record_context(stream);
    return st.ok_status() ? record_action(stream) : st;
}

modalities::Status NativeGraphOwner::record_graph(
    void* user, NativeGraphKind kind, void* stream) {
    auto* owner = static_cast<NativeGraphOwner*>(user);
    return owner->record(kind, stream);
}

modalities::Status NativeGraphOwner::set_prompt_length(int prompt_tokens) {
    modalities::Status st = attention_.set_fixed_prompt_length(prompt_tokens);
    if (!st.ok_status()) return st;
    return workspace_.update_decoder_rope(prompt_tokens);
}

int NativeGraphOwner::replay(NativeGraphKind kind) const {
    return graphs_.replay(kind);
}

modalities::Status NativeGraphOwner::synchronize() const {
    return graphs_.synchronize();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

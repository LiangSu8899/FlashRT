#include "flashrt/cpp/models/pi05/native_graph_owner.h"

#include "flashrt/cpp/models/pi05/native_style_precompute.h"
#include "flashrt/cpp/models/pi05/native_weight_materializer.h"

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

}  // namespace

NativeGraphOwner::NativeGraphOwner(
    frt_ctx ctx,
    const NativeGraphConfig& config)
    : graphs_(ctx),
      config_(config),
      weights_(ctx),
      workspace_(ctx),
      attention_(ctx),
      forward_(&driver_) {}

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
    modalities::Status st = owner->initialize(checkpoint_path);
    if (!st.ok_status()) {
        if (status) *status = st;
        return nullptr;
    }
    if (status) *status = modalities::Status::ok();
    return owner;
}

modalities::Status NativeGraphOwner::initialize(
    const std::string& checkpoint_path) {
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
    options.merge_decoder_gate_up = false;
    options.include_embedding = true;
    modalities::Status st = materializer.materialize_all(options);
    if (!st.ok_status()) return st;
    report("materialize");

    NativeWorkspaceConfig workspace_config;
    workspace_config.num_views = config_.num_views;
    workspace_config.max_prompt_tokens = config_.max_prompt_tokens;
    workspace_config.chunk_size = config_.chunk_size;
    workspace_config.num_steps = config_.num_steps;
    workspace_config.vision_pool_factor = config_.vision_pool_factor;
    st = workspace_.allocate(workspace_config);
    if (!st.ok_status()) return st;
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
    report("workspace_style");

    for (const char* name : {"observation_images_normalized",
                             "prompt_embedding", "diffusion_noise"}) {
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

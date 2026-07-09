#include "flashrt/cpp/models/pi05/native_graph_owner.h"

#include "flashrt/cpp/models/pi05/native_style_precompute.h"
#include "flashrt/cpp/models/pi05/native_weight_materializer.h"

#include <cuda_runtime_api.h>

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
    : ctx_(ctx),
      config_(config),
      weights_(ctx),
      workspace_(ctx),
      attention_(ctx),
      forward_(&driver_),
      capture_status_(modalities::Status::ok()) {}

NativeGraphOwner::~NativeGraphOwner() {
    if (replay_stream_) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(replay_stream_));
        cudaStreamDestroy(static_cast<cudaStream_t>(replay_stream_));
        replay_stream_ = nullptr;
    }
    if (ctx_) {
        frt_ctx_destroy(ctx_);
        ctx_ = nullptr;
    }
}

std::unique_ptr<NativeGraphOwner> NativeGraphOwner::create(
    const std::string& checkpoint_path,
    const NativeGraphConfig& config,
    modalities::Status* status) {
    if (config.num_views < 1 || config.num_views > 3 ||
        config.max_prompt_tokens < 1 || config.chunk_size < 1 ||
        config.num_steps < 1 ||
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
    loader::SafetensorsFile source;
    if (!source.open(checkpoint_path + "/model.safetensors")) {
        return modalities::Status::error(modalities::StatusCode::kNotFound,
                                         source.error());
    }
    NativeWeightMaterializer materializer(source, &weights_);
    NativeMaterializationOptions options;
    options.num_steps = config_.num_steps;
    options.merge_decoder_gate_up = false;
    options.include_embedding = true;
    modalities::Status st = materializer.materialize_all(options);
    if (!st.ok_status()) return st;

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

    infer_graph_ = frt_graph_create(ctx_, "infer", 1);
    const NativeWorkspaceBuffer* images =
        workspace_.find("observation_images_normalized");
    const NativeWorkspaceBuffer* prompt = workspace_.find("prompt_embedding");
    const NativeWorkspaceBuffer* encoder = workspace_.find("encoder_x");
    const NativeWorkspaceBuffer* noise = workspace_.find("diffusion_noise");
    if (!infer_graph_ || !images || !prompt || !encoder || !noise ||
        frt_graph_bind(infer_graph_, "images", images->buffer) != FRT_OK ||
        frt_graph_bind(infer_graph_, "prompt", prompt->buffer) != FRT_OK ||
        frt_graph_bind(infer_graph_, "encoder_x", encoder->buffer) != FRT_OK ||
        frt_graph_bind(infer_graph_, "noise", noise->buffer) != FRT_OK) {
        return backend("native graph binding failed");
    }
    capture_status_ = modalities::Status::ok();
    if (frt_graph_capture(infer_graph_, 0, record_graph, this) != FRT_OK) {
        return capture_status_.ok_status()
                   ? backend("native full graph capture failed")
                   : capture_status_;
    }
    if (!capture_status_.ok_status() ||
        frt_graph_variant_count(infer_graph_) != 1) {
        return capture_status_.ok_status()
                   ? backend("native full graph variant is missing")
                   : capture_status_;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        return backend("native replay stream creation failed");
    }
    replay_stream_ = stream;
    stream_id_ = frt_ctx_wrap_stream(ctx_, replay_stream_);
    if (stream_id_ < 0) return backend("native replay stream wrapping failed");
    return modalities::Status::ok();
}

modalities::Status NativeGraphOwner::record(void* stream) {
    const NativeWorkspaceBuffer* prompt = workspace_.find("prompt_embedding");
    const NativeWorkspaceBuffer* encoder = workspace_.find("encoder_x");
    if (!prompt || !encoder) return invalid("native prompt buffers are missing");
    const std::size_t prompt_bytes = frt_buffer_bytes(prompt->buffer);
    const std::size_t prompt_offset =
        static_cast<std::size_t>(workspace_.encoder_vision_sequence()) * 2048 *
        sizeof(std::uint16_t);
    if (prompt_offset > frt_buffer_bytes(encoder->buffer) ||
        prompt_bytes > frt_buffer_bytes(encoder->buffer) - prompt_offset) {
        return invalid("native prompt window exceeds encoder storage");
    }
    auto* destination =
        static_cast<unsigned char*>(frt_buffer_dptr(encoder->buffer)) +
        prompt_offset;
    if (cudaMemcpyAsync(destination, frt_buffer_dptr(prompt->buffer),
                        prompt_bytes, cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream)) != cudaSuccess) {
        return backend("native prompt graph copy failed");
    }
    modalities::Status st = forward_.vision(
        weights_, &workspace_, &attention_, attention_driver_.get(),
        reinterpret_cast<std::uintptr_t>(stream));
    if (!st.ok_status()) return st;
    st = forward_.encoder(weights_, &workspace_, &attention_,
                          attention_driver_.get(),
                          reinterpret_cast<std::uintptr_t>(stream));
    if (!st.ok_status()) return st;
    return forward_.diffusion(weights_, &workspace_, &attention_,
                              attention_driver_.get(),
                              reinterpret_cast<std::uintptr_t>(stream));
}

void NativeGraphOwner::record_graph(void* user, void* stream) {
    auto* owner = static_cast<NativeGraphOwner*>(user);
    owner->capture_status_ = owner->record(stream);
}

modalities::Status NativeGraphOwner::set_prompt_length(int prompt_tokens) {
    modalities::Status st = attention_.set_fixed_prompt_length(prompt_tokens);
    if (!st.ok_status()) return st;
    return workspace_.update_decoder_rope(prompt_tokens);
}

int NativeGraphOwner::replay() const {
    if (!infer_graph_ || stream_id_ < 0) return FRT_ERR_INVALID;
    return frt_graph_replay(infer_graph_, 0, stream_id_);
}

modalities::Status NativeGraphOwner::synchronize() const {
    if (!replay_stream_) return invalid("native replay stream is missing");
    const cudaError_t rc =
        cudaStreamSynchronize(static_cast<cudaStream_t>(replay_stream_));
    return rc == cudaSuccess ? modalities::Status::ok()
                             : backend(cudaGetErrorString(rc));
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

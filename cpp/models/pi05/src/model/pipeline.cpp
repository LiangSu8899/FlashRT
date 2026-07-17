#include "flashrt/cpp/models/pi05/model/pipeline.h"

#include "flashrt/cpp/models/pi05/model/spec.h"

#include <cuda_runtime_api.h>

#include <cstdint>

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

Pi05Pipeline::Pi05Pipeline(frt_ctx context, const Pi05PipelineConfig& config)
    : graphs_(context, static_cast<std::size_t>(GraphKind::kCount)),
      config_(config) {}

const char* pipeline_graph_name(GraphKind kind) {
    switch (kind) {
        case GraphKind::kInfer: return "infer";
        case GraphKind::kDecodeOnly: return "decode_only";
        case GraphKind::kContext: return "context";
        case GraphKind::kCount: break;
    }
    return nullptr;
}

modalities::Status capture_pipeline_graph(
    native::CudaGraphSet* graphs,
    GraphKind kind,
    const NativeWorkspace& workspace,
    std::initializer_list<const char*> bindings,
    native::CudaGraphSet::RecordFn record,
    void* owner) {
    if (!graphs || !pipeline_graph_name(kind)) {
        return invalid("native graph capture request is invalid");
    }
    std::vector<native::CudaGraphBinding> resolved;
    resolved.reserve(bindings.size());
    for (const char* binding : bindings) {
        const NativeWorkspaceBuffer* buffer = workspace.find(binding);
        if (!buffer) {
            return backend("native graph binding failed");
        }
        resolved.push_back({binding, buffer->buffer});
    }
    return graphs->capture(static_cast<std::size_t>(kind),
                           pipeline_graph_name(kind), resolved, record, owner);
}

modalities::Status copy_prompt_to_encoder(NativeWorkspace* workspace,
                                          void* stream) {
    if (!workspace) return invalid("native workspace is missing");
    const NativeWorkspaceBuffer* prompt = workspace->find("prompt_embedding");
    const NativeWorkspaceBuffer* encoder = workspace->find("encoder_x");
    if (!prompt || !encoder) return invalid("native prompt buffers are missing");
    const std::size_t prompt_bytes = frt_buffer_bytes(prompt->buffer);
    const std::size_t prompt_offset =
        static_cast<std::size_t>(workspace->encoder_vision_sequence()) *
        kEncoderWidth * sizeof(std::uint16_t);
    if (prompt_offset > frt_buffer_bytes(encoder->buffer) ||
        prompt_bytes > frt_buffer_bytes(encoder->buffer) - prompt_offset) {
        return invalid("native prompt window exceeds encoder storage");
    }
    auto* destination =
        static_cast<unsigned char*>(frt_buffer_dptr(encoder->buffer)) +
        prompt_offset;
    const cudaError_t rc = cudaMemcpyAsync(
        destination, frt_buffer_dptr(prompt->buffer), prompt_bytes,
        cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream));
    return rc == cudaSuccess
               ? modalities::Status::ok()
               : backend("native prompt graph copy failed");
}

modalities::Status resolve_pipeline_artifacts(
    const NativeWorkspace& workspace,
    const NativeDeviceWeightStore& weights,
    NativeWeightDType embedding_dtype,
    PipelineArtifacts* artifacts) {
    if (!artifacts) {
        return invalid("native runtime artifacts destination is null");
    }
    PipelineArtifacts result;
    result.images = workspace.find("observation_images_normalized");
    result.noise = workspace.find("diffusion_noise");
    result.encoder = workspace.find("encoder_x");
    result.previous_actions = workspace.find("rtc_prev_action_chunk");
    result.prefix_weights = workspace.find("rtc_prefix_weights");
    result.guidance_weight = workspace.find("rtc_guidance_weight");
    result.prompt_embedding = workspace.find("prompt_embedding");
    result.embedding_table = weights.find("embedding_weight");
    if (!result.images || !result.noise || !result.encoder ||
        !result.previous_actions || !result.prefix_weights ||
        !result.guidance_weight || !result.prompt_embedding ||
        !result.embedding_table ||
        result.embedding_table->dtype != embedding_dtype ||
        result.embedding_table->shape.size() != 2 ||
        result.embedding_table->shape[1] != kEncoderWidth) {
        return backend("native graph export buffers are incomplete");
    }
    *artifacts = result;
    return modalities::Status::ok();
}
modalities::Status Pi05Pipeline::initialize_capture_inputs() {
    for (const char* name : {"observation_images_normalized",
                             "prompt_embedding", "encoder_x",
                             "diffusion_noise"}) {
        const NativeWorkspaceBuffer* buffer = workspace().find(name);
        if (!buffer ||
            cudaMemset(frt_buffer_dptr(buffer->buffer), 0,
                       frt_buffer_bytes(buffer->buffer)) != cudaSuccess) {
            return backend("Pi0.5 graph input initialization failed");
        }
    }
    return cudaDeviceSynchronize() == cudaSuccess
               ? modalities::Status::ok()
               : backend("Pi0.5 graph setup synchronization failed");
}

modalities::Status record_pi05_context(
    Pi05Operations& operations,
    NativeWorkspace* workspace,
    void* stream) {
    if (!workspace) return invalid("Pi0.5 context workspace is missing");
    modalities::Status st = copy_prompt_to_encoder(workspace, stream);
    if (!st.ok_status()) return st;
    st = operations.record_vision_begin(stream);
    if (!st.ok_status()) return st;
    for (int layer = 0; layer < kVisionLayers; ++layer) {
        st = operations.record_vision_layer(layer, stream);
        if (!st.ok_status()) return st;
    }
    st = operations.record_vision_end(stream);
    if (!st.ok_status()) return st;
    for (int layer = 0; layer < kEncoderLayers; ++layer) {
        st = operations.record_encoder_layer(layer, stream);
        if (!st.ok_status()) return st;
    }
    return modalities::Status::ok();
}

modalities::Status record_pi05_decode(
    Pi05Operations& operations,
    int num_steps,
    void* stream) {
    if (num_steps < 1) return invalid("Pi0.5 decode step count is invalid");
    for (int step = 0; step < num_steps; ++step) {
        modalities::Status st =
            operations.record_diffusion_begin(step, stream);
        if (!st.ok_status()) return st;
        for (int layer = 0; layer < kDecoderLayers; ++layer) {
            st = operations.record_decoder_layer(step, layer, stream);
            if (!st.ok_status()) return st;
        }
        st = operations.record_diffusion_end(step, stream);
        if (!st.ok_status()) return st;
    }
    return modalities::Status::ok();
}

modalities::Status Pi05Pipeline::record_context(void* stream) {
    return record_pi05_context(*this, &workspace(), stream);
}

modalities::Status Pi05Pipeline::record_decode(void* stream) {
    return record_pi05_decode(*this, config_.num_steps, stream);
}

modalities::Status Pi05Pipeline::record(GraphKind kind, void* stream) {
    if (kind == GraphKind::kContext) return record_context(stream);
    if (kind == GraphKind::kDecodeOnly) return record_decode(stream);
    if (kind != GraphKind::kInfer) {
        return invalid("Pi0.5 graph kind is invalid");
    }
    modalities::Status st = record_context(stream);
    return st.ok_status() ? record_decode(stream) : st;
}

modalities::Status Pi05Pipeline::record_graph(
    void* owner, std::size_t slot, void* stream) {
    auto* pipeline = static_cast<Pi05Pipeline*>(owner);
    if (!pipeline || slot >= static_cast<std::size_t>(GraphKind::kCount)) {
        return invalid("Pi0.5 graph record request is invalid");
    }
    return pipeline->record(static_cast<GraphKind>(slot), stream);
}

modalities::Status Pi05Pipeline::finish_prepare(
    bool warmup_before_capture) {
    modalities::Status st = initialize_capture_inputs();
    if (!st.ok_status()) return st;

    if (warmup_before_capture) {
        // Initialize vendor-library tactics before entering CUDA capture.
        st = record(GraphKind::kInfer, nullptr);
        if (!st.ok_status()) return st;
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return backend("Pi0.5 graph warmup synchronization failed");
        }
        const NativeWorkspaceBuffer* noise =
            workspace().find("diffusion_noise");
        if (!noise ||
            cudaMemset(frt_buffer_dptr(noise->buffer), 0,
                       frt_buffer_bytes(noise->buffer)) != cudaSuccess) {
            return backend("Pi0.5 graph warmup reset failed");
        }
    }

    st = capture_pipeline_graph(
        &graphs_, GraphKind::kInfer, workspace(),
        {"observation_images_normalized", "prompt_embedding", "encoder_x",
         "diffusion_noise", "rtc_prev_action_chunk", "rtc_prefix_weights",
         "rtc_guidance_weight"},
        record_graph, this);
    if (!st.ok_status()) return st;
    st = capture_pipeline_graph(
        &graphs_, GraphKind::kDecodeOnly, workspace(),
        {"encoder_x", "diffusion_noise", "rtc_prev_action_chunk",
         "rtc_prefix_weights", "rtc_guidance_weight"},
        record_graph, this);
    if (!st.ok_status()) return st;
    st = capture_pipeline_graph(
        &graphs_, GraphKind::kContext, workspace(),
        {"observation_images_normalized", "prompt_embedding", "encoder_x"},
        record_graph, this);
    if (!st.ok_status()) return st;
    return graphs_.create_replay_stream();
}

int Pi05Pipeline::replay(GraphKind kind) const {
    return graphs_.replay(static_cast<std::size_t>(kind));
}

modalities::Status Pi05Pipeline::synchronize() const {
    return graphs_.synchronize();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

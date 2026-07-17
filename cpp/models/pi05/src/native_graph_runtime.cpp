#include "flashrt/cpp/models/pi05/native_graph_runtime.h"

#include "flashrt/cpp/models/pi05/spec.h"

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

const char* native_graph_name(NativeGraphKind kind) {
    switch (kind) {
        case NativeGraphKind::kInfer: return "infer";
        case NativeGraphKind::kDecodeOnly: return "decode_only";
        case NativeGraphKind::kContext: return "context";
        case NativeGraphKind::kCount: break;
    }
    return nullptr;
}

modalities::Status capture_native_graph(
    native::CudaGraphSet* graphs,
    NativeGraphKind kind,
    const NativeWorkspace& workspace,
    std::initializer_list<const char*> bindings,
    native::CudaGraphSet::RecordFn record,
    void* owner) {
    if (!graphs || !native_graph_name(kind)) {
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
                           native_graph_name(kind), resolved, record, owner);
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

modalities::Status resolve_native_runtime_artifacts(
    const NativeWorkspace& workspace,
    const NativeDeviceWeightStore& weights,
    NativeWeightDType embedding_dtype,
    NativeRuntimeArtifacts* artifacts) {
    if (!artifacts) {
        return invalid("native runtime artifacts destination is null");
    }
    NativeRuntimeArtifacts result;
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

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

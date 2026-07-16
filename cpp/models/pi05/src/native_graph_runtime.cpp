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

std::size_t graph_index(NativeGraphKind kind) {
    return static_cast<std::size_t>(kind);
}

}  // namespace

struct NativeGraphCatalog::CaptureCall {
    RecordFn record = nullptr;
    void* owner = nullptr;
    NativeGraphKind kind = NativeGraphKind::kInfer;
    modalities::Status status = modalities::Status::ok();
};

NativeGraphCatalog::~NativeGraphCatalog() {
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

const char* NativeGraphCatalog::name(NativeGraphKind kind) {
    switch (kind) {
        case NativeGraphKind::kInfer: return "infer";
        case NativeGraphKind::kDecodeOnly: return "decode_only";
        case NativeGraphKind::kContext: return "context";
        case NativeGraphKind::kCount: break;
    }
    return nullptr;
}

frt_graph NativeGraphCatalog::graph(NativeGraphKind kind) const {
    const std::size_t index = graph_index(kind);
    return index < graph_index(NativeGraphKind::kCount)
               ? graphs_[index]
               : nullptr;
}

void NativeGraphCatalog::record_graph(void* user, void* stream) {
    auto* call = static_cast<CaptureCall*>(user);
    if (!call || !call->record) return;
    call->status = call->record(call->owner, call->kind, stream);
}

modalities::Status NativeGraphCatalog::capture(
    NativeGraphKind kind, const NativeWorkspace& workspace,
    std::initializer_list<const char*> bindings,
    RecordFn record, void* owner) {
    const std::size_t index = graph_index(kind);
    const char* graph_name = name(kind);
    if (!ctx_ || !graph_name || index >= graph_index(NativeGraphKind::kCount) ||
        graphs_[index] || !record || !owner) {
        return invalid("native graph capture request is invalid");
    }
    frt_graph captured = frt_graph_create(ctx_, graph_name, 1);
    if (!captured) return backend("native graph creation failed");
    graphs_[index] = captured;
    for (const char* binding : bindings) {
        const NativeWorkspaceBuffer* buffer = workspace.find(binding);
        if (!buffer ||
            frt_graph_bind(captured, binding, buffer->buffer) != FRT_OK) {
            frt_graph_destroy(captured);
            graphs_[index] = nullptr;
            return backend("native graph binding failed");
        }
    }
    CaptureCall call;
    call.record = record;
    call.owner = owner;
    call.kind = kind;
    const int rc = frt_graph_capture(captured, 0, record_graph, &call);
    if (!call.status.ok_status() || rc != FRT_OK ||
        frt_graph_variant_count(captured) != 1) {
        frt_graph_destroy(captured);
        graphs_[index] = nullptr;
        return call.status.ok_status()
                   ? backend("native graph capture failed")
                   : call.status;
    }
    return modalities::Status::ok();
}

modalities::Status NativeGraphCatalog::create_replay_stream() {
    if (!ctx_ || replay_stream_) {
        return invalid("native replay stream request is invalid");
    }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        return backend("native replay stream creation failed");
    }
    const int wrapped = frt_ctx_wrap_stream(ctx_, stream);
    if (wrapped < 0) {
        cudaStreamDestroy(stream);
        return backend("native replay stream wrapping failed");
    }
    replay_stream_ = stream;
    stream_id_ = wrapped;
    return modalities::Status::ok();
}

int NativeGraphCatalog::replay(NativeGraphKind kind) const {
    frt_graph selected = graph(kind);
    if (!selected || stream_id_ < 0) return FRT_ERR_INVALID;
    return frt_graph_replay(selected, 0, stream_id_);
}

modalities::Status NativeGraphCatalog::synchronize() const {
    if (!replay_stream_) return invalid("native replay stream is missing");
    const cudaError_t rc =
        cudaStreamSynchronize(static_cast<cudaStream_t>(replay_stream_));
    return rc == cudaSuccess
               ? modalities::Status::ok()
               : backend(cudaGetErrorString(rc));
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

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

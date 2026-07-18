#include "flashrt/cpp/native/cuda_graph_set.h"

#include <cuda_runtime_api.h>

namespace flashrt {
namespace native {
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

struct CudaGraphSet::CaptureCall {
    RecordFn record = nullptr;
    void* owner = nullptr;
    std::size_t slot = 0;
    modalities::Status status = modalities::Status::ok();
};

CudaGraphSet::CudaGraphSet(frt_ctx context, std::size_t graph_count)
    : context_(context), graphs_(graph_count, nullptr) {}

CudaGraphSet::~CudaGraphSet() {
    if (replay_stream_) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(replay_stream_));
        cudaStreamDestroy(static_cast<cudaStream_t>(replay_stream_));
        replay_stream_ = nullptr;
    }
    if (context_) {
        frt_ctx_destroy(context_);
        context_ = nullptr;
    }
}

frt_graph CudaGraphSet::graph(std::size_t slot) const {
    return slot < graphs_.size() ? graphs_[slot] : nullptr;
}

void CudaGraphSet::record_graph(void* user, void* stream) {
    auto* call = static_cast<CaptureCall*>(user);
    if (!call || !call->record) return;
    call->status = call->record(call->owner, call->slot, stream);
}

modalities::Status CudaGraphSet::capture(
    std::size_t slot,
    const char* name,
    const std::vector<CudaGraphBinding>& bindings,
    RecordFn record,
    void* owner) {
    if (!context_ || slot >= graphs_.size() || !name || !*name ||
        graphs_[slot] || !record || !owner) {
        return invalid("native graph capture request is invalid");
    }

    frt_graph captured = frt_graph_create(context_, name, 1);
    if (!captured) return backend("native graph creation failed");
    graphs_[slot] = captured;
    for (const CudaGraphBinding& binding : bindings) {
        if (!binding.name || !binding.buffer ||
            frt_graph_bind(captured, binding.name, binding.buffer) != FRT_OK) {
            frt_graph_destroy(captured);
            graphs_[slot] = nullptr;
            return backend("native graph binding failed");
        }
    }

    CaptureCall call;
    call.record = record;
    call.owner = owner;
    call.slot = slot;
    const int rc = frt_graph_capture(captured, 0, record_graph, &call);
    if (!call.status.ok_status() || rc != FRT_OK ||
        frt_graph_variant_count(captured) != 1) {
        frt_graph_destroy(captured);
        graphs_[slot] = nullptr;
        return call.status.ok_status()
                   ? backend("native graph capture failed")
                   : call.status;
    }
    return modalities::Status::ok();
}

modalities::Status CudaGraphSet::create_replay_stream() {
    if (!context_ || replay_stream_) {
        return invalid("native replay stream request is invalid");
    }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        return backend("native replay stream creation failed");
    }
    const int wrapped = frt_ctx_wrap_stream(context_, stream);
    if (wrapped < 0) {
        cudaStreamDestroy(stream);
        return backend("native replay stream wrapping failed");
    }
    replay_stream_ = stream;
    stream_id_ = wrapped;
    return modalities::Status::ok();
}

int CudaGraphSet::replay(std::size_t slot) const {
    const frt_graph selected = graph(slot);
    if (!selected || stream_id_ < 0) return FRT_ERR_INVALID;
    return frt_graph_replay(selected, 0, stream_id_);
}

modalities::Status CudaGraphSet::synchronize() const {
    if (!replay_stream_) return invalid("native replay stream is missing");
    const cudaError_t rc =
        cudaStreamSynchronize(static_cast<cudaStream_t>(replay_stream_));
    return rc == cudaSuccess
               ? modalities::Status::ok()
               : backend(cudaGetErrorString(rc));
}

}  // namespace native
}  // namespace flashrt

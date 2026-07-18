#include "flashrt/cpp/models/pi05/model/captured_program.h"

#include <cuda_runtime_api.h>

#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status missing(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kNotFound,
                                     message);
}

modalities::Status backend(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

}  // namespace

modalities::Status Pi05ResolvedGraphBindings::bind(
    Pi05GraphBindingId id,
    frt_buffer buffer) {
    const std::size_t index = static_cast<std::size_t>(id);
    if (index >= buffers_.size() || !buffer || buffers_[index]) {
        return invalid("PI0.5 graph binding request is invalid");
    }
    buffers_[index] = buffer;
    return modalities::Status::ok();
}

frt_buffer Pi05ResolvedGraphBindings::get(Pi05GraphBindingId id) const {
    const std::size_t index = static_cast<std::size_t>(id);
    return index < buffers_.size() ? buffers_[index] : nullptr;
}

struct Pi05CapturedProgram::RecordCall {
    const Pi05SemanticPipeline* pipeline = nullptr;
    Pi05OperationSink* operations = nullptr;
};

Pi05CapturedProgram::Pi05CapturedProgram(frt_ctx context)
    : graphs_(context, static_cast<std::size_t>(Pi05GraphId::kCount)) {}

modalities::Status Pi05CapturedProgram::record_body(
    const Pi05SemanticPipeline& pipeline,
    Pi05OperationSink& operations,
    Pi05RecordBody body,
    Pi05Stream stream) {
    switch (body) {
        case Pi05RecordBody::kFull:
            return pipeline.record_full(operations, stream);
        case Pi05RecordBody::kDecode:
            return pipeline.record_decode(operations, stream);
        case Pi05RecordBody::kContext:
            return pipeline.record_context(operations, stream);
    }
    return invalid("PI0.5 graph record body is invalid");
}

modalities::Status Pi05CapturedProgram::record_graph(
    void* owner,
    std::size_t slot,
    void* stream) {
    auto* call = static_cast<RecordCall*>(owner);
    const Pi05GraphDescriptor* graph = pi05_graph_descriptor(
        static_cast<Pi05GraphId>(slot));
    if (!call || !call->pipeline || !call->operations || !graph || !stream) {
        return invalid("PI0.5 graph record request is invalid");
    }
    return record_body(*call->pipeline, *call->operations, graph->body,
                       reinterpret_cast<Pi05Stream>(stream));
}

modalities::Status Pi05CapturedProgram::warmup(
    const Pi05SemanticPipeline& pipeline,
    Pi05OperationSink& operations) {
    if (!graphs_.context() || capture_attempted_) {
        return invalid("PI0.5 graph warmup request is invalid");
    }
    const Pi05GraphDescriptor* graph =
        pi05_graph_descriptor(Pi05GraphId::kInfer);
    if (!graph) return invalid("PI0.5 warmup graph descriptor is missing");
    modalities::Status status =
        record_body(pipeline, operations, graph->body, 0);
    if (!status.ok_status()) return status;
    const cudaError_t result = cudaDeviceSynchronize();
    return result == cudaSuccess
               ? modalities::Status::ok()
               : backend(cudaGetErrorString(result));
}

modalities::Status Pi05CapturedProgram::capture(
    const Pi05SemanticPipeline& pipeline,
    Pi05OperationSink& operations,
    const Pi05ResolvedGraphBindings& bindings) {
    if (!graphs_.context() || capture_attempted_) {
        return invalid("PI0.5 graph capture request is invalid");
    }

    std::size_t graph_count = 0;
    const Pi05GraphDescriptor* catalog = pi05_graph_catalog(&graph_count);
    if (!catalog || graph_count !=
                        static_cast<std::size_t>(Pi05GraphId::kCount)) {
        return invalid("PI0.5 graph catalog is invalid");
    }
    for (std::size_t slot = 0; slot < graph_count; ++slot) {
        const Pi05GraphDescriptor& graph = catalog[slot];
        if (static_cast<std::size_t>(graph.id) != slot || !graph.name ||
            !*graph.name || !graph.bindings || !graph.binding_count) {
            return invalid("PI0.5 graph descriptor is invalid");
        }
        for (std::size_t i = 0; i < graph.binding_count; ++i) {
            if (!pi05_graph_binding_name(graph.bindings[i]) ||
                !bindings.get(graph.bindings[i])) {
                return missing("PI0.5 graph binding is missing");
            }
        }
    }

    capture_attempted_ = true;
    RecordCall call{&pipeline, &operations};
    for (std::size_t slot = 0; slot < graph_count; ++slot) {
        const Pi05GraphDescriptor& graph = catalog[slot];
        std::vector<native::CudaGraphBinding> resolved;
        resolved.reserve(graph.binding_count);
        for (std::size_t i = 0; i < graph.binding_count; ++i) {
            const Pi05GraphBindingId id = graph.bindings[i];
            resolved.push_back(
                {pi05_graph_binding_name(id), bindings.get(id)});
        }
        modalities::Status status = graphs_.capture(
            slot, graph.name, resolved, record_graph, &call);
        if (!status.ok_status()) return status;
    }

    modalities::Status status = graphs_.create_replay_stream();
    if (!status.ok_status()) return status;
    captured_ = true;
    return modalities::Status::ok();
}

frt_graph Pi05CapturedProgram::graph(Pi05GraphId id) const {
    if (!captured_) return nullptr;
    const std::size_t slot = static_cast<std::size_t>(id);
    return slot < static_cast<std::size_t>(Pi05GraphId::kCount)
               ? graphs_.graph(slot)
               : nullptr;
}

int Pi05CapturedProgram::replay(Pi05GraphId id) const {
    if (!captured_) return FRT_ERR_INVALID;
    const std::size_t slot = static_cast<std::size_t>(id);
    return slot < static_cast<std::size_t>(Pi05GraphId::kCount)
               ? graphs_.replay(slot)
               : FRT_ERR_INVALID;
}

modalities::Status Pi05CapturedProgram::synchronize() const {
    return captured_ ? graphs_.synchronize()
                     : invalid("PI0.5 captured program is incomplete");
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

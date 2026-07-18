#ifndef FLASHRT_CPP_MODELS_PI05_MODEL_CAPTURED_PROGRAM_H
#define FLASHRT_CPP_MODELS_PI05_MODEL_CAPTURED_PROGRAM_H

#include "flashrt/cpp/models/pi05/model/execution_plan.h"
#include "flashrt/cpp/models/pi05/model/semantic_pipeline.h"
#include "flashrt/cpp/native/cuda_graph_set.h"

#include <array>
#include <cstddef>

namespace flashrt {
namespace models {
namespace pi05 {

class Pi05ResolvedGraphBindings final {
public:
    modalities::Status bind(Pi05GraphBindingId id, frt_buffer buffer);
    frt_buffer get(Pi05GraphBindingId id) const;

private:
    std::array<frt_buffer,
               static_cast<std::size_t>(Pi05GraphBindingId::kCount)>
        buffers_{};
};

class Pi05CapturedProgram final {
public:
    // The captured program takes ownership of context.
    explicit Pi05CapturedProgram(frt_ctx context);

    Pi05CapturedProgram(const Pi05CapturedProgram&) = delete;
    Pi05CapturedProgram& operator=(const Pi05CapturedProgram&) = delete;

    modalities::Status warmup(const Pi05SemanticPipeline& pipeline,
                               Pi05OperationSink& operations);
    modalities::Status capture(
        const Pi05SemanticPipeline& pipeline,
        Pi05OperationSink& operations,
        const Pi05ResolvedGraphBindings& bindings);

    frt_ctx context() const { return graphs_.context(); }
    frt_graph graph(Pi05GraphId id) const;
    int stream_id() const { return graphs_.stream_id(); }
    void* native_stream() const { return graphs_.native_stream(); }
    int replay(Pi05GraphId id) const;
    modalities::Status synchronize() const;

private:
    struct RecordCall;

    static modalities::Status record_graph(
        void* owner, std::size_t slot, void* stream);
    static modalities::Status record_body(
        const Pi05SemanticPipeline& pipeline,
        Pi05OperationSink& operations,
        Pi05RecordBody body,
        Pi05Stream stream);

    native::CudaGraphSet graphs_;
    bool capture_attempted_ = false;
    bool captured_ = false;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_MODEL_CAPTURED_PROGRAM_H

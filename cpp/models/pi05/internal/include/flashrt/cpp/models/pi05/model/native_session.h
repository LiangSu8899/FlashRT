#ifndef FLASHRT_CPP_MODELS_PI05_MODEL_NATIVE_SESSION_H
#define FLASHRT_CPP_MODELS_PI05_MODEL_NATIVE_SESSION_H

#include "flashrt/cpp/models/pi05/model/captured_program.h"
#include "flashrt/cpp/models/pi05/model/resolved_resources.h"

#include <memory>

namespace flashrt {
namespace models {
namespace pi05 {

class Pi05TargetBundle : public Pi05OperationSink {
public:
    Pi05TargetBundle(frt_ctx context, bool warmup_before_capture)
        : context_(context),
          warmup_before_capture_(warmup_before_capture) {}
    ~Pi05TargetBundle() override = default;

    Pi05TargetBundle(const Pi05TargetBundle&) = delete;
    Pi05TargetBundle& operator=(const Pi05TargetBundle&) = delete;

    frt_ctx context() const { return context_; }
    bool warmup_before_capture() const {
        return warmup_before_capture_;
    }

    virtual modalities::Status initialize_resources() = 0;
    virtual modalities::Status resolve_resources(
        Pi05ResolvedResources* out) = 0;
    virtual modalities::Status finalize_setup() = 0;
    virtual modalities::Status initialize_capture_inputs() = 0;
    virtual modalities::Status reset_after_warmup() = 0;
    virtual modalities::Status set_prompt_length(int prompt_tokens) = 0;

private:
    frt_ctx context_ = nullptr;
    bool warmup_before_capture_ = false;
};

class Pi05NativeSession final {
public:
    static std::unique_ptr<Pi05NativeSession> create(
        frt_ctx context,
        Pi05ResolvedShape shape,
        std::unique_ptr<Pi05TargetBundle> target,
        modalities::Status* status);
    ~Pi05NativeSession() = default;

    Pi05NativeSession(const Pi05NativeSession&) = delete;
    Pi05NativeSession& operator=(const Pi05NativeSession&) = delete;

    const Pi05ResolvedShape& shape() const { return pipeline_.shape(); }
    const Pi05ResolvedResources& resources() const { return resources_; }
    frt_ctx context() const { return program_.context(); }
    frt_graph graph(Pi05GraphId id) const { return program_.graph(id); }
    int stream_id() const { return program_.stream_id(); }
    void* native_stream() const { return program_.native_stream(); }

    int replay(Pi05GraphId id = Pi05GraphId::kInfer) const {
        return program_.replay(id);
    }
    modalities::Status synchronize() const {
        return program_.synchronize();
    }
    modalities::Status set_prompt_length(int prompt_tokens);

private:
    Pi05NativeSession(frt_ctx context,
                      Pi05ResolvedShape shape,
                      std::unique_ptr<Pi05TargetBundle> target);
    modalities::Status initialize();

    // Reverse destruction tears down the target before the owned context.
    Pi05SemanticPipeline pipeline_;
    Pi05CapturedProgram program_;
    std::unique_ptr<Pi05TargetBundle> target_;
    Pi05ResolvedResources resources_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_MODEL_NATIVE_SESSION_H

#ifndef FLASHRT_CPP_MODELS_PI05_MODEL_NATIVE_SESSION_H
#define FLASHRT_CPP_MODELS_PI05_MODEL_NATIVE_SESSION_H

#include "flashrt/cpp/models/pi05/model/captured_program.h"
#include "flashrt/cpp/models/pi05/model/frontend_ops.h"
#include "flashrt/cpp/models/pi05/model/target_bundle.h"

#include <memory>

namespace flashrt {
namespace models {
namespace pi05 {

enum class Pi05SessionMode {
    kCaptured = 0,
    kUncaptured,
};

class Pi05NativeSession final {
public:
    static std::unique_ptr<Pi05NativeSession> create(
        frt_ctx context,
        Pi05ResolvedShape shape,
        std::unique_ptr<Pi05TargetBundle> target,
        modalities::Status* status,
        Pi05SessionMode mode = Pi05SessionMode::kCaptured);
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
    modalities::Status execute(Pi05GraphId id = Pi05GraphId::kInfer);
    modalities::Status synchronize() const;
    modalities::Status set_prompt_length(int prompt_tokens);
    modalities::Status reset_observer();
    modalities::Status download_observer(Pi05ObservedScales* out) const;
    bool observes_activations() const {
        return target_ && target_->observes_activations();
    }

private:
    Pi05NativeSession(frt_ctx context,
                      Pi05ResolvedShape shape,
                      std::unique_ptr<Pi05TargetBundle> target,
                      Pi05SessionMode mode);
    modalities::Status initialize();

    // Reverse destruction tears down the target before the owned context.
    Pi05SemanticPipeline pipeline_;
    Pi05CapturedProgram program_;
    std::unique_ptr<Pi05TargetBundle> target_;
    Pi05SessionMode mode_ = Pi05SessionMode::kCaptured;
    Pi05ResolvedResources resources_;
    Pi05ForwardExecution forward_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_MODEL_NATIVE_SESSION_H

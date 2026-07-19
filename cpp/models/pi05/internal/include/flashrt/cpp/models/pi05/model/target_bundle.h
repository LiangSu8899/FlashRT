#ifndef FLASHRT_CPP_MODELS_PI05_MODEL_TARGET_BUNDLE_H
#define FLASHRT_CPP_MODELS_PI05_MODEL_TARGET_BUNDLE_H

#include "flashrt/cpp/models/pi05/model/resolved_resources.h"

#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

struct Pi05ForwardExecution;

struct Pi05ObservedScales final {
    std::vector<float> vision;
    std::vector<float> encoder;
    std::vector<float> decoder;
};

class Pi05TargetBundle {
public:
    Pi05TargetBundle(frt_ctx context, bool warmup_before_capture)
        : context_(context),
          warmup_before_capture_(warmup_before_capture) {}
    virtual ~Pi05TargetBundle() = default;

    Pi05TargetBundle(const Pi05TargetBundle&) = delete;
    Pi05TargetBundle& operator=(const Pi05TargetBundle&) = delete;

    frt_ctx context() const { return context_; }
    bool warmup_before_capture() const {
        return warmup_before_capture_;
    }

    virtual modalities::Status initialize_resources() = 0;
    virtual modalities::Status resolve_resources(
        Pi05ResolvedResources* out) = 0;
    virtual modalities::Status make_prepare_execution(
        Pi05PrepareExecution* out) = 0;
    virtual modalities::Status complete_prepare() = 0;
    virtual modalities::Status finalize_setup() = 0;
    virtual modalities::Status make_forward_execution(
        Pi05ForwardExecution* out) = 0;
    virtual modalities::Status initialize_capture_inputs() = 0;
    virtual modalities::Status reset_after_warmup() = 0;
    virtual modalities::Status set_prompt_length(int prompt_tokens) = 0;
    virtual bool observes_activations() const { return false; }
    virtual modalities::Status reset_observer(Pi05Stream) {
        return modalities::Status::error(
            modalities::StatusCode::kUnsupported,
            "target activation observer is unavailable");
    }
    virtual modalities::Status download_observer(
        Pi05ObservedScales*) const {
        return modalities::Status::error(
            modalities::StatusCode::kUnsupported,
            "target activation observer is unavailable");
    }

private:
    frt_ctx context_ = nullptr;
    bool warmup_before_capture_ = false;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_MODEL_TARGET_BUNDLE_H

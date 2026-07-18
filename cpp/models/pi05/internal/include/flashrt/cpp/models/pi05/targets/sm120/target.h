#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_TARGET_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_TARGET_H

#include "flashrt/cpp/models/pi05/model/target_bundle.h"

#include <cstddef>
#include <memory>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {

class Sm120TargetBundle final : public Pi05TargetBundle {
public:
    static std::unique_ptr<Sm120TargetBundle> create(
        frt_ctx context,
        const Pi05ResolvedShape& shape,
        std::string checkpoint_path,
        modalities::Status* status);
    ~Sm120TargetBundle() override;

    Sm120TargetBundle(const Sm120TargetBundle&) = delete;
    Sm120TargetBundle& operator=(const Sm120TargetBundle&) = delete;

    modalities::Status initialize_resources() override;
    modalities::Status resolve_resources(Pi05ResolvedResources* out) override;
    modalities::Status finalize_setup() override;
    modalities::Status initialize_capture_inputs() override;
    modalities::Status reset_after_warmup() override;
    modalities::Status set_prompt_length(int prompt_tokens) override;
    modalities::Status record(const Pi05OperationCall& call,
                              const Pi05ResolvedShape& shape,
                              Pi05Stream stream) override;

    const Pi05ResolvedResources* resolved_resources() const;
    std::size_t materialized_weight_count() const;
    std::size_t prepare_call_count() const;
    bool ready_for_capture() const;

private:
    struct Impl;

    Sm120TargetBundle(frt_ctx context, std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_TARGET_H

#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_TARGET_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_TARGET_H

#include "flashrt/cpp/models/pi05/model/target_bundle.h"
#include "flashrt/cpp/models/pi05/support/native_calibration.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {

enum class Sm120Precision {
    kBf16 = 0,
    kStaticFp8E4M3,
};

struct Sm120TargetConfig final {
    std::string checkpoint_path;
    Sm120Precision precision = Sm120Precision::kBf16;
    std::optional<NativeCalibrationArtifact> calibration;
};

class Sm120TargetBundle final : public Pi05TargetBundle {
public:
    static std::unique_ptr<Sm120TargetBundle> create(
        frt_ctx context,
        const Pi05ResolvedShape& shape,
        Sm120TargetConfig config,
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
    std::size_t packed_weight_count() const;
    std::size_t autotuned_shape_count() const;
    std::size_t prepare_call_count() const;
    Sm120Precision precision() const;
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

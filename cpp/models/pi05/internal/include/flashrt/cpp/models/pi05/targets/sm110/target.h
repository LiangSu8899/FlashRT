#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_TARGET_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_TARGET_H

#include "flashrt/cpp/models/pi05/model/target_bundle.h"
#include "flashrt/cpp/models/pi05/support/native_calibration.h"

#include <cstddef>
#include <memory>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {

struct Pi05NativeSupportBuffers;

namespace targets {
namespace sm110 {

class Sm110Fp8WeightPacker;
class Sm110OperationDriver;
class Sm110PhysicalResources;

struct Sm110TargetConfig final {
    std::string checkpoint_path;
    NativeCalibrationArtifact calibration;
};

// C2 exposes resource construction for validation only. Operation recording is
// connected to the semantic pipeline in C3 before this target is published.
class Sm110TargetBundle final : public Pi05TargetBundle {
public:
    static std::unique_ptr<Sm110TargetBundle> create(
        frt_ctx context,
        const Pi05ResolvedShape& shape,
        Sm110TargetConfig config,
        modalities::Status* status);
    ~Sm110TargetBundle() override;

    Sm110TargetBundle(const Sm110TargetBundle&) = delete;
    Sm110TargetBundle& operator=(const Sm110TargetBundle&) = delete;

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
    const Pi05NativeSupportBuffers* support_buffers() const;
    const Sm110PhysicalResources* physical_resources() const;
    const Sm110Fp8WeightPacker* weight_packer() const;
    const Sm110OperationDriver* operation_driver() const;
    std::size_t materialized_weight_count() const;
    std::size_t logical_workspace_count() const;
    std::size_t logical_workspace_allocation_count() const;
    std::size_t logical_workspace_bytes() const;
    bool resources_ready() const;

private:
    struct Impl;

    Sm110TargetBundle(frt_ctx context, std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

}  // namespace sm110
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_TARGET_H

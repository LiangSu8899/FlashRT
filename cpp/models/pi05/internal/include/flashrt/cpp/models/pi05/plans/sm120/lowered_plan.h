#ifndef FLASHRT_CPP_MODELS_PI05_PLANS_SM120_LOWERED_PLAN_H
#define FLASHRT_CPP_MODELS_PI05_PLANS_SM120_LOWERED_PLAN_H

#include "flashrt/cpp/models/pi05/plans/sm120/native_bf16_forward.h"
#include "flashrt/cpp/models/pi05/support/native_calibration.h"
#include "flashrt/cpp/models/pi05/model/pipeline.h"

#include <memory>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {

NativeWorkspaceRequirements make_sm120_workspace_requirements(
    const NativeWorkspaceConfig& config,
    bool fp8);

class Sm120LoweredPlan final : public Pi05Pipeline {
public:
    static std::unique_ptr<Sm120LoweredPlan> create(
        const std::string& checkpoint_path, const Pi05PipelineConfig& config,
        modalities::Status* status);
    static std::unique_ptr<Sm120LoweredPlan> create(
        const std::string& checkpoint_path, const Pi05PipelineConfig& config,
        const NativeCalibrationArtifact& calibration,
        modalities::Status* status);
    static std::unique_ptr<Sm120LoweredPlan> create_calibration(
        const std::string& checkpoint_path, const Pi05PipelineConfig& config,
        modalities::Status* status);

    ~Sm120LoweredPlan() override;

    Sm120LoweredPlan(const Sm120LoweredPlan&) = delete;
    Sm120LoweredPlan& operator=(const Sm120LoweredPlan&) = delete;

    NativeDeviceWeightStore& weights() { return weights_; }
    const NativeDeviceWeightStore& weights() const { return weights_; }
    NativeWorkspace& workspace() override { return workspace_; }
    const NativeWorkspace& workspace() const override { return workspace_; }
    const PipelineArtifacts& artifacts() const override {
        return artifacts_;
    }
    NativeRtxAttentionWorkspace& attention() { return attention_; }
    const NativeRtxAttentionWorkspace& attention() const { return attention_; }

    modalities::Status set_prompt_length(int prompt_tokens) override;

private:
    Sm120LoweredPlan(frt_ctx ctx, const Pi05PipelineConfig& config,
                     NativeRtxLinearMode linear_mode);
    modalities::Status initialize(
        const std::string& checkpoint_path,
        const NativeCalibrationArtifact* calibration);
    modalities::Status record_vision(void* stream) override;
    modalities::Status record_encoder(void* stream) override;
    modalities::Status record_diffusion_step(int step,
                                              void* stream) override;

    NativeDeviceWeightStore weights_;
    NativeWorkspace workspace_;
    PipelineArtifacts artifacts_;
    NativeRtxAttentionWorkspace attention_;
    NativeKernelDriver driver_;
    NativeRtxLinear linear_;
    NativeBf16Forward forward_;
    std::unique_ptr<NativeRtxAttentionDriver> attention_driver_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_PLANS_SM120_LOWERED_PLAN_H

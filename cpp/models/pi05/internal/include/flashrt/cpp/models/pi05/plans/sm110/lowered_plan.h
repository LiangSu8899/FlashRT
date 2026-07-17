#ifndef FLASHRT_CPP_MODELS_PI05_PLANS_SM110_LOWERED_PLAN_H
#define FLASHRT_CPP_MODELS_PI05_PLANS_SM110_LOWERED_PLAN_H

#include "flashrt/cpp/models/pi05/support/native_calibration.h"
#include "flashrt/cpp/models/pi05/model/pipeline.h"
#include "flashrt/cpp/models/pi05/plans/sm110/native_thor_fp8_forward.h"

#include <memory>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

NativeWorkspaceRequirements make_sm110_workspace_requirements(
    const NativeWorkspaceConfig& config,
    bool enable_calibration);

modalities::Status initialize_sm110_workspace(NativeWorkspace* workspace);

class Sm110LoweredPlan final : public Pi05Pipeline {
public:
    static std::unique_ptr<Sm110LoweredPlan> create(
        const std::string& checkpoint_path,
        const Pi05PipelineConfig& config,
        const NativeCalibrationArtifact& calibration,
        modalities::Status* status);

    ~Sm110LoweredPlan() override;

    Sm110LoweredPlan(const Sm110LoweredPlan&) = delete;
    Sm110LoweredPlan& operator=(const Sm110LoweredPlan&) = delete;

    NativeDeviceWeightStore& weights() { return weights_; }
    const NativeDeviceWeightStore& weights() const { return weights_; }
    NativeWorkspace& workspace() override { return workspace_; }
    const NativeWorkspace& workspace() const override { return workspace_; }
    const PipelineArtifacts& artifacts() const override {
        return artifacts_;
    }

    modalities::Status set_prompt_length(int prompt_tokens) override;

private:
    Sm110LoweredPlan(frt_ctx ctx, const Pi05PipelineConfig& config);
    modalities::Status initialize(
        const std::string& checkpoint_path,
        const NativeCalibrationArtifact& calibration);
    modalities::Status record_vision_begin(void* stream) override;
    modalities::Status record_vision_layer(int layer, void* stream) override;
    modalities::Status record_vision_end(void* stream) override;
    modalities::Status record_encoder_layer(int layer, void* stream) override;
    modalities::Status record_diffusion_begin(int step,
                                              void* stream) override;
    modalities::Status record_decoder_layer(int step, int layer,
                                            void* stream) override;
    modalities::Status record_diffusion_end(int step,
                                            void* stream) override;

    NativeDeviceWeightStore weights_;
    NativeWorkspace workspace_;
    PipelineArtifacts artifacts_;
    NativeThorKernelDriver driver_;
    NativeThorFp8Forward forward_;
    NativeThorWeightScales weight_scales_;
    std::vector<float> encoder_alphas_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_PLANS_SM110_LOWERED_PLAN_H

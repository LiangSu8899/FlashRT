#ifndef FLASHRT_CPP_MODELS_PI05_BACKENDS_SM110_SESSION_H
#define FLASHRT_CPP_MODELS_PI05_BACKENDS_SM110_SESSION_H

#include "flashrt/cpp/models/pi05/support/native_calibration.h"
#include "flashrt/cpp/models/pi05/backend/session.h"
#include "flashrt/cpp/models/pi05/backends/sm110/native_thor_fp8_forward.h"

#include <memory>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

class Sm110BackendSession final : public Pi05Pipeline {
public:
    static std::unique_ptr<Sm110BackendSession> create(
        const std::string& checkpoint_path,
        const BackendConfig& config,
        const NativeCalibrationArtifact& calibration,
        modalities::Status* status);

    ~Sm110BackendSession() override;

    Sm110BackendSession(const Sm110BackendSession&) = delete;
    Sm110BackendSession& operator=(const Sm110BackendSession&) = delete;

    NativeDeviceWeightStore& weights() { return weights_; }
    const NativeDeviceWeightStore& weights() const { return weights_; }
    NativeWorkspace& workspace() override { return workspace_; }
    const NativeWorkspace& workspace() const override { return workspace_; }
    const BackendArtifacts& artifacts() const override {
        return artifacts_;
    }

    modalities::Status set_prompt_length(int prompt_tokens) override;

private:
    Sm110BackendSession(frt_ctx ctx, const BackendConfig& config);
    modalities::Status initialize(
        const std::string& checkpoint_path,
        const NativeCalibrationArtifact& calibration);
    modalities::Status record_vision(void* stream) override;
    modalities::Status record_encoder(void* stream) override;
    modalities::Status record_diffusion(void* stream) override;

    NativeDeviceWeightStore weights_;
    NativeWorkspace workspace_;
    BackendArtifacts artifacts_;
    NativeThorKernelDriver driver_;
    NativeThorFp8Forward forward_;
    NativeThorWeightScales weight_scales_;
    std::vector<float> encoder_alphas_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_BACKENDS_SM110_SESSION_H

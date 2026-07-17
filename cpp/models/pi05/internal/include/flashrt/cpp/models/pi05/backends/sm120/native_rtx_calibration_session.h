#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_CALIBRATION_SESSION_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_CALIBRATION_SESSION_H

#include "flashrt/cpp/models/pi05/backend/native_calibration_session.h"

#include <memory>

namespace flashrt {
namespace models {
namespace pi05 {

class NativeRtxCalibrationSession final : public NativeCalibrationSession {
public:
    static std::unique_ptr<NativeRtxCalibrationSession> create(
        const NativeCalibrationConfig& config,
        double percentile,
        modalities::Status* status);

    ~NativeRtxCalibrationSession() override;

    NativeRtxCalibrationSession(const NativeRtxCalibrationSession&) = delete;
    NativeRtxCalibrationSession& operator=(
        const NativeRtxCalibrationSession&) = delete;

    modalities::Status observe(
        const std::string& prompt,
        const float* state,
        std::uint64_t n_state,
        const std::vector<modalities::VisionFrame>& frames,
        const float* noise,
        std::uint64_t n_noise,
        std::uint64_t noise_seed) override;
    modalities::Status finalize(
        const std::string& artifact_path) const override;
    std::uint64_t sample_count() const override;

private:
    struct Impl;
    explicit NativeRtxCalibrationSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_CALIBRATION_SESSION_H

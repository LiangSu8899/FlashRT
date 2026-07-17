#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_CALIBRATION_SESSION_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_CALIBRATION_SESSION_H

#include "flashrt/cpp/models/pi05/backend/native_calibration_session.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

using NativeThorCalibrationConfig = NativeCalibrationConfig;

class NativeThorCalibrationSession final : public NativeCalibrationSession {
public:
    static std::unique_ptr<NativeThorCalibrationSession> create(
        const NativeThorCalibrationConfig& config,
        double percentile,
        modalities::Status* status);

    ~NativeThorCalibrationSession() override;

    NativeThorCalibrationSession(const NativeThorCalibrationSession&) = delete;
    NativeThorCalibrationSession& operator=(
        const NativeThorCalibrationSession&) = delete;

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
    explicit NativeThorCalibrationSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_CALIBRATION_SESSION_H

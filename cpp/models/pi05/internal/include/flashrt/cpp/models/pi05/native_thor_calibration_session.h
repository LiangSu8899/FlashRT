#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_CALIBRATION_SESSION_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_CALIBRATION_SESSION_H

#include "flashrt/cpp/modalities/vision.h"
#include "flashrt/cpp/modalities/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

struct NativeThorCalibrationConfig {
    std::string checkpoint_path;
    std::string tokenizer_model_path;
    int max_prompt_tokens = 200;
    int state_dim = 0;
    int num_views = 2;
    int chunk_size = 10;
    int num_steps = 10;
    int vision_pool_factor = 1;
    int max_frame_width = 1280;
    int max_frame_height = 720;
    std::vector<float> state_q01;
    std::vector<float> state_q99;
};

class NativeThorCalibrationSession {
public:
    static std::unique_ptr<NativeThorCalibrationSession> create(
        const NativeThorCalibrationConfig& config,
        double percentile,
        modalities::Status* status);

    ~NativeThorCalibrationSession();

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
        std::uint64_t noise_seed);
    modalities::Status finalize(const std::string& artifact_path) const;
    std::uint64_t sample_count() const;

private:
    struct Impl;
    explicit NativeThorCalibrationSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_CALIBRATION_SESSION_H

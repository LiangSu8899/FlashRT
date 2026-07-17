#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_CALIBRATION_SESSION_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_CALIBRATION_SESSION_H

#include "flashrt/cpp/modalities/vision.h"
#include "flashrt/cpp/modalities/types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

struct NativeCalibrationConfig {
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

class NativeCalibrationSession {
public:
    virtual ~NativeCalibrationSession() = default;

    virtual modalities::Status observe(
        const std::string& prompt,
        const float* state,
        std::uint64_t n_state,
        const std::vector<modalities::VisionFrame>& frames,
        const float* noise,
        std::uint64_t n_noise,
        std::uint64_t noise_seed) = 0;
    virtual modalities::Status finalize(
        const std::string& artifact_path) const = 0;
    virtual std::uint64_t sample_count() const = 0;
};

bool valid_native_calibration_config(const NativeCalibrationConfig& config);

modalities::Status normalize_native_calibration_state(
    const NativeCalibrationConfig& config,
    const float* state,
    std::uint64_t n_state,
    std::vector<float>* output);

modalities::Status prepare_native_calibration_noise(
    const float* noise,
    std::uint64_t n_noise,
    std::uint64_t seed,
    std::size_t elements,
    modalities::DType dtype,
    std::vector<std::uint16_t>* output);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_CALIBRATION_SESSION_H

#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_CALIBRATION_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_CALIBRATION_H

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

struct NativeCalibrationArtifact {
    std::string activation_dtype = "float16";
    std::string hardware;
    std::string weights_sha256;
    std::string tokenizer_sha256;
    int num_views = 0;
    int max_prompt_tokens = 0;
    int state_dim = 0;
    int chunk_size = 0;
    int num_steps = 0;
    int vision_pool_factor = 0;
    std::uint64_t sample_count = 0;
    double percentile = 100.0;
    std::vector<float> vision_scales;
    std::vector<float> encoder_scales;
    std::vector<float> decoder_scales;
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

modalities::Status validate_native_calibration_artifact(
    const NativeCalibrationArtifact& artifact);

modalities::Status save_native_calibration_artifact(
    const std::string& path,
    const NativeCalibrationArtifact& artifact);

modalities::Status load_native_calibration_artifact(
    const std::string& path,
    NativeCalibrationArtifact* artifact);

modalities::Status reduce_native_calibration_samples(
    const std::vector<std::vector<float>>& samples,
    double percentile,
    std::vector<float>* reduced);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_CALIBRATION_H

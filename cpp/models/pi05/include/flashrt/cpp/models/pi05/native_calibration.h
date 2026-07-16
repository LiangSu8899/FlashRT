#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_CALIBRATION_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_CALIBRATION_H

#include "flashrt/cpp/modalities/types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

struct NativeCalibrationArtifact {
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
    std::vector<float> encoder_scales;
    std::vector<float> decoder_scales;
};

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

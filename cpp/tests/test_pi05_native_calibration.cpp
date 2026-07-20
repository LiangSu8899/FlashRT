#include "flashrt/cpp/models/pi05/model/dims.h"
#include "flashrt/cpp/models/pi05/support/native_calibration.h"
#include "flashrt/cpp/models/pi05/support/native_float16.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

std::string temp_path() {
    char path[] = "/tmp/frt_pi05_calibration_XXXXXX";
    const int fd = ::mkstemp(path);
    assert(fd >= 0);
    ::close(fd);
    assert(::unlink(path) == 0);
    return path;
}

}  // namespace

int main() {
    using flashrt::models::pi05::NativeCalibrationArtifact;
    using flashrt::models::pi05::NativeCalibrationConfig;
    using flashrt::models::pi05::load_native_calibration_artifact;
    using flashrt::models::pi05::normalize_native_calibration_state;
    using flashrt::models::pi05::prepare_native_calibration_noise;
    using flashrt::models::pi05::reduce_native_calibration_samples;
    using flashrt::models::pi05::save_native_calibration_artifact;
    using flashrt::models::pi05::valid_native_calibration_config;

    const auto to_float16 = flashrt::models::pi05::float_to_float16_rne;
    assert(to_float16(1.00048828125f) == 0x3c00u);
    assert(to_float16(1.00146484375f) == 0x3c02u);
    assert(to_float16(-1.00048828125f) == 0xbc00u);
    assert(to_float16(-1.00146484375f) == 0xbc02u);
    const float half_min_subnormal = std::ldexp(1.0f, -25);
    assert(to_float16(half_min_subnormal) == 0x0000u);
    assert(to_float16(std::nextafter(
               half_min_subnormal, std::numeric_limits<float>::infinity())) ==
           0x0001u);
    assert(to_float16(3.0f * std::ldexp(1.0f, -25)) == 0x0002u);
    assert(to_float16(std::ldexp(1.0f, -14) -
                      std::ldexp(1.0f, -25)) == 0x0400u);
    assert(to_float16(std::numeric_limits<float>::infinity()) == 0x7c00u);
    assert(to_float16(-std::numeric_limits<float>::infinity()) == 0xfc00u);
    const std::uint16_t nan =
        to_float16(std::numeric_limits<float>::quiet_NaN());
    assert((nan & 0x7c00u) == 0x7c00u && (nan & 0x03ffu) != 0);

    NativeCalibrationConfig config;
    config.checkpoint_path = "checkpoint";
    config.tokenizer_model_path = "tokenizer.model";
    config.state_dim = 2;
    config.num_views = 3;
    config.state_q01 = {-2.0f, 0.0f};
    config.state_q99 = {2.0f, 4.0f};
    assert(valid_native_calibration_config(config));
    NativeCalibrationConfig invalid_config = config;
    invalid_config.state_q99[0] = invalid_config.state_q01[0];
    assert(!valid_native_calibration_config(invalid_config));
    invalid_config = config;
    invalid_config.max_frame_width = 0;
    assert(!valid_native_calibration_config(invalid_config));
    std::vector<float> normalized;
    const float state[] = {0.0f, 1.0f};
    assert(normalize_native_calibration_state(
               config, state, 2, &normalized)
               .ok_status());
    assert(normalized.size() == 2);
    assert(std::fabs(normalized[0]) < 1e-6f);
    assert(std::fabs(normalized[1] + 0.5f) < 1e-6f);
    const float invalid_state[] = {0.0f, INFINITY};
    assert(!normalize_native_calibration_state(
                config, invalid_state, 2, &normalized)
                .ok_status());

    const float explicit_noise[] = {-1.0f, 0.0f, 1.0f};
    std::vector<std::uint16_t> encoded;
    assert(prepare_native_calibration_noise(
               explicit_noise, 3, 0, 3,
               flashrt::modalities::DType::kBFloat16, &encoded)
               .ok_status());
    assert(encoded == std::vector<std::uint16_t>({
                          flashrt::modalities::float_to_bfloat16(-1.0f),
                          flashrt::modalities::float_to_bfloat16(0.0f),
                          flashrt::modalities::float_to_bfloat16(1.0f)}));
    std::vector<std::uint16_t> generated;
    assert(prepare_native_calibration_noise(
               nullptr, 0, 1234, 7,
               flashrt::modalities::DType::kFloat16, &generated)
               .ok_status());
    std::vector<std::uint16_t> repeated;
    assert(prepare_native_calibration_noise(
               nullptr, 0, 1234, 7,
               flashrt::modalities::DType::kFloat16, &repeated)
               .ok_status());
    assert(generated == repeated);
    assert(!prepare_native_calibration_noise(
                explicit_noise, 2, 0, 3,
                flashrt::modalities::DType::kFloat16, &encoded)
                .ok_status());
    const float nonfinite_noise[] = {0.0f, NAN, 1.0f};
    assert(!prepare_native_calibration_noise(
                nonfinite_noise, 3, 0, 3,
                flashrt::modalities::DType::kFloat16, &encoded)
                .ok_status());

    std::vector<float> reduced;
    assert(reduce_native_calibration_samples(
               {{1.0f, 10.0f}, {2.0f, 20.0f}, {4.0f, 40.0f},
                {8.0f, 80.0f}},
               25.0, &reduced)
               .ok_status());
    assert(reduced.size() == 2);
    assert(std::fabs(reduced[0] - 1.75f) < 1e-6f);
    assert(std::fabs(reduced[1] - 17.5f) < 1e-6f);
    assert(reduce_native_calibration_samples({{3.0f}}, 99.9, &reduced)
               .ok_status());
    assert(reduced == std::vector<float>({3.0f}));
    assert(!reduce_native_calibration_samples({}, 99.9, &reduced)
                .ok_status());
    assert(!reduce_native_calibration_samples(
                {{1.0f}, {1.0f, 2.0f}}, 99.9, &reduced)
                .ok_status());

    NativeCalibrationArtifact expected;
    expected.hardware = "sm110";
    expected.weights_sha256 = std::string(64, 'a');
    expected.tokenizer_sha256 = std::string(64, 'b');
    expected.num_views = 2;
    expected.max_prompt_tokens = 200;
    expected.state_dim = 8;
    expected.chunk_size = 10;
    expected.num_steps = 10;
    expected.vision_pool_factor = 1;
    expected.sample_count = 8;
    expected.percentile = 99.9;
    constexpr std::size_t kObservedScalesPerLayer = 4;
    expected.encoder_scales.resize(
        flashrt::models::pi05::kPi05ModelDims.encoder_layers *
        kObservedScalesPerLayer);
    expected.decoder_scales.resize(
        expected.num_steps *
        flashrt::models::pi05::kPi05ModelDims.decoder_layers *
        kObservedScalesPerLayer);
    for (std::size_t i = 0; i < expected.encoder_scales.size(); ++i) {
        expected.encoder_scales[i] = 0.001f * static_cast<float>(i + 1);
    }
    for (std::size_t i = 0; i < expected.decoder_scales.size(); ++i) {
        expected.decoder_scales[i] = 0.0001f * static_cast<float>(i + 1);
    }

    const std::string path = temp_path();
    assert(save_native_calibration_artifact(path, expected).ok_status());
    NativeCalibrationArtifact loaded;
    assert(load_native_calibration_artifact(path, &loaded).ok_status());
    assert(loaded.hardware == expected.hardware);
    assert(loaded.weights_sha256 == expected.weights_sha256);
    assert(loaded.tokenizer_sha256 == expected.tokenizer_sha256);
    assert(loaded.num_views == expected.num_views);
    assert(loaded.max_prompt_tokens == expected.max_prompt_tokens);
    assert(loaded.state_dim == expected.state_dim);
    assert(loaded.chunk_size == expected.chunk_size);
    assert(loaded.num_steps == expected.num_steps);
    assert(loaded.vision_pool_factor == expected.vision_pool_factor);
    assert(loaded.sample_count == expected.sample_count);
    assert(loaded.activation_dtype == expected.activation_dtype);
    assert(loaded.vision_scales == expected.vision_scales);
    assert(std::fabs(loaded.percentile - expected.percentile) < 1e-12);
    assert(loaded.encoder_scales == expected.encoder_scales);
    assert(loaded.decoder_scales == expected.decoder_scales);
    assert(::unlink(path.c_str()) == 0);

    expected.activation_dtype = "bfloat16";
    expected.hardware = "sm120";
    expected.vision_scales.resize(
        flashrt::models::pi05::kPi05ModelDims.vision_layers *
            kObservedScalesPerLayer +
        1);
    for (std::size_t i = 0; i < expected.vision_scales.size(); ++i) {
        expected.vision_scales[i] = 0.002f * static_cast<float>(i + 1);
    }
    assert(save_native_calibration_artifact(path, expected).ok_status());
    assert(load_native_calibration_artifact(path, &loaded).ok_status());
    assert(loaded.activation_dtype == expected.activation_dtype);
    assert(loaded.hardware == expected.hardware);
    assert(loaded.vision_scales == expected.vision_scales);
    assert(loaded.encoder_scales == expected.encoder_scales);
    assert(loaded.decoder_scales == expected.decoder_scales);
    assert(::truncate(path.c_str(), 16) == 0);
    assert(!load_native_calibration_artifact(path, &loaded).ok_status());
    assert(::unlink(path.c_str()) == 0);

    NativeCalibrationArtifact invalid_artifact = expected;
    invalid_artifact.decoder_scales.pop_back();
    assert(!save_native_calibration_artifact(path, invalid_artifact)
                .ok_status());
    invalid_artifact = expected;
    invalid_artifact.encoder_scales[0] = 0.0f;
    assert(!save_native_calibration_artifact(path, invalid_artifact)
                .ok_status());
    expected.weights_sha256 = "short";
    assert(!save_native_calibration_artifact(path, expected).ok_status());
    std::printf("PASS - Pi0.5 native calibration artifact\n");
    return 0;
}

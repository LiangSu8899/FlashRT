#include "flashrt/cpp/models/pi05/native_calibration.h"

#include <cassert>
#include <cmath>
#include <cstdio>
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
    using flashrt::models::pi05::load_native_calibration_artifact;
    using flashrt::models::pi05::reduce_native_calibration_samples;
    using flashrt::models::pi05::save_native_calibration_artifact;

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
    expected.encoder_scales.resize(18 * 4);
    expected.decoder_scales.resize(10 * 18 * 4);
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
    expected.vision_scales.resize(27 * 4 + 1);
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
    assert(::unlink(path.c_str()) == 0);

    expected.weights_sha256 = "short";
    assert(!save_native_calibration_artifact(path, expected).ok_status());
    std::printf("PASS - Pi0.5 native calibration artifact\n");
    return 0;
}

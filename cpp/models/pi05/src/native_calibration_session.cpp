#include "flashrt/cpp/models/pi05/native_calibration_session.h"

#include <cmath>
#include <limits>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

std::uint64_t splitmix64(std::uint64_t* state) {
    std::uint64_t value = (*state += 0x9e3779b97f4a7c15ull);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

double uniform_open(std::uint64_t* state) {
    constexpr double kDenominator = 9007199254740993.0;
    return (static_cast<double>(splitmix64(state) >> 11) + 1.0) /
           kDenominator;
}

std::uint16_t encode(float value, modalities::DType dtype) {
    return dtype == modalities::DType::kFloat16
               ? modalities::float_to_float16(value)
               : modalities::float_to_bfloat16(value);
}

}  // namespace

bool valid_native_calibration_config(const NativeCalibrationConfig& config) {
    const std::uint64_t width =
        static_cast<std::uint64_t>(config.max_frame_width);
    const std::uint64_t height =
        static_cast<std::uint64_t>(config.max_frame_height);
    bool valid_quantiles =
        config.state_q01.size() ==
            static_cast<std::size_t>(config.state_dim) &&
        config.state_q99.size() == config.state_q01.size();
    for (std::size_t i = 0;
         valid_quantiles && i < config.state_q01.size(); ++i) {
        valid_quantiles = std::isfinite(config.state_q01[i]) &&
                          std::isfinite(config.state_q99[i]) &&
                          config.state_q99[i] > config.state_q01[i];
    }
    return !config.checkpoint_path.empty() &&
           !config.tokenizer_model_path.empty() && config.state_dim > 0 &&
           config.num_views >= 1 && config.num_views <= 3 &&
           config.max_prompt_tokens >= 1 && config.chunk_size > 0 &&
           config.num_steps > 0 &&
           (config.vision_pool_factor == 1 ||
            config.vision_pool_factor == 2 ||
            config.vision_pool_factor == 4) &&
           static_cast<std::uint64_t>(config.max_prompt_tokens) +
                   static_cast<std::uint64_t>(config.chunk_size) +
                   static_cast<std::uint64_t>(config.num_views) * 256 <=
               static_cast<std::uint64_t>(
                   std::numeric_limits<int>::max()) &&
           config.max_frame_width > 0 && config.max_frame_height > 0 &&
           width <= std::numeric_limits<std::uint64_t>::max() / height / 4 &&
           valid_quantiles;
}

modalities::Status normalize_native_calibration_state(
    const NativeCalibrationConfig& config,
    const float* state,
    std::uint64_t n_state,
    std::vector<float>* output) {
    if (!state || !output ||
        n_state != static_cast<std::uint64_t>(config.state_dim)) {
        return invalid("native calibration state shape is invalid");
    }
    output->resize(static_cast<std::size_t>(config.state_dim));
    for (std::size_t i = 0; i < output->size(); ++i) {
        if (!std::isfinite(state[i])) {
            return invalid("native calibration state contains non-finite data");
        }
        const float lo = config.state_q01[i];
        const float hi = config.state_q99[i];
        (*output)[i] = ((state[i] - lo) / (hi - lo + 1e-6f)) * 2.0f - 1.0f;
    }
    return modalities::Status::ok();
}

modalities::Status prepare_native_calibration_noise(
    const float* noise,
    std::uint64_t n_noise,
    std::uint64_t seed,
    std::size_t elements,
    modalities::DType dtype,
    std::vector<std::uint16_t>* output) {
    if (!output || !elements ||
        (dtype != modalities::DType::kFloat16 &&
         dtype != modalities::DType::kBFloat16) ||
        (noise && n_noise != elements) || (!noise && n_noise != 0)) {
        return invalid("native calibration noise shape is invalid");
    }
    output->resize(elements);
    if (noise) {
        for (std::size_t i = 0; i < elements; ++i) {
            if (!std::isfinite(noise[i])) {
                return invalid(
                    "native calibration noise contains non-finite data");
            }
            (*output)[i] = encode(noise[i], dtype);
        }
        return modalities::Status::ok();
    }

    constexpr double kTwoPi = 6.283185307179586476925286766559;
    std::uint64_t state = seed ^ 0x243f6a8885a308d3ull;
    for (std::size_t i = 0; i < elements; i += 2) {
        const double radius = std::sqrt(-2.0 * std::log(uniform_open(&state)));
        const double angle = kTwoPi * uniform_open(&state);
        (*output)[i] = encode(
            static_cast<float>(radius * std::cos(angle)), dtype);
        if (i + 1 < elements) {
            (*output)[i + 1] = encode(
                static_cast<float>(radius * std::sin(angle)), dtype);
        }
    }
    return modalities::Status::ok();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

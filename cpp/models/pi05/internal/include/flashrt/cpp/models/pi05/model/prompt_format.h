#ifndef FLASHRT_CPP_MODELS_PI05_PROMPT_FORMAT_H
#define FLASHRT_CPP_MODELS_PI05_PROMPT_FORMAT_H

#include <cstdint>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

std::vector<std::int64_t> discretize_state_prompt_bins(
    const float* state, std::uint64_t n);

std::string clean_task_prompt(const std::string& prompt);

std::string format_state_prompt(const std::string& prompt,
                                const float* state,
                                std::uint64_t n_state);

void format_state_prompt_into(const std::string& prompt,
                              const float* state,
                              std::uint64_t n_state,
                              std::string* out);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_PROMPT_FORMAT_H

#include "flashrt/cpp/models/pi05/prompt_format.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

std::vector<float> make_openpi_bins() {
    std::vector<float> bins;
    bins.reserve(256);
    for (int i = 0; i < 256; ++i) {
        bins.push_back(-1.0f + static_cast<float>(i) * (2.0f / 256.0f));
    }
    return bins;
}

bool ascii_space(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

}  // namespace

std::vector<std::int64_t> discretize_state_prompt_bins(
    const float* state, std::uint64_t n) {
    static const std::vector<float> bins = make_openpi_bins();
    std::vector<std::int64_t> out;
    out.reserve(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
        const auto it = std::upper_bound(bins.begin(), bins.end(), state[i]);
        out.push_back(static_cast<std::int64_t>(it - bins.begin()) - 1);
    }
    return out;
}

std::string clean_task_prompt(const std::string& prompt) {
    auto begin = prompt.begin();
    auto end = prompt.end();
    while (begin != end && ascii_space(*begin)) ++begin;
    while (begin != end && ascii_space(*(end - 1))) --end;

    std::string cleaned(begin, end);
    for (char& c : cleaned) {
        if (c == '_' || c == '\n') c = ' ';
    }
    return cleaned;
}

std::string format_state_prompt(const std::string& prompt,
                                const float* state,
                                std::uint64_t n_state) {
    const std::string cleaned = clean_task_prompt(prompt);
    if (!state) return cleaned;

    const auto tokens = discretize_state_prompt_bins(state, n_state);
    std::ostringstream oss;
    oss << "Task: " << cleaned << ", State: ";
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i) oss << ' ';
        oss << tokens[i];
    }
    oss << ";\nAction: ";
    return oss.str();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

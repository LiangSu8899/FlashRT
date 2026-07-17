#include "flashrt/cpp/models/pi05/model/prompt_format.h"

#include <algorithm>
#include <cctype>
#include <charconv>

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
    std::string out;
    out.reserve(prompt.size() + static_cast<std::size_t>(n_state) * 5 + 32);
    format_state_prompt_into(prompt, state, n_state, &out);
    return out;
}

void format_state_prompt_into(const std::string& prompt,
                              const float* state,
                              std::uint64_t n_state,
                              std::string* out) {
    if (!out) return;
    out->clear();
    auto begin = prompt.begin();
    auto end = prompt.end();
    while (begin != end && ascii_space(*begin)) ++begin;
    while (begin != end && ascii_space(*(end - 1))) --end;

    if (state) out->append("Task: ");
    for (auto it = begin; it != end; ++it) {
        out->push_back(*it == '_' || *it == '\n' ? ' ' : *it);
    }
    if (!state) return;

    static const std::vector<float> bins = make_openpi_bins();
    out->append(", State: ");
    char number[24];
    for (std::uint64_t i = 0; i < n_state; ++i) {
        if (i) out->push_back(' ');
        const auto bin = static_cast<std::int64_t>(
                             std::upper_bound(bins.begin(), bins.end(), state[i]) -
                             bins.begin()) -
                         1;
        const auto result = std::to_chars(number, number + sizeof(number), bin);
        out->append(number, result.ptr);
    }
    out->append(";\nAction: ");
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

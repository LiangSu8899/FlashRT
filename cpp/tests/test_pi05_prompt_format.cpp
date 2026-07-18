#include "flashrt/cpp/models/pi05/model/prompt_format.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void test_discretize_matches_python_reference() {
    const float state[] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f};
    const auto bins = flashrt::models::pi05::discretize_state_prompt_bins(
        state, 5);
    const std::vector<std::int64_t> expected = {0, 128, 255, 255, -1};
    assert(bins == expected);
    assert(flashrt::models::pi05::discretize_state_prompt_bins(nullptr, 1)
               .empty());
}

void test_prompt_format_matches_python_reference() {
    const float state[] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f};
    const std::string out = flashrt::models::pi05::format_state_prompt(
        "pick_up\nred", state, 5);
    assert(out ==
           "Task: pick up red, State: 0 128 255 255 -1;\nAction: ");
    std::string workspace;
    workspace.reserve(128);
    const std::size_t capacity = workspace.capacity();
    for (int i = 0; i < 1000; ++i) {
        flashrt::models::pi05::format_state_prompt_into(
            "pick_up\nred", state, 5, &workspace);
        assert(workspace == out);
        assert(workspace.capacity() == capacity);
    }
}

void test_prompt_without_state_keeps_text_only_format() {
    const std::string out = flashrt::models::pi05::format_state_prompt(
        " pick_up\nred ", nullptr, 0);
    assert(out == "pick up red");
}

void test_boundary_values() {
    const float eps = 1.0f / 1024.0f;
    const float state[] = {
        -1.0f - eps,
        -1.0f,
        -1.0f + eps,
        1.0f - eps,
        1.0f,
        std::numeric_limits<float>::quiet_NaN(),
    };
    const auto bins = flashrt::models::pi05::discretize_state_prompt_bins(
        state, 6);
    assert(bins[0] == -1);
    assert(bins[1] == 0);
    assert(bins[2] == 0);
    assert(bins[3] == 255);
    assert(bins[4] == 255);
    assert(bins[5] == 255);
}

}  // namespace

int main() {
    test_discretize_matches_python_reference();
    test_prompt_format_matches_python_reference();
    test_prompt_without_state_keeps_text_only_format();
    test_boundary_values();
    std::cout << "PASS - Pi05 prompt formatter\n";
    return 0;
}

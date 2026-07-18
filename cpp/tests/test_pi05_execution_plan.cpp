#include "flashrt/cpp/models/pi05/model/execution_plan.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace pi05 = flashrt::models::pi05;

namespace {

[[noreturn]] void fail(const char* expression, int line) {
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    std::abort();
}

#define CHECK(expression)                              \
    do {                                               \
        if (!(expression)) fail(#expression, __LINE__); \
    } while (false)

bool same_text(const char* lhs, const char* rhs) {
    return lhs && rhs && std::strcmp(lhs, rhs) == 0;
}

void test_binding_names() {
    constexpr std::array<const char*, 7> kExpected = {
        "observation_images_normalized",
        "prompt_embedding",
        "encoder_x",
        "diffusion_noise",
        "rtc_prev_action_chunk",
        "rtc_prefix_weights",
        "rtc_guidance_weight",
    };
    CHECK(kExpected.size() ==
          static_cast<std::size_t>(pi05::Pi05GraphBindingId::kCount));
    for (std::size_t i = 0; i < kExpected.size(); ++i) {
        CHECK(same_text(pi05::pi05_graph_binding_name(
                            static_cast<pi05::Pi05GraphBindingId>(i)),
                        kExpected[i]));
    }
    CHECK(pi05::pi05_graph_binding_name(
              pi05::Pi05GraphBindingId::kCount) == nullptr);
}

void check_bindings(
    const pi05::Pi05GraphDescriptor& graph,
    const pi05::Pi05GraphBindingId* expected,
    std::size_t expected_count) {
    CHECK(graph.bindings != nullptr);
    CHECK(graph.binding_count == expected_count);
    for (std::size_t i = 0; i < expected_count; ++i) {
        CHECK(graph.bindings[i] == expected[i]);
    }
}

void test_graph_catalog() {
    constexpr pi05::Pi05GraphBindingId kInfer[] = {
        pi05::Pi05GraphBindingId::kImages,
        pi05::Pi05GraphBindingId::kPromptEmbedding,
        pi05::Pi05GraphBindingId::kEncoderState,
        pi05::Pi05GraphBindingId::kNoise,
        pi05::Pi05GraphBindingId::kPreviousActions,
        pi05::Pi05GraphBindingId::kPrefixWeights,
        pi05::Pi05GraphBindingId::kGuidanceWeight,
    };
    constexpr pi05::Pi05GraphBindingId kDecode[] = {
        pi05::Pi05GraphBindingId::kEncoderState,
        pi05::Pi05GraphBindingId::kNoise,
        pi05::Pi05GraphBindingId::kPreviousActions,
        pi05::Pi05GraphBindingId::kPrefixWeights,
        pi05::Pi05GraphBindingId::kGuidanceWeight,
    };
    constexpr pi05::Pi05GraphBindingId kContext[] = {
        pi05::Pi05GraphBindingId::kImages,
        pi05::Pi05GraphBindingId::kPromptEmbedding,
        pi05::Pi05GraphBindingId::kEncoderState,
    };

    std::size_t count = 0;
    const pi05::Pi05GraphDescriptor* catalog =
        pi05::pi05_graph_catalog(&count);
    CHECK(catalog != nullptr);
    CHECK(count == 3);

    CHECK(catalog[0].id == pi05::Pi05GraphId::kInfer);
    CHECK(same_text(catalog[0].name, "infer"));
    CHECK(catalog[0].body == pi05::Pi05RecordBody::kFull);
    check_bindings(catalog[0], kInfer, sizeof(kInfer) / sizeof(kInfer[0]));

    CHECK(catalog[1].id == pi05::Pi05GraphId::kDecodeOnly);
    CHECK(same_text(catalog[1].name, "decode_only"));
    CHECK(catalog[1].body == pi05::Pi05RecordBody::kDecode);
    check_bindings(catalog[1], kDecode,
                   sizeof(kDecode) / sizeof(kDecode[0]));

    CHECK(catalog[2].id == pi05::Pi05GraphId::kContext);
    CHECK(same_text(catalog[2].name, "context"));
    CHECK(catalog[2].body == pi05::Pi05RecordBody::kContext);
    check_bindings(catalog[2], kContext,
                   sizeof(kContext) / sizeof(kContext[0]));

    for (std::size_t i = 0; i < count; ++i) {
        CHECK(pi05::pi05_graph_descriptor(
                  static_cast<pi05::Pi05GraphId>(i)) == &catalog[i]);
    }
    CHECK(pi05::pi05_graph_descriptor(pi05::Pi05GraphId::kCount) == nullptr);
}

void test_stage_plans() {
    const pi05::Pi05ExecutionPlanDescriptor* full =
        pi05::pi05_execution_plan("full");
    CHECK(full != nullptr);
    CHECK(same_text(full->name, "full"));
    CHECK(full->stage_count == 1);
    CHECK(same_text(full->stages[0].name, "infer"));
    CHECK(full->stages[0].graph == pi05::Pi05GraphId::kInfer);
    CHECK(full->stages[0].after == nullptr);
    CHECK(full->stages[0].after_count == 0);

    const pi05::Pi05ExecutionPlanDescriptor* split =
        pi05::pi05_execution_plan("context_action");
    CHECK(split != nullptr);
    CHECK(same_text(split->name, "context_action"));
    CHECK(split->stage_count == 2);
    CHECK(same_text(split->stages[0].name, "context"));
    CHECK(split->stages[0].graph == pi05::Pi05GraphId::kContext);
    CHECK(split->stages[0].after == nullptr);
    CHECK(split->stages[0].after_count == 0);
    CHECK(same_text(split->stages[1].name, "action"));
    CHECK(split->stages[1].graph == pi05::Pi05GraphId::kDecodeOnly);
    CHECK(split->stages[1].after != nullptr);
    CHECK(split->stages[1].after_count == 1);
    CHECK(split->stages[1].after[0] == 0);

    CHECK(pi05::pi05_execution_plan(nullptr) == nullptr);
    CHECK(pi05::pi05_execution_plan("") == nullptr);
    CHECK(pi05::pi05_execution_plan("unknown") == nullptr);
}

}  // namespace

int main() {
    test_binding_names();
    test_graph_catalog();
    test_stage_plans();
    std::cout << "PASS - PI0.5 execution plan contract\n";
    return 0;
}

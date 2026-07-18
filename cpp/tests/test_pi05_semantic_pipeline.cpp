#include "flashrt/cpp/models/pi05/model/semantic_pipeline.h"
#include "flashrt/cpp/models/pi05/spec.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace pi05 = flashrt::models::pi05;

namespace {

[[noreturn]] void fail(const char* expression, int line) {
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    std::abort();
}

#define CHECK(expression)                     \
    do {                                      \
        if (!(expression)) fail(#expression, __LINE__); \
    } while (false)

pi05::Pi05ResolvedShape canonical_shape(int steps = 10) {
    pi05::Pi05ShapeConfig config;
    config.num_views = 3;
    config.max_prompt_tokens = 64;
    config.chunk = 10;
    config.num_steps = steps;
    config.vision_pool_factor = 1;
    config.state_dim = 8;
    config.robot_action_dim = 7;
    pi05::Pi05ResolvedShape shape;
    CHECK(pi05::resolve_pi05_shape(config, &shape).ok_status());
    return shape;
}

bool same_call(const pi05::Pi05OperationCall& lhs,
               const pi05::Pi05OperationCall& rhs) {
    return lhs.id == rhs.id && lhs.layer == rhs.layer &&
           lhs.step == rhs.step &&
           lhs.input_generation == rhs.input_generation &&
           lhs.output_generation == rhs.output_generation;
}

struct ExpectedCall {
    pi05::Pi05OperationId id;
    int layer;
    int step;
};

std::vector<ExpectedCall> expected_context() {
    std::vector<ExpectedCall> expected;
    expected.push_back({pi05::Pi05OperationId::kComposePrompt, -1, -1});
    expected.push_back({pi05::Pi05OperationId::kVisionEmbed, -1, -1});
    for (int layer = 0; layer < 27; ++layer) {
        expected.push_back(
            {pi05::Pi05OperationId::kVisionAttention, layer, -1});
        expected.push_back({pi05::Pi05OperationId::kVisionMlp, layer, -1});
    }
    expected.push_back({pi05::Pi05OperationId::kVisionProject, -1, -1});
    for (int layer = 0; layer < 17; ++layer) {
        expected.push_back(
            {pi05::Pi05OperationId::kEncoderAttention, layer, -1});
        expected.push_back({pi05::Pi05OperationId::kEncoderMlp, layer, -1});
    }
    expected.push_back(
        {pi05::Pi05OperationId::kEncoderCacheFinalize, 17, -1});
    return expected;
}

std::vector<ExpectedCall> expected_decode(int steps) {
    std::vector<ExpectedCall> expected;
    for (int step = 0; step < steps; ++step) {
        expected.push_back(
            {pi05::Pi05OperationId::kDiffusionInputProject, -1, step});
        for (int layer = 0; layer < 18; ++layer) {
            expected.push_back(
                {pi05::Pi05OperationId::kDecoderAttention, layer, step});
            expected.push_back(
                {pi05::Pi05OperationId::kDecoderMlp, layer, step});
        }
        expected.push_back(
            {pi05::Pi05OperationId::kActionProject, -1, step});
        expected.push_back(
            {pi05::Pi05OperationId::kDiffusionUpdate, -1, step});
    }
    return expected;
}

class TraceSink final : public pi05::Pi05OperationSink {
public:
    explicit TraceSink(bool has_context_cache = false,
                       std::size_t fail_at = std::numeric_limits<std::size_t>::max())
        : fail_at_(fail_at) {
        if (has_context_cache) {
            generation_[static_cast<std::size_t>(pi05::Pi05ValueId::kKeyCache)] =
                18;
            generation_[static_cast<std::size_t>(
                pi05::Pi05ValueId::kValueCache)] = 18;
        }
    }

    flashrt::modalities::Status record(
        const pi05::Pi05OperationCall& call,
        const pi05::Pi05ResolvedShape& shape,
        pi05::Pi05Stream stream) override {
        CHECK(stream == 73);
        CHECK(pi05::validate_pi05_operation_call(call, shape).ok_status());
        const pi05::Pi05OperationContract* contract =
            pi05::pi05_operation_contract(call.id);
        CHECK(contract != nullptr);
        for (std::size_t i = 0; i < contract->input_count; ++i) {
            const std::size_t value =
                static_cast<std::size_t>(contract->inputs[i]);
            CHECK(call.input_generation[i] == generation_[value]);
        }
        if (calls.size() == fail_at_) {
            return flashrt::modalities::Status::error(
                flashrt::modalities::StatusCode::kBackend,
                "injected trace failure");
        }
        for (std::size_t i = 0; i < contract->output_count; ++i) {
            const std::size_t value =
                static_cast<std::size_t>(contract->outputs[i]);
            if (contract->output_alias_input[i] == pi05::kPi05NoAlias) {
                CHECK(call.output_generation[i] == generation_[value] + 1);
            }
            generation_[value] = call.output_generation[i];
        }
        calls.push_back(call);
        return flashrt::modalities::Status::ok();
    }

    std::vector<pi05::Pi05OperationCall> calls;

private:
    std::array<std::uint64_t,
               static_cast<std::size_t>(pi05::Pi05ValueId::kCount)>
        generation_{};
    std::size_t fail_at_;
};

void check_sequence(const std::vector<pi05::Pi05OperationCall>& actual,
                    const std::vector<ExpectedCall>& expected) {
    CHECK(actual.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK(actual[i].id == expected[i].id);
        CHECK(actual[i].layer == expected[i].layer);
        CHECK(actual[i].step == expected[i].step);
    }
}

void test_model_dims_and_shape_resolution() {
    static_assert(pi05::kPi05ModelDims.vision_heads *
                          pi05::kPi05ModelDims.vision_head_dim ==
                      pi05::kPi05ModelDims.vision_width,
                  "vision attention width must be coherent");
    static_assert(pi05::kPi05ModelDims.encoder_heads *
                          pi05::kPi05ModelDims.encoder_head_dim ==
                      pi05::kPi05ModelDims.encoder_width,
                  "encoder attention width must be coherent");
    static_assert(pi05::kPi05ModelDims.decoder_heads *
                          pi05::kPi05ModelDims.decoder_head_dim ==
                      pi05::kPi05ModelDims.encoder_width,
                  "decoder attention output width must be coherent");
    static_assert(pi05::kImageSize == pi05::kPi05ModelDims.image_width,
                  "legacy image constant must match model dims");
    static_assert(pi05::kDefaultChunk == 10,
                  "legacy default chunk changed unexpectedly");
    static_assert(pi05::kModelActionDim ==
                      pi05::kPi05ModelDims.action_width,
                  "legacy action constant must match model dims");

    const pi05::Pi05ResolvedShape shape = canonical_shape();
    CHECK(shape.num_views == 3);
    CHECK(shape.pool_area == 1);
    CHECK(shape.vision_sequence == 768);
    CHECK(shape.encoder_vision_sequence == 768);
    CHECK(shape.encoder_sequence == 832);
    CHECK(shape.total_attention_keys == 842);
    CHECK(pi05::validate_pi05_resolved_shape(shape).ok_status());

    for (const int pool : {1, 2, 4}) {
        pi05::Pi05ShapeConfig config;
        config.num_views = 3;
        config.max_prompt_tokens = 64;
        config.chunk = 10;
        config.num_steps = 1;
        config.vision_pool_factor = pool;
        config.state_dim = 8;
        config.robot_action_dim = 7;
        pi05::Pi05ResolvedShape pooled;
        CHECK(pi05::resolve_pi05_shape(config, &pooled).ok_status());
        CHECK(pooled.encoder_vision_sequence == 768 / (pool * pool));
        CHECK(pooled.encoder_sequence ==
              pooled.encoder_vision_sequence + 64);
    }

    pi05::Pi05ShapeConfig invalid;
    invalid.num_views = 3;
    invalid.max_prompt_tokens = 64;
    invalid.chunk = 10;
    invalid.num_steps = 10;
    invalid.vision_pool_factor = 1;
    invalid.state_dim = 8;
    invalid.robot_action_dim = 7;
    for (int field = 0; field < 6; ++field) {
        pi05::Pi05ShapeConfig candidate = invalid;
        switch (field) {
            case 0: candidate.num_views = 4; break;
            case 1: candidate.max_prompt_tokens = 0; break;
            case 2: candidate.chunk = 0; break;
            case 3: candidate.num_steps = 0; break;
            case 4: candidate.state_dim = 0; break;
            case 5: candidate.robot_action_dim = 33; break;
        }
        pi05::Pi05ResolvedShape untouched;
        untouched.num_views = 99;
        CHECK(!pi05::resolve_pi05_shape(candidate, &untouched).ok_status());
        CHECK(untouched.num_views == 99);
    }
    invalid.vision_pool_factor = 3;
    pi05::Pi05ResolvedShape untouched;
    untouched.num_views = 99;
    CHECK(!pi05::resolve_pi05_shape(invalid, &untouched).ok_status());
    CHECK(untouched.num_views == 99);

    invalid = {};
    invalid.num_views = 1;
    invalid.max_prompt_tokens = std::numeric_limits<int>::max();
    invalid.chunk = 1;
    invalid.num_steps = 1;
    invalid.vision_pool_factor = 1;
    invalid.state_dim = 1;
    invalid.robot_action_dim = 1;
    CHECK(!pi05::resolve_pi05_shape(invalid, &untouched).ok_status());

    pi05::Pi05ResolvedShape inconsistent = shape;
    ++inconsistent.encoder_sequence;
    CHECK(!pi05::validate_pi05_resolved_shape(inconsistent).ok_status());
}

void test_value_specs() {
    const pi05::Pi05ResolvedShape shape = canonical_shape();
    struct Expected {
        pi05::Pi05ValueId id;
        pi05::Pi05ScalarKind scalar;
        std::uint8_t rank;
        std::array<std::uint64_t, 4> dimensions;
    };
    const Expected expected[] = {
        {pi05::Pi05ValueId::kImages, pi05::Pi05ScalarKind::kActivation, 4,
         {3, 224, 224, 3}},
        {pi05::Pi05ValueId::kPromptEmbedding,
         pi05::Pi05ScalarKind::kActivation, 2, {64, 2048, 0, 0}},
        {pi05::Pi05ValueId::kVisionState,
         pi05::Pi05ScalarKind::kActivation, 2, {768, 1152, 0, 0}},
        {pi05::Pi05ValueId::kEncoderState,
         pi05::Pi05ScalarKind::kActivation, 2, {832, 2048, 0, 0}},
        {pi05::Pi05ValueId::kKeyCache,
         pi05::Pi05ScalarKind::kActivation, 3, {18, 842, 256, 0}},
        {pi05::Pi05ValueId::kValueCache,
         pi05::Pi05ScalarKind::kActivation, 3, {18, 842, 256, 0}},
        {pi05::Pi05ValueId::kNoise, pi05::Pi05ScalarKind::kActivation, 2,
         {10, 32, 0, 0}},
        {pi05::Pi05ValueId::kDecoderState,
         pi05::Pi05ScalarKind::kActivation, 2, {10, 1024, 0, 0}},
        {pi05::Pi05ValueId::kActionDelta,
         pi05::Pi05ScalarKind::kActionUpdate, 2, {10, 32, 0, 0}},
    };
    for (const Expected& item : expected) {
        pi05::Pi05TensorSpec spec;
        CHECK(pi05::pi05_value_spec(item.id, shape, &spec).ok_status());
        CHECK(spec.scalar == item.scalar);
        CHECK(spec.rank == item.rank);
        CHECK(spec.dimensions == item.dimensions);
        CHECK(pi05::pi05_value_name(item.id) != nullptr);
    }
    pi05::Pi05TensorSpec untouched;
    untouched.rank = 9;
    CHECK(!pi05::pi05_value_spec(pi05::Pi05ValueId::kCount, shape,
                                 &untouched)
               .ok_status());
    CHECK(untouched.rank == 9);
}

void test_operation_contracts() {
    using V = pi05::Pi05ValueId;
    using O = pi05::Pi05OperationId;
    struct Expected {
        O id;
        pi05::Pi05IndexDomain domain;
        std::vector<V> inputs;
        std::vector<V> outputs;
        std::vector<std::uint8_t> aliases;
    };
    const Expected expected[] = {
        {O::kComposePrompt, pi05::Pi05IndexDomain::kNone,
         {V::kPromptEmbedding}, {V::kEncoderState}, {pi05::kPi05NoAlias}},
        {O::kVisionEmbed, pi05::Pi05IndexDomain::kNone,
         {V::kImages}, {V::kVisionState}, {pi05::kPi05NoAlias}},
        {O::kVisionAttention, pi05::Pi05IndexDomain::kVisionLayer,
         {V::kVisionState}, {V::kVisionState}, {0}},
        {O::kVisionMlp, pi05::Pi05IndexDomain::kVisionLayer,
         {V::kVisionState}, {V::kVisionState}, {0}},
        {O::kVisionProject, pi05::Pi05IndexDomain::kNone,
         {V::kVisionState, V::kEncoderState}, {V::kEncoderState}, {1}},
        {O::kEncoderAttention, pi05::Pi05IndexDomain::kEncoderLayer,
         {V::kEncoderState, V::kKeyCache, V::kValueCache},
         {V::kEncoderState, V::kKeyCache, V::kValueCache}, {0, 1, 2}},
        {O::kEncoderMlp, pi05::Pi05IndexDomain::kEncoderLayer,
         {V::kEncoderState}, {V::kEncoderState}, {0}},
        {O::kEncoderCacheFinalize,
         pi05::Pi05IndexDomain::kEncoderFinalLayer,
         {V::kEncoderState, V::kKeyCache, V::kValueCache},
         {V::kKeyCache, V::kValueCache}, {1, 2}},
        {O::kDiffusionInputProject, pi05::Pi05IndexDomain::kDiffusionStep,
         {V::kNoise}, {V::kDecoderState}, {pi05::kPi05NoAlias}},
        {O::kDecoderAttention, pi05::Pi05IndexDomain::kDecoderLayer,
         {V::kDecoderState, V::kKeyCache, V::kValueCache},
         {V::kDecoderState, V::kKeyCache, V::kValueCache}, {0, 1, 2}},
        {O::kDecoderMlp, pi05::Pi05IndexDomain::kDecoderLayer,
         {V::kDecoderState}, {V::kDecoderState}, {0}},
        {O::kActionProject, pi05::Pi05IndexDomain::kDiffusionStep,
         {V::kDecoderState}, {V::kActionDelta}, {pi05::kPi05NoAlias}},
        {O::kDiffusionUpdate, pi05::Pi05IndexDomain::kDiffusionStep,
         {V::kNoise, V::kActionDelta}, {V::kNoise}, {0}},
    };
    CHECK(sizeof(expected) / sizeof(expected[0]) ==
          static_cast<std::size_t>(O::kCount));
    for (const Expected& item : expected) {
        const pi05::Pi05OperationContract* contract =
            pi05::pi05_operation_contract(item.id);
        CHECK(contract != nullptr);
        CHECK(contract->id == item.id);
        CHECK(contract->index_domain == item.domain);
        CHECK(contract->input_count == item.inputs.size());
        CHECK(contract->output_count == item.outputs.size());
        CHECK(pi05::pi05_operation_name(item.id) != nullptr);
        for (std::size_t i = 0; i < item.inputs.size(); ++i) {
            CHECK(contract->inputs[i] == item.inputs[i]);
        }
        for (std::size_t i = 0; i < item.outputs.size(); ++i) {
            CHECK(contract->outputs[i] == item.outputs[i]);
            CHECK(contract->output_alias_input[i] == item.aliases[i]);
        }
    }
}

void test_semantic_traces() {
    const pi05::Pi05ResolvedShape shape = canonical_shape();
    const pi05::Pi05SemanticPipeline pipeline(shape);

    TraceSink context;
    CHECK(pipeline.record_context(context, 73).ok_status());
    const std::vector<ExpectedCall> context_expected = expected_context();
    CHECK(context_expected.size() == 92);
    check_sequence(context.calls, context_expected);

    TraceSink decode(true);
    CHECK(pipeline.record_decode(decode, 73).ok_status());
    const std::vector<ExpectedCall> decode_expected = expected_decode(10);
    CHECK(decode_expected.size() == 390);
    check_sequence(decode.calls, decode_expected);

    TraceSink full;
    CHECK(pipeline.record_full(full, 73).ok_status());
    std::vector<ExpectedCall> full_expected = context_expected;
    full_expected.insert(full_expected.end(), decode_expected.begin(),
                         decode_expected.end());
    CHECK(full_expected.size() == 482);
    check_sequence(full.calls, full_expected);
    CHECK(full.calls.size() == context.calls.size() + decode.calls.size());
    for (std::size_t i = 0; i < context.calls.size(); ++i) {
        CHECK(same_call(full.calls[i], context.calls[i]));
    }
    for (std::size_t i = 0; i < decode.calls.size(); ++i) {
        CHECK(same_call(full.calls[context.calls.size() + i], decode.calls[i]));
    }

    TraceSink failure(false, 37);
    const flashrt::modalities::Status status =
        pipeline.record_full(failure, 73);
    CHECK(!status.ok_status());
    CHECK(status.code == flashrt::modalities::StatusCode::kBackend);
    CHECK(failure.calls.size() == 37);

    pi05::Pi05OperationCall bad = full.calls[2];
    bad.layer = 27;
    CHECK(!pi05::validate_pi05_operation_call(bad, shape).ok_status());
    bad = full.calls[2];
    ++bad.output_generation[0];
    CHECK(!pi05::validate_pi05_operation_call(bad, shape).ok_status());
    bad = full.calls[0];
    bad.input_generation[1] = 1;
    CHECK(!pi05::validate_pi05_operation_call(bad, shape).ok_status());
}

}  // namespace

int main() {
    test_model_dims_and_shape_resolution();
    test_value_specs();
    test_operation_contracts();
    test_semantic_traces();
    std::cout << "PASS - PI0.5 semantic pipeline contract\n";
    return 0;
}

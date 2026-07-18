#include "flashrt/cpp/models/pi05/model/linear_weight_groups.h"

#include "pi05_resolved_fixture.h"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace pi05 = flashrt::models::pi05;
namespace fixture = flashrt::tests::pi05_fixture;

namespace {

[[noreturn]] void fail(const char* expression, int line) {
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    std::abort();
}

#define CHECK(expression)                                      \
    do {                                                       \
        if (!(expression)) fail(#expression, __LINE__);        \
    } while (false)

class TraceSink final : public pi05::Pi05LinearWeightGroupSink {
public:
    explicit TraceSink(
        std::size_t fail_at = std::numeric_limits<std::size_t>::max())
        : fail_at_(fail_at) {}

    flashrt::modalities::Status record(
        const pi05::Pi05LinearWeightGroup& group) override {
        if (groups.size() == fail_at_) {
            return flashrt::modalities::Status::error(
                flashrt::modalities::StatusCode::kBackend,
                "injected weight visitor failure");
        }
        groups.push_back(group);
        return flashrt::modalities::Status::ok();
    }

    std::vector<pi05::Pi05LinearWeightGroup> groups;

private:
    std::size_t fail_at_;
};

void check_single(
    const pi05::Pi05LinearWeightGroup& group,
    pi05::Pi05LinearDomain domain,
    pi05::Pi05LinearRole role,
    int layer,
    pi05::Pi05ResolvedWeight* weight) {
    CHECK(group.key.domain == domain);
    CHECK(group.key.role == role);
    CHECK(group.key.layer == layer);
    CHECK(group.first == weight);
    CHECK(group.second == nullptr);
    CHECK(group.fused == nullptr);
}

void check_gate_up(
    const pi05::Pi05LinearWeightGroup& group,
    pi05::Pi05LinearDomain domain,
    int layer,
    pi05::Pi05FeedForwardWeights* weights) {
    CHECK(group.key.domain == domain);
    CHECK(group.key.role == pi05::Pi05LinearRole::kMlpGateUpGroup);
    CHECK(group.key.layer == layer);
    CHECK(group.first == &weights->gate_weight);
    CHECK(group.second == &weights->up_weight);
    CHECK(group.fused == &weights->gate_up_weight);
}

void test_canonical_order() {
    pi05::Pi05ResolvedWeights weights;
    const pi05::Pi05ResolvedShape shape = fixture::canonical_shape();
    fixture::fill_weights(
        &weights, reinterpret_cast<void*>(0x1000), shape,
        flashrt::modalities::DType::kBFloat16, false);

    TraceSink sink;
    CHECK(pi05::visit_pi05_linear_weight_groups(&weights, &sink)
              .ok_status());
    CHECK(sink.groups.size() == 253);

    std::size_t cursor = 0;
    for (std::size_t i = 0; i < weights.vision_layers.size(); ++i) {
        pi05::Pi05VisionLayerWeights& layer = weights.vision_layers[i];
        const int index = static_cast<int>(i);
        check_single(
            sink.groups[cursor++], pi05::Pi05LinearDomain::kVision,
            pi05::Pi05LinearRole::kAttentionQkv, index,
            &layer.attention_qkv_weight);
        check_single(
            sink.groups[cursor++], pi05::Pi05LinearDomain::kVision,
            pi05::Pi05LinearRole::kAttentionOutput, index,
            &layer.attention_output_weight);
        check_single(
            sink.groups[cursor++], pi05::Pi05LinearDomain::kVision,
            pi05::Pi05LinearRole::kMlpUp, index, &layer.mlp_up_weight);
        check_single(
            sink.groups[cursor++], pi05::Pi05LinearDomain::kVision,
            pi05::Pi05LinearRole::kMlpDown, index,
            &layer.mlp_down_weight);
    }
    CHECK(cursor == 108);
    check_single(
        sink.groups[cursor++], pi05::Pi05LinearDomain::kVision,
        pi05::Pi05LinearRole::kProjector, -1,
        &weights.vision.projector_weight);
    CHECK(cursor == 109);

    for (std::size_t i = 0; i < weights.encoder_layers.size(); ++i) {
        pi05::Pi05EncoderLayerWeights& layer = weights.encoder_layers[i];
        const int index = static_cast<int>(i);
        check_single(
            sink.groups[cursor++], pi05::Pi05LinearDomain::kEncoder,
            pi05::Pi05LinearRole::kAttentionQkv, index,
            &layer.attention_qkv_weight);
        check_single(
            sink.groups[cursor++], pi05::Pi05LinearDomain::kEncoder,
            pi05::Pi05LinearRole::kAttentionOutput, index,
            &layer.attention_output_weight);
        check_gate_up(
            sink.groups[cursor++], pi05::Pi05LinearDomain::kEncoder,
            index, &layer.mlp);
        check_single(
            sink.groups[cursor++], pi05::Pi05LinearDomain::kEncoder,
            pi05::Pi05LinearRole::kMlpDown, index,
            &layer.mlp.down_weight);
    }
    CHECK(cursor == 181);

    for (std::size_t i = 0; i < weights.decoder_layers.size(); ++i) {
        pi05::Pi05DecoderLayerWeights& layer = weights.decoder_layers[i];
        const int index = static_cast<int>(i);
        check_single(
            sink.groups[cursor++], pi05::Pi05LinearDomain::kDecoder,
            pi05::Pi05LinearRole::kAttentionQkv, index,
            &layer.attention_qkv_weight);
        check_single(
            sink.groups[cursor++], pi05::Pi05LinearDomain::kDecoder,
            pi05::Pi05LinearRole::kAttentionOutput, index,
            &layer.attention_output_weight);
        check_gate_up(
            sink.groups[cursor++], pi05::Pi05LinearDomain::kDecoder,
            index, &layer.mlp);
        check_single(
            sink.groups[cursor++], pi05::Pi05LinearDomain::kDecoder,
            pi05::Pi05LinearRole::kMlpDown, index,
            &layer.mlp.down_weight);
    }
    CHECK(cursor == sink.groups.size());
}

void test_failure_boundary() {
    pi05::Pi05ResolvedWeights weights;
    const pi05::Pi05ResolvedShape shape = fixture::canonical_shape();
    fixture::fill_weights(
        &weights, reinterpret_cast<void*>(0x1000), shape,
        flashrt::modalities::DType::kBFloat16, false);
    TraceSink sink(109);
    const flashrt::modalities::Status status =
        pi05::visit_pi05_linear_weight_groups(&weights, &sink);
    CHECK(!status.ok_status());
    CHECK(status.code == flashrt::modalities::StatusCode::kBackend);
    CHECK(sink.groups.size() == 109);
    CHECK(!pi05::visit_pi05_linear_weight_groups(nullptr, &sink)
               .ok_status());
    CHECK(!pi05::visit_pi05_linear_weight_groups(&weights, nullptr)
               .ok_status());
}

}  // namespace

int main() {
    test_canonical_order();
    test_failure_boundary();
    return 0;
}

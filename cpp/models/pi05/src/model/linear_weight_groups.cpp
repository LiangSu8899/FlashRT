#include "flashrt/cpp/models/pi05/model/linear_weight_groups.h"

#include <cstddef>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status record_single(
    Pi05LinearWeightGroupSink* sink,
    Pi05LinearDomain domain,
    Pi05LinearRole role,
    int layer,
    Pi05ResolvedWeight* weight) {
    if (!sink || !weight) {
        return invalid("PI0.5 linear weight group is invalid");
    }
    return sink->record({{domain, role, layer}, weight, nullptr, nullptr});
}

modalities::Status record_gate_up(
    Pi05LinearWeightGroupSink* sink,
    Pi05LinearDomain domain,
    int layer,
    Pi05FeedForwardWeights* weights) {
    if (!sink || !weights) {
        return invalid("PI0.5 gate/up weight group is invalid");
    }
    return sink->record({
        {domain, Pi05LinearRole::kMlpGateUpGroup, layer},
        &weights->gate_weight,
        &weights->up_weight,
        &weights->gate_up_weight,
    });
}

}  // namespace

modalities::Status visit_pi05_linear_weight_groups(
    Pi05ResolvedWeights* weights,
    Pi05LinearWeightGroupSink* sink) {
    if (!weights || !sink) {
        return invalid("PI0.5 linear weight visitor is invalid");
    }
    modalities::Status status;
    for (std::size_t i = 0; i < weights->vision_layers.size(); ++i) {
        Pi05VisionLayerWeights& layer = weights->vision_layers[i];
        const int index = static_cast<int>(i);
        status = record_single(
            sink, Pi05LinearDomain::kVision,
            Pi05LinearRole::kAttentionQkv, index,
            &layer.attention_qkv_weight);
        if (!status.ok_status()) return status;
        status = record_single(
            sink, Pi05LinearDomain::kVision,
            Pi05LinearRole::kAttentionOutput, index,
            &layer.attention_output_weight);
        if (!status.ok_status()) return status;
        status = record_single(
            sink, Pi05LinearDomain::kVision,
            Pi05LinearRole::kMlpUp, index, &layer.mlp_up_weight);
        if (!status.ok_status()) return status;
        status = record_single(
            sink, Pi05LinearDomain::kVision,
            Pi05LinearRole::kMlpDown, index, &layer.mlp_down_weight);
        if (!status.ok_status()) return status;
    }
    status = record_single(
        sink, Pi05LinearDomain::kVision, Pi05LinearRole::kProjector, -1,
        &weights->vision.projector_weight);
    if (!status.ok_status()) return status;

    for (std::size_t i = 0; i < weights->encoder_layers.size(); ++i) {
        Pi05EncoderLayerWeights& layer = weights->encoder_layers[i];
        const int index = static_cast<int>(i);
        status = record_single(
            sink, Pi05LinearDomain::kEncoder,
            Pi05LinearRole::kAttentionQkv, index,
            &layer.attention_qkv_weight);
        if (!status.ok_status()) return status;
        status = record_single(
            sink, Pi05LinearDomain::kEncoder,
            Pi05LinearRole::kAttentionOutput, index,
            &layer.attention_output_weight);
        if (!status.ok_status()) return status;
        status = record_gate_up(
            sink, Pi05LinearDomain::kEncoder, index, &layer.mlp);
        if (!status.ok_status()) return status;
        status = record_single(
            sink, Pi05LinearDomain::kEncoder,
            Pi05LinearRole::kMlpDown, index, &layer.mlp.down_weight);
        if (!status.ok_status()) return status;
    }

    for (std::size_t i = 0; i < weights->decoder_layers.size(); ++i) {
        Pi05DecoderLayerWeights& layer = weights->decoder_layers[i];
        const int index = static_cast<int>(i);
        status = record_single(
            sink, Pi05LinearDomain::kDecoder,
            Pi05LinearRole::kAttentionQkv, index,
            &layer.attention_qkv_weight);
        if (!status.ok_status()) return status;
        status = record_single(
            sink, Pi05LinearDomain::kDecoder,
            Pi05LinearRole::kAttentionOutput, index,
            &layer.attention_output_weight);
        if (!status.ok_status()) return status;
        status = record_gate_up(
            sink, Pi05LinearDomain::kDecoder, index, &layer.mlp);
        if (!status.ok_status()) return status;
        status = record_single(
            sink, Pi05LinearDomain::kDecoder,
            Pi05LinearRole::kMlpDown, index, &layer.mlp.down_weight);
        if (!status.ok_status()) return status;
    }
    return modalities::Status::ok();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

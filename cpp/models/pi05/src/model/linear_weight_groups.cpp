#include "flashrt/cpp/models/pi05/model/linear_weight_groups.h"

#include <cstddef>
#include <limits>

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

int linear_slot(Pi05LinearDomain domain, Pi05LinearRole role) {
    switch (role) {
        case Pi05LinearRole::kAttentionQkv: return 0;
        case Pi05LinearRole::kAttentionOutput: return 1;
        case Pi05LinearRole::kMlpUp:
            return domain == Pi05LinearDomain::kVision ? 2 : -1;
        case Pi05LinearRole::kMlpGateUpGroup:
            return domain == Pi05LinearDomain::kVision ? -1 : 2;
        case Pi05LinearRole::kMlpDown: return 3;
        case Pi05LinearRole::kProjector: return -1;
    }
    return -1;
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

modalities::Status resolve_pi05_linear_scale_layout(
    const Pi05ResolvedShape& shape,
    Pi05LinearScaleLayout* out) {
    if (!out) return invalid("PI0.5 linear scale layout output is null");
    modalities::Status status = validate_pi05_resolved_shape(shape);
    if (!status.ok_status()) return status;
    const std::size_t decoder_layers =
        static_cast<std::size_t>(kPi05ModelDims.decoder_layers);
    const std::size_t steps = static_cast<std::size_t>(shape.num_steps);
    if (steps > std::numeric_limits<std::size_t>::max() /
                    decoder_layers / kPi05LinearScalesPerLayer) {
        return invalid("PI0.5 decoder scale layout overflows");
    }
    Pi05LinearScaleLayout layout;
    layout.vision =
        static_cast<std::size_t>(kPi05ModelDims.vision_layers) *
            kPi05LinearScalesPerLayer +
        1;
    layout.encoder =
        static_cast<std::size_t>(kPi05ModelDims.encoder_layers) *
        kPi05LinearScalesPerLayer;
    layout.decoder =
        steps * decoder_layers * kPi05LinearScalesPerLayer;
    *out = layout;
    return modalities::Status::ok();
}

modalities::Status resolve_pi05_linear_activation_site(
    const Pi05LinearWeightKey& key,
    int step,
    const Pi05ResolvedShape& shape,
    Pi05LinearActivationSite* out) {
    if (!out) return invalid("PI0.5 linear activation site output is null");
    Pi05LinearScaleLayout layout;
    modalities::Status status =
        resolve_pi05_linear_scale_layout(shape, &layout);
    if (!status.ok_status()) return status;
    Pi05LinearActivationSite site;
    site.domain = key.domain;
    if (key.domain == Pi05LinearDomain::kVision &&
        key.role == Pi05LinearRole::kProjector && key.layer == -1 &&
        step == -1) {
        site.index = layout.vision - 1;
        *out = site;
        return modalities::Status::ok();
    }

    const int slot = linear_slot(key.domain, key.role);
    int layers = 0;
    switch (key.domain) {
        case Pi05LinearDomain::kVision:
            layers = kPi05ModelDims.vision_layers;
            if (step != -1) {
                return invalid("PI0.5 vision scale step is invalid");
            }
            break;
        case Pi05LinearDomain::kEncoder:
            layers = kPi05ModelDims.encoder_layers;
            if (step != -1) {
                return invalid("PI0.5 encoder scale step is invalid");
            }
            break;
        case Pi05LinearDomain::kDecoder:
            layers = kPi05ModelDims.decoder_layers;
            if (step < 0 || step >= shape.num_steps) {
                return invalid("PI0.5 decoder scale step is invalid");
            }
            break;
    }
    if (slot < 0 || key.layer < 0 || key.layer >= layers) {
        return invalid("PI0.5 linear activation key is invalid");
    }
    std::size_t linear_layer = static_cast<std::size_t>(key.layer);
    if (key.domain == Pi05LinearDomain::kDecoder) {
        linear_layer += static_cast<std::size_t>(step) *
                        static_cast<std::size_t>(layers);
    }
    site.index = linear_layer * kPi05LinearScalesPerLayer +
                 static_cast<std::size_t>(slot);
    const std::size_t count =
        key.domain == Pi05LinearDomain::kVision
            ? layout.vision
            : key.domain == Pi05LinearDomain::kEncoder ? layout.encoder
                                                       : layout.decoder;
    if (site.index >= count) {
        return invalid("PI0.5 linear activation site is out of range");
    }
    *out = site;
    return modalities::Status::ok();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

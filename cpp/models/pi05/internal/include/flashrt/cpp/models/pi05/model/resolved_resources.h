#ifndef FLASHRT_CPP_MODELS_PI05_MODEL_RESOLVED_RESOURCES_H
#define FLASHRT_CPP_MODELS_PI05_MODEL_RESOLVED_RESOURCES_H

#include "flashrt/cpp/models/pi05/model/semantic_pipeline.h"
#include "flashrt/exec.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace flashrt {
namespace models {
namespace pi05 {

class Pi05ResolvedGraphBindings;

struct Pi05ResolvedBuffer final {
    frt_buffer buffer = nullptr;
    modalities::DType physical_dtype = modalities::DType::kUInt8;
    modalities::Shape physical_shape;
    std::uint64_t physical_bytes = 0;
    Pi05TensorSpec logical_spec;
    const void* storage_identity = nullptr;
    std::uint64_t storage_bytes = 0;
    std::uint64_t storage_offset = 0;
};

struct Pi05ResolvedBuffers final {
    Pi05ResolvedBuffer images;
    Pi05ResolvedBuffer prompt_embedding;
    Pi05ResolvedBuffer vision_state;
    Pi05ResolvedBuffer encoder_state;
    Pi05ResolvedBuffer key_cache;
    Pi05ResolvedBuffer value_cache;
    Pi05ResolvedBuffer noise;
    Pi05ResolvedBuffer decoder_state;
    Pi05ResolvedBuffer action_delta;
    Pi05ResolvedBuffer time_state;
    Pi05ResolvedBuffer attention_style;
    Pi05ResolvedBuffer mlp_style;
    Pi05ResolvedBuffer final_style;

    Pi05ResolvedBuffer encoder_rope;
    Pi05ResolvedBuffer decoder_rope;
    Pi05ResolvedBuffer encoder_valid_tokens;
    Pi05ResolvedBuffer decoder_valid_tokens;
    Pi05ResolvedBuffer decoder_position;
    Pi05ResolvedBuffer previous_actions;
    Pi05ResolvedBuffer prefix_weights;
    Pi05ResolvedBuffer guidance_weight;
};

enum class Pi05WeightStorage : std::uint8_t {
    kBFloat16 = 0,
    kFloat16,
    kFp8E4M3,
};

struct Pi05ResolvedWeight final {
    const void* device_data = nullptr;
    const float* scale_data = nullptr;
    std::uint64_t bytes = 0;
    Pi05WeightStorage storage = Pi05WeightStorage::kBFloat16;
    modalities::Shape shape;
};

struct Pi05VisionGlobalWeights final {
    Pi05ResolvedWeight patch_weight;
    Pi05ResolvedWeight patch_bias;
    Pi05ResolvedWeight position_embedding;
    Pi05ResolvedWeight final_norm_weight;
    Pi05ResolvedWeight final_norm_bias;
    Pi05ResolvedWeight projector_weight;
    Pi05ResolvedWeight projector_bias;
};

struct Pi05DecoderGlobalWeights final {
    Pi05ResolvedWeight time_embeddings;
    Pi05ResolvedWeight time_mlp_in_weight;
    Pi05ResolvedWeight time_mlp_in_bias;
    Pi05ResolvedWeight time_mlp_out_weight;
    Pi05ResolvedWeight time_mlp_out_bias;
    Pi05ResolvedWeight final_norm_mod_weight;
    Pi05ResolvedWeight final_norm_mod_bias;
    Pi05ResolvedWeight action_in_weight;
    Pi05ResolvedWeight action_in_bias;
    Pi05ResolvedWeight action_out_weight;
    Pi05ResolvedWeight action_out_bias;
};

struct Pi05VisionLayerWeights final {
    Pi05ResolvedWeight pre_attention_norm_weight;
    Pi05ResolvedWeight pre_attention_norm_bias;
    Pi05ResolvedWeight attention_qkv_weight;
    Pi05ResolvedWeight attention_qkv_bias;
    Pi05ResolvedWeight attention_output_weight;
    Pi05ResolvedWeight attention_output_bias;
    Pi05ResolvedWeight pre_mlp_norm_weight;
    Pi05ResolvedWeight pre_mlp_norm_bias;
    Pi05ResolvedWeight mlp_up_weight;
    Pi05ResolvedWeight mlp_up_bias;
    Pi05ResolvedWeight mlp_down_weight;
    Pi05ResolvedWeight mlp_down_bias;
};

struct Pi05FeedForwardWeights final {
    Pi05ResolvedWeight gate_weight;
    Pi05ResolvedWeight up_weight;
    Pi05ResolvedWeight gate_up_weight;
    Pi05ResolvedWeight down_weight;
};

struct Pi05EncoderLayerWeights final {
    Pi05ResolvedWeight attention_qkv_weight;
    Pi05ResolvedWeight attention_output_weight;
    Pi05FeedForwardWeights mlp;
};

struct Pi05DecoderLayerWeights final {
    Pi05ResolvedWeight attention_qkv_weight;
    Pi05ResolvedWeight attention_output_weight;
    Pi05FeedForwardWeights mlp;
    Pi05ResolvedWeight attention_mod_weight;
    Pi05ResolvedWeight attention_mod_bias;
    Pi05ResolvedWeight mlp_mod_weight;
    Pi05ResolvedWeight mlp_mod_bias;
};

struct Pi05ResolvedWeights final {
    Pi05ResolvedWeight embedding_table;
    Pi05VisionGlobalWeights vision;
    Pi05DecoderGlobalWeights decoder;
    std::array<Pi05VisionLayerWeights,
               static_cast<std::size_t>(kPi05ModelDims.vision_layers)>
        vision_layers;
    std::array<Pi05EncoderLayerWeights,
               static_cast<std::size_t>(kPi05ModelDims.encoder_layers)>
        encoder_layers;
    std::array<Pi05DecoderLayerWeights,
               static_cast<std::size_t>(kPi05ModelDims.decoder_layers)>
        decoder_layers;
};

struct Pi05ResolvedResources final {
    Pi05ResolvedBuffers buffers;
    Pi05ResolvedWeights weights;
};

Pi05ResolvedBuffer* pi05_resolved_buffer(Pi05ResolvedBuffers* buffers,
                                         Pi05ValueId id);
const Pi05ResolvedBuffer* pi05_resolved_buffer(
    const Pi05ResolvedBuffers& buffers,
    Pi05ValueId id);

modalities::Status validate_pi05_resolved_buffers(
    const Pi05ResolvedBuffers& buffers,
    const Pi05ResolvedShape& shape);
modalities::Status validate_pi05_resolved_weights(
    const Pi05ResolvedWeights& weights,
    const Pi05ResolvedShape& shape,
    modalities::DType activation_dtype);
modalities::Status validate_pi05_resolved_resources(
    const Pi05ResolvedResources& resources,
    const Pi05ResolvedShape& shape);
modalities::Status make_pi05_graph_bindings(
    const Pi05ResolvedBuffers& buffers,
    Pi05ResolvedGraphBindings* out);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_MODEL_RESOLVED_RESOURCES_H

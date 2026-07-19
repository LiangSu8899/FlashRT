#ifndef FLASHRT_CPP_MODELS_PI05_MODEL_FRONTEND_OPS_H
#define FLASHRT_CPP_MODELS_PI05_MODEL_FRONTEND_OPS_H

#include "flashrt/cpp/models/pi05/model/linear_weight_groups.h"

#include <cstddef>

namespace flashrt {
namespace models {
namespace pi05 {

struct Pi05TargetProfile final {
    modalities::DType activation_dtype = modalities::DType::kUInt8;
};

enum class Pi05LinearEpilogueKind {
    kNone = 0,
    kBias,
    kBiasGelu,
    kBiasResidual,
};

struct Pi05LinearEpilogue final {
    Pi05LinearEpilogueKind kind = Pi05LinearEpilogueKind::kNone;
    const Pi05ResolvedWeight* bias = nullptr;
    void* residual = nullptr;
};

struct Pi05LinearInput final {
    const void* data = nullptr;
    bool prequantized = false;
};

using Pi05LinearPrimitive = modalities::Status (*)(
    void* state,
    const Pi05ResolvedWeight& weight,
    const void* input,
    void* output,
    int rows,
    int input_width,
    int output_width,
    Pi05Stream stream);
using Pi05ProjectedLinearPrimitive = modalities::Status (*)(
    void* state,
    const Pi05ResolvedWeight& weight,
    Pi05LinearWeightKey key,
    int step,
    const void* input,
    void* output,
    int rows,
    int input_width,
    int output_width,
    bool prequantized,
    const Pi05LinearEpilogue& epilogue,
    Pi05Stream stream);
using Pi05BiasPrimitive = modalities::Status (*)(
    void* state,
    void* values,
    const Pi05ResolvedWeight& bias,
    int rows,
    int columns,
    Pi05Stream stream);
using Pi05UnaryPrimitive = modalities::Status (*)(
    void* state,
    void* values,
    std::size_t elements,
    Pi05Stream stream);
using Pi05CopyPrimitive = modalities::Status (*)(
    void* state,
    void* destination,
    const void* source,
    std::size_t bytes,
    Pi05Stream stream);
using Pi05PatchifyPrimitive = modalities::Status (*)(
    void* state,
    const void* images,
    void* patches,
    int views,
    Pi05Stream stream);
using Pi05BiasResidualPrimitive = modalities::Status (*)(
    void* state,
    void* residual,
    const void* values,
    const Pi05ResolvedWeight& bias,
    int rows,
    int columns,
    Pi05Stream stream);
using Pi05LayerNormPrimitive = modalities::Status (*)(
    void* state,
    const void* values,
    const Pi05ResolvedWeight& weight,
    const Pi05ResolvedWeight& bias,
    void* output,
    int rows,
    int columns,
    float epsilon,
    bool quantize,
    Pi05LinearInput* linear_input,
    Pi05Stream stream);
using Pi05QkvSplitPrimitive = modalities::Status (*)(
    void* state,
    const void* qkv,
    void* query,
    void* key,
    void* value,
    int rows,
    int query_width,
    int key_width,
    int value_width,
    Pi05Stream stream);
using Pi05NormalizeForLinearPrimitive = modalities::Status (*)(
    void* state,
    void* residual,
    const void* update,
    const Pi05ResolvedBuffer& weight,
    void* normalized,
    Pi05LinearWeightKey key,
    int step,
    int rows,
    int columns,
    float epsilon,
    const void** linear_input,
    bool* prequantized,
    Pi05Stream stream);
using Pi05QkvRopePrimitive = modalities::Status (*)(
    void* state,
    const void* qkv,
    const Pi05ResolvedBuffer& rope,
    const Pi05ResolvedBuffer* position,
    void* query,
    void* key,
    void* value,
    int rows,
    int query_width,
    int key_width,
    int value_width,
    int head_width,
    Pi05Stream stream);
using Pi05AdaptiveNormalizePrimitive = modalities::Status (*)(
    void* state,
    void* residual,
    const void* update,
    const void* update_gate,
    const Pi05ResolvedBuffer& weight,
    const void* style,
    void* normalized,
    void* gate,
    Pi05LinearWeightKey key,
    int step,
    int rows,
    int columns,
    float epsilon,
    bool quantize,
    const void** linear_input,
    bool* prequantized,
    Pi05Stream stream);
using Pi05DiffusionUpdatePrimitive = modalities::Status (*)(
    void* state,
    void* residual,
    const void* update,
    const Pi05ResolvedWeight& bias,
    int rows,
    int columns,
    int num_steps,
    Pi05Stream stream);
using Pi05AttentionPrimitive = modalities::Status (*)(
    void* state,
    Pi05LinearDomain domain,
    int layer,
    const void* query,
    const void* key,
    const void* value,
    void* output,
    int batches,
    int query_rows,
    int key_rows,
    int query_heads,
    int key_heads,
    int head_width,
    Pi05Stream stream);
using Pi05GateUpPrimitive = modalities::Status (*)(
    void* state,
    const Pi05FeedForwardWeights& weights,
    Pi05LinearWeightKey key,
    int step,
    const void* input,
    bool prequantized,
    void* gate,
    void* up,
    int rows,
    int width,
    int hidden_width,
    bool* merged,
    Pi05Stream stream);
using Pi05GatedActivationPrimitive = modalities::Status (*)(
    void* state,
    const void* gate,
    const void* up,
    bool merged,
    void* output,
    int rows,
    int hidden_width,
    Pi05LinearWeightKey output_key,
    int step,
    const void** linear_input,
    bool* prequantized,
    Pi05Stream stream);
using Pi05GeluPrimitive = modalities::Status (*)(
    void* state,
    void* values,
    std::size_t elements,
    Pi05Stream stream);
using Pi05VisionPoolPrimitive = modalities::Status (*)(
    void* state,
    const void* input,
    void* output,
    int views,
    int grid_height,
    int grid_width,
    int columns,
    int factor,
    Pi05Stream stream);

struct Pi05PrimitiveSet final {
    void* state = nullptr;
    Pi05LinearPrimitive linear = nullptr;
    Pi05ProjectedLinearPrimitive projected_linear = nullptr;
    Pi05BiasPrimitive add_bias = nullptr;
    Pi05UnaryPrimitive silu = nullptr;
    Pi05CopyPrimitive copy = nullptr;
    Pi05PatchifyPrimitive patchify = nullptr;
    Pi05BiasResidualPrimitive bias_residual = nullptr;
    Pi05LayerNormPrimitive layer_norm = nullptr;
    Pi05QkvSplitPrimitive qkv_split = nullptr;
    Pi05NormalizeForLinearPrimitive normalize_for_linear = nullptr;
    Pi05QkvRopePrimitive qkv_rope = nullptr;
    Pi05AdaptiveNormalizePrimitive adaptive_normalize = nullptr;
    Pi05DiffusionUpdatePrimitive diffusion_update = nullptr;
    Pi05AttentionPrimitive attention = nullptr;
    Pi05GateUpPrimitive gate_up = nullptr;
    Pi05GatedActivationPrimitive gated_activation = nullptr;
    Pi05GeluPrimitive gelu = nullptr;
    Pi05VisionPoolPrimitive vision_pool = nullptr;
};

struct Pi05FrontendOps final {
    Pi05TargetProfile profile;
    Pi05PrimitiveSet bf16;
    Pi05PrimitiveSet f16;
};

struct Pi05PrepareScratch final {
    void* mlp_hidden = nullptr;
    void* mlp_output = nullptr;
    std::size_t bytes = 0;
};

struct Pi05PrepareExecution final {
    const Pi05ResolvedResources* resources = nullptr;
    const Pi05FrontendOps* ops = nullptr;
    Pi05PrepareScratch scratch;
};

struct Pi05VisionExecution final {
    void* patches = nullptr;
    void* expanded_position = nullptr;
    void* pooled = nullptr;
    void* normalized = nullptr;
    void* qkv = nullptr;
    void* hidden = nullptr;
    void* query = nullptr;
    void* key = nullptr;
    void* value = nullptr;
    void* attention_output = nullptr;
    Pi05LinearInput normalized_input;
};

struct Pi05EncoderExecution final {
    Pi05ResolvedBuffer rms_weight;
    void* normalized = nullptr;
    void* qkv = nullptr;
    void* gate = nullptr;
    void* hidden = nullptr;
    void* query = nullptr;
    void* attention_output = nullptr;
    const void* mlp_input = nullptr;
    bool mlp_input_prequantized = false;
    const void* pending_update = nullptr;
};

struct Pi05DecoderExecution final {
    Pi05ResolvedBuffer rms_weight;
    void* normalized = nullptr;
    void* gate = nullptr;
    void* qkv = nullptr;
    void* gate_projection = nullptr;
    void* hidden = nullptr;
    void* query = nullptr;
    void* attention_output = nullptr;
    const void* pending_update = nullptr;
    const void* pending_gate = nullptr;
};

struct Pi05ForwardExecution final : Pi05OperationSink {
    const Pi05ResolvedResources* resources = nullptr;
    const Pi05FrontendOps* ops = nullptr;
    Pi05VisionExecution vision;
    Pi05EncoderExecution encoder;
    Pi05DecoderExecution decoder;
    modalities::Status record(const Pi05OperationCall& call,
                              const Pi05ResolvedShape& shape,
                              Pi05Stream stream) override;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_MODEL_FRONTEND_OPS_H

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
using Pi05VisionAttentionPrimitive = modalities::Status (*)(
    void* state,
    const void* query,
    const void* key,
    const void* value,
    void* output,
    int views,
    int rows,
    int heads,
    int head_width,
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
    Pi05VisionAttentionPrimitive vision_attention = nullptr;
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
};

struct Pi05ForwardExecution final : Pi05OperationSink {
    const Pi05ResolvedResources* resources = nullptr;
    const Pi05FrontendOps* ops = nullptr;
    Pi05VisionExecution vision;
    Pi05OperationSink* fallback = nullptr;

    modalities::Status record(const Pi05OperationCall& call,
                              const Pi05ResolvedShape& shape,
                              Pi05Stream stream) override;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_MODEL_FRONTEND_OPS_H

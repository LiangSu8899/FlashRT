#ifndef FLASHRT_CPP_MODELS_PI05_MODEL_FRONTEND_OPS_H
#define FLASHRT_CPP_MODELS_PI05_MODEL_FRONTEND_OPS_H

#include "flashrt/cpp/models/pi05/model/resolved_resources.h"

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

struct Pi05PrimitiveSet final {
    void* state = nullptr;
    Pi05LinearPrimitive linear = nullptr;
    Pi05BiasPrimitive add_bias = nullptr;
    Pi05UnaryPrimitive silu = nullptr;
    Pi05CopyPrimitive copy = nullptr;
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

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_MODEL_FRONTEND_OPS_H

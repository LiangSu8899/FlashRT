#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_RESOURCE_RESOLVER_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_RESOURCE_RESOLVER_H

#include "flashrt/cpp/models/pi05/model/resolved_resources.h"
#include "flashrt/cpp/models/pi05/support/native_device_weights.h"
#include "flashrt/cpp/models/pi05/support/native_workspace.h"

#include <cstdint>

namespace flashrt {
namespace models {
namespace pi05 {

enum class NativeFeedForwardLayout : std::uint8_t {
    kSeparateGateUp = 0,
    kFusedGateUp,
};

struct Pi05NativeWeightLayout final {
    NativeFeedForwardLayout encoder =
        NativeFeedForwardLayout::kSeparateGateUp;
    NativeFeedForwardLayout decoder =
        NativeFeedForwardLayout::kSeparateGateUp;
};

struct Pi05TargetBufferBindings final {
    Pi05ResolvedBuffer key_cache;
    Pi05ResolvedBuffer value_cache;
    Pi05ResolvedBuffer encoder_valid_tokens;
    Pi05ResolvedBuffer decoder_valid_tokens;
    Pi05ResolvedBuffer decoder_position;
};

modalities::Status resolve_pi05_native_buffers(
    const NativeWorkspace& workspace,
    const Pi05TargetBufferBindings& target,
    const Pi05ResolvedShape& shape,
    Pi05ResolvedBuffers* out);

modalities::Status resolve_pi05_materialized_weights(
    const NativeDeviceWeightStore& store,
    const Pi05ResolvedShape& shape,
    modalities::DType activation_dtype,
    Pi05NativeWeightLayout layout,
    Pi05ResolvedWeights* out);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_RESOURCE_RESOLVER_H

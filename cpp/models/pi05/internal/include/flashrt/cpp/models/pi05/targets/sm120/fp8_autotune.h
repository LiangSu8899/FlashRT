#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_FP8_AUTOTUNE_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_FP8_AUTOTUNE_H

#include "flashrt/cpp/models/pi05/support/native_resource_resolver.h"
#include "flashrt/cpp/models/pi05/targets/sm120/bf16_scratch.h"
#include "flashrt/cpp/models/pi05/targets/sm120/fp8_linear.h"

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {

modalities::Status autotune_sm120_fp8(
    const Pi05ResolvedShape& shape,
    Pi05ResolvedResources* resources,
    const Sm120Bf16ScratchBacking& scratch,
    Sm120Fp8Linear* linear);

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_FP8_AUTOTUNE_H

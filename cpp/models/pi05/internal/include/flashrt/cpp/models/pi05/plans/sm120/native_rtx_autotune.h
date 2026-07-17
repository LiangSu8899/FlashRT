#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_AUTOTUNE_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_AUTOTUNE_H

#include "flashrt/cpp/models/pi05/plans/sm120/native_rtx_linear.h"

namespace flashrt {
namespace models {
namespace pi05 {

modalities::Status autotune_native_rtx_fp8(
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    const NativeRtxLinear& linear,
    int num_views,
    int chunk_size);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_AUTOTUNE_H

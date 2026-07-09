#include "flashrt/cpp/models/pi05/native_quantization.h"

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status unavailable() {
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "native weight quantization requires the CUDA kernels build");
}

}  // namespace

modalities::Status native_quantize_fp8_e4m3(
    const NativeFloatTensor&,
    bool,
    NativeFp8Tensor*) {
    return unavailable();
}

modalities::Status native_quantize_int8_per_output(
    const NativeFloatTensor&,
    NativeInt8Tensor*) {
    return unavailable();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

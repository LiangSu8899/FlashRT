#include "flashrt/cpp/models/pi05/native_rope.h"

namespace flashrt {
namespace models {
namespace pi05 {

modalities::Status generate_native_thor_rope_f16(
    void*, int, int, std::uintptr_t) {
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "Thor RoPE generation requires the CUDA kernels build");
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

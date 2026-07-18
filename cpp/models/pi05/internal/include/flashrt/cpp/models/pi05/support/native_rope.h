#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_ROPE_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_ROPE_H

#include "flashrt/cpp/modalities/types.h"

#include <cstdint>

namespace flashrt {
namespace models {
namespace pi05 {

modalities::Status generate_native_rope_f16(
    void* output, int start_position, int positions, std::uintptr_t stream);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_ROPE_H

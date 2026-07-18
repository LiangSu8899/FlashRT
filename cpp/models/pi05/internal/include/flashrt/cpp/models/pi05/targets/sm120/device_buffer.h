#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_DEVICE_BUFFER_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_DEVICE_BUFFER_H

#include "flashrt/cpp/modalities/types.h"
#include "flashrt/exec.h"

#include <cstddef>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {

struct Sm120DeviceBuffer final {
    frt_buffer buffer = nullptr;
    modalities::DType dtype = modalities::DType::kUInt8;
    modalities::Shape shape;

    void* device_data() const {
        return buffer ? frt_buffer_dptr(buffer) : nullptr;
    }
    std::size_t bytes() const {
        return buffer ? frt_buffer_bytes(buffer) : 0;
    }
};

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_DEVICE_BUFFER_H

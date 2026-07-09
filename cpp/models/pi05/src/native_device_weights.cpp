#include "flashrt/cpp/models/pi05/native_device_weights.h"

#ifdef FLASHRT_CPP_WITH_CUDA_STAGING
#include <cuda_runtime_api.h>
#endif

#include <limits>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

bool element_count(const std::vector<std::uint64_t>& shape,
                   std::size_t* out) {
    std::size_t count = 1;
    for (std::uint64_t dim : shape) {
        if (dim > std::numeric_limits<std::size_t>::max() ||
            (dim && count > std::numeric_limits<std::size_t>::max() /
                                static_cast<std::size_t>(dim))) {
            return false;
        }
        count *= static_cast<std::size_t>(dim);
    }
    if (out) *out = count;
    return true;
}

}  // namespace

modalities::Status NativeDeviceWeightStore::upload(
    const std::string& name,
    const NativeBf16Tensor& tensor) {
    if (!ctx_ || name.empty()) return invalid("invalid device weight store");
    if (weights_.find(name) != weights_.end()) {
        return invalid("duplicate device weight name");
    }
    std::size_t elements = 0;
    if (!element_count(tensor.shape, &elements) ||
        elements != tensor.values.size() ||
        elements > std::numeric_limits<std::size_t>::max() /
                       sizeof(std::uint16_t)) {
        return invalid("device weight shape does not match BF16 payload");
    }
    const std::size_t bytes = elements * sizeof(std::uint16_t);
    if (!bytes) return invalid("device weight payload is empty");

#ifndef FLASHRT_CPP_WITH_CUDA_STAGING
    (void)tensor;
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "device weight upload requires the CUDA build");
#else
    frt_buffer buffer = frt_buffer_alloc(ctx_, name.c_str(), bytes);
    if (!buffer) {
        return modalities::Status::error(modalities::StatusCode::kBackend,
                                         "device weight allocation failed");
    }
    const cudaError_t rc = cudaMemcpy(frt_buffer_dptr(buffer),
                                      tensor.values.data(), bytes,
                                      cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) {
        return modalities::Status::error(
            modalities::StatusCode::kBackend,
            std::string("device weight upload failed: ") +
                cudaGetErrorString(rc));
    }
    weights_.emplace(name, NativeDeviceWeight{buffer, tensor.shape});
    return modalities::Status::ok();
#endif
}

const NativeDeviceWeight* NativeDeviceWeightStore::find(
    const std::string& name) const {
    const auto it = weights_.find(name);
    return it == weights_.end() ? nullptr : &it->second;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

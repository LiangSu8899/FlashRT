#include "flashrt/cpp/models/pi05/support/native_device_weights.h"

#include "flashrt/cpp/models/pi05/support/native_weight_ops.h"

#ifdef FLASHRT_CPP_WITH_CUDA_STAGING
#include <cuda_runtime_api.h>
#endif

#include <limits>
#include <sstream>
#include <utility>

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

std::size_t element_bytes(NativeWeightDType dtype) {
    switch (dtype) {
        case NativeWeightDType::kBf16: return sizeof(std::uint16_t);
        case NativeWeightDType::kFloat16: return sizeof(std::uint16_t);
        case NativeWeightDType::kFp8E4M3: return sizeof(std::uint8_t);
        case NativeWeightDType::kInt8: return sizeof(std::int8_t);
        case NativeWeightDType::kFloat32: return sizeof(float);
    }
    return 0;
}

}  // namespace

modalities::Status NativeDeviceWeightStore::upload(
    const std::string& name,
    const NativeBf16Tensor& tensor) {
    return upload_bytes(name, tensor.shape, NativeWeightDType::kBf16,
                        tensor.values.data(),
                        tensor.values.size() * sizeof(std::uint16_t));
}

modalities::Status NativeDeviceWeightStore::upload(
    const std::string& name,
    const NativeF16Tensor& tensor) {
    return upload_bytes(name, tensor.shape, NativeWeightDType::kFloat16,
                        tensor.values.data(),
                        tensor.values.size() * sizeof(std::uint16_t));
}

modalities::Status NativeDeviceWeightStore::upload_bytes(
    const std::string& name,
    const std::vector<std::uint64_t>& shape,
    NativeWeightDType dtype,
    const void* data,
    std::size_t bytes) {
    std::lock_guard<std::mutex> lock(upload_mutex_);
    if (!ctx_ || name.empty()) return invalid("invalid device weight store");
    if (weights_.find(name) != weights_.end()) {
        return invalid("duplicate device weight name");
    }
    std::size_t elements = 0;
    const std::size_t width = element_bytes(dtype);
    if (!data || !width || !element_count(shape, &elements) ||
        elements > std::numeric_limits<std::size_t>::max() / width ||
        elements * width != bytes) {
        return invalid("device weight shape does not match typed payload");
    }
    if (!bytes) return invalid("device weight payload is empty");

#ifndef FLASHRT_CPP_WITH_CUDA_STAGING
    (void)data;
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "device weight upload requires the CUDA build");
#else
    frt_buffer buffer = frt_buffer_alloc(ctx_, name.c_str(), bytes);
    if (!buffer) {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        std::ostringstream message;
        message << "device weight allocation failed: " << name
                << " requested=" << bytes << " free=" << free_bytes
                << " total=" << total_bytes;
        return modalities::Status::error(modalities::StatusCode::kBackend,
                                         message.str());
    }
    const cudaError_t rc = cudaMemcpy(frt_buffer_dptr(buffer),
                                      data, bytes,
                                      cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) {
        return modalities::Status::error(
            modalities::StatusCode::kBackend,
            std::string("device weight upload failed: ") +
                cudaGetErrorString(rc));
    }
    weights_.emplace(name, NativeDeviceWeight{buffer, shape, dtype});
    return modalities::Status::ok();
#endif
}

modalities::Status NativeDeviceWeightStore::allocate(
    const std::string& name,
    const std::vector<std::uint64_t>& shape,
    NativeWeightDType dtype) {
    std::lock_guard<std::mutex> lock(upload_mutex_);
    if (!ctx_ || name.empty() || weights_.find(name) != weights_.end()) {
        return invalid("invalid device weight allocation");
    }
    std::size_t elements = 0;
    const std::size_t width = element_bytes(dtype);
    if (!width || !element_count(shape, &elements) || !elements ||
        elements > std::numeric_limits<std::size_t>::max() / width) {
        return invalid("device weight allocation shape is invalid");
    }
#ifndef FLASHRT_CPP_WITH_CUDA_STAGING
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "device weight allocation requires the CUDA build");
#else
    const std::size_t bytes = elements * width;
    frt_buffer buffer = frt_buffer_alloc(ctx_, name.c_str(), bytes);
    if (!buffer) {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        std::ostringstream message;
        message << "device weight allocation failed: " << name
                << " requested=" << bytes << " free=" << free_bytes
                << " total=" << total_bytes;
        return modalities::Status::error(modalities::StatusCode::kBackend,
                                         message.str());
    }
    weights_.emplace(name, NativeDeviceWeight{buffer, shape, dtype});
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

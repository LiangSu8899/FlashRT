#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_DEVICE_WEIGHTS_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_DEVICE_WEIGHTS_H

#include "flashrt/cpp/modalities/types.h"
#include "flashrt/exec.h"

#include <cstddef>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

struct NativeBf16Tensor;
struct NativeF16Tensor;

enum class NativeWeightDType {
    kBf16,
    kFloat16,
    kFp8E4M3,
    kInt8,
    kFloat32,
};

struct NativeDeviceWeight {
    frt_buffer buffer = nullptr;
    std::vector<std::uint64_t> shape;
    NativeWeightDType dtype = NativeWeightDType::kBf16;
};

class NativeDeviceWeightStore {
public:
    explicit NativeDeviceWeightStore(frt_ctx ctx) : ctx_(ctx) {}

    NativeDeviceWeightStore(const NativeDeviceWeightStore&) = delete;
    NativeDeviceWeightStore& operator=(const NativeDeviceWeightStore&) = delete;

    modalities::Status upload(const std::string& name,
                              const NativeBf16Tensor& tensor);
    modalities::Status upload(const std::string& name,
                              const NativeF16Tensor& tensor);
    modalities::Status upload_bytes(
        const std::string& name,
        const std::vector<std::uint64_t>& shape,
        NativeWeightDType dtype,
        const void* data,
        std::size_t bytes);
    modalities::Status allocate(
        const std::string& name,
        const std::vector<std::uint64_t>& shape,
        NativeWeightDType dtype);
    const NativeDeviceWeight* find(const std::string& name) const;
    std::size_t size() const { return weights_.size(); }

private:
    frt_ctx ctx_ = nullptr;  // borrowed; the context owns every buffer
    std::map<std::string, NativeDeviceWeight> weights_;
    std::mutex upload_mutex_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_DEVICE_WEIGHTS_H

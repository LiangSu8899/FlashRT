#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_LINEAR_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_LINEAR_H

#include "flashrt/cpp/models/pi05/native_device_weights.h"
#include "flashrt/cpp/models/pi05/native_kernel_driver.h"
#include "flashrt/cpp/models/pi05/native_workspace.h"

#include <cstdint>
#include <initializer_list>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {

enum class NativeRtxLinearMode {
    kBf16,
    kFp8Dynamic,
    kFp8Static,
};

enum class NativeRtxScaleDomain {
    kVision,
    kEncoder,
    kDecoder,
};

struct NativeRtxScaleSite {
    NativeRtxScaleDomain domain = NativeRtxScaleDomain::kVision;
    int index = 0;
};

class NativeRtxLinear {
public:
    NativeRtxLinear(const NativeKernelDriver* driver,
                    NativeRtxLinearMode mode)
        : driver_(driver), mode_(mode) {}

    bool fp8() const { return mode_ != NativeRtxLinearMode::kBf16; }
    bool static_fp8() const {
        return mode_ == NativeRtxLinearMode::kFp8Static;
    }
    bool dynamic_fp8() const {
        return mode_ == NativeRtxLinearMode::kFp8Dynamic;
    }

    const NativeDeviceWeight* find_weight(
        const NativeDeviceWeightStore& weights,
        const std::string& name) const;
    bool weight_shape_is(
        const NativeDeviceWeightStore& weights,
        const std::string& name,
        std::initializer_list<std::uint64_t> shape) const;

    modalities::Status run(
        const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace,
        const std::string& weight_name,
        NativeRtxScaleSite site,
        const void* input,
        void* output,
        int m,
        int n,
        int k,
        std::uintptr_t stream) const;
    modalities::Status autotune(
        const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace,
        const std::string& weight_name,
        NativeRtxScaleSite site,
        void* output,
        int m,
        int n,
        int k) const;
    modalities::Status run_prequantized(
        const NativeDeviceWeightStore& weights,
        const std::string& weight_name,
        NativeRtxScaleSite site,
        const NativeWorkspace& workspace,
        const void* input,
        void* output,
        int m,
        int n,
        int k,
        std::uintptr_t stream) const;
    const float* scale(
        const NativeWorkspace& workspace,
        NativeRtxScaleSite site) const;

private:
    const NativeWorkspaceBuffer* scale_buffer(
        const NativeWorkspace& workspace,
        NativeRtxScaleDomain domain) const;

    const NativeKernelDriver* driver_ = nullptr;
    NativeRtxLinearMode mode_ = NativeRtxLinearMode::kBf16;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_LINEAR_H

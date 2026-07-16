#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_BF16_FORWARD_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_BF16_FORWARD_H

#include "flashrt/cpp/models/pi05/native_kernel_driver.h"
#include "flashrt/cpp/models/pi05/native_rtx_linear.h"
#include "flashrt/cpp/models/pi05/native_rtx_attention.h"
#include "flashrt/cpp/models/pi05/native_rtx_attention_driver.h"
#include "flashrt/cpp/models/pi05/native_workspace.h"

namespace flashrt {
namespace models {
namespace pi05 {

class NativeBf16Forward {
public:
    explicit NativeBf16Forward(const NativeKernelDriver* driver)
        : driver_(driver), fallback_linear_(driver, NativeRtxLinearMode::kBf16),
          linear_(&fallback_linear_) {}
    NativeBf16Forward(const NativeKernelDriver* driver,
                      const NativeRtxLinear* linear)
        : driver_(driver), fallback_linear_(driver, NativeRtxLinearMode::kBf16),
          linear_(linear ? linear : &fallback_linear_) {}

    modalities::Status encoder_qkv(
        int layer, const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace, NativeRtxAttentionWorkspace* attention,
        std::uintptr_t stream) const;
#ifdef FLASHRT_CPP_WITH_FA2
    modalities::Status vision_layer(
        int layer, const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace, NativeRtxAttentionWorkspace* attention,
        const NativeRtxAttentionDriver* attention_driver,
        std::uintptr_t stream) const;
    modalities::Status vision(
        const NativeDeviceWeightStore& weights, NativeWorkspace* workspace,
        NativeRtxAttentionWorkspace* attention,
        const NativeRtxAttentionDriver* attention_driver,
        std::uintptr_t stream) const;
    modalities::Status encoder_layer(
        int layer, const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace, NativeRtxAttentionWorkspace* attention,
        const NativeRtxAttentionDriver* attention_driver,
        std::uintptr_t stream) const;
    modalities::Status encoder(
        const NativeDeviceWeightStore& weights, NativeWorkspace* workspace,
        NativeRtxAttentionWorkspace* attention,
        const NativeRtxAttentionDriver* attention_driver,
        std::uintptr_t stream) const;
    modalities::Status decoder_layer(
        int layer, int step, const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace, NativeRtxAttentionWorkspace* attention,
        const NativeRtxAttentionDriver* attention_driver,
        std::uintptr_t stream) const;
    modalities::Status diffusion_step(
        int step, const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace, NativeRtxAttentionWorkspace* attention,
        const NativeRtxAttentionDriver* attention_driver,
        std::uintptr_t stream) const;
    modalities::Status diffusion(
        const NativeDeviceWeightStore& weights, NativeWorkspace* workspace,
        NativeRtxAttentionWorkspace* attention,
        const NativeRtxAttentionDriver* attention_driver,
        std::uintptr_t stream) const;
#endif

private:
    const NativeKernelDriver* driver_ = nullptr;
    NativeRtxLinear fallback_linear_;
    const NativeRtxLinear* linear_ = nullptr;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_BF16_FORWARD_H

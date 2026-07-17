#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_FP8_FORWARD_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_FP8_FORWARD_H

#include "flashrt/cpp/models/pi05/plans/sm110/native_thor_kernel_driver.h"
#include "flashrt/cpp/models/pi05/plans/sm110/native_thor_weight_materializer.h"
#include "flashrt/cpp/models/pi05/support/native_workspace.h"

#include <cstdint>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

class NativeThorFp8Forward {
public:
    explicit NativeThorFp8Forward(const NativeThorKernelDriver* driver)
        : driver_(driver) {}

    modalities::Status vision_begin(
        const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace,
        std::uintptr_t stream) const;
    modalities::Status vision_layer(
        int layer,
        const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace,
        const NativeThorWeightScales& weight_scales,
        std::uintptr_t stream) const;
    modalities::Status vision_end(
        const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace,
        std::uintptr_t stream) const;
    modalities::Status encoder_layer(
        int layer, const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace,
        const std::vector<float>& activation_weight_alphas,
        std::uintptr_t stream) const;
    modalities::Status diffusion_begin(
        int step, const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace,
        std::uintptr_t stream) const;
    modalities::Status decoder_layer(
        int step, int layer, const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace, std::uintptr_t stream) const;
    modalities::Status diffusion_end(
        int step, const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace, std::uintptr_t stream) const;

    modalities::Status calibrate_encoder(
        const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace,
        const NativeThorWeightScales& weight_scales,
        std::vector<float>* sample_scales,
        std::uintptr_t stream) const;
    modalities::Status calibrate_decoder(
        const NativeDeviceWeightStore& weights,
        NativeWorkspace* workspace,
        std::vector<float>* sample_scales,
        std::uintptr_t stream) const;

private:
    const NativeThorKernelDriver* driver_ = nullptr;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_FP8_FORWARD_H

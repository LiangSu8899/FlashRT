#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_GRAPH_OWNER_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_GRAPH_OWNER_H

#include "flashrt/cpp/models/pi05/native_calibration.h"
#include "flashrt/cpp/models/pi05/native_graph_runtime.h"
#include "flashrt/cpp/models/pi05/native_thor_fp8_forward.h"

#include <memory>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

class NativeThorGraphOwner final : public NativeGraphRuntime {
public:
    static std::unique_ptr<NativeThorGraphOwner> create(
        const std::string& checkpoint_path,
        const NativeGraphConfig& config,
        const NativeCalibrationArtifact& calibration,
        modalities::Status* status);

    ~NativeThorGraphOwner() override;

    NativeThorGraphOwner(const NativeThorGraphOwner&) = delete;
    NativeThorGraphOwner& operator=(const NativeThorGraphOwner&) = delete;

    frt_ctx context() const override { return graphs_.context(); }
    frt_graph graph(NativeGraphKind kind) const override {
        return graphs_.graph(static_cast<std::size_t>(kind));
    }
    int stream_id() const override { return graphs_.stream_id(); }
    void* native_stream() const override { return graphs_.native_stream(); }
    NativeDeviceWeightStore& weights() override { return weights_; }
    const NativeDeviceWeightStore& weights() const override { return weights_; }
    NativeWorkspace& workspace() override { return workspace_; }
    const NativeWorkspace& workspace() const override { return workspace_; }

    modalities::Status set_prompt_length(int prompt_tokens) override;
    int replay(NativeGraphKind kind = NativeGraphKind::kInfer) const override;
    modalities::Status synchronize() const override;

private:
    NativeThorGraphOwner(frt_ctx ctx, const NativeGraphConfig& config);
    modalities::Status initialize(
        const std::string& checkpoint_path,
        const NativeCalibrationArtifact& calibration);
    modalities::Status record(NativeGraphKind kind, void* stream);
    modalities::Status record_context(void* stream);
    modalities::Status record_action(void* stream);
    static modalities::Status record_graph(
        void* user, std::size_t slot, void* stream);

    native::CudaGraphSet graphs_;
    NativeGraphConfig config_;
    NativeDeviceWeightStore weights_;
    NativeWorkspace workspace_;
    NativeThorKernelDriver driver_;
    NativeThorFp8Forward forward_;
    NativeThorWeightScales weight_scales_;
    std::vector<float> encoder_alphas_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_GRAPH_OWNER_H

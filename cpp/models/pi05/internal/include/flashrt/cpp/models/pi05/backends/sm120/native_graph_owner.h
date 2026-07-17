#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_OWNER_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_OWNER_H

#include "flashrt/cpp/models/pi05/backends/sm120/native_bf16_forward.h"
#include "flashrt/cpp/models/pi05/support/native_calibration.h"
#include "flashrt/cpp/models/pi05/backend/native_graph_runtime.h"

#include <memory>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {

class NativeGraphOwner final : public NativeGraphRuntime {
public:
    static std::unique_ptr<NativeGraphOwner> create(
        const std::string& checkpoint_path, const NativeGraphConfig& config,
        modalities::Status* status);
    static std::unique_ptr<NativeGraphOwner> create(
        const std::string& checkpoint_path, const NativeGraphConfig& config,
        const NativeCalibrationArtifact& calibration,
        modalities::Status* status);
    static std::unique_ptr<NativeGraphOwner> create_calibration(
        const std::string& checkpoint_path, const NativeGraphConfig& config,
        modalities::Status* status);

    ~NativeGraphOwner() override;

    NativeGraphOwner(const NativeGraphOwner&) = delete;
    NativeGraphOwner& operator=(const NativeGraphOwner&) = delete;

    frt_ctx context() const override { return graphs_.context(); }
    frt_graph graph(NativeGraphKind kind) const override {
        return graphs_.graph(static_cast<std::size_t>(kind));
    }
    int stream_id() const override { return graphs_.stream_id(); }
    void* native_stream() const override { return graphs_.native_stream(); }
    const NativeGraphConfig& config() const { return config_; }
    NativeDeviceWeightStore& weights() { return weights_; }
    const NativeDeviceWeightStore& weights() const { return weights_; }
    NativeWorkspace& workspace() { return workspace_; }
    const NativeWorkspace& workspace() const { return workspace_; }
    const NativeRuntimeArtifacts& artifacts() const override {
        return artifacts_;
    }
    NativeRtxAttentionWorkspace& attention() { return attention_; }
    const NativeRtxAttentionWorkspace& attention() const { return attention_; }

    modalities::Status set_prompt_length(int prompt_tokens) override;
    int replay(NativeGraphKind kind = NativeGraphKind::kInfer) const override;
    modalities::Status synchronize() const override;

private:
    NativeGraphOwner(frt_ctx ctx, const NativeGraphConfig& config,
                     NativeRtxLinearMode linear_mode);
    modalities::Status initialize(
        const std::string& checkpoint_path,
        const NativeCalibrationArtifact* calibration);
    modalities::Status record(NativeGraphKind kind, void* stream);
    modalities::Status record_context(void* stream);
    modalities::Status record_action(void* stream);
    static modalities::Status record_graph(
        void* user, std::size_t slot, void* stream);

    native::CudaGraphSet graphs_;
    NativeGraphConfig config_;
    NativeDeviceWeightStore weights_;
    NativeWorkspace workspace_;
    NativeRuntimeArtifacts artifacts_;
    NativeRtxAttentionWorkspace attention_;
    NativeKernelDriver driver_;
    NativeRtxLinear linear_;
    NativeBf16Forward forward_;
    std::unique_ptr<NativeRtxAttentionDriver> attention_driver_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_OWNER_H

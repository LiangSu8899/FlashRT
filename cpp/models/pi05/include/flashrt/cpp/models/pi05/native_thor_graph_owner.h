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

    frt_ctx context() const override { return ctx_; }
    frt_graph infer_graph() const override { return infer_graph_; }
    int stream_id() const override { return stream_id_; }
    void* native_stream() const override { return replay_stream_; }
    NativeDeviceWeightStore& weights() override { return weights_; }
    const NativeDeviceWeightStore& weights() const override { return weights_; }
    NativeWorkspace& workspace() override { return workspace_; }
    const NativeWorkspace& workspace() const override { return workspace_; }

    modalities::Status set_prompt_length(int prompt_tokens) override;
    int replay() const;
    modalities::Status synchronize() const;

private:
    NativeThorGraphOwner(frt_ctx ctx, const NativeGraphConfig& config);
    modalities::Status initialize(
        const std::string& checkpoint_path,
        const NativeCalibrationArtifact& calibration);
    modalities::Status record(void* stream);
    static void record_graph(void* user, void* stream);

    frt_ctx ctx_ = nullptr;
    NativeGraphConfig config_;
    NativeDeviceWeightStore weights_;
    NativeWorkspace workspace_;
    NativeThorKernelDriver driver_;
    NativeThorFp8Forward forward_;
    NativeThorWeightScales weight_scales_;
    std::vector<float> encoder_alphas_;
    frt_graph infer_graph_ = nullptr;
    void* replay_stream_ = nullptr;
    int stream_id_ = -1;
    modalities::Status capture_status_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_THOR_GRAPH_OWNER_H

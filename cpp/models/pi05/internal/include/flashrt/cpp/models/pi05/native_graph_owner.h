#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_OWNER_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_OWNER_H

#include "flashrt/cpp/models/pi05/native_bf16_forward.h"
#include "flashrt/cpp/models/pi05/native_graph_runtime.h"

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

    ~NativeGraphOwner() override;

    NativeGraphOwner(const NativeGraphOwner&) = delete;
    NativeGraphOwner& operator=(const NativeGraphOwner&) = delete;

    frt_ctx context() const override { return ctx_; }
    frt_graph infer_graph() const override { return infer_graph_; }
    int stream_id() const override { return stream_id_; }
    void* native_stream() const override { return replay_stream_; }
    const NativeGraphConfig& config() const { return config_; }
    NativeDeviceWeightStore& weights() override { return weights_; }
    const NativeDeviceWeightStore& weights() const override { return weights_; }
    NativeWorkspace& workspace() override { return workspace_; }
    const NativeWorkspace& workspace() const override { return workspace_; }
    NativeRtxAttentionWorkspace& attention() { return attention_; }
    const NativeRtxAttentionWorkspace& attention() const { return attention_; }

    modalities::Status set_prompt_length(int prompt_tokens) override;
    int replay() const;
    modalities::Status synchronize() const;

private:
    explicit NativeGraphOwner(frt_ctx ctx, const NativeGraphConfig& config);
    modalities::Status initialize(const std::string& checkpoint_path);
    modalities::Status record(void* stream);
    static void record_graph(void* user, void* stream);

    frt_ctx ctx_ = nullptr;
    NativeGraphConfig config_;
    NativeDeviceWeightStore weights_;
    NativeWorkspace workspace_;
    NativeRtxAttentionWorkspace attention_;
    NativeKernelDriver driver_;
    NativeBf16Forward forward_;
    std::unique_ptr<NativeRtxAttentionDriver> attention_driver_;
    frt_graph infer_graph_ = nullptr;
    void* replay_stream_ = nullptr;
    int stream_id_ = -1;
    modalities::Status capture_status_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_OWNER_H

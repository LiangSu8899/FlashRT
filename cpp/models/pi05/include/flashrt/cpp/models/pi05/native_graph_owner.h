#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_OWNER_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_OWNER_H

#include "flashrt/cpp/models/pi05/native_bf16_forward.h"

#include <memory>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {

struct NativeGraphConfig {
    int num_views = 2;
    int max_prompt_tokens = 200;
    int chunk_size = 10;
    int num_steps = 10;
    int vision_pool_factor = 1;
};

class NativeGraphOwner {
public:
    static std::unique_ptr<NativeGraphOwner> create(
        const std::string& checkpoint_path, const NativeGraphConfig& config,
        modalities::Status* status);

    ~NativeGraphOwner();

    NativeGraphOwner(const NativeGraphOwner&) = delete;
    NativeGraphOwner& operator=(const NativeGraphOwner&) = delete;

    frt_ctx context() const { return ctx_; }
    frt_graph infer_graph() const { return infer_graph_; }
    int stream_id() const { return stream_id_; }
    void* native_stream() const { return replay_stream_; }
    const NativeGraphConfig& config() const { return config_; }
    NativeDeviceWeightStore& weights() { return weights_; }
    const NativeDeviceWeightStore& weights() const { return weights_; }
    NativeWorkspace& workspace() { return workspace_; }
    const NativeWorkspace& workspace() const { return workspace_; }
    NativeRtxAttentionWorkspace& attention() { return attention_; }
    const NativeRtxAttentionWorkspace& attention() const { return attention_; }

    modalities::Status set_prompt_length(int prompt_tokens);
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

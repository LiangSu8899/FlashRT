#ifndef FLASHRT_CPP_MODELS_PI05_MODEL_PIPELINE_H
#define FLASHRT_CPP_MODELS_PI05_MODEL_PIPELINE_H

#include "flashrt/cpp/models/pi05/support/native_device_weights.h"
#include "flashrt/cpp/models/pi05/support/native_workspace.h"
#include "flashrt/cpp/native/cuda_graph_set.h"

#include <cstddef>
#include <initializer_list>

namespace flashrt {
namespace models {
namespace pi05 {

enum class Pi05Precision {
    kBf16,
    kFp8E4M3,
};

struct Pi05PipelineConfig {
    int num_views = 2;
    int max_prompt_tokens = 200;
    int chunk_size = 10;
    int num_steps = 10;
    int vision_pool_factor = 1;
    Pi05Precision precision = Pi05Precision::kBf16;
};

enum class GraphKind : std::size_t {
    kInfer = 0,
    kDecodeOnly = 1,
    kContext = 2,
    kCount = 3,
};

const char* pipeline_graph_name(GraphKind kind);

modalities::Status capture_pipeline_graph(
    native::CudaGraphSet* graphs,
    GraphKind kind,
    const NativeWorkspace& workspace,
    std::initializer_list<const char*> bindings,
    native::CudaGraphSet::RecordFn record,
    void* owner);

modalities::Status copy_prompt_to_encoder(NativeWorkspace* workspace,
                                          void* stream);

struct PipelineArtifacts {
    const NativeWorkspaceBuffer* images = nullptr;
    const NativeWorkspaceBuffer* noise = nullptr;
    const NativeWorkspaceBuffer* encoder = nullptr;
    const NativeWorkspaceBuffer* previous_actions = nullptr;
    const NativeWorkspaceBuffer* prefix_weights = nullptr;
    const NativeWorkspaceBuffer* guidance_weight = nullptr;
    const NativeWorkspaceBuffer* prompt_embedding = nullptr;
    const NativeDeviceWeight* embedding_table = nullptr;
};

modalities::Status resolve_pipeline_artifacts(
    const NativeWorkspace& workspace,
    const NativeDeviceWeightStore& weights,
    NativeWeightDType embedding_dtype,
    PipelineArtifacts* artifacts);

class Pi05Pipeline {
public:
    Pi05Pipeline(frt_ctx context, const Pi05PipelineConfig& config);
    virtual ~Pi05Pipeline() = default;

    Pi05Pipeline(const Pi05Pipeline&) = delete;
    Pi05Pipeline& operator=(const Pi05Pipeline&) = delete;

    frt_ctx context() const { return graphs_.context(); }
    frt_graph graph(GraphKind kind) const {
        return graphs_.graph(static_cast<std::size_t>(kind));
    }
    frt_graph infer_graph() const { return graph(GraphKind::kInfer); }
    int stream_id() const { return graphs_.stream_id(); }
    void* native_stream() const { return graphs_.native_stream(); }
    const Pi05PipelineConfig& config() const { return config_; }
    virtual const PipelineArtifacts& artifacts() const = 0;
    virtual modalities::Status set_prompt_length(int prompt_tokens) = 0;
    int replay(GraphKind kind = GraphKind::kInfer) const;
    modalities::Status synchronize() const;

protected:
    modalities::Status finish_prepare(bool warmup_before_capture);
    virtual NativeWorkspace& workspace() = 0;
    virtual const NativeWorkspace& workspace() const = 0;
    virtual modalities::Status record_vision(void* stream) = 0;
    virtual modalities::Status record_encoder(void* stream) = 0;
    virtual modalities::Status record_diffusion_step(int step,
                                                      void* stream) = 0;

private:
    modalities::Status initialize_capture_inputs();
    modalities::Status record(GraphKind kind, void* stream);
    modalities::Status record_context(void* stream);
    modalities::Status record_decode(void* stream);
    static modalities::Status record_graph(
        void* owner, std::size_t slot, void* stream);

    native::CudaGraphSet graphs_;
    Pi05PipelineConfig config_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_MODEL_PIPELINE_H

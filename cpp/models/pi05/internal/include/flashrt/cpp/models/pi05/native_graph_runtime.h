#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_RUNTIME_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_RUNTIME_H

#include "flashrt/cpp/models/pi05/native_device_weights.h"
#include "flashrt/cpp/models/pi05/native_workspace.h"

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

class NativeGraphRuntime {
public:
    virtual ~NativeGraphRuntime() = default;

    virtual frt_ctx context() const = 0;
    virtual frt_graph infer_graph() const = 0;
    virtual int stream_id() const = 0;
    virtual void* native_stream() const = 0;
    virtual NativeDeviceWeightStore& weights() = 0;
    virtual const NativeDeviceWeightStore& weights() const = 0;
    virtual NativeWorkspace& workspace() = 0;
    virtual const NativeWorkspace& workspace() const = 0;
    virtual modalities::Status set_prompt_length(int prompt_tokens) = 0;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_RUNTIME_H

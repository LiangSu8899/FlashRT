#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_RUNTIME_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_RUNTIME_H

#include "flashrt/cpp/models/pi05/native_device_weights.h"
#include "flashrt/cpp/models/pi05/native_workspace.h"
#include "flashrt/cpp/native/cuda_graph_set.h"

#include <cstddef>
#include <initializer_list>

namespace flashrt {
namespace models {
namespace pi05 {

enum class NativeGraphPrecision {
    kBf16,
    kFp8E4M3,
};

struct NativeGraphConfig {
    int num_views = 2;
    int max_prompt_tokens = 200;
    int chunk_size = 10;
    int num_steps = 10;
    int vision_pool_factor = 1;
    NativeGraphPrecision precision = NativeGraphPrecision::kBf16;
};

enum class NativeGraphKind : std::size_t {
    kInfer = 0,
    kDecodeOnly = 1,
    kContext = 2,
    kCount = 3,
};

const char* native_graph_name(NativeGraphKind kind);

modalities::Status capture_native_graph(
    native::CudaGraphSet* graphs,
    NativeGraphKind kind,
    const NativeWorkspace& workspace,
    std::initializer_list<const char*> bindings,
    native::CudaGraphSet::RecordFn record,
    void* owner);

modalities::Status copy_prompt_to_encoder(NativeWorkspace* workspace,
                                          void* stream);

class NativeGraphRuntime {
public:
    virtual ~NativeGraphRuntime() = default;

    virtual frt_ctx context() const = 0;
    virtual frt_graph graph(NativeGraphKind kind) const = 0;
    frt_graph infer_graph() const { return graph(NativeGraphKind::kInfer); }
    virtual int stream_id() const = 0;
    virtual void* native_stream() const = 0;
    virtual NativeDeviceWeightStore& weights() = 0;
    virtual const NativeDeviceWeightStore& weights() const = 0;
    virtual NativeWorkspace& workspace() = 0;
    virtual const NativeWorkspace& workspace() const = 0;
    virtual modalities::Status set_prompt_length(int prompt_tokens) = 0;
    virtual int replay(NativeGraphKind kind = NativeGraphKind::kInfer) const = 0;
    virtual modalities::Status synchronize() const = 0;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_RUNTIME_H

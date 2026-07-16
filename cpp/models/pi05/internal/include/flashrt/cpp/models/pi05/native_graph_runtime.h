#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_RUNTIME_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_GRAPH_RUNTIME_H

#include "flashrt/cpp/models/pi05/native_device_weights.h"
#include "flashrt/cpp/models/pi05/native_workspace.h"

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

class NativeGraphCatalog {
public:
    using RecordFn = modalities::Status (*)(
        void* owner, NativeGraphKind kind, void* stream);

    explicit NativeGraphCatalog(frt_ctx ctx) : ctx_(ctx) {}
    ~NativeGraphCatalog();

    NativeGraphCatalog(const NativeGraphCatalog&) = delete;
    NativeGraphCatalog& operator=(const NativeGraphCatalog&) = delete;

    modalities::Status capture(
        NativeGraphKind kind, const NativeWorkspace& workspace,
        std::initializer_list<const char*> bindings,
        RecordFn record, void* owner);
    modalities::Status create_replay_stream();

    frt_ctx context() const { return ctx_; }
    frt_graph graph(NativeGraphKind kind) const;
    int stream_id() const { return stream_id_; }
    void* native_stream() const { return replay_stream_; }
    int replay(NativeGraphKind kind) const;
    modalities::Status synchronize() const;

    static const char* name(NativeGraphKind kind);

private:
    struct CaptureCall;
    static void record_graph(void* user, void* stream);

    frt_ctx ctx_ = nullptr;
    frt_graph graphs_[static_cast<std::size_t>(NativeGraphKind::kCount)] = {};
    void* replay_stream_ = nullptr;
    int stream_id_ = -1;
};

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

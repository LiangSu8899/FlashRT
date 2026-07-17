#include "flashrt/cpp/native/cuda_graph_set.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cassert>
#include <cstddef>

namespace {

struct RecordCall {
    void* destination = nullptr;
    std::size_t bytes = 0;
};

flashrt::modalities::Status record_fill(
    void* user, std::size_t slot, void* stream) {
    auto* call = static_cast<RecordCall*>(user);
    if (!call || slot != 0 || !stream || !call->destination) {
        return flashrt::modalities::Status::error(
            flashrt::modalities::StatusCode::kInvalidArgument,
            "invalid graph test record request");
    }
    const cudaError_t result = cudaMemsetAsync(
        call->destination, 0x5a, call->bytes,
        static_cast<cudaStream_t>(stream));
    return result == cudaSuccess
               ? flashrt::modalities::Status::ok()
               : flashrt::modalities::Status::error(
                     flashrt::modalities::StatusCode::kBackend,
                     cudaGetErrorString(result));
}

}  // namespace

int main() {
    frt_ctx context = frt_ctx_create();
    assert(context);
    flashrt::native::CudaGraphSet graphs(context, 1);

    constexpr std::size_t kBytes = 64;
    frt_buffer output = frt_buffer_alloc(context, "output", kBytes);
    assert(output);
    RecordCall call{frt_buffer_dptr(output), kBytes};
    const std::vector<flashrt::native::CudaGraphBinding> bindings = {
        {"output", output},
    };
    assert(graphs.capture(0, "fill", bindings, record_fill, &call)
               .ok_status());
    assert(graphs.graph(0));
    assert(frt_graph_variant_count(graphs.graph(0)) == 1);
    assert(graphs.create_replay_stream().ok_status());
    assert(graphs.replay(0) == FRT_OK);
    assert(graphs.synchronize().ok_status());

    std::array<unsigned char, kBytes> result{};
    assert(cudaMemcpy(result.data(), frt_buffer_dptr(output), kBytes,
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    for (unsigned char value : result) assert(value == 0x5a);
    return 0;
}

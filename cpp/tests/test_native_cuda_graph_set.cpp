#include "flashrt/cpp/native/cuda_graph_set.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

[[noreturn]] void fail(const char* expression, int line) {
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    std::abort();
}

#define CHECK(expression)                              \
    do {                                               \
        if (!(expression)) fail(#expression, __LINE__); \
    } while (false)

struct RecordCall {
    void* destination = nullptr;
    std::size_t bytes = 0;
    std::size_t expected_slot = 0;
    bool inject_failure = false;
    std::size_t calls = 0;
};

flashrt::modalities::Status record_fill(
    void* user, std::size_t slot, void* stream) {
    auto* call = static_cast<RecordCall*>(user);
    if (!call || slot != call->expected_slot || !stream ||
        !call->destination) {
        return flashrt::modalities::Status::error(
            flashrt::modalities::StatusCode::kInvalidArgument,
            "invalid graph test record request");
    }
    ++call->calls;
    if (call->inject_failure) {
        return flashrt::modalities::Status::error(
            flashrt::modalities::StatusCode::kBackend,
            "injected graph record failure");
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
    CHECK(context != nullptr);
    flashrt::native::CudaGraphSet graphs(context, 2);

    constexpr std::size_t kBytes = 64;
    frt_buffer output = frt_buffer_alloc(context, "output", kBytes);
    CHECK(output != nullptr);
    const std::vector<flashrt::native::CudaGraphBinding> bindings = {
        {"output", output},
    };

    RecordCall rejected{frt_buffer_dptr(output), kBytes, 1, true, 0};
    flashrt::modalities::Status status =
        graphs.capture(1, "rejected", bindings, record_fill, &rejected);
    CHECK(!status.ok_status());
    CHECK(status.code == flashrt::modalities::StatusCode::kBackend);
    CHECK(rejected.calls == 1);
    CHECK(graphs.graph(1) == nullptr);

    RecordCall accepted{frt_buffer_dptr(output), kBytes, 0, false, 0};
    CHECK(graphs.capture(0, "fill", bindings, record_fill, &accepted)
              .ok_status());
    CHECK(accepted.calls == 1);
    CHECK(graphs.graph(0) != nullptr);
    CHECK(frt_graph_variant_count(graphs.graph(0)) == 1);
    CHECK(!graphs.capture(0, "duplicate", bindings, record_fill, &accepted)
               .ok_status());
    CHECK(graphs.replay(0) == FRT_ERR_INVALID);
    CHECK(graphs.replay(2) == FRT_ERR_INVALID);
    CHECK(!graphs.synchronize().ok_status());

    CHECK(graphs.create_replay_stream().ok_status());
    CHECK(!graphs.create_replay_stream().ok_status());
    CHECK(graphs.replay(0) == FRT_OK);
    CHECK(graphs.synchronize().ok_status());

    std::array<unsigned char, kBytes> result{};
    CHECK(cudaMemcpy(result.data(), frt_buffer_dptr(output), kBytes,
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    for (unsigned char value : result) CHECK(value == 0x5a);

    std::cout << "PASS - native CUDA graph set\n";
    return 0;
}

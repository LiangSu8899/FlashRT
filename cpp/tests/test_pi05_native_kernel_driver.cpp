#include "flashrt/cpp/models/pi05/native_kernel_driver.h"
#include "flashrt/cpp/modalities/types.h"
#include "flashrt/exec.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct CaptureArgs {
    flashrt::models::pi05::NativeKernelDriver* driver = nullptr;
    void* a = nullptr;
    void* b = nullptr;
    void* output = nullptr;
    bool recorded = false;
};

bool has_cuda_device() {
    int count = 0;
    const cudaError_t rc = cudaGetDeviceCount(&count);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

void record_gemm(void* user, void* stream) {
    auto* args = static_cast<CaptureArgs*>(user);
    args->recorded = args->driver
                         ->bf16_nn(args->a, args->b, args->output,
                                   2, 2, 3,
                                   reinterpret_cast<std::uintptr_t>(stream))
                         .ok_status();
}

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    using flashrt::modalities::bfloat16_to_float;
    using flashrt::modalities::float_to_bfloat16;

    flashrt::models::pi05::NativeKernelDriver driver;
    assert(driver.status().ok_status());
    frt_ctx ctx = frt_ctx_create();
    assert(ctx);
    frt_buffer a = frt_buffer_alloc(ctx, "a", 6 * sizeof(std::uint16_t));
    frt_buffer b = frt_buffer_alloc(ctx, "b", 6 * sizeof(std::uint16_t));
    frt_buffer output =
        frt_buffer_alloc(ctx, "output", 4 * sizeof(std::uint16_t));
    assert(a && b && output);
    const std::vector<std::uint16_t> host_a = {
        float_to_bfloat16(1), float_to_bfloat16(2), float_to_bfloat16(3),
        float_to_bfloat16(4), float_to_bfloat16(5), float_to_bfloat16(6)};
    const std::vector<std::uint16_t> host_b = {
        float_to_bfloat16(1), float_to_bfloat16(2),
        float_to_bfloat16(3), float_to_bfloat16(4),
        float_to_bfloat16(5), float_to_bfloat16(6)};
    assert(cudaMemcpy(frt_buffer_dptr(a), host_a.data(),
                      host_a.size() * sizeof(std::uint16_t),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(frt_buffer_dptr(b), host_b.data(),
                      host_b.size() * sizeof(std::uint16_t),
                      cudaMemcpyHostToDevice) == cudaSuccess);

    cudaStream_t stream = nullptr;
    assert(cudaStreamCreate(&stream) == cudaSuccess);
    assert(driver.bf16_nn(frt_buffer_dptr(a), frt_buffer_dptr(b),
                          frt_buffer_dptr(output), 2, 2, 3,
                          reinterpret_cast<std::uintptr_t>(stream))
               .ok_status());
    assert(cudaStreamSynchronize(stream) == cudaSuccess);

    frt_graph graph = frt_graph_create(ctx, "native_bf16_gemm", 1);
    assert(graph);
    assert(frt_graph_bind(graph, "a", a) == FRT_OK);
    assert(frt_graph_bind(graph, "b", b) == FRT_OK);
    assert(frt_graph_bind(graph, "output", output) == FRT_OK);
    CaptureArgs capture{&driver, frt_buffer_dptr(a), frt_buffer_dptr(b),
                        frt_buffer_dptr(output), false};
    assert(frt_graph_capture(graph, 1, record_gemm, &capture) == FRT_OK);
    assert(capture.recorded);
    assert(frt_graph_variant_count(graph) == 1);
    const int stream_id = frt_ctx_wrap_stream(ctx, stream);
    assert(stream_id >= 0);
    for (int i = 0; i < 100; ++i) {
        assert(frt_graph_replay(graph, 1, stream_id) == FRT_OK);
    }
    assert(frt_graph_variant_count(graph) == 1);
    assert(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<std::uint16_t> host_output(4);
    assert(cudaMemcpy(host_output.data(), frt_buffer_dptr(output),
                      host_output.size() * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    const float expected[] = {22, 28, 49, 64};
    for (std::size_t i = 0; i < host_output.size(); ++i) {
        assert(std::fabs(bfloat16_to_float(host_output[i]) - expected[i]) <
               0.01f);
    }
    frt_graph_destroy(graph);
    assert(cudaStreamDestroy(stream) == cudaSuccess);
    frt_ctx_destroy(ctx);
    std::printf("PASS - Pi0.5 native kernel driver capture\n");
    return 0;
}

#include "rope.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

__global__ void reference_rope_table(
    __half* output, int start_position, int positions,
    int frequencies, float theta) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= positions * frequencies) return;
    const int row = index / frequencies;
    const int frequency = index - row * frequencies;
    const float exponent = static_cast<float>(frequency) / frequencies;
    const float denominator = powf(theta, exponent);
    float inverse_frequency = __fdiv_rn(1.0f, denominator);
    inverse_frequency = __bfloat162float(
        __float2bfloat16_rn(inverse_frequency));
    const float phase = __fmul_rn(
        static_cast<float>(start_position + row), inverse_frequency);
    output[2 * index] = __float2half_rn(cosf(phase));
    output[2 * index + 1] = __float2half_rn(sinf(phase));
}

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

bool has_cuda_device() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) cudaGetLastError();
    return status == cudaSuccess && count > 0;
}

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }

    constexpr int kStartPosition = 641;
    constexpr int kPositions = 64;
    constexpr int kFrequencies = 128;
    constexpr float kTheta = 10000.0f;
    constexpr int kValues = kPositions * kFrequencies * 2;
    const std::size_t bytes = static_cast<std::size_t>(kValues) * sizeof(__half);

    __half* actual = nullptr;
    __half* expected = nullptr;
    if (!check_cuda(cudaMalloc(&actual, bytes), "cudaMalloc(actual)") ||
        !check_cuda(cudaMalloc(&expected, bytes), "cudaMalloc(expected)")) {
        cudaFree(actual);
        cudaFree(expected);
        return 1;
    }

    generate_rope_table_f16(actual, kStartPosition, kPositions,
                            kFrequencies, kTheta);
    reference_rope_table<<<(kPositions * kFrequencies + 255) / 256, 256>>>(
        expected, kStartPosition, kPositions, kFrequencies, kTheta);
    if (!check_cuda(cudaGetLastError(), "RoPE table launch") ||
        !check_cuda(cudaDeviceSynchronize(), "RoPE table synchronize")) {
        cudaFree(actual);
        cudaFree(expected);
        return 1;
    }

    std::vector<__half> actual_host(kValues);
    std::vector<__half> expected_host(kValues);
    const bool copied =
        check_cuda(cudaMemcpy(actual_host.data(), actual, bytes,
                              cudaMemcpyDeviceToHost), "copy actual") &&
        check_cuda(cudaMemcpy(expected_host.data(), expected, bytes,
                              cudaMemcpyDeviceToHost), "copy expected");
    cudaFree(actual);
    cudaFree(expected);
    if (!copied) return 1;

    if (std::memcmp(actual_host.data(), expected_host.data(), bytes) != 0) {
        std::fprintf(stderr,
                     "RoPE table differs from precise device-math reference\n");
        return 1;
    }
    return 0;
}

#include "activation.cuh"
#include "elementwise.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

__global__ void reference_silu_bf16(__nv_bfloat16* values, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = __bfloat162float(values[index]);
        values[index] =
            __float2bfloat16(value / (1.0f + expf(-value)));
    }
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

bool test_silu() {
    const float inputs[] = {
        -10.0f, -3.0f, -1.0f, -0.5f, -0.0f, 0.0f,
        0.5f, 1.0f, 3.0f, 10.0f, 0.00390625f,
    };
    constexpr int kCount = static_cast<int>(sizeof(inputs) / sizeof(float));
    std::vector<__nv_bfloat16> host(kCount);
    for (int i = 0; i < kCount; ++i) {
        host[static_cast<std::size_t>(i)] = __float2bfloat16(inputs[i]);
    }

    __nv_bfloat16* actual = nullptr;
    __nv_bfloat16* expected = nullptr;
    const std::size_t bytes = host.size() * sizeof(__nv_bfloat16);
    if (!check_cuda(cudaMalloc(&actual, bytes), "cudaMalloc(actual)") ||
        !check_cuda(cudaMalloc(&expected, bytes), "cudaMalloc(expected)") ||
        !check_cuda(cudaMemcpy(actual, host.data(), bytes,
                              cudaMemcpyHostToDevice), "copy actual input") ||
        !check_cuda(cudaMemcpy(expected, host.data(), bytes,
                              cudaMemcpyHostToDevice), "copy expected input")) {
        cudaFree(actual);
        cudaFree(expected);
        return false;
    }

    silu_inplace_bf16(actual, kCount);
    reference_silu_bf16<<<1, 256>>>(expected, kCount);
    if (!check_cuda(cudaGetLastError(), "SiLU launches") ||
        !check_cuda(cudaDeviceSynchronize(), "SiLU synchronize")) {
        cudaFree(actual);
        cudaFree(expected);
        return false;
    }

    std::vector<std::uint16_t> actual_host(kCount);
    std::vector<std::uint16_t> expected_host(kCount);
    const bool copied =
        check_cuda(cudaMemcpy(actual_host.data(), actual, bytes,
                              cudaMemcpyDeviceToHost), "copy actual output") &&
        check_cuda(cudaMemcpy(expected_host.data(), expected, bytes,
                              cudaMemcpyDeviceToHost), "copy expected output");
    cudaFree(actual);
    cudaFree(expected);
    return copied &&
           std::memcmp(actual_host.data(), expected_host.data(), bytes) == 0;
}

bool test_negative_infinity() {
    constexpr std::size_t kCount = 513;
    float* values = nullptr;
    if (!check_cuda(cudaMalloc(&values, kCount * sizeof(float)),
                    "cudaMalloc(negative infinity)")) {
        return false;
    }
    fill_negative_infinity_f32(nullptr, 0);
    fill_negative_infinity_f32(values, kCount);
    if (!check_cuda(cudaGetLastError(), "negative infinity launch") ||
        !check_cuda(cudaDeviceSynchronize(),
                    "negative infinity synchronize")) {
        cudaFree(values);
        return false;
    }
    std::vector<std::uint32_t> host(kCount);
    const bool copied = check_cuda(
        cudaMemcpy(host.data(), values, kCount * sizeof(float),
                   cudaMemcpyDeviceToHost), "copy negative infinity");
    cudaFree(values);
    if (!copied) return false;
    for (std::uint32_t bits : host) {
        if (bits != 0xff800000u) return false;
    }
    return true;
}

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    if (!test_silu()) {
        std::fprintf(stderr, "BF16 SiLU differs from the scalar reference\n");
        return 1;
    }
    if (!test_negative_infinity()) {
        std::fprintf(stderr, "F32 negative-infinity fill is not bit exact\n");
        return 1;
    }
    return 0;
}

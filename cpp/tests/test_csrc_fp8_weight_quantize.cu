#include "quantize.cuh"
#include "flashrt/runtime.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kCount = 4096;
constexpr std::uint32_t kScaleBits = 0x3d092492u;
constexpr std::uint64_t kOutputHash = 0x68ea83e06e2dce5cull;
constexpr int kMidpointIndex = 402;
constexpr std::uint8_t kMidpointOutput = 0xf2u;

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

std::vector<__nv_bfloat16> golden_input() {
    constexpr int kTrial = 11;
    std::vector<__nv_bfloat16> values(kCount);
    std::uint32_t state = 0x9e3779b9u ^ kTrial;
    for (int i = 0; i < kCount; ++i) {
        state = state * 1664525u + 1013904223u;
        const int centered =
            static_cast<int>((state >> 8) % 2000001u) - 1000000;
        constexpr float kDenominator = 65537.0f + 97.0f * kTrial;
        values[static_cast<std::size_t>(i)] = __float2bfloat16_rn(
            static_cast<float>(centered) / kDenominator);
    }
    values[0] = __float2bfloat16_rn(7.0f + kTrial * 0.03125f);
    values[1] = __float2bfloat16_rn(-6.5f - kTrial * 0.015625f);
    return values;
}

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }

    const std::vector<__nv_bfloat16> input = golden_input();
    __nv_bfloat16* device_input = nullptr;
    __nv_fp8_e4m3* device_output = nullptr;
    float* device_scale = nullptr;
    if (!check_cuda(cudaMalloc(&device_input,
                               input.size() * sizeof(__nv_bfloat16)),
                    "cudaMalloc(input)") ||
        !check_cuda(cudaMalloc(&device_output,
                               input.size() * sizeof(__nv_fp8_e4m3)),
                    "cudaMalloc(output)") ||
        !check_cuda(cudaMalloc(&device_scale, sizeof(float)),
                    "cudaMalloc(scale)") ||
        !check_cuda(cudaMemcpy(device_input, input.data(),
                               input.size() * sizeof(__nv_bfloat16),
                               cudaMemcpyHostToDevice),
                    "copy input")) {
        cudaFree(device_input);
        cudaFree(device_output);
        cudaFree(device_scale);
        return 1;
    }

    quantize_fp8_weight_device(device_input, device_output, device_scale,
                               kCount);
    if (!check_cuda(cudaGetLastError(), "FP8 weight quantize launch") ||
        !check_cuda(cudaDeviceSynchronize(),
                    "FP8 weight quantize synchronize")) {
        cudaFree(device_input);
        cudaFree(device_output);
        cudaFree(device_scale);
        return 1;
    }

    std::vector<std::uint8_t> output(kCount);
    float scale = 0.0f;
    const bool copied =
        check_cuda(cudaMemcpy(output.data(), device_output, output.size(),
                              cudaMemcpyDeviceToHost), "copy output") &&
        check_cuda(cudaMemcpy(&scale, device_scale, sizeof(scale),
                              cudaMemcpyDeviceToHost), "copy scale");
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_scale);
    if (!copied) return 1;

    std::uint32_t scale_bits = 0;
    std::memcpy(&scale_bits, &scale, sizeof(scale_bits));
    if (scale_bits != kScaleBits ||
        output[kMidpointIndex] != kMidpointOutput ||
        frt_runtime_fingerprint(output.data(), output.size()) != kOutputHash) {
        std::fprintf(stderr,
                     "precise FP8 weight quantize golden mismatch\n");
        return 1;
    }
    return 0;
}

#include "flashrt/native_cpp/operations.h"
#include "fp8_exact.cuh"
#include "flashrt/runtime.h"
#include "quantize.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kCount = 4096;
constexpr std::uint32_t kScaleBits = 0x3d092492u;
constexpr std::uint32_t kCommonDynamicF16ScaleBits = 0x3d97924au;
constexpr std::uint32_t kDynamicF16ScaleBits = 0x3d979249u;
constexpr std::uint64_t kOutputHash = 0x68ea83e06e2dce5cull;
constexpr int kMidpointIndex = 402;
constexpr std::uint8_t kMidpointOutput = 0xf2u;

__global__ void check_exact_e4m3_conversion_kernel(
    const float* values, int count, int* result) {
    for (int index = 0; index < count; ++index) {
        const float value = values[index];
        const unsigned int bits = __float_as_uint(value);
        const float canonical = (bits & 0x7fffffffu) > 0x7f800000u
                                    ? __uint_as_float(0x7fffffffu)
                                    : value;
        const __nv_fp8_storage_t expected = __nv_cvt_double_to_fp8(
            static_cast<double>(canonical), __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8_storage_t actual =
            flashrt_native_fp8_e4m3_storage_rn(value);
        if (actual != expected) {
            if (result[0] == 0) {
                result[1] = index;
                result[2] = actual;
                result[3] = expected;
            }
            ++result[0];
        }
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

bool check_exact_e4m3_conversion() {
    std::vector<float> values;
    values.reserve(65536 + 8192 + 16);
    for (std::uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        __half value;
        const std::uint16_t storage = static_cast<std::uint16_t>(bits);
        std::memcpy(&value, &storage, sizeof(storage));
        values.push_back(__half2float(value));
    }
    const float boundaries[] = {
        -INFINITY, -464.0f, -448.0f, -1.0f / 64.0f,
        -1.0f / 1024.0f, -0.0f, 0.0f, 1.0f / 1024.0f,
        1.0f / 64.0f, 448.0f, 464.0f, INFINITY, NAN,
    };
    values.insert(values.end(), std::begin(boundaries), std::end(boundaries));
    std::uint32_t state = 0x243f6a88u;
    for (int index = 0; index < 8192; ++index) {
        state = state * 1664525u + 1013904223u;
        float value = 0.0f;
        std::memcpy(&value, &state, sizeof(value));
        values.push_back(value);
    }

    float* device_values = nullptr;
    int* device_result = nullptr;
    const std::size_t bytes = values.size() * sizeof(float);
    if (!check_cuda(cudaMalloc(&device_values, bytes),
                    "cudaMalloc(exact conversion values)") ||
        !check_cuda(cudaMalloc(&device_result, 4 * sizeof(int)),
                    "cudaMalloc(exact conversion result)") ||
        !check_cuda(cudaMemcpy(device_values, values.data(), bytes,
                               cudaMemcpyHostToDevice),
                    "copy exact conversion values") ||
        !check_cuda(cudaMemset(device_result, 0, 4 * sizeof(int)),
                    "clear exact conversion result")) {
        cudaFree(device_values);
        cudaFree(device_result);
        return false;
    }
    check_exact_e4m3_conversion_kernel<<<1, 1>>>(
        device_values, static_cast<int>(values.size()), device_result);
    int result[4] = {};
    const bool copied =
        check_cuda(cudaGetLastError(), "exact conversion launch") &&
        check_cuda(cudaMemcpy(result, device_result, sizeof(result),
                              cudaMemcpyDeviceToHost),
                   "copy exact conversion result");
    cudaFree(device_values);
    cudaFree(device_result);
    if (!copied) return false;
    if (result[0] != 0) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &values[static_cast<std::size_t>(result[1])],
                    sizeof(bits));
        std::fprintf(stderr,
                     "exact E4M3 conversion mismatches: %d; first=%08x "
                     "actual=%02x expected=%02x\n",
                     result[0], bits, result[2], result[3]);
        return false;
    }
    return true;
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

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>((bits + bias) >> 16);
}

bool check_setup_scale(const std::vector<__nv_bfloat16>& input) {
    __nv_bfloat16* device_input = nullptr;
    __nv_bfloat16* device_output = nullptr;
    const std::size_t bytes = input.size() * sizeof(__nv_bfloat16);
    if (!check_cuda(cudaMalloc(&device_input, bytes),
                    "cudaMalloc(scale input)") ||
        !check_cuda(cudaMalloc(&device_output, bytes),
                    "cudaMalloc(scale output)") ||
        !check_cuda(cudaMemcpy(device_input, input.data(), bytes,
                               cudaMemcpyHostToDevice),
                    "copy scale input")) {
        cudaFree(device_input);
        cudaFree(device_output);
        return false;
    }
    constexpr float kScale = -0.1f;
    flashrt_native_scale_bf16_weight(
        device_input, device_output, kScale, static_cast<int>(input.size()));
    std::vector<std::uint16_t> output(input.size());
    const bool copied =
        check_cuda(cudaGetLastError(), "BF16 setup scale launch") &&
        check_cuda(cudaMemcpy(output.data(), device_output, bytes,
                              cudaMemcpyDeviceToHost),
                   "copy scale output");
    cudaFree(device_input);
    cudaFree(device_output);
    if (!copied) return false;

    for (std::size_t i = 0; i < output.size(); ++i) {
        std::uint16_t input_bits = 0;
        std::memcpy(&input_bits, &input[i], sizeof(input_bits));
        const std::uint16_t expected = float_to_bf16(
            bf16_to_float(input_bits) * kScale);
        if (output[i] != expected) {
            std::fprintf(stderr,
                         "BF16 setup scale mismatch at %zu: %04x != %04x\n",
                         i, output[i], expected);
            return false;
        }
    }
    return true;
}

bool check_dynamic_f16_scale() {
    const __half input[] = {
        __float2half_rn(33.15625f),
        __float2half_rn(-1.0f),
    };
    __half* device_input = nullptr;
    __nv_fp8_e4m3* device_output = nullptr;
    float* device_scale = nullptr;
    if (!check_cuda(cudaMalloc(&device_input, sizeof(input)),
                    "cudaMalloc(dynamic F16 input)") ||
        !check_cuda(cudaMalloc(&device_output, sizeof(input) / sizeof(__half)),
                    "cudaMalloc(dynamic F16 output)") ||
        !check_cuda(cudaMalloc(&device_scale, sizeof(float)),
                    "cudaMalloc(dynamic F16 scale)") ||
        !check_cuda(cudaMemcpy(device_input, input, sizeof(input),
                               cudaMemcpyHostToDevice),
                    "copy dynamic F16 input")) {
        cudaFree(device_input);
        cudaFree(device_output);
        cudaFree(device_scale);
        return false;
    }

    quantize_fp8_device_fp16(device_input, device_output, device_scale, 2);
    float common_scale = 0.0f;
    bool copied =
        check_cuda(cudaGetLastError(), "dynamic F16 quantize launch") &&
        check_cuda(cudaMemcpy(&common_scale, device_scale, sizeof(common_scale),
                              cudaMemcpyDeviceToHost),
                   "copy common dynamic F16 scale");
    flashrt_native_quantize_fp8_device_f16_precise(
        device_input, device_output, device_scale, 2);
    float native_scale = 0.0f;
    copied = copied &&
        check_cuda(cudaGetLastError(), "native dynamic F16 quantize launch") &&
        check_cuda(cudaMemcpy(&native_scale, device_scale,
                              sizeof(native_scale), cudaMemcpyDeviceToHost),
                   "copy native dynamic F16 scale");
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_scale);
    if (!copied) return false;

    std::uint32_t common_bits = 0;
    std::uint32_t native_bits = 0;
    std::memcpy(&common_bits, &common_scale, sizeof(common_bits));
    std::memcpy(&native_bits, &native_scale, sizeof(native_bits));
    if (common_bits != kCommonDynamicF16ScaleBits ||
        native_bits != kDynamicF16ScaleBits) {
        std::fprintf(stderr,
                     "dynamic F16 scales mismatch: common=%08x native=%08x\n",
                     common_bits, native_bits);
        return false;
    }
    return true;
}

bool check_f16_pack(bool pair, bool transpose) {
    constexpr int kRows = 3;
    constexpr int kColumns = 5;
    constexpr int kElements = kRows * kColumns;
    std::vector<__half> first(kElements);
    std::vector<__half> second(kElements);
    for (int i = 0; i < kElements; ++i) {
        first[static_cast<std::size_t>(i)] =
            __float2half_rn((static_cast<float>(i) - 6.0f) * 0.1875f);
        second[static_cast<std::size_t>(i)] =
            __float2half_rn((9.0f - static_cast<float>(i)) * 0.3125f);
    }
    const int output_elements = pair ? 2 * kElements : kElements;
    __half* device_first = nullptr;
    __half* device_second = nullptr;
    __nv_fp8_e4m3* device_output = nullptr;
    float* device_scale = nullptr;
    if (!check_cuda(cudaMalloc(&device_first, sizeof(__half) * kElements),
                    "cudaMalloc(F16 first)") ||
        (pair && !check_cuda(
            cudaMalloc(&device_second, sizeof(__half) * kElements),
            "cudaMalloc(F16 second)")) ||
        !check_cuda(cudaMalloc(&device_output, output_elements),
                    "cudaMalloc(F16 packed)") ||
        !check_cuda(cudaMalloc(&device_scale, sizeof(float)),
                    "cudaMalloc(F16 scale)") ||
        !check_cuda(cudaMemcpy(device_first, first.data(),
                               sizeof(__half) * kElements,
                               cudaMemcpyHostToDevice),
                    "copy F16 first") ||
        (pair && !check_cuda(cudaMemcpy(
            device_second, second.data(), sizeof(__half) * kElements,
            cudaMemcpyHostToDevice), "copy F16 second"))) {
        cudaFree(device_first);
        cudaFree(device_second);
        cudaFree(device_output);
        cudaFree(device_scale);
        return false;
    }
    if (pair) {
        flashrt_native_quantize_fp8_weight_f16_pair(
            device_first, device_second, device_output, device_scale,
            kRows, kColumns, transpose);
    } else {
        flashrt_native_quantize_fp8_weight_f16(
            device_first, device_output, device_scale,
            kRows, kColumns, transpose);
    }
    std::vector<std::uint8_t> output(
        static_cast<std::size_t>(output_elements));
    float scale = 0.0f;
    const bool copied =
        check_cuda(cudaGetLastError(), "F16 weight pack launch") &&
        check_cuda(cudaMemcpy(output.data(), device_output, output.size(),
                              cudaMemcpyDeviceToHost),
                   "copy F16 packed") &&
        check_cuda(cudaMemcpy(&scale, device_scale, sizeof(scale),
                              cudaMemcpyDeviceToHost),
                   "copy F16 scale");
    cudaFree(device_first);
    cudaFree(device_second);
    cudaFree(device_output);
    cudaFree(device_scale);
    if (!copied) return false;

    float amax = 0.0f;
    for (const __half value : first) {
        amax = std::max(amax, std::fabs(__half2float(value)));
    }
    if (pair) {
        for (const __half value : second) {
            amax = std::max(amax, std::fabs(__half2float(value)));
        }
    }
    const float expected_scale = std::max(amax / 448.0f, 1.0e-12f);
    std::uint32_t scale_bits = 0;
    std::uint32_t expected_scale_bits = 0;
    std::memcpy(&scale_bits, &scale, sizeof(scale_bits));
    std::memcpy(&expected_scale_bits, &expected_scale,
                sizeof(expected_scale_bits));
    if (scale_bits != expected_scale_bits) {
        std::fprintf(stderr, "F16 weight scale mismatch\n");
        return false;
    }

    const int pair_columns = pair ? 2 * kColumns : kColumns;
    const float inverse_scale = 1.0f / expected_scale;
    for (int destination = 0; destination < output_elements; ++destination) {
        const int row = transpose ? destination % kRows
                                  : destination / pair_columns;
        int column = transpose ? destination / kRows
                               : destination % pair_columns;
        const bool use_second = column >= kColumns;
        if (use_second) column -= kColumns;
        const std::vector<__half>& source = use_second ? second : first;
        const float value = __half2float(
            source[static_cast<std::size_t>(row * kColumns + column)]);
        const float scaled = std::max(
            -448.0f, std::min(448.0f, value * inverse_scale));
        const std::uint8_t expected = __nv_fp8_e4m3(scaled).__x;
        if (output[static_cast<std::size_t>(destination)] != expected) {
            std::fprintf(stderr,
                         "F16 weight pack mismatch pair=%d transpose=%d "
                         "index=%d\n",
                         pair, transpose, destination);
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    if (!check_exact_e4m3_conversion()) return 1;

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

    flashrt_native_quantize_fp8_weight_bf16(
        device_input, device_output, device_scale, kCount);
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
    return check_setup_scale(input) && check_dynamic_f16_scale() &&
                   check_f16_pack(false, false) &&
                   check_f16_pack(false, true) &&
                   check_f16_pack(true, false) &&
                   check_f16_pack(true, true)
               ? 0
               : 1;
}

#include "flashrt/native_cpp/operations.h"

#include "fp8_exact.cuh"

#include "common.cuh"

#include <cuda_runtime.h>

namespace {

template <typename T>
__global__ void absmax_packed_kernel(
    const T* input, float* maximum, int elements) {
    using T2 = typename packed2<T>::type;
    extern __shared__ float shared[];
    const T2* input2 = reinterpret_cast<const T2*>(input);
    const int elements2 = elements >> 1;
    float local_maximum = 0.0f;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements2;
         index += gridDim.x * blockDim.x) {
        const T2 value = input2[index];
        local_maximum = fmaxf(
            local_maximum,
            fmaxf(fabsf(to_f32(value.x)), fabsf(to_f32(value.y))));
    }
    local_maximum = warp_reduce_max(local_maximum);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = local_maximum;
    __syncthreads();
    const int warps = blockDim.x >> 5;
    local_maximum = threadIdx.x < warps ? shared[threadIdx.x] : 0.0f;
    if (warp == 0) local_maximum = warp_reduce_max(local_maximum);
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<int*>(maximum),
                  __float_as_int(local_maximum));
    }
}

template <typename T>
__global__ void absmax_scalar_kernel(
    const T* input, float* maximum, int elements) {
    extern __shared__ float shared[];
    float local_maximum = 0.0f;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements;
         index += gridDim.x * blockDim.x) {
        local_maximum = fmaxf(
            local_maximum, fabsf(to_f32(input[index])));
    }
    local_maximum = warp_reduce_max(local_maximum);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = local_maximum;
    __syncthreads();
    const int warps = blockDim.x >> 5;
    local_maximum = threadIdx.x < warps ? shared[threadIdx.x] : 0.0f;
    if (warp == 0) local_maximum = warp_reduce_max(local_maximum);
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<int*>(maximum),
                  __float_as_int(local_maximum));
    }
}

__global__ void compute_scale_kernel(
    const float* maximum, float* scale) {
    float value = __fdiv_rn(*maximum, 448.0f);
    if (value < 1e-12f) value = 1e-12f;
    *scale = value;
}

template <typename T>
__global__ void quantize_activation_kernel(
    const T* input, __nv_fp8_e4m3* output,
    const float* scale, int elements) {
    using T2 = typename packed2<T>::type;
    const T2* input2 = reinterpret_cast<const T2*>(input);
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (elements >> 1)) return;
    const float inverse_scale = 1.0f / *scale;
    const T2 value = input2[index];
    const float first = to_f32(value.x) * inverse_scale;
    const float second = to_f32(value.y) * inverse_scale;
    output[2 * index] = __nv_fp8_e4m3(
        fminf(fmaxf(first, -448.0f), 448.0f));
    output[2 * index + 1] = __nv_fp8_e4m3(
        fminf(fmaxf(second, -448.0f), 448.0f));
}

__global__ void quantize_weight_bf16_kernel(
    const __nv_bfloat16* input, __nv_fp8_e4m3* output,
    const float* scale, int elements) {
    using T2 = typename packed2<__nv_bfloat16>::type;
    const T2* input2 = reinterpret_cast<const T2*>(input);
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (elements >> 1)) return;
    const float inverse_scale = __fdiv_rn(1.0f, *scale);
    const T2 value = input2[index];
    const float first = __fmul_rn(to_f32(value.x), inverse_scale);
    const float second = __fmul_rn(to_f32(value.y), inverse_scale);
    reinterpret_cast<__nv_fp8_storage_t*>(output)[2 * index] =
        flashrt_native_fp8_e4m3_storage_rn(
            fminf(fmaxf(first, -448.0f), 448.0f));
    reinterpret_cast<__nv_fp8_storage_t*>(output)[2 * index + 1] =
        flashrt_native_fp8_e4m3_storage_rn(
            fminf(fmaxf(second, -448.0f), 448.0f));
}

__global__ void scale_bf16_weight_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    float scale, int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __float2bfloat16_rn(
            __fmul_rn(__bfloat162float(input[index]), scale));
    }
}

__global__ void quantize_weight_f16_layout_kernel(
    const __half* first, const __half* second,
    __nv_fp8_e4m3* output, const float* scale,
    int rows, int columns, bool transpose) {
    const int pair_width = second ? 2 * columns : columns;
    const int elements = rows * pair_width;
    const float inverse_scale = __fdiv_rn(1.0f, *scale);
    for (int destination = blockIdx.x * blockDim.x + threadIdx.x;
         destination < elements;
         destination += gridDim.x * blockDim.x) {
        int source_row = 0;
        int source_column = 0;
        if (transpose) {
            source_row = destination % rows;
            source_column = destination / rows;
        } else {
            source_row = destination / pair_width;
            source_column = destination % pair_width;
        }
        const bool use_second = source_column >= columns;
        if (use_second) source_column -= columns;
        const __half* source = use_second ? second : first;
        const float value = __fmul_rn(
            __half2float(source[source_row * columns + source_column]),
            inverse_scale);
        reinterpret_cast<__nv_fp8_storage_t*>(output)[destination] =
            flashrt_native_fp8_e4m3_storage_rn(
                fminf(fmaxf(value, -448.0f), 448.0f));
    }
}

int reduction_blocks(int elements) {
    int blocks = (elements + 255) / 256;
    return blocks > 1024 ? 1024 : blocks;
}

}  // namespace

void flashrt_native_quantize_fp8_device_precise(
    const __nv_bfloat16* input, __nv_fp8_e4m3* output,
    float* device_scale, int elements, cudaStream_t stream) {
    cudaMemsetAsync(device_scale, 0, sizeof(float), stream);
    const int blocks = reduction_blocks(elements);
    absmax_packed_kernel<<<blocks, 256, 256 * sizeof(float), stream>>>(
        input, device_scale, elements);
    compute_scale_kernel<<<1, 1, 0, stream>>>(device_scale, device_scale);
    const int output_blocks = ((elements >> 1) + 255) / 256;
    quantize_activation_kernel<<<output_blocks, 256, 0, stream>>>(
        input, output, device_scale, elements);
}

void flashrt_native_quantize_fp8_weight_bf16(
    const __nv_bfloat16* input, __nv_fp8_e4m3* output,
    float* device_scale, int elements, cudaStream_t stream) {
    cudaMemsetAsync(device_scale, 0, sizeof(float), stream);
    const int blocks = reduction_blocks(elements);
    absmax_packed_kernel<<<blocks, 256, 256 * sizeof(float), stream>>>(
        input, device_scale, elements);
    compute_scale_kernel<<<1, 1, 0, stream>>>(device_scale, device_scale);
    const int output_blocks = ((elements >> 1) + 255) / 256;
    quantize_weight_bf16_kernel<<<output_blocks, 256, 0, stream>>>(
        input, output, device_scale, elements);
}

void flashrt_native_scale_bf16_weight(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    float scale, int elements, cudaStream_t stream) {
    scale_bf16_weight_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
        input, output, scale, elements);
}

void flashrt_native_quantize_fp8_weight_f16(
    const __half* input, __nv_fp8_e4m3* output,
    float* device_scale, int rows, int columns, bool transpose,
    cudaStream_t stream) {
    const int elements = rows * columns;
    const int blocks = reduction_blocks(elements);
    cudaMemsetAsync(device_scale, 0, sizeof(float), stream);
    absmax_scalar_kernel<<<blocks, 256, 256 * sizeof(float), stream>>>(
        input, device_scale, elements);
    compute_scale_kernel<<<1, 1, 0, stream>>>(device_scale, device_scale);
    quantize_weight_f16_layout_kernel<<<blocks, 256, 0, stream>>>(
        input, nullptr, output, device_scale, rows, columns, transpose);
}

void flashrt_native_quantize_fp8_weight_f16_pair(
    const __half* first, const __half* second,
    __nv_fp8_e4m3* output, float* device_scale,
    int rows, int columns, bool transpose, cudaStream_t stream) {
    const int elements = rows * columns;
    const int blocks = reduction_blocks(elements);
    cudaMemsetAsync(device_scale, 0, sizeof(float), stream);
    absmax_scalar_kernel<<<blocks, 256, 256 * sizeof(float), stream>>>(
        first, device_scale, elements);
    absmax_scalar_kernel<<<blocks, 256, 256 * sizeof(float), stream>>>(
        second, device_scale, elements);
    compute_scale_kernel<<<1, 1, 0, stream>>>(device_scale, device_scale);
    int output_blocks = (2 * elements + 255) / 256;
    if (output_blocks > 4096) output_blocks = 4096;
    quantize_weight_f16_layout_kernel<<<output_blocks, 256, 0, stream>>>(
        first, second, output, device_scale, rows, columns, transpose);
}

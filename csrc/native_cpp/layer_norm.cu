#include "flashrt/native_cpp/operations.h"

#include "fp8_exact.cuh"

namespace {

__global__ void layer_norm_fp8_f16_kernel(
    const __half* input, __nv_fp8_e4m3* output,
    const __half* gamma, const __half* beta,
    int rows, int columns, float epsilon) {
    const int row_index = blockIdx.x;
    if (row_index >= rows) return;
    const __half* row = input + row_index * columns;
    __nv_fp8_e4m3* output_row = output + row_index * columns;

    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns;
         column += blockDim.x) {
        sum += __half2float(row[column]);
    }
    __shared__ float shared[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    if (!lane) shared[warp] = sum;
    __syncthreads();
    if (!warp) {
        sum = lane < (blockDim.x + 31) / 32 ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
    }
    __syncthreads();
    if (!threadIdx.x) shared[0] = sum;
    __syncthreads();
    const float mean = shared[0] / columns;

    float variance = 0.0f;
    for (int column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const float value = __half2float(row[column]) - mean;
        variance += value * value;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        variance += __shfl_xor_sync(0xffffffff, variance, offset);
    }
    if (!lane) shared[warp] = variance;
    __syncthreads();
    if (!warp) {
        variance = lane < (blockDim.x + 31) / 32 ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            variance += __shfl_xor_sync(0xffffffff, variance, offset);
        }
    }
    __syncthreads();
    if (!threadIdx.x) shared[0] = variance;
    __syncthreads();
    const float inverse_stddev = rsqrtf(shared[0] / columns + epsilon);

    for (int column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const float normalized =
            (__half2float(row[column]) - mean) * inverse_stddev;
        const float value =
            normalized * __half2float(gamma[column]) +
            __half2float(beta[column]);
        output_row[column] = flashrt_native_fp8_e4m3_rn(value);
    }
}

}  // namespace

void flashrt_native_layer_norm_fp8_f16(
    const __half* input, __nv_fp8_e4m3* output,
    const __half* gamma, const __half* beta,
    int rows, int columns, float epsilon, cudaStream_t stream) {
    layer_norm_fp8_f16_kernel<<<rows, 256, 0, stream>>>(
        input, output, gamma, beta, rows, columns, epsilon);
}

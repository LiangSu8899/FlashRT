#include "flashrt/native_cpp/operations.h"

#include <cuda_runtime.h>

namespace {

__global__ void silu_bf16_kernel(__nv_bfloat16* values, int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float value = __bfloat162float(values[index]);
    values[index] = __float2bfloat16(
        value / (1.0f + expf(-value)));
}

__global__ void fill_negative_infinity_f32_kernel(
    float* values, std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = __int_as_float(0xff800000);
}

__global__ void generate_rope_table_f16_kernel(
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

}  // namespace

void flashrt_native_silu_inplace_bf16(
    __nv_bfloat16* values, int elements, cudaStream_t stream) {
    if (elements <= 0) return;
    silu_bf16_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
        values, elements);
}

void flashrt_native_fill_negative_infinity_f32(
    float* values, std::size_t count, cudaStream_t stream) {
    if (!count) return;
    fill_negative_infinity_f32_kernel<<<(count + 255) / 256, 256, 0,
                                        stream>>>(values, count);
}

void flashrt_native_generate_rope_table_f16(
    __half* output, int start_position, int positions,
    int frequencies, float theta, cudaStream_t stream) {
    const int elements = positions * frequencies;
    generate_rope_table_f16_kernel<<<(elements + 255) / 256, 256, 0,
                                     stream>>>(
        output, start_position, positions, frequencies, theta);
}

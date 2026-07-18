#include "rope.cuh"

// Generated tables are calibration inputs, so this translation unit must stay
// outside targets that enable approximate device math.
__global__ void generate_rope_table_f16_kernel(
    __half* output, int start_position, int positions,
    int frequencies, float theta) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= positions * frequencies) return;
    int row = index / frequencies;
    int frequency = index - row * frequencies;
    float exponent = static_cast<float>(frequency) / frequencies;
    float denominator = powf(theta, exponent);
    float inverse_frequency = __fdiv_rn(1.0f, denominator);
    inverse_frequency = __bfloat162float(
        __float2bfloat16_rn(inverse_frequency));
    float phase = __fmul_rn(
        static_cast<float>(start_position + row), inverse_frequency);
    output[2 * index] = __float2half_rn(cosf(phase));
    output[2 * index + 1] = __float2half_rn(sinf(phase));
}

void generate_rope_table_f16(__half* output, int start_position,
                             int positions, int frequencies, float theta,
                             cudaStream_t stream) {
    int elements = positions * frequencies;
    generate_rope_table_f16_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
        output, start_position, positions, frequencies, theta);
}

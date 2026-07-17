#include "flashrt/cpp/models/pi05/support/native_rope.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <limits>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

constexpr int kFrequencies = 128;

__global__ void generate_rope_kernel(__half* output, int start_position,
                                     int positions) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= positions * kFrequencies) return;
    const int row = index / kFrequencies;
    const int frequency = index - row * kFrequencies;
    const float exponent = static_cast<float>(frequency) / kFrequencies;
    const float denominator = powf(10000.0f, exponent);
    float inverse_frequency = __fdiv_rn(1.0f, denominator);
    inverse_frequency = __bfloat162float(
        __float2bfloat16_rn(inverse_frequency));
    const float phase = __fmul_rn(
        static_cast<float>(start_position + row), inverse_frequency);
    output[2 * index] = __float2half_rn(cosf(phase));
    output[2 * index + 1] = __float2half_rn(sinf(phase));
}

}  // namespace

modalities::Status generate_native_thor_rope_f16(
    void* output, int start_position, int positions, std::uintptr_t stream) {
    constexpr int kMax = std::numeric_limits<int>::max();
    if (!output || start_position < 0 || positions <= 0 ||
        positions > kMax / kFrequencies ||
        start_position > kMax - (positions - 1)) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "Thor RoPE generation arguments are invalid");
    }
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int elements = positions * kFrequencies;
    generate_rope_kernel<<<(elements + 255) / 256, 256, 0, cuda_stream>>>(
        static_cast<__half*>(output), start_position, positions);
    cudaError_t rc = cudaGetLastError();
    if (rc == cudaSuccess) rc = cudaStreamSynchronize(cuda_stream);
    return rc == cudaSuccess
               ? modalities::Status::ok()
               : modalities::Status::error(
                     modalities::StatusCode::kBackend,
                     std::string("Thor RoPE generation failed: ") +
                         cudaGetErrorString(rc));
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace {

__global__ void precise_silu_fp16_kernel(__half* values,
                                         std::size_t elements) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float value = __half2float(values[index]);
    values[index] = __float2half_rn(
        value / (1.0f + expf(-value)));
}

}  // namespace

extern "C" cudaError_t flashrt_silu_inplace_fp16_precise(
    __half* values, std::size_t elements, cudaStream_t stream) {
    if (!values || !elements) return cudaErrorInvalidValue;
    constexpr unsigned int kBlock = 256;
    precise_silu_fp16_kernel<<<
        static_cast<unsigned int>((elements + kBlock - 1) / kBlock),
        kBlock, 0, stream>>>(values, elements);
    return cudaGetLastError();
}

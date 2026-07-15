// SPDX-License-Identifier: Apache-2.0
// Channels-last (NDHWC) RMSNorm + RMS_SiLU kernels for Wan VAE.

#include "fp16_rms_norm_ndhwc.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {
namespace {

// One warp per spatial position.  4 warps per block (128 threads).
constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * 32;

template <bool kApplySilu>
__global__ void fp16_rms_norm_cl_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ y,
    int B, int C, int T, int H, int W,
    int total_spatial,   // B * T * H * W
    float eps)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    const int spatial_idx = blockIdx.x * kWarpsPerBlock + warp_id;
    if (spatial_idx >= total_spatial) return;

    // In NDHWC, C values for each spatial position are contiguous.
    const long long base = (long long)spatial_idx * C;
    const __half* x_row = x + base;
    __half* y_row = y + base;

    // Phase 1: read C values, compute sum-of-squares.
    float sum_sq = 0.0f;
    for (int c = lane; c < C; c += 32) {
        float v = __half2float(x_row[c]);
        sum_sq = fmaf(v, v, sum_sq);
    }

    // Warp-level reduction (no shared memory needed).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);

    const float inv_rms = rsqrtf(sum_sq * (1.0f / static_cast<float>(C)) + eps);

    // Phase 2: normalise, apply gamma+bias, optional silu, write.
    for (int c = lane; c < C; c += 32) {
        float v = __half2float(x_row[c]) * inv_rms
                * __half2float(gamma[c]);
        if (bias != nullptr)
            v += __half2float(bias[c]);
        if constexpr (kApplySilu) {
            // silu(v) = v / (1 + exp(-v))
            v = v * (1.0f / (1.0f + __expf(-v)));
        }
        y_row[c] = __float2half(v);
    }
}

}  // namespace

int fp16_rms_norm_ndhwc(
    const void* x_fp16, const void* gamma_fp16, const void* bias_fp16,
    void* y_fp16,
    int B, int C, int T, int H, int W,
    float eps, cudaStream_t stream)
{
    if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
    const long long total_spatial = (long long)B * T * H * W;
    if (total_spatial <= 0 || total_spatial > (long long)INT32_MAX) return -4;

    const unsigned n_blocks = (unsigned)((total_spatial + kWarpsPerBlock - 1)
                                         / kWarpsPerBlock);
    fp16_rms_norm_cl_kernel<false><<<n_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        reinterpret_cast<const __half*>(gamma_fp16),
        bias_fp16 ? reinterpret_cast<const __half*>(bias_fp16) : nullptr,
        reinterpret_cast<__half*>(y_fp16),
        B, C, T, H, W, (int)total_spatial, eps);
    return static_cast<int>(cudaGetLastError());
}

int fp16_rms_silu_ndhwc(
    const void* x_fp16, const void* gamma_fp16, const void* bias_fp16,
    void* y_fp16,
    int B, int C, int T, int H, int W,
    float eps, cudaStream_t stream)
{
    if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
    const long long total_spatial = (long long)B * T * H * W;
    if (total_spatial <= 0 || total_spatial > (long long)INT32_MAX) return -4;

    const unsigned n_blocks = (unsigned)((total_spatial + kWarpsPerBlock - 1)
                                         / kWarpsPerBlock);
    fp16_rms_norm_cl_kernel<true><<<n_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        reinterpret_cast<const __half*>(gamma_fp16),
        bias_fp16 ? reinterpret_cast<const __half*>(bias_fp16) : nullptr,
        reinterpret_cast<__half*>(y_fp16),
        B, C, T, H, W, (int)total_spatial, eps);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt

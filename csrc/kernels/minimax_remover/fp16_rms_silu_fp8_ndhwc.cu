// ================================================================
// flash_rt — MiniMax-Remover fused RMSNorm+SiLU with amax / FP8
// quantize (channels-last NDHWC).
//
// Built on the same warp-per-spatial-position pattern as
// fp16_rms_norm_ndhwc, with three entry points:
//
//   1. fp16_rms_silu_amax_ndhwc       — norm+silu+amax → fp16 + amax
//   2. fp16_rms_silu_quant_fp8_ndhwc  — norm+silu+quant → fp8 (pre-comp amax)
//   3. fp16_rms_silu_amax_quant_fp8_ndhwc — 2-pass: → fp8 + scale
//
// Template kernel controls:
//   kWriteFp16 : write the fp16 normed+silu output
//   kComputeAmax : accumulate |output| via atomicMax
//   kQuantizeFp8 : read device amax and write fp8 output
// ================================================================

#include "fp16_rms_silu_fp8_ndhwc.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * 32;

__device__ __forceinline__ float silu_f32(float x) {
    return x * (1.0f / (1.0f + __expf(-x)));
}

//atomicMax on reinterpreted int preserves ordering for non-negative floats.
__device__ __forceinline__ void atomic_max_f32(float* addr, float val) {
    if (val > 0.0f) {
        atomicMax(reinterpret_cast<int*>(addr), __float_as_int(val));
    }
}

template <bool kWriteFp16, bool kComputeAmax, bool kQuantizeFp8>
__global__ void fused_rms_silu_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ y_fp16,
    __nv_fp8_e4m3* __restrict__ y_fp8,
    float* __restrict__ amax_buf,
    const float* __restrict__ amax_in,
    float* __restrict__ scale_out,
    int B, int C, int T, int H, int W,
    int total_spatial,
    float eps)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    const int spatial_idx = blockIdx.x * kWarpsPerBlock + warp_id;
    if (spatial_idx >= total_spatial) return;

    const long long base = (long long)spatial_idx * C;
    const __half* x_row = x + base;

    // ── Phase 1: read x, compute sum-of-squares ──────────────────
    float sum_sq = 0.0f;
    for (int c = lane; c < C; c += 32) {
        float v = __half2float(x_row[c]);
        sum_sq = fmaf(v, v, sum_sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);

    const float inv_rms =
        rsqrtf(sum_sq * (1.0f / static_cast<float>(C)) + eps);

    // ── Optional: broadcast amax → inv_scale for FP8 quantize ────
    float inv_scale = 0.0f;
    if constexpr (kQuantizeFp8) {
        __shared__ float s_inv_scale;
        if (threadIdx.x == 0) {
            float amax = fmaxf(*amax_in, 0.0f);
            float scale = amax * (1.0f / 448.0f);
            if (scale < 1e-6f) scale = 1e-6f;
            if (scale_out) *scale_out = scale;
            s_inv_scale = 1.0f / scale;
        }
        __syncthreads();
        inv_scale = s_inv_scale;
    }

    // ── Phase 2: normalise, apply gamma+bias, silu, write ────────
    float local_amax = 0.0f;
    for (int c = lane; c < C; c += 32) {
        float v = __half2float(x_row[c]) * inv_rms
                * __half2float(gamma[c]);
        if (bias != nullptr)
            v += __half2float(bias[c]);
        v = silu_f32(v);

        if constexpr (kWriteFp16) {
            y_fp16[base + c] = __float2half_rn(v);
        }
        if constexpr (kComputeAmax) {
            local_amax = fmaxf(local_amax, fabsf(v));
        }
        if constexpr (kQuantizeFp8) {
            float q = v * inv_scale;
            q = fminf(fmaxf(q, -448.0f), 448.0f);
            y_fp8[base + c] = __nv_fp8_e4m3(q);
        }
    }

    // ── Warp-level amax reduction + atomicMax ────────────────────
    if constexpr (kComputeAmax) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            local_amax = fmaxf(local_amax, __shfl_xor_sync(0xFFFFFFFF, local_amax, off));
        // lane 0 has the warp-wide amax
        if (lane == 0 && local_amax > 0.0f) {
            atomic_max_f32(amax_buf, local_amax);
        }
    }
}

}  // namespace

// ── (1) Fused norm+silu+amax → fp16 + amax ──────────────────────
int fp16_rms_silu_amax_ndhwc(
    const void* x_fp16, const void* gamma_fp16, const void* bias_fp16,
    void* y_fp16, void* amax_buf,
    int B, int C, int T, int H, int W,
    float eps, cudaStream_t stream)
{
    if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
    const long long total_spatial = (long long)B * T * H * W;
    if (total_spatial <= 0 || total_spatial > (long long)INT32_MAX) return -4;

    const unsigned n_blocks = (unsigned)((total_spatial + kWarpsPerBlock - 1)
                                         / kWarpsPerBlock);
    fused_rms_silu_kernel<true, true, false><<<n_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        reinterpret_cast<const __half*>(gamma_fp16),
        bias_fp16 ? reinterpret_cast<const __half*>(bias_fp16) : nullptr,
        reinterpret_cast<__half*>(y_fp16),
        nullptr,  // y_fp8
        reinterpret_cast<float*>(amax_buf),
        nullptr, nullptr,  // amax_in, scale_out
        B, C, T, H, W, (int)total_spatial, eps);
    return static_cast<int>(cudaGetLastError());
}

// ── (2) Fused norm+silu+quant → fp8 (pre-computed amax) ─────────
int fp16_rms_silu_quant_fp8_ndhwc(
    const void* x_fp16, const void* gamma_fp16, const void* bias_fp16,
    void* y_fp8, const void* amax_buf, void* scale_out,
    int B, int C, int T, int H, int W,
    float eps, cudaStream_t stream)
{
    if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
    const long long total_spatial = (long long)B * T * H * W;
    if (total_spatial <= 0 || total_spatial > (long long)INT32_MAX) return -4;

    const unsigned n_blocks = (unsigned)((total_spatial + kWarpsPerBlock - 1)
                                         / kWarpsPerBlock);
    fused_rms_silu_kernel<false, false, true><<<n_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        reinterpret_cast<const __half*>(gamma_fp16),
        bias_fp16 ? reinterpret_cast<const __half*>(bias_fp16) : nullptr,
        nullptr,  // y_fp16
        reinterpret_cast<__nv_fp8_e4m3*>(y_fp8),
        nullptr,  // amax_buf (not computing)
        reinterpret_cast<const float*>(amax_buf),
        reinterpret_cast<float*>(scale_out),
        B, C, T, H, W, (int)total_spatial, eps);
    return static_cast<int>(cudaGetLastError());
}

// ── (3) 2-pass: norm+silu+amax → fp8+scale (no fp16 write) ──────
int fp16_rms_silu_amax_quant_fp8_ndhwc(
    const void* x_fp16, const void* gamma_fp16, const void* bias_fp16,
    void* y_fp8, void* scale_out, void* amax_buf,
    int B, int C, int T, int H, int W,
    float eps, cudaStream_t stream)
{
    if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
    const long long total_spatial = (long long)B * T * H * W;
    if (total_spatial <= 0 || total_spatial > (long long)INT32_MAX) return -4;

    cudaError_t e;
    e = cudaMemsetAsync(amax_buf, 0, sizeof(float), stream);
    if (e != cudaSuccess) return -2;

    const unsigned n_blocks = (unsigned)((total_spatial + kWarpsPerBlock - 1)
                                         / kWarpsPerBlock);

    // Pass 1: norm+silu+amax (no write).
    fused_rms_silu_kernel<false, true, false><<<n_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        reinterpret_cast<const __half*>(gamma_fp16),
        bias_fp16 ? reinterpret_cast<const __half*>(bias_fp16) : nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<float*>(amax_buf),
        nullptr, nullptr,
        B, C, T, H, W, (int)total_spatial, eps);

    // Pass 2: norm+silu+quant (reads amax from pass 1).
    fused_rms_silu_kernel<false, false, true><<<n_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        reinterpret_cast<const __half*>(gamma_fp16),
        bias_fp16 ? reinterpret_cast<const __half*>(bias_fp16) : nullptr,
        nullptr,
        reinterpret_cast<__nv_fp8_e4m3*>(y_fp8),
        nullptr,
        reinterpret_cast<const float*>(amax_buf),
        reinterpret_cast<float*>(scale_out),
        B, C, T, H, W, (int)total_spatial, eps);

    e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -3;
}

int fp16_rms_silu_amax_quant_fp8_ndhwc_nozero(
    const void* x_fp16, const void* gamma_fp16, const void* bias_fp16,
    void* y_fp8, void* scale_out, void* amax_buf,
    int B, int C, int T, int H, int W,
    float eps, cudaStream_t stream)
{
    if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
    const long long total_spatial = (long long)B * T * H * W;
    if (total_spatial <= 0 || total_spatial > (long long)INT32_MAX) return -4;

    const unsigned n_blocks = (unsigned)((total_spatial + kWarpsPerBlock - 1)
                                         / kWarpsPerBlock);

    fused_rms_silu_kernel<false, true, false><<<n_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        reinterpret_cast<const __half*>(gamma_fp16),
        bias_fp16 ? reinterpret_cast<const __half*>(bias_fp16) : nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<float*>(amax_buf),
        nullptr, nullptr,
        B, C, T, H, W, (int)total_spatial, eps);

    fused_rms_silu_kernel<false, false, true><<<n_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        reinterpret_cast<const __half*>(gamma_fp16),
        bias_fp16 ? reinterpret_cast<const __half*>(bias_fp16) : nullptr,
        nullptr,
        reinterpret_cast<__nv_fp8_e4m3*>(y_fp8),
        nullptr,
        reinterpret_cast<const float*>(amax_buf),
        reinterpret_cast<float*>(scale_out),
        B, C, T, H, W, (int)total_spatial, eps);

    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -3;
}

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt

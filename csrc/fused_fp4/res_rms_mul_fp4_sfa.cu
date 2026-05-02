// ============================================================================
//  F3 + per-channel-mul fused: residual_add + rms_norm + inv_s multiply + fp4
//  quant + SFA write, all in one kernel launch.
//
//  Semantics: identical to the 3-kernel sequence
//      residual_add_rms_norm_noweight_fp16(x, fg, x_normed)
//      per_channel_mul_fp16(x_normed, inv_s)
//      quantize_fp4_dynamic_sfa_fp16(x_normed, packed, sfa)
//  ...combined into a single launch. Used by the AWQ frontend path.
//
//  Layout: strict copy of F3 v1 (norm_silu_fp4_sfa.cu), extended with a
//  per-channel inv_s multiply applied to the normalized value BEFORE it is
//  written to shared memory for stage-4 per-block quant. Bit-exact with the
//  3-kernel reference path under fp16 intermediate rounding.
//
//  Additive: does NOT modify existing F3 (norm_silu_fp4_sfa.cu).
// ============================================================================
#include "fused_fp4/norm_silu_fp4_sfa.cuh"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(__CUDA_ARCH__)
#  include "cutlass/cutlass.h"
#  include "cutlass/detail/sm100_blockscaled_layout.hpp"
#  include "cute/tensor.hpp"
#  define FV_HAVE_CUTLASS 1
#else
#  define FV_HAVE_CUTLASS 0
#endif

namespace flash_rt {
namespace fused_fp4 {

#if FV_HAVE_CUTLASS

using CfgF3M = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ uint8_t fp32_to_e2m1_f3m(float x) {
    uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
    float ax = fabsf(x);
    uint8_t mant;
    if      (ax <= 0.25f) mant = 0u;
    else if (ax <= 0.75f) mant = 1u;
    else if (ax <= 1.25f) mant = 2u;
    else if (ax <= 1.75f) mant = 3u;
    else if (ax <= 2.5f)  mant = 4u;
    else if (ax <= 3.5f)  mant = 5u;
    else if (ax <= 5.0f)  mant = 6u;
    else                  mant = 7u;
    return sign | mant;
}

template <class LayoutSF>
__device__ __forceinline__ void quant_block_from_smem_f3m(
    const __half* smem_normed,
    uint8_t* packed_row,
    uint8_t* dst_sfa,
    LayoutSF layout,
    int row, int block_idx) {
    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = __half2float(smem_normed[block_idx * 16 + i]);
        float a = fabsf(vals[i]);
        if (a > amax) amax = a;
    }
    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    float bs_dq = static_cast<float>(bs_q);
    int sfa_off = layout(row, block_idx * 16, 0);
    dst_sfa[sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);

    const float inv_bs = 1.f / bs_dq;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t lo = fp32_to_e2m1_f3m(vals[2 * p    ] * inv_bs);
        uint8_t hi = fp32_to_e2m1_f3m(vals[2 * p + 1] * inv_bs);
        packed_row[block_idx * 8 + p] = lo | (hi << 4);
    }
}

template <class LayoutSF>
__global__ void f3m_res_rms_mul_fp4_sfa_kernel(
    __half* __restrict__ residual,
    const __half* __restrict__ x,
    const __half* __restrict__ inv_s,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int D) {
    const int r = blockIdx.x;
    __half* res_row = residual + r * D;
    const __half* x_row = x + r * D;
    uint8_t* packed_row = packed + r * (D / 2);
    const int D2 = D / 2;

    __half2* res2w = reinterpret_cast<__half2*>(res_row);
    const __half2* res2 = reinterpret_cast<const __half2*>(res_row);
    const __half2* x2 = reinterpret_cast<const __half2*>(x_row);
    const __half2* inv_s2 = reinterpret_cast<const __half2*>(inv_s);

    extern __shared__ __half smem_normed[];

    constexpr int ELEMS_PER_THREAD = 8;
    float cache[ELEMS_PER_THREAD];
    float ssq = 0;
    int c2_base[ELEMS_PER_THREAD / 2];

    // Stage 1: load + residual add + ssq
    #pragma unroll
    for (int it = 0; it < ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        c2_base[it] = c2;
        if (c2 < D2) {
            __half2 rv2 = res2[c2], xv2 = x2[c2];
            float a = __half2float(rv2.x) + __half2float(xv2.x);
            float b = __half2float(rv2.y) + __half2float(xv2.y);
            cache[it*2]   = a;
            cache[it*2+1] = b;
            res2w[c2] = __halves2half2(__float2half(a), __float2half(b));
            ssq += a*a + b*b;
        } else {
            cache[it*2] = 0; cache[it*2+1] = 0;
        }
    }

    // Stage 2: block-wide ssq reduce
    __shared__ float sh[16];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    if (!lane) sh[wid] = ssq; __syncthreads();
    if (!wid) { ssq = (lane < (blockDim.x/32)) ? sh[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o); }
    __syncthreads(); if (!threadIdx.x) sh[0] = ssq; __syncthreads();

    float rms = __frsqrt_rn(sh[0] / D + 1e-6f);

    // Stage 3: normalize * inv_s (per-channel), write fp16 to shared mem.
    // AWQ: pre-multiply by inv_s[c] BEFORE FP4 quant so per-block amax
    // reflects the post-AWQ activation distribution.
    #pragma unroll
    for (int it = 0; it < ELEMS_PER_THREAD / 2; it++) {
        int c2 = c2_base[it];
        if (c2 < D2) {
            __half2 is2 = inv_s2[c2];
            float isx = __half2float(is2.x);
            float isy = __half2float(is2.y);
            float v0 = (cache[it*2]   * rms) * isx;
            float v1 = (cache[it*2+1] * rms) * isy;
            __half2 h2 = __halves2half2(__float2half(v0), __float2half(v1));
            reinterpret_cast<__half2*>(smem_normed)[c2] = h2;
        }
    }
    __syncthreads();

    // Stage 4: per-16-block quant + SFA write
    const int n_blocks = D / 16;
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        quant_block_from_smem_f3m(smem_normed, packed_row, dst_sfa, layout, r, b);
    }
}

#endif

void residual_add_rms_norm_mul_fp4_sfa_fp16(
    __half* residual, const __half* x, const __half* inv_s,
    uint8_t* packed, uint8_t* sfa,
    int seq_len, int dim, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = CfgF3M::tile_atom_to_shape_SFA(shape);
    const size_t smem_bytes = dim * sizeof(__half);
    f3m_res_rms_mul_fp4_sfa_kernel<<<seq_len, 256, smem_bytes, stream>>>(
        residual, x, inv_s, packed, sfa, layout, dim);
#else
    (void)residual; (void)x; (void)inv_s; (void)packed; (void)sfa;
    (void)seq_len; (void)dim; (void)stream;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt

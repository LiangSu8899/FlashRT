// ============================================================================
//  F4 v2 + per-channel-mul fused: gate_silu_mul + inv_s multiply + fp4 quant
//  + SFA write, single-launch register-only design.
//
//  Semantics: identical to the 3-kernel sequence
//      gate_silu_mul_merged_fp16(merged, hid_fp16)
//      per_channel_mul_fp16(hid_fp16, inv_s)
//      quantize_fp4_dynamic_sfa_fp16(hid_fp16, packed, sfa)
//  ...combined. Used by the AWQ frontend path for the Down-GEMM pre-step.
//
//  Layout: strict copy of F4 v2 (silu_mul_fp4_sfa_v2.cu), with the inv_s[col]
//  multiply applied inline after silu_mul and before amax/quant.
//
//  Additive: does NOT modify existing F4 v2 kernel.
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

using CfgF4M = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ uint8_t fp32_to_e2m1_f4m(float x) {
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

__device__ __forceinline__ float silu_gelu_mul_f4m(float g, float u) {
    // Same formula as gate_silu_mul_merged_kernel (tanh-approx GELU constant).
    float gelu = g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
    return gelu * u;
}

template <class LayoutSF>
__global__ void f4m_silu_mul_mul_fp4_sfa_kernel(
    const __half* __restrict__ merged,   // [S, 2H]
    const __half* __restrict__ inv_s,    // [H]
    uint8_t* __restrict__ packed,        // [S, H/2]
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int H) {
    // One thread → one NVFP4 block (16 output elements).
    const int block_idx = blockIdx.y * blockDim.x + threadIdx.x;
    const int row       = blockIdx.x;
    const int n_blocks  = H / 16;
    if (block_idx >= n_blocks) return;

    const int col_base = block_idx * 16;
    const __half* merged_row = merged + row * 2 * H;
    const __half* gate_ptr = merged_row + col_base;
    const __half* up_ptr   = merged_row + H + col_base;
    const __half* inv_s_blk = inv_s + col_base;

    const __half2* gate2 = reinterpret_cast<const __half2*>(gate_ptr);
    const __half2* up2   = reinterpret_cast<const __half2*>(up_ptr);
    const __half2* isf2  = reinterpret_cast<const __half2*>(inv_s_blk);

    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        __half2 g2 = gate2[i];
        __half2 u2 = up2[i];
        __half2 s2 = isf2[i];
        float v0 = silu_gelu_mul_f4m(__half2float(g2.x), __half2float(u2.x)) * __half2float(s2.x);
        float v1 = silu_gelu_mul_f4m(__half2float(g2.y), __half2float(u2.y)) * __half2float(s2.y);
        vals[2*i]   = v0;
        vals[2*i+1] = v1;
        float a0 = fabsf(v0), a1 = fabsf(v1);
        if (a0 > amax) amax = a0;
        if (a1 > amax) amax = a1;
    }

    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    float bs_dq = static_cast<float>(bs_q);
    int sfa_off = layout(row, col_base, 0);
    dst_sfa[sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);

    uint8_t* packed_row = packed + row * (H / 2);
    const float inv_bs = 1.f / bs_dq;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t lo = fp32_to_e2m1_f4m(vals[2 * p    ] * inv_bs);
        uint8_t hi = fp32_to_e2m1_f4m(vals[2 * p + 1] * inv_bs);
        packed_row[block_idx * 8 + p] = lo | (hi << 4);
    }
}

#endif

void gate_silu_mul_mul_fp4_sfa_v2_fp16(
    const __half* merged, const __half* inv_s,
    uint8_t* packed, uint8_t* sfa,
    int seq_len, int half_dim, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    auto shape = cute::make_shape(seq_len, 1, half_dim, 1);
    auto layout = CfgF4M::tile_atom_to_shape_SFA(shape);
    const int n_blocks = half_dim / 16;
    const int threads = 256;
    const int y_groups = (n_blocks + threads - 1) / threads;
    dim3 grid(seq_len, y_groups);
    dim3 block(threads);
    f4m_silu_mul_mul_fp4_sfa_kernel<<<grid, block, 0, stream>>>(
        merged, inv_s, packed, sfa, layout, half_dim);
#else
    (void)merged; (void)inv_s; (void)packed; (void)sfa;
    (void)seq_len; (void)half_dim; (void)stream;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt

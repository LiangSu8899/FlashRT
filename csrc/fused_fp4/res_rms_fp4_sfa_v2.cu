// ============================================================================
//  F3 v2 — residual_add + rms_norm + fp4_quant + SFA, no-smem 2-pass design.
//
//  v1 design (norm_silu_fp4_sfa.cu): 1-block-per-row, 256 threads, smem-staged
//  D=2048 fp16 between RMS and per-16-block quant. Extra smem roundtrip.
//
//  v2 design: same 1-block-per-row, but stage 2 doesn't read from smem. Each
//  thread owns 8 consecutive fp16 elements in registers after stage 1 (from
//  coalesced strided loads). We reshuffle via warp shuffles so 2 consecutive
//  threads collaborate on one NVFP4 block (16 elements). No smem.
//
//  Actually the CLEANEST design at D=2048: blockDim = 128 threads, each
//  thread owns 16 consecutive fp16 elements = one NVFP4 block. Stage 1 does
//  the load+add+ssq reduction; stage 2 computes rms_scale, normalizes +
//  quantizes + writes the thread's 16 elements independently. Coalesced load
//  via half2 strided by 128.
//
//  D=2048 / 16 = 128 NVFP4 blocks per row. blockDim=128. Perfect 1-1 mapping.
// ============================================================================
#include "fused_fp4/norm_silu_fp4_sfa.cuh"

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

using CfgF3V2 = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ uint8_t fp32_to_e2m1_f3v2(float x) {
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

// F3 v2 requires D = 16 * blockDim.x exactly. We hardcode blockDim=128 for
// Pi0.5's D=2048. If other D values are needed, add variants.
// Each thread owns 16 consecutive fp16 elements of one row, which is exactly
// one NVFP4 block.
template <class LayoutSF>
__global__ void f3v2_res_rms_fp4_sfa_kernel(
    __half* __restrict__ residual,
    const __half* __restrict__ x,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int D) {
    const int r = blockIdx.x;
    __half* res_row = residual + r * D;
    const __half* x_row = x + r * D;
    uint8_t* packed_row = packed + r * (D / 2);

    // Each thread loads its 16 consecutive fp16 via 8 half2.
    const int col_base = threadIdx.x * 16;  // 0, 16, 32, ..., (blockDim-1)*16
    if (col_base >= D) return;

    __half2* res2 = reinterpret_cast<__half2*>(res_row + col_base);
    const __half2* x2 = reinterpret_cast<const __half2*>(x_row + col_base);

    // Stage 1: load+add+store residual+ssq (16 elements per thread).
    float vals[16];
    float local_ssq = 0.f;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        __half2 rv = res2[p];
        __half2 xv = x2[p];
        float a = __half2float(rv.x) + __half2float(xv.x);
        float b = __half2float(rv.y) + __half2float(xv.y);
        vals[2*p]   = a;
        vals[2*p+1] = b;
        // Update residual in place (fp16).
        res2[p] = __halves2half2(__float2half(a), __float2half(b));
        local_ssq += a*a + b*b;
    }

    // Block-wide ssq reduction (same pattern as existing RMS kernels).
    __shared__ float sh[16];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) local_ssq += __shfl_xor_sync(0xffffffff, local_ssq, o);
    if (!lane) sh[wid] = local_ssq;
    __syncthreads();
    float ssq;
    if (!wid) {
        ssq = (lane < (blockDim.x / 32)) ? sh[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    }
    __syncthreads();
    if (!threadIdx.x) sh[0] = ssq;
    __syncthreads();

    float rms = __frsqrt_rn(sh[0] / D + 1e-6f);

    // Stage 2: normalize + per-block amax + UE4M3 quant + e2m1 pack + SFA write.
    // All in registers, no smem roundtrip.
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] *= rms;
        float a = fabsf(vals[i]);
        if (a > amax) amax = a;
    }

    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    float bs_dq = static_cast<float>(bs_q);

    int block_idx = threadIdx.x;
    int sfa_off = layout(r, block_idx * 16, 0);
    dst_sfa[sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);

    const float inv_bs = 1.f / bs_dq;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t lo = fp32_to_e2m1_f3v2(vals[2 * p    ] * inv_bs);
        uint8_t hi = fp32_to_e2m1_f3v2(vals[2 * p + 1] * inv_bs);
        packed_row[block_idx * 8 + p] = lo | (hi << 4);
    }
}

#endif

void residual_add_rms_norm_fp4_sfa_v2_fp16(
    __half* residual, const __half* x,
    uint8_t* packed, uint8_t* sfa,
    int seq_len, int dim, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    // Require D divisible by 16 and block fits D/16 threads.
    // For D=2048 → 128 threads. For D=1024 → 64. Capped at 512.
    int threads = dim / 16;
    if (threads <= 0 || threads > 1024) return;

    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = CfgF3V2::tile_atom_to_shape_SFA(shape);
    f3v2_res_rms_fp4_sfa_kernel<<<seq_len, threads, 0, stream>>>(
        residual, x, packed, sfa, layout, dim);
#else
    (void)residual; (void)x; (void)packed; (void)sfa;
    (void)seq_len; (void)dim; (void)stream;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt

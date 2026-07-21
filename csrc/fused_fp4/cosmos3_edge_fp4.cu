// ============================================================================
//  Cosmos3-Edge model-specific NVFP4 fused quant kernels (additive).
//
//  1. cosmos3_edge_res_rms_fp4_sfa_bf16:
//     residual(bf16, updated in place) += x(bf16); RMSNorm with gamma weight
//     (bf16, eps parameterized); NVFP4 quantize + CUTLASS tile-interleaved SFA.
//     Register-only F3-v2 design: 1 thread per NVFP4 block, D = 16*blockDim.
//
//  2. cosmos3_edge_relu2_fp4_sfa_fp16:
//     x(fp16, up-proj GEMM output) -> relu(x)^2 -> NVFP4 quantize + SFA.
//     1 thread per NVFP4 block, F4-v2 design.
//
//  Both feed cutlass_fp4_sq_fp16 (A-side packed + SFA).
// ============================================================================
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

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

using CfgEdge = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ uint8_t fp32_to_e2m1_edge(float x) {
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

__device__ __forceinline__ void quant_pack_sfa_edge(
    float (&vals)[16], float amax, int row, int col_base, int block_idx,
    uint8_t* __restrict__ packed_row, uint8_t* __restrict__ dst_sfa, int sfa_off) {
    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    const float inv_bs = 1.f / static_cast<float>(bs_q);
    dst_sfa[sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t lo = fp32_to_e2m1_edge(vals[2 * p    ] * inv_bs);
        uint8_t hi = fp32_to_e2m1_edge(vals[2 * p + 1] * inv_bs);
        packed_row[block_idx * 8 + p] = lo | (hi << 4);
    }
}

template <class LayoutSF>
__global__ void edge_res_rms_fp4_sfa_bf16_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int D,
    float eps) {
    const int r = blockIdx.x;
    __nv_bfloat16* res_row = residual + r * D;
    const __nv_bfloat16* x_row = x + r * D;
    uint8_t* packed_row = packed + r * (D / 2);

    const int col_base = threadIdx.x * 16;
    if (col_base >= D) return;

    __nv_bfloat162* res2 = reinterpret_cast<__nv_bfloat162*>(res_row + col_base);
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x_row + col_base);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(weight + col_base);

    float vals[16];
    float local_ssq = 0.f;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        __nv_bfloat162 rv = res2[p];
        __nv_bfloat162 xv = x2[p];
        float a = __bfloat162float(rv.x) + __bfloat162float(xv.x);
        float b = __bfloat162float(rv.y) + __bfloat162float(xv.y);
        vals[2 * p]     = a;
        vals[2 * p + 1] = b;
        res2[p] = __floats2bfloat162_rn(a, b);
        local_ssq += a * a + b * b;
    }

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

    const float rms = __frsqrt_rn(sh[0] / D + eps);

    float amax = 0.f;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        __nv_bfloat162 wv = w2[p];
        float v0 = vals[2 * p]     * rms * __bfloat162float(wv.x);
        float v1 = vals[2 * p + 1] * rms * __bfloat162float(wv.y);
        vals[2 * p]     = v0;
        vals[2 * p + 1] = v1;
        float a0 = fabsf(v0), a1 = fabsf(v1);
        if (a0 > amax) amax = a0;
        if (a1 > amax) amax = a1;
    }

    const int block_idx = threadIdx.x;
    quant_pack_sfa_edge(vals, amax, r, col_base, block_idx, packed_row, dst_sfa,
                        layout(r, col_base, 0));
}

template <class LayoutSF>
__global__ void edge_relu2_fp4_sfa_fp16_kernel(
    const __half* __restrict__ x,   // [S, H]
    uint8_t* __restrict__ packed,   // [S, H/2]
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int H) {
    const int block_idx = blockIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.x;
    const int n_blocks = H / 16;
    if (block_idx >= n_blocks) return;

    const int col_base = block_idx * 16;
    const __half2* x2 = reinterpret_cast<const __half2*>(x + row * H + col_base);

    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        __half2 v2 = x2[i];
        float a = fmaxf(__half2float(v2.x), 0.f);
        float b = fmaxf(__half2float(v2.y), 0.f);
        a = a * a;
        b = b * b;
        vals[2 * i]     = a;
        vals[2 * i + 1] = b;
        if (a > amax) amax = a;
        if (b > amax) amax = b;
    }

    uint8_t* packed_row = packed + row * (H / 2);
    quant_pack_sfa_edge(vals, amax, row, col_base, block_idx, packed_row, dst_sfa,
                        layout(row, col_base, 0));
}

#endif  // FV_HAVE_CUTLASS

void cosmos3_edge_res_rms_fp4_sfa_bf16(
    __nv_bfloat16* residual, const __nv_bfloat16* x, const __nv_bfloat16* weight,
    uint8_t* packed, uint8_t* sfa,
    int seq_len, int dim, float eps, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    int threads = dim / 16;
    if (threads <= 0 || threads > 1024 || (dim % 16) != 0) return;
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = CfgEdge::tile_atom_to_shape_SFA(shape);
    edge_res_rms_fp4_sfa_bf16_kernel<<<seq_len, threads, 0, stream>>>(
        residual, x, weight, packed, sfa, layout, dim, eps);
#else
    (void)residual; (void)x; (void)weight; (void)packed; (void)sfa;
    (void)seq_len; (void)dim; (void)eps; (void)stream;
#endif
}

void cosmos3_edge_relu2_fp4_sfa_fp16(
    const __half* x, uint8_t* packed, uint8_t* sfa,
    int seq_len, int dim, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    if ((dim % 16) != 0) return;
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = CfgEdge::tile_atom_to_shape_SFA(shape);
    const int n_blocks = dim / 16;
    const int threads = 256;
    const int y_groups = (n_blocks + threads - 1) / threads;
    dim3 grid(seq_len, y_groups);
    edge_relu2_fp4_sfa_fp16_kernel<<<grid, dim3(threads), 0, stream>>>(
        x, packed, sfa, layout, dim);
#else
    (void)x; (void)packed; (void)sfa; (void)seq_len; (void)dim; (void)stream;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt

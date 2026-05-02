// ============================================================================
//  FlashVLA — silu_mul_two_fp4_to_fp4: P1 split-GU FFN combiner.
//
//  Reads two FP4 inputs (gate, up — each packed [S, H/2] + SFA) produced
//  by separate NVFP4 GEMMs, dequantizes each, computes silu(gate) * up in
//  fp32 registers, requantizes to FP4 + SFA for the next-stage Down GEMM.
//
//  Replaces the merged-GU + F4 v2+mul pair in the AWQ baseline with a
//  third leg that reads HALF the activation DRAM (8.6 MB combined fp4 vs
//  31.6 MB fp16 today).
//
//  Layout:
//    gate_packed[s, b*8+p] = byte holding gate elements (s, b*16 + 2p),
//                            (s, b*16 + 2p + 1)
//    gate_SFA[layout(s, b*16, 0)] = UE4M3 fp8 scale for that 16-block
//    Same for up. Output packed[s, b*8+p] + SFA same shape.
//
//  Design: 1 thread = 1 NVFP4 block (16 elements). Mirrors F4 v2 v2_kernel
//  (silu_mul_fp4_sfa_v2.cu) but reads two fp4 inputs instead of one
//  fp16 [S, 2H] merged buffer.
//
//  Additive: does NOT modify existing kernels.
// ============================================================================
#include "fused_fp4/silu_mul_two_fp4_to_fp4.cuh"

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

using CfgF4P1 = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ float e2m1_to_fp32_p1(uint8_t v) {
    // magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6} indexed by v & 0x7
    static constexpr float mags[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float m = mags[v & 0x7];
    return (v & 0x8) ? -m : m;
}

__device__ __forceinline__ uint8_t fp32_to_e2m1_p1(float x) {
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

__device__ __forceinline__ float silu_mul_p1(float g, float u) {
    // Same formula as csrc/kernels/activation.cu and silu_mul_fp4_sfa_v2.cu.
    float gelu = g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
    return gelu * u;
}

template <class LayoutSF>
__global__ void silu_mul_two_fp4_to_fp4_kernel(
    const uint8_t* __restrict__ gate_packed,   // [S, H/2]
    const uint8_t* __restrict__ gate_sfa,
    const uint8_t* __restrict__ up_packed,     // [S, H/2]
    const uint8_t* __restrict__ up_sfa,
    uint8_t* __restrict__ out_packed,          // [S, H/2]
    uint8_t* __restrict__ out_sfa,
    LayoutSF layout_in,                        // SFA layout for inputs (S, H)
    LayoutSF layout_out,                       // SFA layout for output  (S, H)
    int H) {
    // 1 thread = 1 NVFP4 block (16 elements).
    const int block_idx = blockIdx.y * blockDim.x + threadIdx.x;
    const int row       = blockIdx.x;
    const int n_blocks  = H / 16;
    if (block_idx >= n_blocks) return;

    const int col_base = block_idx * 16;

    // Read SFA scales (UE4M3 = positive fp8_e4m3 bit pattern).
    int sfa_off = layout_in(row, col_base, 0);
    uint8_t gate_sf_byte = gate_sfa[sfa_off];
    uint8_t up_sf_byte   = up_sfa[sfa_off];
    // Reinterpret as fp8 e4m3 → float.
    __nv_fp8_e4m3 gate_bs_q, up_bs_q;
    *reinterpret_cast<uint8_t*>(&gate_bs_q) = gate_sf_byte;
    *reinterpret_cast<uint8_t*>(&up_bs_q)   = up_sf_byte;
    float gate_scale = static_cast<float>(gate_bs_q);
    float up_scale   = static_cast<float>(up_bs_q);

    // Read 8 packed bytes per input (16 elements each).
    const uint8_t* gp = gate_packed + row * (H / 2) + block_idx * 8;
    const uint8_t* up = up_packed   + row * (H / 2) + block_idx * 8;

    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t gb = gp[p];
        uint8_t ub = up[p];
        float g_lo = e2m1_to_fp32_p1(gb & 0xF) * gate_scale;
        float g_hi = e2m1_to_fp32_p1(gb >> 4)  * gate_scale;
        float u_lo = e2m1_to_fp32_p1(ub & 0xF) * up_scale;
        float u_hi = e2m1_to_fp32_p1(ub >> 4)  * up_scale;
        float v0 = silu_mul_p1(g_lo, u_lo);
        float v1 = silu_mul_p1(g_hi, u_hi);
        vals[2*p]   = v0;
        vals[2*p+1] = v1;
        float a0 = fabsf(v0), a1 = fabsf(v1);
        if (a0 > amax) amax = a0;
        if (a1 > amax) amax = a1;
    }

    // Per-block scale + quantize + pack + SFA write
    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    float bs_dq = static_cast<float>(bs_q);

    int out_sfa_off = layout_out(row, col_base, 0);
    out_sfa[out_sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);

    uint8_t* op = out_packed + row * (H / 2) + block_idx * 8;
    const float inv_bs = 1.f / bs_dq;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t lo = fp32_to_e2m1_p1(vals[2*p]   * inv_bs);
        uint8_t hi = fp32_to_e2m1_p1(vals[2*p+1] * inv_bs);
        op[p] = lo | (hi << 4);
    }
}

// AWQ-Down variant: fuse per-input-channel inv_s multiply between
// silu_mul and FP4 quant. Used by P1 split-GU path with use_awq=True.
template <class LayoutSF>
__global__ void silu_mul_two_mul_fp4_to_fp4_kernel(
    const uint8_t* __restrict__ gate_packed,
    const uint8_t* __restrict__ gate_sfa,
    const uint8_t* __restrict__ up_packed,
    const uint8_t* __restrict__ up_sfa,
    const __half*  __restrict__ inv_s,           // [H], shared across rows
    uint8_t* __restrict__ out_packed,
    uint8_t* __restrict__ out_sfa,
    LayoutSF layout_in, LayoutSF layout_out,
    int H) {
    const int block_idx = blockIdx.y * blockDim.x + threadIdx.x;
    const int row       = blockIdx.x;
    const int n_blocks  = H / 16;
    if (block_idx >= n_blocks) return;
    const int col_base = block_idx * 16;

    int sfa_off = layout_in(row, col_base, 0);
    uint8_t gsf = gate_sfa[sfa_off];
    uint8_t usf = up_sfa[sfa_off];
    __nv_fp8_e4m3 gq, uq;
    *reinterpret_cast<uint8_t*>(&gq) = gsf;
    *reinterpret_cast<uint8_t*>(&uq) = usf;
    float gs = static_cast<float>(gq);
    float us = static_cast<float>(uq);

    const uint8_t* gp = gate_packed + row * (H/2) + block_idx * 8;
    const uint8_t* up = up_packed   + row * (H/2) + block_idx * 8;
    const __half* inv = inv_s + col_base;

    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t gb = gp[p];
        uint8_t ub = up[p];
        float g_lo = e2m1_to_fp32_p1(gb & 0xF) * gs;
        float g_hi = e2m1_to_fp32_p1(gb >> 4)  * gs;
        float u_lo = e2m1_to_fp32_p1(ub & 0xF) * us;
        float u_hi = e2m1_to_fp32_p1(ub >> 4)  * us;
        float v0 = silu_mul_p1(g_lo, u_lo) * __half2float(inv[2*p]);
        float v1 = silu_mul_p1(g_hi, u_hi) * __half2float(inv[2*p+1]);
        vals[2*p]   = v0;
        vals[2*p+1] = v1;
        float a0 = fabsf(v0), a1 = fabsf(v1);
        if (a0 > amax) amax = a0;
        if (a1 > amax) amax = a1;
    }

    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    float bs_dq = static_cast<float>(bs_q);
    int out_sfa_off = layout_out(row, col_base, 0);
    out_sfa[out_sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);

    uint8_t* op = out_packed + row * (H/2) + block_idx * 8;
    const float inv_bs = 1.f / bs_dq;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t lo = fp32_to_e2m1_p1(vals[2*p]   * inv_bs);
        uint8_t hi = fp32_to_e2m1_p1(vals[2*p+1] * inv_bs);
        op[p] = lo | (hi << 4);
    }
}

#endif

void silu_mul_two_fp4_to_fp4(
    const uint8_t* gate_packed, const uint8_t* gate_sfa,
    const uint8_t* up_packed,   const uint8_t* up_sfa,
    uint8_t* out_packed, uint8_t* out_sfa,
    int seq_len, int H, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    auto shape = cute::make_shape(seq_len, 1, H, 1);
    auto layout = CfgF4P1::tile_atom_to_shape_SFA(shape);

    const int n_blocks = H / 16;
    const int threads = 256;
    const int y_groups = (n_blocks + threads - 1) / threads;
    dim3 grid(seq_len, y_groups);
    dim3 block(threads);
    silu_mul_two_fp4_to_fp4_kernel<<<grid, block, 0, stream>>>(
        gate_packed, gate_sfa, up_packed, up_sfa,
        out_packed, out_sfa, layout, layout, H);
#else
    (void)gate_packed; (void)gate_sfa; (void)up_packed; (void)up_sfa;
    (void)out_packed; (void)out_sfa; (void)seq_len; (void)H; (void)stream;
#endif
}

void silu_mul_two_mul_fp4_to_fp4(
    const uint8_t* gate_packed, const uint8_t* gate_sfa,
    const uint8_t* up_packed,   const uint8_t* up_sfa,
    const __half*  inv_s,
    uint8_t* out_packed, uint8_t* out_sfa,
    int seq_len, int H, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    auto shape = cute::make_shape(seq_len, 1, H, 1);
    auto layout = CfgF4P1::tile_atom_to_shape_SFA(shape);
    const int n_blocks = H / 16;
    const int threads = 256;
    const int y_groups = (n_blocks + threads - 1) / threads;
    dim3 grid(seq_len, y_groups);
    dim3 block(threads);
    silu_mul_two_mul_fp4_to_fp4_kernel<<<grid, block, 0, stream>>>(
        gate_packed, gate_sfa, up_packed, up_sfa, inv_s,
        out_packed, out_sfa, layout, layout, H);
#else
    (void)gate_packed; (void)gate_sfa; (void)up_packed; (void)up_sfa;
    (void)inv_s; (void)out_packed; (void)out_sfa;
    (void)seq_len; (void)H; (void)stream;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt

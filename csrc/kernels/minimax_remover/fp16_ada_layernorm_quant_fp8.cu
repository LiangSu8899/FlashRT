// ================================================================
// flash_rt_minimax_remover — fused adaLN + FP8 quantise CUDA kernel.
// See fp16_ada_layernorm_quant_fp8.cuh for semantics.
//
// Implementation notes:
//   * One CUDA block per row (S rows total).  Each row's D elements
//     are processed cooperatively via a fp32 reduction.
//   * fp16x8 (uint4) vector loads for x; fp16x2 (__half2 loads);
//     one warp shuffles + a single-warp reduction across warps via
//     shared memory (no atomic in the hot path).
//   * scale/shift/x_norm arithmetic done in fp32 to match the
//     reference FP32LayerNorm path bit-for-bit within fp16 tolerance.
//   * Output is packed as fp8x4 uint32 stores.
//
// D must be a multiple of 8 (all MiniMax-Remover Linears satisfy this:
// D ∈ {1536}).  Kernel assumes D <= 4096 * threads * 8 which covers
// every practical value; caller sizes the block accordingly.
// ================================================================
#include "fp16_ada_layernorm_quant_fp8.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

namespace {

constexpr int VEC = 8;   // 8 fp16 per uint4 load

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// One block per row; blockDim.x = THREADS (128 = 4 warps).  Each thread
// walks the row in VEC-strided chunks: thread t handles elements
// t*VEC, (t + THREADS)*VEC, ...  D is a multiple of VEC.
template <int THREADS>
__global__ void ada_layernorm_quant_fp8_kernel(
    const __half* __restrict__ x_in,
    const float*  __restrict__ scale_vec,
    const float*  __restrict__ shift_vec,
    const float*  __restrict__ act_scale_ptr,
    __nv_fp8_e4m3* __restrict__ out,
    int S, int D, float eps)
{
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int NWARPS = THREADS / 32;

    const __half* xp = x_in + s * D;
    __nv_fp8_e4m3* op = out + s * D;

    const int D_vec = D / VEC;           // number of uint4 chunks
    // Pass 1 + 2 fused: single-load streaming stats (Welford-style, but
    // we prefer a simple two-pass with L2 reuse since D is small enough
    // that x is fully resident after first pass).  Two passes are used
    // here to keep the code short and match the reference fp32
    // LayerNorm bit-for-bit.

    // ── Pass 1: sum ──────────────────────────────────────────
    float local_sum = 0.f;
    for (int v = tid; v < D_vec; v += THREADS) {
        const uint4* p = reinterpret_cast<const uint4*>(xp + v * VEC);
        uint4 raw = *p;
        __half2 h0 = *reinterpret_cast<__half2*>(&raw.x);
        __half2 h1 = *reinterpret_cast<__half2*>(&raw.y);
        __half2 h2 = *reinterpret_cast<__half2*>(&raw.z);
        __half2 h3 = *reinterpret_cast<__half2*>(&raw.w);
        float2 f0 = __half22float2(h0);
        float2 f1 = __half22float2(h1);
        float2 f2 = __half22float2(h2);
        float2 f3 = __half22float2(h3);
        local_sum += f0.x + f0.y + f1.x + f1.y +
                     f2.x + f2.y + f3.x + f3.y;
    }
    // Warp reduce, then cross-warp via smem.
    __shared__ float smem_sum[NWARPS];
    float wsum = warp_reduce_sum(local_sum);
    if (lane == 0) smem_sum[warp] = wsum;
    __syncthreads();
    float mean;
    if (warp == 0) {
        float v = (lane < NWARPS) ? smem_sum[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem_sum[0] = v / (float)D;
    }
    __syncthreads();
    mean = smem_sum[0];

    // ── Pass 2: sum of squared deviations ─────────────────────
    float local_sq = 0.f;
    for (int v = tid; v < D_vec; v += THREADS) {
        const uint4* p = reinterpret_cast<const uint4*>(xp + v * VEC);
        uint4 raw = *p;
        __half2 h0 = *reinterpret_cast<__half2*>(&raw.x);
        __half2 h1 = *reinterpret_cast<__half2*>(&raw.y);
        __half2 h2 = *reinterpret_cast<__half2*>(&raw.z);
        __half2 h3 = *reinterpret_cast<__half2*>(&raw.w);
        float2 f0 = __half22float2(h0);
        float2 f1 = __half22float2(h1);
        float2 f2 = __half22float2(h2);
        float2 f3 = __half22float2(h3);
        float d0 = f0.x - mean, d1 = f0.y - mean;
        float d2 = f1.x - mean, d3 = f1.y - mean;
        float d4 = f2.x - mean, d5 = f2.y - mean;
        float d6 = f3.x - mean, d7 = f3.y - mean;
        local_sq += d0*d0 + d1*d1 + d2*d2 + d3*d3 +
                    d4*d4 + d5*d5 + d6*d6 + d7*d7;
    }
    __shared__ float smem_sq[NWARPS];
    float wsq = warp_reduce_sum(local_sq);
    if (lane == 0) smem_sq[warp] = wsq;
    __syncthreads();
    float rstd;
    if (warp == 0) {
        float v = (lane < NWARPS) ? smem_sq[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem_sq[0] = rsqrtf(v / (float)D + eps);
    }
    __syncthreads();
    rstd = smem_sq[0];

    // ── Pass 3: normalise + adaLN modulate + fp8 quantise ────
    const float act_scale = *act_scale_ptr;
    const float inv_a = 1.0f / fmaxf(act_scale, 1e-12f);
    for (int v = tid; v < D_vec; v += THREADS) {
        const int d0 = v * VEC;
        const uint4* px = reinterpret_cast<const uint4*>(xp + d0);
        uint4 raw = *px;
        __half2 h0 = *reinterpret_cast<__half2*>(&raw.x);
        __half2 h1 = *reinterpret_cast<__half2*>(&raw.y);
        __half2 h2 = *reinterpret_cast<__half2*>(&raw.z);
        __half2 h3 = *reinterpret_cast<__half2*>(&raw.w);
        float xv[VEC];
        {
            float2 f0 = __half22float2(h0);
            float2 f1 = __half22float2(h1);
            float2 f2 = __half22float2(h2);
            float2 f3 = __half22float2(h3);
            xv[0]=f0.x; xv[1]=f0.y; xv[2]=f1.x; xv[3]=f1.y;
            xv[4]=f2.x; xv[5]=f2.y; xv[6]=f3.x; xv[7]=f3.y;
        }
        // scale/shift are fp32, read sequentially.
        float sv[VEC], bv[VEC];
        // 8 fp32 loads via 2×float4:
        const float4* pS = reinterpret_cast<const float4*>(scale_vec + d0);
        const float4* pB = reinterpret_cast<const float4*>(shift_vec + d0);
        float4 s0 = pS[0], s1 = pS[1];
        float4 b0 = pB[0], b1 = pB[1];
        sv[0]=s0.x; sv[1]=s0.y; sv[2]=s0.z; sv[3]=s0.w;
        sv[4]=s1.x; sv[5]=s1.y; sv[6]=s1.z; sv[7]=s1.w;
        bv[0]=b0.x; bv[1]=b0.y; bv[2]=b0.z; bv[3]=b0.w;
        bv[4]=b1.x; bv[5]=b1.y; bv[6]=b1.z; bv[7]=b1.w;

        __nv_fp8_e4m3 fp8_pack[VEC];
        #pragma unroll
        for (int i = 0; i < VEC; i++) {
            float xn = (xv[i] - mean) * rstd;
            float y  = xn * (1.0f + sv[i]) + bv[i];
            float yq = y * inv_a;
            yq = fminf(fmaxf(yq, -448.0f), 448.0f);
            fp8_pack[i] = __nv_fp8_e4m3(yq);
        }
        // Pack 8 fp8 into 2×uint32 stores.
        uint32_t* out_u32 = reinterpret_cast<uint32_t*>(op + d0);
        out_u32[0] = *reinterpret_cast<const uint32_t*>(&fp8_pack[0]);
        out_u32[1] = *reinterpret_cast<const uint32_t*>(&fp8_pack[4]);
    }
}

}  // anonymous namespace

int fp16_ada_layernorm_quant_fp8(
    const void* x_fp16,
    const void* scale_fp32,
    const void* shift_fp32,
    const void* act_scale_fp32,
    void* out_fp8,
    int S, int D, float eps,
    cudaStream_t stream)
{
    if (!x_fp16 || !scale_fp32 || !shift_fp32 || !act_scale_fp32 || !out_fp8)
        return -1;
    if (S <= 0 || D <= 0 || (D % VEC) != 0) return -2;
    constexpr int THREADS = 128;
    ada_layernorm_quant_fp8_kernel<THREADS><<<S, THREADS, 0, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        reinterpret_cast<const float*>(scale_fp32),
        reinterpret_cast<const float*>(shift_fp32),
        reinterpret_cast<const float*>(act_scale_fp32),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
        S, D, eps);
    return 0;
}

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt

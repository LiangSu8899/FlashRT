// ================================================================
// flash_rt_minimax_remover — fused RMSNorm + interleaved RoPE.
// See fp16_rmsnorm_rope.cuh for semantics.
//
// Implementation:
//   * One CUDA block per token (B*S tokens total).  Each block
//     reduces D in fp32 across THREADS threads, then applies affine
//     weight + interleaved RoPE.
//   * fp16x8 (uint4) vector loads/stores for x.
//   * cos/sin tables indexed by (seq_idx, pair_idx) — same table
//     across heads, so a per-token block re-uses the same base_cos_sin
//     pointer for each head.
// ================================================================
#include "fp16_rmsnorm_rope.cuh"

#include <cuda_fp16.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

namespace {

constexpr int VEC = 8;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

template <int THREADS>
__global__ void rmsnorm_rope_kernel(
    __half* __restrict__ x,
    const __half* __restrict__ weight,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    int B, int S, int H, int Dd,
    float eps)
{
    const int tok = blockIdx.x;        // 0 .. B*S - 1
    const int s   = tok % S;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int NWARPS = THREADS / 32;

    const int D = H * Dd;
    __half* xrow = x + tok * D;

    // ── Pass 1: sum of squares over full D (across all heads) ────────
    const int D_vec = D / VEC;
    float local_sq = 0.f;
    for (int v = tid; v < D_vec; v += THREADS) {
        const uint4* p = reinterpret_cast<const uint4*>(xrow + v * VEC);
        uint4 raw = *p;
        __half2 h0 = *reinterpret_cast<__half2*>(&raw.x);
        __half2 h1 = *reinterpret_cast<__half2*>(&raw.y);
        __half2 h2 = *reinterpret_cast<__half2*>(&raw.z);
        __half2 h3 = *reinterpret_cast<__half2*>(&raw.w);
        float2 f0 = __half22float2(h0);
        float2 f1 = __half22float2(h1);
        float2 f2 = __half22float2(h2);
        float2 f3 = __half22float2(h3);
        local_sq += f0.x*f0.x + f0.y*f0.y + f1.x*f1.x + f1.y*f1.y +
                    f2.x*f2.x + f2.y*f2.y + f3.x*f3.x + f3.y*f3.y;
    }
    __shared__ float smem[NWARPS];
    float ws = warp_reduce_sum(local_sq);
    if (lane == 0) smem[warp] = ws;
    __syncthreads();
    float rstd;
    if (warp == 0) {
        float v = (lane < NWARPS) ? smem[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem[0] = rsqrtf(v / (float)D + eps);
    }
    __syncthreads();
    rstd = smem[0];

    // ── Pass 2: (x*rstd*w) then RoPE within each head ────────────────
    //
    // Layout: xrow[h*Dd + i], for h in [0,H), i in [0,Dd).
    // Weight[h*Dd + i] provides the affine multiplier.
    // RoPE pairs (i, i+1) share cos[s, i/2], sin[s, i/2] across h.
    //
    // Process VEC=8 elements per thread, which spans (VEC/2)=4 RoPE
    // pairs.  Guarantee Dd is a multiple of VEC so within-head pair
    // alignment holds.  Since D = H*Dd is a multiple of VEC (H*Dd
    // multiple of 8 required by caller — H%1 free, Dd%8 required),
    // vector index v corresponds to head h = (v*VEC) / Dd and offset
    // off_in_head = (v*VEC) % Dd.
    const float* cos_row = cos_tab + s * (Dd / 2);
    const float* sin_row = sin_tab + s * (Dd / 2);
    for (int v = tid; v < D_vec; v += THREADS) {
        const int base_d = v * VEC;
        const int off = base_d % Dd;      // 0 .. Dd-1 within head
        // Load x, weight
        const uint4* px = reinterpret_cast<const uint4*>(xrow + base_d);
        uint4 raw = *px;
        __half2 h0 = *reinterpret_cast<__half2*>(&raw.x);
        __half2 h1 = *reinterpret_cast<__half2*>(&raw.y);
        __half2 h2 = *reinterpret_cast<__half2*>(&raw.z);
        __half2 h3 = *reinterpret_cast<__half2*>(&raw.w);

        const uint4* pw = reinterpret_cast<const uint4*>(weight + base_d);
        uint4 wraw = *pw;
        __half2 w0 = *reinterpret_cast<__half2*>(&wraw.x);
        __half2 w1 = *reinterpret_cast<__half2*>(&wraw.y);
        __half2 w2 = *reinterpret_cast<__half2*>(&wraw.z);
        __half2 w3 = *reinterpret_cast<__half2*>(&wraw.w);

        // Normalise + affine (fp32)
        float2 f0 = __half22float2(h0);
        float2 f1 = __half22float2(h1);
        float2 f2 = __half22float2(h2);
        float2 f3 = __half22float2(h3);
        float2 wf0 = __half22float2(w0);
        float2 wf1 = __half22float2(w1);
        float2 wf2 = __half22float2(w2);
        float2 wf3 = __half22float2(w3);
        float xv[VEC];
        xv[0] = f0.x * rstd * wf0.x;
        xv[1] = f0.y * rstd * wf0.y;
        xv[2] = f1.x * rstd * wf1.x;
        xv[3] = f1.y * rstd * wf1.y;
        xv[4] = f2.x * rstd * wf2.x;
        xv[5] = f2.y * rstd * wf2.y;
        xv[6] = f3.x * rstd * wf3.x;
        xv[7] = f3.y * rstd * wf3.y;

        // Load cos/sin for the 4 pairs (off, off+2, off+4, off+6).
        // These are within the same head (guaranteed since Dd % VEC == 0).
        const int pair0 = off / 2;
        float cs[4], sn[4];
        // 4 fp32 pairs; use float4 loads.
        const float4* pc = reinterpret_cast<const float4*>(cos_row + pair0);
        const float4* ps = reinterpret_cast<const float4*>(sin_row + pair0);
        float4 c4 = *pc, s4 = *ps;
        cs[0] = c4.x; cs[1] = c4.y; cs[2] = c4.z; cs[3] = c4.w;
        sn[0] = s4.x; sn[1] = s4.y; sn[2] = s4.z; sn[3] = s4.w;

        // Apply RoPE (interleaved): (x0, x1) -> (x0*c - x1*s, x0*s + x1*c)
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float x0 = xv[2*j], x1 = xv[2*j + 1];
            float c = cs[j], sn_j = sn[j];
            xv[2*j    ] = x0 * c    - x1 * sn_j;
            xv[2*j + 1] = x0 * sn_j + x1 * c;
        }

        // Pack back to fp16 and store.
        __half2 o0 = __floats2half2_rn(xv[0], xv[1]);
        __half2 o1 = __floats2half2_rn(xv[2], xv[3]);
        __half2 o2 = __floats2half2_rn(xv[4], xv[5]);
        __half2 o3 = __floats2half2_rn(xv[6], xv[7]);
        uint4 out;
        out.x = *reinterpret_cast<const uint32_t*>(&o0);
        out.y = *reinterpret_cast<const uint32_t*>(&o1);
        out.z = *reinterpret_cast<const uint32_t*>(&o2);
        out.w = *reinterpret_cast<const uint32_t*>(&o3);
        *reinterpret_cast<uint4*>(xrow + base_d) = out;
    }
}

}  // anonymous namespace

int fp16_rmsnorm_rope_bshd(
    void* x_fp16,
    const void* weight_fp16,
    const void* cos_fp32,
    const void* sin_fp32,
    int B, int S, int H, int Dd,
    float eps,
    cudaStream_t stream)
{
    if (!x_fp16 || !weight_fp16 || !cos_fp32 || !sin_fp32) return -1;
    if (B <= 0 || S <= 0 || H <= 0 || Dd <= 0) return -2;
    const int D = H * Dd;
    if ((D % VEC) != 0 || (Dd % VEC) != 0) return -3;
    constexpr int THREADS = 128;
    const int tokens = B * S;
    rmsnorm_rope_kernel<THREADS><<<tokens, THREADS, 0, stream>>>(
        reinterpret_cast<__half*>(x_fp16),
        reinterpret_cast<const __half*>(weight_fp16),
        reinterpret_cast<const float*>(cos_fp32),
        reinterpret_cast<const float*>(sin_fp32),
        B, S, H, Dd, eps);
    return 0;
}

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt

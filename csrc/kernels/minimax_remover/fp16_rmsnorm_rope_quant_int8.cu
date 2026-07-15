// ================================================================
// Fused RMSNorm + interleaved RoPE + int8 per-warp/per-block quantization.
//
// Two-kernel pipeline:
//   Kernel A (rmsnorm_rstd_kernel): 1 block/token, computes rstd across
//       D=H*Dd and writes to rstd[B*S] global buffer.
//   Kernel B (norm_rope_quant_kernel): 1 block per (group, head, batch),
//       reads x fp16 + rstd, applies norm+rope, reduces max-abs across
//       the group, quantizes to int8 in one pass over shared memory.
//
// Eliminates the intermediate fp16 tensor between rmsnorm+rope and
// the int8 quantization — saves 2×S×D bytes per Q/K tensor (read+write).
// ================================================================

#include "fp16_rmsnorm_rope_quant_int8.cuh"
#include <cuda_fp16.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

namespace {

constexpr int VEC_RSTD = 8;

__device__ __forceinline__ float warp_reduce_sum_rq(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// ─── Kernel A: compute rstd per token ─────────────────────────────
// Grid: (B*S), Block: 128 threads
// Reads D fp16 elements, reduces sum-of-squares, writes 1 float.
template <int THREADS>
__global__ void rmsnorm_rstd_kernel(
    const __half* __restrict__ x,   // [B*S, D]
    const __half* __restrict__ bias, // [D] or nullptr — added before rmsnorm
    float* __restrict__ rstd_out,   // [B*S]
    int D, float eps)
{
    const int tok = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int NWARPS = THREADS / 32;

    const __half* xrow = x + (long long)tok * D;

    float local_sq = 0.f;
    const int D_vec = D / VEC_RSTD;
    for (int v = tid; v < D_vec; v += THREADS) {
        const uint4* p = reinterpret_cast<const uint4*>(xrow + v * VEC_RSTD);
        uint4 raw = *p;
        __half2 h0 = *reinterpret_cast<__half2*>(&raw.x);
        __half2 h1 = *reinterpret_cast<__half2*>(&raw.y);
        __half2 h2 = *reinterpret_cast<__half2*>(&raw.z);
        __half2 h3 = *reinterpret_cast<__half2*>(&raw.w);
        float2 f0 = __half22float2(h0);
        float2 f1 = __half22float2(h1);
        float2 f2 = __half22float2(h2);
        float2 f3 = __half22float2(h3);
        if (bias) {
            const uint4* pb = reinterpret_cast<const uint4*>(bias + v * VEC_RSTD);
            uint4 rb = *pb;
            float2 b0 = __half22float2(*reinterpret_cast<__half2*>(&rb.x));
            float2 b1 = __half22float2(*reinterpret_cast<__half2*>(&rb.y));
            float2 b2 = __half22float2(*reinterpret_cast<__half2*>(&rb.z));
            float2 b3 = __half22float2(*reinterpret_cast<__half2*>(&rb.w));
            f0.x += b0.x; f0.y += b0.y; f1.x += b1.x; f1.y += b1.y;
            f2.x += b2.x; f2.y += b2.y; f3.x += b3.x; f3.y += b3.y;
        }
        local_sq += f0.x*f0.x + f0.y*f0.y + f1.x*f1.x + f1.y*f1.y +
                    f2.x*f2.x + f2.y*f2.y + f3.x*f3.x + f3.y*f3.y;
    }

    __shared__ float smem[NWARPS];
    float ws = warp_reduce_sum_rq(local_sq);
    if (lane == 0) smem[warp] = ws;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < NWARPS) ? smem[lane] : 0.f;
        v = warp_reduce_sum_rq(v);
        if (lane == 0)
            rstd_out[tok] = rsqrtf(v / (float)D + eps);
    }
}

// ─── Kernel B: norm + rope + max-abs + quantize ──────────────────
// Grid: (num_groups, H, B)
// Block: TPB threads (256)
// Each block processes GROUP_SIZE tokens × Dd dims for one head.
//
// Shared memory:
//   normed[GROUP_SIZE * Dd] fp32 — holds normalized+RoPE'd values
//   reduce_buf[TPB] fp32 — for max-abs reduction
template <int GROUP_SIZE, int TPB>
__global__ void norm_rope_quant_kernel(
    const __half* __restrict__ x,       // [B*S, D]
    const __half* __restrict__ weight,   // [D]
    const __half* __restrict__ bias,     // [D] or nullptr — added before rmsnorm
    const float* __restrict__ cos_tab,   // [S, Dd/2]
    const float* __restrict__ sin_tab,   // [S, Dd/2]
    const float* __restrict__ rstd,      // [B*S]
    const __half* __restrict__ km,       // [B, H, Dd] or nullptr
    int8_t* __restrict__ out,            // [B*S, D]
    float* __restrict__ scale_out,       // [B, H, num_groups]
    int B, int S, int H, int Dd,
    float sm_scale_factor)
{
    const int group_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const int D = H * Dd;

    const int tok_start = group_idx * GROUP_SIZE;
    const int tok_end = min(tok_start + GROUP_SIZE, S);
    const int n_toks = tok_end - tok_start;
    const int total_elems = n_toks * Dd;

    extern __shared__ char smem_raw[];
    float* normed = reinterpret_cast<float*>(smem_raw);
    float* reduce_buf = normed + GROUP_SIZE * Dd;

    // ── Phase 1: Load, normalize, apply RoPE, store to shared ──
    float local_max = 0.f;

    // Process elements in pairs (for RoPE correctness)
    // Each element at even dim d needs its partner at d+1 and vice versa.
    // Process in pairs: iterate over (tok, pair_idx) where pair_idx = d/2.
    const int total_pairs = n_toks * (Dd / 2);

    for (int idx = tid; idx < total_pairs; idx += TPB) {
        const int lt = idx / (Dd / 2);       // local token
        const int pair = idx % (Dd / 2);     // pair index within head
        const int d0 = pair * 2;
        const int d1 = d0 + 1;
        const int s = tok_start + lt;        // sequence position
        const int global_tok = b * S + s;

        const float my_rstd = rstd[global_tok];

        // Load x values (+ bias, fused: avoids the separate add_bias kernel)
        float x0 = __half2float(x[(long long)global_tok * D + h * Dd + d0]);
        float x1 = __half2float(x[(long long)global_tok * D + h * Dd + d1]);
        if (bias) {
            x0 += __half2float(bias[h * Dd + d0]);
            x1 += __half2float(bias[h * Dd + d1]);
        }

        // Load weights
        float w0 = __half2float(weight[h * Dd + d0]);
        float w1 = __half2float(weight[h * Dd + d1]);

        // RMSNorm + affine
        x0 = x0 * my_rstd * w0;
        x1 = x1 * my_rstd * w1;

        // Interleaved RoPE: (x0, x1) → (x0*c - x1*s, x0*s + x1*c)
        float c = cos_tab[s * (Dd / 2) + pair];
        float sn = sin_tab[s * (Dd / 2) + pair];
        float r0 = x0 * c - x1 * sn;
        float r1 = x0 * sn + x1 * c;

        // Apply sm_scale (for Q quantization, folds softmax scale)
        r0 *= sm_scale_factor;
        r1 *= sm_scale_factor;

        // Subtract key mean if applicable (smooth_k)
        if (km != nullptr) {
            float km0 = __half2float(km[(long long)b * H * Dd + h * Dd + d0]);
            float km1 = __half2float(km[(long long)b * H * Dd + h * Dd + d1]);
            r0 -= km0 * sm_scale_factor;
            r1 -= km1 * sm_scale_factor;
        }

        // Store to shared memory
        normed[lt * Dd + d0] = r0;
        normed[lt * Dd + d1] = r1;

        // Track max absolute value
        local_max = fmaxf(local_max, fmaxf(fabsf(r0), fabsf(r1)));
    }
    __syncthreads();

    // ── Phase 2: Reduce max-abs across all threads ──
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int stride = TPB / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + stride]);
        __syncthreads();
    }
    float group_max = reduce_buf[0];
    float scale = group_max / 127.0f + 1e-7f;
    float inv_scale = 1.0f / scale;

    // Write scale
    if (tid == 0) {
        const int num_groups = (S + GROUP_SIZE - 1) / GROUP_SIZE;
        scale_out[(long long)b * H * num_groups + h * num_groups + group_idx] = scale;
    }

    // ── Phase 3: Quantize from shared memory and write int8 ──
    for (int idx = tid; idx < total_elems; idx += TPB) {
        const int lt = idx / Dd;
        const int d  = idx % Dd;
        const int global_tok = b * S + tok_start + lt;

        float val = normed[lt * Dd + d];
        int ival = __float2int_rn(val * inv_scale);
        ival = max(-128, min(127, ival));
        out[(long long)global_tok * D + h * Dd + d] = static_cast<int8_t>(ival);
    }
}

}  // anonymous namespace

int fp16_rmsnorm_rope_quant_int8_q(
    const void* x_fp16,
    const void* weight_fp16,
    const void* bias_fp16,
    const void* cos_fp32,
    const void* sin_fp32,
    void* out_int8,
    void* scale_fp32,
    int B, int S, int H, int Dd,
    float eps, float sm_scale,
    void* rstd_buf,
    cudaStream_t stream)
{
    if (!x_fp16 || !weight_fp16 || !cos_fp32 || !sin_fp32 ||
        !out_int8 || !scale_fp32)
        return -1;
    if (B <= 0 || S <= 0 || H <= 0 || Dd <= 0) return -2;
    const int D = H * Dd;
    if ((D % VEC_RSTD) != 0) return -3;

    // rstd scratch: prefer the caller-owned buffer (hot path, zero alloc).
    // Fall back to a stream-ordered transient allocation only when the
    // caller passes nullptr (one-off / non-hot-path use).
    float* rstd = static_cast<float*>(rstd_buf);
    float* transient = nullptr;
    if (rstd == nullptr) {
        cudaError_t e = cudaMallocAsync(&transient,
                                        B * S * sizeof(float), stream);
        if (e != cudaSuccess) return -4;
        rstd = transient;
    }

    // Kernel A: compute rstd (over x+bias)
    constexpr int THREADS_A = 128;
    rmsnorm_rstd_kernel<THREADS_A><<<B * S, THREADS_A, 0, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        bias_fp16 ? reinterpret_cast<const __half*>(bias_fp16) : nullptr,
        rstd, D, eps);

    // Kernel B: norm + rope + quantize (Q: GROUP_SIZE=32)
    constexpr int GROUP_SIZE = 32;
    constexpr int TPB = 256;
    const int num_groups = (S + GROUP_SIZE - 1) / GROUP_SIZE;
    dim3 grid(num_groups, H, B);
    size_t smem = (GROUP_SIZE * Dd + TPB) * sizeof(float);

    norm_rope_quant_kernel<GROUP_SIZE, TPB><<<grid, TPB, smem, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        reinterpret_cast<const __half*>(weight_fp16),
        bias_fp16 ? reinterpret_cast<const __half*>(bias_fp16) : nullptr,
        reinterpret_cast<const float*>(cos_fp32),
        reinterpret_cast<const float*>(sin_fp32),
        rstd,
        nullptr,  // no km for Q
        reinterpret_cast<int8_t*>(out_int8),
        reinterpret_cast<float*>(scale_fp32),
        B, S, H, Dd, sm_scale);

    if (transient != nullptr)
        cudaFreeAsync(transient, stream);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -5;
}

int fp16_rmsnorm_rope_quant_int8_k(
    const void* x_fp16,
    const void* weight_fp16,
    const void* bias_fp16,
    const void* cos_fp32,
    const void* sin_fp32,
    const void* km_fp16,
    void* out_int8,
    void* scale_fp32,
    int B, int S, int H, int Dd,
    float eps, float sm_scale,
    void* rstd_buf,
    cudaStream_t stream)
{
    if (!x_fp16 || !weight_fp16 || !cos_fp32 || !sin_fp32 ||
        !out_int8 || !scale_fp32)
        return -1;
    if (B <= 0 || S <= 0 || H <= 0 || Dd <= 0) return -2;
    const int D = H * Dd;
    if ((D % VEC_RSTD) != 0) return -3;

    // rstd scratch: prefer the caller-owned buffer (hot path, zero alloc).
    float* rstd = static_cast<float*>(rstd_buf);
    float* transient = nullptr;
    if (rstd == nullptr) {
        cudaError_t e = cudaMallocAsync(&transient,
                                        B * S * sizeof(float), stream);
        if (e != cudaSuccess) return -4;
        rstd = transient;
    }

    // Kernel A: compute rstd (over x+bias)
    constexpr int THREADS_A = 128;
    rmsnorm_rstd_kernel<THREADS_A><<<B * S, THREADS_A, 0, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        bias_fp16 ? reinterpret_cast<const __half*>(bias_fp16) : nullptr,
        rstd, D, eps);

    // Kernel B: norm + rope + quantize (K: GROUP_SIZE=64, with smooth_k)
    constexpr int GROUP_SIZE = 64;
    constexpr int TPB = 256;
    const int num_groups = (S + GROUP_SIZE - 1) / GROUP_SIZE;
    dim3 grid(num_groups, H, B);
    size_t smem = (GROUP_SIZE * Dd + TPB) * sizeof(float);

    norm_rope_quant_kernel<GROUP_SIZE, TPB><<<grid, TPB, smem, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        reinterpret_cast<const __half*>(weight_fp16),
        bias_fp16 ? reinterpret_cast<const __half*>(bias_fp16) : nullptr,
        reinterpret_cast<const float*>(cos_fp32),
        reinterpret_cast<const float*>(sin_fp32),
        rstd,
        km_fp16 ? reinterpret_cast<const __half*>(km_fp16) : nullptr,
        reinterpret_cast<int8_t*>(out_int8),
        reinterpret_cast<float*>(scale_fp32),
        B, S, H, Dd, sm_scale);

    if (transient != nullptr)
        cudaFreeAsync(transient, stream);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -5;
}

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt

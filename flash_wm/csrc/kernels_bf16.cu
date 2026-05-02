// ================================================================
// flash_wm_kernels — BF16 kernel variants for BAGEL world model
//
// These are BF16 ports of FlashRT's FP16 kernels.
// BAGEL requires BF16 because its residual stream exceeds FP16 range
// (absmax ~480K >> FP16 max 65504).
//
// Kernels:
//   1. attention_mha_bf16    — cuBLAS batched attention
//   2. softmax_bf16          — warp-level row softmax
//   3. rope_rotate_half_bf16 — Qwen-style RoPE (in-place)
//   4. silu_mul_split_fp8_bf16 — SiLU(gate)*up → FP8
//   5. gpu_fill_neginf_bf16  — fill buffer with -inf
//   6. gpu_repeat_interleave_heads_bf16 — GQA head repeat
// ================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cmath>

// ── Softmax BF16 ────────────────────────────────────────────────

#define SM_WARP_SIZE 32
#define SM_MAX_COLS 1024
#define SM_ITERS (SM_MAX_COLS / SM_WARP_SIZE)

__global__ void softmax_bf16_kernel(__nv_bfloat16* data, int rows, int cols) {
    int lane = threadIdx.x % SM_WARP_SIZE;
    int row = blockIdx.x;
    if (row >= rows) return;

    __nv_bfloat16* src = data + row * cols;
    int cols2 = cols / 2;
    __nv_bfloat162* src2 = reinterpret_cast<__nv_bfloat162*>(src);

    float reg[SM_ITERS];
    float mx = -1e30f;

    #pragma unroll
    for (int it = 0; it < SM_ITERS / 2; it++) {
        int c2 = it * SM_WARP_SIZE + lane;
        if (c2 < cols2) {
            __nv_bfloat162 v2 = src2[c2];
            reg[it*2]   = __bfloat162float(v2.x);
            reg[it*2+1] = __bfloat162float(v2.y);
            mx = fmaxf(mx, fmaxf(reg[it*2], reg[it*2+1]));
        } else {
            reg[it*2] = -1e30f;
            reg[it*2+1] = -1e30f;
        }
    }
    if ((cols & 1) && lane == 0) {
        float v = __bfloat162float(src[cols-1]);
        reg[SM_ITERS-1] = v;
        mx = fmaxf(mx, v);
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));

    float sm = 0;
    #pragma unroll
    for (int it = 0; it < SM_ITERS; it++) {
        reg[it] = __expf(reg[it] - mx);
        sm += reg[it];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sm += __shfl_xor_sync(0xffffffff, sm, o);

    float inv = 1.f / (sm + 1e-8f);
    #pragma unroll
    for (int it = 0; it < SM_ITERS / 2; it++) {
        int c2 = it * SM_WARP_SIZE + lane;
        if (c2 < cols2) {
            __nv_bfloat162 v2;
            v2.x = __float2bfloat16(reg[it*2] * inv);
            v2.y = __float2bfloat16(reg[it*2+1] * inv);
            src2[c2] = v2;
        }
    }
    if ((cols & 1) && lane == 0) {
        src[cols-1] = __float2bfloat16(reg[SM_ITERS-1] * inv);
    }
}

void softmax_bf16(__nv_bfloat16* data, int rows, int cols, cudaStream_t stream) {
    softmax_bf16_kernel<<<rows, SM_WARP_SIZE, 0, stream>>>(data, rows, cols);
}

// ── Attention MHA BF16 ──────────────────────────────────────────

void attention_mha_bf16(
    cublasHandle_t handle,
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* logits,
    __nv_bfloat16* out,
    int S_q, int S_kv, int NH, int HD,
    float attn_scale,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);
    int S_kv_pad = ((S_kv + 7) / 8) * 8;
    float zero = 0.0f;
    long long strideC = (long long)S_q * S_kv_pad;

    // Step 1: QK^T
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        S_kv, S_q, HD,
        &attn_scale,
        K, CUDA_R_16BF, NH * HD, (long long)HD,
        Q, CUDA_R_16BF, NH * HD, (long long)HD,
        &zero,
        logits, CUDA_R_16BF, S_kv_pad, strideC,
        NH,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Step 2: softmax
    softmax_bf16(logits, NH * S_q, S_kv_pad, stream);

    // Step 3: PV
    float one = 1.0f;
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, S_q, S_kv,
        &one,
        V, CUDA_R_16BF, NH * HD, (long long)HD,
        logits, CUDA_R_16BF, S_kv_pad, strideC,
        &zero,
        out, CUDA_R_16BF, NH * HD, (long long)HD,
        NH,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ── RoPE rotate_half BF16 ───────────────────────────────────────

__global__ void rope_rotate_half_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_t,
    const __nv_bfloat16* __restrict__ sin_t,
    int S, int NH, int HD)
{
    int half_hd = HD / 2;
    int total_pairs = S * NH * half_hd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int d = idx % half_hd;
    int remainder = idx / half_hd;
    int h = remainder % NH;
    int s = remainder / NH;

    int rope_idx = s * HD + d;
    float c  = __bfloat162float(cos_t[rope_idx]);
    float si = __bfloat162float(sin_t[rope_idx]);

    int base = s * NH * HD + h * HD;
    float x_lo = __bfloat162float(x[base + d]);
    float x_hi = __bfloat162float(x[base + d + half_hd]);

    x[base + d]           = __float2bfloat16(x_lo * c - x_hi * si);
    x[base + d + half_hd] = __float2bfloat16(x_hi * c + x_lo * si);
}

void rope_rotate_half_bf16(
    __nv_bfloat16* x,
    const __nv_bfloat16* cos_table,
    const __nv_bfloat16* sin_table,
    int S, int NH, int HD,
    cudaStream_t stream)
{
    int total_pairs = S * NH * (HD / 2);
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    rope_rotate_half_bf16_kernel<<<blocks, threads, 0, stream>>>(
        x, cos_table, sin_table, S, NH, HD);
}

// ── SiLU(gate) * up → FP8 (BF16 input) ─────────────────────────

__global__ void silu_mul_split_fp8_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_fp8_e4m3* __restrict__ out,
    int n, const float* __restrict__ d_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float inv_scale = 1.0f / (*d_scale);
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    float val = silu_g * u * inv_scale;
    out[idx] = __nv_fp8_e4m3(fminf(fmaxf(val, -448.0f), 448.0f));
}

void silu_mul_split_fp8_bf16(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_fp8_e4m3* out,
    int n, const float* d_scale,
    cudaStream_t stream)
{
    silu_mul_split_fp8_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        gate, up, out, n, d_scale);
}

// ── Tiny BF16 matmul for M=2 (MoT text path) ────────────────────
// Computes C[2, N] = A[2, K] * B[N, K]^T (row-major B — weight matrix layout)
// Optimized for the MoT text path: M is always 2, N and K are large
// (up to 18944 and 3584). cuBLAS has ~500 us setup overhead per call; a
// hand-rolled kernel hits DRAM-bandwidth limit at ~half that.
//
// Launch config: grid=(N,), block=(32,). One warp per output column.
// Each lane streams K/32 * bf162 (= K/16 elements) of one B row, dot-
// products both A rows, warp-reduces, lane 0 writes C[0,n] and C[1,n].
// A rows stay in L1 across the N blocks (same 2 rows read by every block);
// B row is unique per block and fetched coalesced bf162 at a time.
//
// Requires K even and K <= 32 * K_MAX_PER_LANE * 2 (K <= ~65k for our use).

__global__ void tiny_bf16_matmul_m2_kernel(
    const __nv_bfloat16* __restrict__ A,   // [2, K]
    const __nv_bfloat16* __restrict__ B,   // [N, K]
    __nv_bfloat16* __restrict__ C,         // [2, N]
    int N, int K)
{
    int n = blockIdx.x;
    int lane = threadIdx.x;
    if (n >= N) return;

    const __nv_bfloat162* A0 = reinterpret_cast<const __nv_bfloat162*>(A);
    const __nv_bfloat162* A1 = reinterpret_cast<const __nv_bfloat162*>(A + K);
    const __nv_bfloat162* Bn = reinterpret_cast<const __nv_bfloat162*>(B + (size_t)n * K);
    int K2 = K >> 1;

    float s0 = 0.0f, s1 = 0.0f;
    #pragma unroll 4
    for (int k = lane; k < K2; k += 32) {
        __nv_bfloat162 a0 = A0[k];
        __nv_bfloat162 a1 = A1[k];
        __nv_bfloat162 bv = Bn[k];
        float2 a0f = __bfloat1622float2(a0);
        float2 a1f = __bfloat1622float2(a1);
        float2 bf  = __bfloat1622float2(bv);
        s0 += a0f.x * bf.x + a0f.y * bf.y;
        s1 += a1f.x * bf.x + a1f.y * bf.y;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s0 += __shfl_xor_sync(0xffffffff, s0, o);
        s1 += __shfl_xor_sync(0xffffffff, s1, o);
    }
    if (lane == 0) {
        C[n]         = __float2bfloat16(s0);
        C[N + n]     = __float2bfloat16(s1);
    }
}

void tiny_bf16_matmul_m2(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int N, int K, cudaStream_t stream)
{
    tiny_bf16_matmul_m2_kernel<<<N, 32, 0, stream>>>(A, B, C, N, K);
}

// ── SiLU(gate) * up → BF16 (BF16 in, BF16 out, no FP8 quantize) ─
// Used for MoT text FFN path (2 rows) where no FP8 quantize is needed.

__global__ void silu_mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    out[idx] = __float2bfloat16(silu_g * u);
}

void silu_mul_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* out, int n, cudaStream_t stream)
{
    silu_mul_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n);
}

// ── SiLU merged BF16 → FP8 (F1: after packed gate+up GEMM) ──────
// Input: packed [Sq, 2*FFN] (row = gate[:FFN] then up[FFN:2*FFN]).
// Output: [Sq*FFN] FP8 = SiLU(gate) * up / d_scale, saturated to e4m3.

__global__ void silu_mul_merged_fp8_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_fp8_e4m3* __restrict__ out,
    int seq, int ffn, const float* __restrict__ d_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * ffn;
    if (idx >= total) return;
    int s = idx / ffn;
    int f = idx - s * ffn;
    int row_base = s * (ffn << 1);
    float g = __bfloat162float(gate_up[row_base + f]);
    float u = __bfloat162float(gate_up[row_base + ffn + f]);
    float inv_scale = 1.0f / (*d_scale);
    float silu_g = g / (1.0f + expf(-g));
    float val = silu_g * u * inv_scale;
    out[idx] = __nv_fp8_e4m3(fminf(fmaxf(val, -448.0f), 448.0f));
}

void silu_mul_merged_fp8_bf16(
    const __nv_bfloat16* gate_up, __nv_fp8_e4m3* out,
    int seq, int ffn, const float* d_scale, cudaStream_t stream)
{
    int total = seq * ffn;
    silu_mul_merged_fp8_bf16_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        gate_up, out, seq, ffn, d_scale);
}

// ── Fill -inf (BF16) ────────────────────────────────────────────

__global__ void fill_neginf_bf16_kernel(__nv_bfloat16* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = __float2bfloat16(-1e30f);
}

void gpu_fill_neginf_bf16(__nv_bfloat16* dst, int n, cudaStream_t stream) {
    fill_neginf_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(dst, n);
}

// ── Add bias (BF16, broadcast over rows) ────────────────────────

__global__ void add_bias_bf16_kernel(
    __nv_bfloat16* data, const __nv_bfloat16* bias,
    int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int c = idx % cols;
    float val = __bfloat162float(data[idx]) + __bfloat162float(bias[c]);
    data[idx] = __float2bfloat16(val);
}

void gpu_add_bias_bf16(__nv_bfloat16* data, const __nv_bfloat16* bias,
                       int rows, int cols, cudaStream_t stream) {
    int total = rows * cols;
    add_bias_bf16_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        data, bias, rows, cols);
}

// ── BF16 GEMM (cuBLAS, for small text token matmul) ─────────────

void bf16_gemm_nn(
    cublasHandle_t handle,
    const __nv_bfloat16* A,  // [M, K]
    const __nv_bfloat16* B,  // [N, K] — stored as B^T [K, N] in row-major = [N, K] col-major
    __nv_bfloat16* C,        // [M, N]
    int M, int N, int K,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    // C[M,N] = A[M,K] * B^T[K,N]
    // In col-major: C[N,M] = B[K,N]^T * A[K,M]
    // But A is row-major [M,K], in col-major it's [K,M]
    // B is row-major [N,K], in col-major it's [K,N]
    // We want C = A * B^T → col-major: C[N,M] = B[K,N]^T * A[K,M] = CUBLAS_OP_T on B, CUBLAS_OP_N on A
    // But cublas(A=B^T[K,N], B=A[K,M]) → C[N,M]
    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, K,    // B^T: [K, N] with lda=K
        A, CUDA_R_16BF, K,    // A: [K, M] with ldb=K
        &beta,
        C, CUDA_R_16BF, N,    // C: [N, M] with ldc=N
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ── Repeat interleave heads (BF16) ──────────────────────────────
// Pure memory movement — works identically to FP16 version

__global__ void repeat_interleave_heads_bf16_kernel(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int S, int NH_src, int HD, int repeat)
{
    int NH_dst = NH_src * repeat;
    int total = S * NH_dst * HD;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d = idx % HD;
    int remainder = idx / HD;
    int h_dst = remainder % NH_dst;
    int s = remainder / NH_dst;

    int h_src = h_dst / repeat;
    dst[s * NH_dst * HD + h_dst * HD + d] = src[s * NH_src * HD + h_src * HD + d];
}

void gpu_repeat_interleave_heads_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int S, int NH_src, int HD, int repeat,
    cudaStream_t stream)
{
    int NH_dst = NH_src * repeat;
    int total = S * NH_dst * HD;
    repeat_interleave_heads_bf16_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        src, dst, S, NH_src, HD, repeat);
}

// ── BF16 → FP16 cast (B5 FP4 FFN bridge) ─────────────────────────
// Only safe on post-rms-norm activations (|x| < ~150 in BAGEL);
// NEVER cast the raw residual stream which reaches 150K+.
#include <cuda_fp16.h>

__global__ void cast_bf16_to_fp16_kernel(
    const __nv_bfloat16* __restrict__ src, __half* __restrict__ dst, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __float2half(__bfloat162float(src[idx]));
}

void cast_bf16_to_fp16(
    const __nv_bfloat16* src, __half* dst, int n, cudaStream_t stream)
{
    cast_bf16_to_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, n);
}

// ── FP16 → BF16 cast (FP4 FFN exit side) ─────────────────────────
// Used to bring the FP4 down-GEMM fp16 output back into the bf16
// b_down buffer, so text-row overlay + parent's F2 fuse-B flow
// (residual_add_rms_norm_fp8) work unchanged.
__global__ void cast_fp16_to_bf16_kernel(
    const __half* __restrict__ src, __nv_bfloat16* __restrict__ dst, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __float2bfloat16(__half2float(src[idx]));
}

void cast_fp16_to_bf16(
    const __half* src, __nv_bfloat16* dst, int n, cudaStream_t stream)
{
    cast_fp16_to_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, n);
}

// ── Fused BF16 residual_add + weighted rms_norm → FP16 out ──────
// Replaces 3 kernels per FP4 layer (residual_add + rms_norm + cast_bf16_to_fp16)
// with a single block-per-row kernel. Math:
//   x_new[i,:] = x[i,:] + r[i,:]            (bf16, in-place on x)
//   out[i,:]   = fp16( x_new[i,:] * rsqrt(mean(x_new[i,:]^2) + eps) * w[:] )
// w (norm weight) is bf16 [D], broadcast across rows.
// Block size = 1024 threads, one block per row; warp reductions for mean.

__global__ void residual_add_rms_norm_bf16_to_fp16_kernel(
    __nv_bfloat16* __restrict__ x,           // [rows, D]  mutated in place
    const __nv_bfloat16* __restrict__ r,     // [rows, D]
    const __nv_bfloat16* __restrict__ w,     // [D]
    __half* __restrict__ out,                // [rows, D]
    int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    __nv_bfloat16* x_row = x + row * D;
    const __nv_bfloat16* r_row = r + row * D;
    __half* out_row = out + row * D;

    // Pass 1: compute x_new = x + r in place, accumulate sum-of-squares.
    float sumsq = 0.0f;
    for (int k = tid; k < D; k += stride) {
        float v = __bfloat162float(x_row[k]) + __bfloat162float(r_row[k]);
        x_row[k] = __float2bfloat16(v);
        sumsq += v * v;
    }
    // Reduce sumsq across block using shared memory.
    __shared__ float sh[32];
    for (int off = 16; off > 0; off >>= 1) sumsq += __shfl_xor_sync(0xffffffff, sumsq, off);
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) sh[warp] = sumsq;
    __syncthreads();
    if (warp == 0) {
        sumsq = (tid < (blockDim.x + 31) / 32) ? sh[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) sumsq += __shfl_xor_sync(0xffffffff, sumsq, off);
        if (lane == 0) sh[0] = sumsq;
    }
    __syncthreads();
    float rstd = rsqrtf(sh[0] / (float)D + eps);

    // Pass 2: multiply by rstd * w and emit fp16.
    for (int k = tid; k < D; k += stride) {
        float v = __bfloat162float(x_row[k]) * rstd * __bfloat162float(w[k]);
        out_row[k] = __float2half(v);
    }
}

void residual_add_rms_norm_bf16_to_fp16(
    __nv_bfloat16* x, const __nv_bfloat16* r, const __nv_bfloat16* w,
    __half* out, int rows, int D, float eps, cudaStream_t stream)
{
    // One block per row, 1024 threads per block. Works for any D divisible
    // by 1024-element chunks; loops internally.
    residual_add_rms_norm_bf16_to_fp16_kernel<<<rows, 1024, 0, stream>>>(
        x, r, w, out, D, eps);
}

// ── SiLU(gate) * up -> FP16 (FP16 in, FP16 out) ─────────────────
// Mirror of silu_mul_bf16. Used after FP4 gate/up GEMM to produce
// the FP4-Down input tile in fp16, ready for quantize_fp4_dynamic_sfa_fp16.
//
// OPTIMIZED: vectorized half2 loads/stores (process 8 fp16 elements per
// thread via four half2). Previous naive 1-element version ran ~643 μs
// per call on [Sq=786, FFN=18944]; vectorized version targets ~130 μs
// (HBM-bound at ~90 MB memory traffic).

__global__ void silu_mul_fp16_kernel_v1(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    out[idx] = __float2half(silu_g * u);
}

// Vectorized: each thread handles 2 fp16 elements via one half2 pair.
__global__ void silu_mul_fp16_kernel_vec2(
    const __half2* __restrict__ gate,
    const __half2* __restrict__ up,
    __half2* __restrict__ out, int n_half2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_half2) return;
    __half2 g = gate[idx];
    __half2 u = up[idx];
    float g0 = __half2float(__low2half(g));
    float g1 = __half2float(__high2half(g));
    float u0 = __half2float(__low2half(u));
    float u1 = __half2float(__high2half(u));
    float s0 = g0 / (1.0f + __expf(-g0));
    float s1 = g1 / (1.0f + __expf(-g1));
    out[idx] = __floats2half2_rn(s0 * u0, s1 * u1);
}

void silu_mul_fp16(
    const __half* gate, const __half* up,
    __half* out, int n, cudaStream_t stream)
{
    if ((n & 1) == 0) {
        int n_half2 = n >> 1;
        int blocks = (n_half2 + 255) / 256;
        silu_mul_fp16_kernel_vec2<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const __half2*>(gate),
            reinterpret_cast<const __half2*>(up),
            reinterpret_cast<__half2*>(out),
            n_half2);
    } else {
        silu_mul_fp16_kernel_v1<<<(n + 255) / 256, 256, 0, stream>>>(
            gate, up, out, n);
    }
}

// ── SiLU(gate) * up + saturate clamp  (FP4 Down-accumulator safety) ─
// Same as silu_mul_fp16 but bounds |out| <= max_abs. Intended as a
// Path C fix for L5, L9 FP4 Down GEMM fp16 overflow: by capping the
// silu*up magnitude, we bound the downstream accumulator sum.
// Semantic change vs silu_mul_fp16: outliers past max_abs are clipped
// (lost precision on extreme values, but no Inf / no NaN).
__global__ void silu_mul_clamp_fp16_kernel(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ out, int n, float max_abs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    float v = silu_g * u;
    if (v >  max_abs) v =  max_abs;
    if (v < -max_abs) v = -max_abs;
    out[idx] = __float2half(v);
}

void silu_mul_clamp_fp16(
    const __half* gate, const __half* up,
    __half* out, int n, float max_abs, cudaStream_t stream)
{
    silu_mul_clamp_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        gate, up, out, n, max_abs);
}

// ── BF16 += FP16 (residual accumulator stays BF16, delta is FP16) ─
// Fuses fp16→fp32 cast + bf16→fp32 load + fp32 add + fp32→bf16 store.
// Used to fold the FP4 FFN down-GEMM output (fp16) back into the BF16
// residual stream. Computes:   acc_bf16[i] += delta_fp16[i]
__global__ void residual_add_fp16_to_bf16_kernel(
    __nv_bfloat16* __restrict__ acc,
    const __half* __restrict__ delta, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float a = __bfloat162float(acc[idx]);
    float d = __half2float(delta[idx]);
    acc[idx] = __float2bfloat16(a + d);
}

void residual_add_fp16_to_bf16(
    __nv_bfloat16* acc, const __half* delta, int n, cudaStream_t stream)
{
    residual_add_fp16_to_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        acc, delta, n);
}

// ── Class 1d: BF16 text row gather / scatter for batched und path ──
// gather: dst[2b]   = src[b*Sq + 0]        (row 0 of batch b)
//         dst[2b+1] = src[b*Sq + Sq-1]     (row Sq-1 of batch b)
// scatter: dst[b*Sq + 0]      = src[2b]
//          dst[b*Sq + Sq-1]   = src[2b+1]
// Replaces 2*B separate gpu_copy launches with ONE kernel launch (grid = 2*B).

__global__ void bf16_text_gather_kernel(
    const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
    int Sq, int N)
{
    int row = blockIdx.x;                 // 0 .. 2*B - 1
    int b   = row >> 1;
    int which = row & 1;                  // 0 = first row, 1 = last row
    int src_row = b * Sq + (which ? (Sq - 1) : 0);
    const __nv_bfloat16* s = src + src_row * N;
    __nv_bfloat16* d = dst + row * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) d[i] = s[i];
}

__global__ void bf16_text_scatter_kernel(
    __nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ src,
    int Sq, int N)
{
    int row = blockIdx.x;
    int b   = row >> 1;
    int which = row & 1;
    int dst_row = b * Sq + (which ? (Sq - 1) : 0);
    __nv_bfloat16* d = dst + dst_row * N;
    const __nv_bfloat16* s = src + row * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) d[i] = s[i];
}

void bf16_text_gather(const __nv_bfloat16* src, __nv_bfloat16* dst,
                       int B, int Sq, int N, cudaStream_t stream)
{
    int threads = (N >= 256) ? 256 : ((N + 31) & ~31);
    if (threads <= 0) threads = 32;
    bf16_text_gather_kernel<<<2 * B, threads, 0, stream>>>(src, dst, Sq, N);
}

void bf16_text_scatter(__nv_bfloat16* dst, const __nv_bfloat16* src,
                        int B, int Sq, int N, cudaStream_t stream)
{
    int threads = (N >= 256) ? 256 : ((N + 31) & ~31);
    if (threads <= 0) threads = 32;
    bf16_text_scatter_kernel<<<2 * B, threads, 0, stream>>>(dst, src, Sq, N);
}

// ── Class D: QK per-head rms_norm + RoPE rotate_half fused (BF16 in-place) ──
// Replaces:  fvk.rms_norm(qk, qn_w, ..., rows*NH, HD, eps)
//         +  fwk.rope_rotate_half_bf16(qk, cos, sin, rows, NH, HD)
// with 1 launch. Assumes HD is a multiple of 64 (uses HD/2 threads per block).
// cos_t/sin_t layout: [rope_table_rows, HD] with first HD/2 columns populated
// (same as the existing rope_rotate_half_bf16 kernel — identical semantics).

__global__ void qk_rmsnorm_rope_fused_kernel(
    __nv_bfloat16* __restrict__ qk,      // [rows*NH, HD] bf16 in-place
    const __nv_bfloat16* __restrict__ w, // [HD] per-head norm weight
    const __nv_bfloat16* __restrict__ cos_t,
    const __nv_bfloat16* __restrict__ sin_t,
    int NH, int HD, float eps)
{
    const int row_h = blockIdx.x;         // 0..rows*NH-1 (head-row index)
    const int tid   = threadIdx.x;        // 0..HD/2-1
    const int half  = HD / 2;
    const int s     = row_h / NH;         // sequence-position index (shared across heads)

    __nv_bfloat16* rp = qk + row_h * HD;
    // Pair load
    float lo = __bfloat162float(rp[tid]);
    float hi = __bfloat162float(rp[tid + half]);

    // Per-thread partial ssq
    float ssq = lo * lo + hi * hi;

    // Warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ssq += __shfl_xor_sync(0xffffffff, ssq, off);

    // Cross-warp reduce (HD/2 threads, so num_warps = HD/64 — ≤ 16 for HD up to 1024)
    __shared__ float sh[16];
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = (half + 31) >> 5;
    if (lane == 0) sh[warp] = ssq;
    __syncthreads();
    if (warp == 0) {
        ssq = (tid < num_warps) ? sh[lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ssq += __shfl_xor_sync(0xffffffff, ssq, off);
        if (tid == 0) sh[0] = ssq;
    }
    __syncthreads();
    const float rms = rsqrtf(sh[0] / (float)HD + eps);

    // Apply per-head norm weight
    const float n_lo = lo * rms * __bfloat162float(w[tid]);
    const float n_hi = hi * rms * __bfloat162float(w[tid + half]);

    // RoPE rotate_half
    const int rope_idx = s * HD + tid;
    const float c  = __bfloat162float(cos_t[rope_idx]);
    const float si = __bfloat162float(sin_t[rope_idx]);
    const float out_lo = n_lo * c - n_hi * si;
    const float out_hi = n_hi * c + n_lo * si;

    rp[tid]        = __float2bfloat16(out_lo);
    rp[tid + half] = __float2bfloat16(out_hi);
}

void qk_rmsnorm_rope_fused_bf16(
    __nv_bfloat16* qk, const __nv_bfloat16* w,
    const __nv_bfloat16* cos_t, const __nv_bfloat16* sin_t,
    int rows, int NH, int HD, float eps, cudaStream_t stream)
{
    dim3 grid(rows * NH);
    dim3 block(HD / 2);
    qk_rmsnorm_rope_fused_kernel<<<grid, block, 0, stream>>>(
        qk, w, cos_t, sin_t, NH, HD, eps);
}

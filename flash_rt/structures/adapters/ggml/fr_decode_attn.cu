// Decomposed tiny-M decode attention for the pi0.5 action expert (Thor).
//
// For q_tokens ≤ 16 over a padded f16 KV of token rows, flash attention's
// stream-k kernel plus its fixup pass is slower than the classic
// decomposition the FlashRT torch pipeline uses: one batched QK^T GEMM
// (all GQA query heads share the single KV head, so the K operand batches
// with stride 0), a masked softmax over the KV axis, and one batched PV
// GEMM whose fp32 output lands directly in the flash-attention node's
// [hd, n_head, n_tok] layout via a strided C.
//
// Numerics follow ggml's fattn contract: scores = scale * q.k + mask (f16
// mask, slope 1 as max_bias must be 0), softmax in fp32 with running max.

#include "fr_kernels.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>

namespace ggml_cuda_flashrt {

namespace {

// gather the permuted f32 Q view into contiguous f16 rows [n_head*n_tok, hd]
// (row r = h*n_tok + t), applying nothing else (scale folds into QK alpha)
__global__ void kernel_q_gather_f16(const float * __restrict__ q,
                                    __half * __restrict__ out,
                                    int hd, int n_tok, int n_head,
                                    int64_t s_d, int64_t s_tok, int64_t s_head) {
    const int r = blockIdx.x;          // h*n_tok + t
    const int h = r / n_tok;
    const int t = r % n_tok;
    const float * src = q + (int64_t) h * s_head + (int64_t) t * s_tok;
    __half * dst = out + (int64_t) r * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        dst[d] = __float2half(src[(int64_t) d * s_d]);
    }
}

// in-place masked softmax over rows of [n_head*n_tok, n_kv] f16 scores.
// mask element for (kv, t) at mask + kv + t*mask_stride (f16, -inf on pads).
__global__ void kernel_mask_softmax_f16(__half * __restrict__ scores,
                                        const __half * __restrict__ mask,
                                        int n_kv, int n_tok, int64_t mask_stride) {
    const int r = blockIdx.x;          // h*n_tok + t
    const int t = r % n_tok;
    __half * row = scores + (int64_t) r * n_kv;
    const __half * mrow = mask + (int64_t) t * mask_stride;

    float m = -INFINITY;
    for (int i = threadIdx.x; i < n_kv; i += blockDim.x) {
        const float v = __half2float(row[i]) + __half2float(mrow[i]);
        m = fmaxf(m, v);
    }
    __shared__ float red[32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    }
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < blockDim.x / 32) ? red[threadIdx.x] : -INFINITY;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
        }
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    m = red[0];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n_kv; i += blockDim.x) {
        const float v = __half2float(row[i]) + __half2float(mrow[i]);
        const float e = expf(v - m);
        row[i] = __float2half(e);
        sum += e;
    }
    __shared__ float red2[32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }
    if (threadIdx.x % 32 == 0) red2[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < blockDim.x / 32) ? red2[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, off);
        }
        if (threadIdx.x == 0) red2[0] = v;
    }
    __syncthreads();
    const float inv = 1.0f / red2[0];
    for (int i = threadIdx.x; i < n_kv; i += blockDim.x) {
        row[i] = __float2half(__half2float(row[i]) * inv);
    }
}

} // namespace

int decode_attn_decomposed(void * cublas_handle,
                           const float * q, int64_t q_sd, int64_t q_stok, int64_t q_shead,
                           const void * k_f16_rows,   // [n_kv, hd] f16 rows
                           const void * v_f16_rows,   // [n_kv, hd] f16 rows
                           const void * mask_f16, int64_t mask_stride,
                           float * dst, int64_t dst_stok, int64_t dst_shead,
                           void * q16_ws, void * scores_ws,
                           int hd, int n_tok, int n_head, int n_kv,
                           float scale, cudaStream_t stream) {
    cublasHandle_t handle = (cublasHandle_t) cublas_handle;
    const int R = n_head * n_tok;

    kernel_q_gather_f16<<<R, 128, 0, stream>>>(
        q, (__half *) q16_ws, hd, n_tok, n_head, q_sd, q_stok, q_shead);

    cublasSetStream(handle, stream);
    // scores_col[n_kv, n_tok] per head = K_col^T [n_kv, hd] x Q_col [hd, n_tok]
    const float beta0 = 0.0f;
    cublasStatus_t st = cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        n_kv, n_tok, hd,
        &scale,
        k_f16_rows, CUDA_R_16F, hd, 0,
        q16_ws,     CUDA_R_16F, hd, (int64_t) n_tok * hd,
        &beta0,
        scores_ws,  CUDA_R_16F, n_kv, (int64_t) n_tok * n_kv,
        n_head, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) return -100 - (int) st;

    kernel_mask_softmax_f16<<<R, 256, 0, stream>>>(
        (__half *) scores_ws, (const __half *) mask_f16, n_kv, n_tok, mask_stride);

    // out_col[hd, n_tok] per head (ldc = dst token stride) =
    //   V_col [hd, n_kv] x P_col [n_kv, n_tok]
    const float one = 1.0f;
    st = cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        hd, n_tok, n_kv,
        &one,
        v_f16_rows, CUDA_R_16F, hd, 0,
        scores_ws,  CUDA_R_16F, n_kv, (int64_t) n_tok * n_kv,
        &beta0,
        dst, CUDA_R_32F, (int) dst_stok, dst_shead,
        n_head, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) return -200 - (int) st;

    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

} // namespace ggml_cuda_flashrt

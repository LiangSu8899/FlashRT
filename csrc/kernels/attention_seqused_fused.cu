// ================================================================
// FlashRT — seqused attention with the mask folded into softmax.
//
// attention_qkv_fp16_seqused runs QK^T, a standalone -inf mask
// kernel over rows [valid, S_kv_max), softmax, then AV. The v2 chain
// folds the seqused bound into the softmax kernel itself: max/sum
// run over [0, valid) and positions beyond valid are written as
// zero probabilities (identical to exp(-65504 - max) ~ 0 in the
// reference), removing one kernel per call.
//
// Additive: attention_cublas.cu is untouched.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "attention_seqused_fused.cuh"

namespace {

#define SMV2_WARP_SIZE 32
#define SMV2_MAX_COLS 1024
#define SMV2_ITERS (SMV2_MAX_COLS / SMV2_WARP_SIZE)

// One warp per logits row; numerics match softmax_fp16_kernel
// (fp32 max-subtract, __expf, sum + 1e-8) with the seqused bound
// applied while loading.
__global__ void softmax_fp16_seqused_kernel(
    __half* data, int rows, int cols, const int* __restrict__ seqused_k) {
    int lane = threadIdx.x % SMV2_WARP_SIZE;
    int row = blockIdx.x;
    if (row >= rows) return;

    int valid = seqused_k[0];
    if (valid < 0) valid = 0;
    if (valid > cols) valid = cols;

    __half* src = data + row * cols;

    float reg[SMV2_ITERS];
    float mx = -1e30f;
    #pragma unroll
    for (int it = 0; it < SMV2_ITERS; it++) {
        int c = it * SMV2_WARP_SIZE + lane;
        if (c < valid) {
            reg[it] = __half2float(src[c]);
            mx = fmaxf(mx, reg[it]);
        } else {
            reg[it] = -1e30f;
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));

    float sm = 0.f;
    #pragma unroll
    for (int it = 0; it < SMV2_ITERS; it++) {
        reg[it] = __expf(reg[it] - mx);
        sm += reg[it];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sm += __shfl_xor_sync(0xffffffff, sm, o);

    float inv = 1.f / (sm + 1e-8f);
    #pragma unroll
    for (int it = 0; it < SMV2_ITERS; it++) {
        int c = it * SMV2_WARP_SIZE + lane;
        if (c < cols) {
            // Positions beyond valid get exp(-1e30 - mx) * inv == 0,
            // matching the reference mask + softmax result.
            src[c] = __float2half(c < valid ? reg[it] * inv : 0.f);
        }
    }
}

}  // namespace

void attention_qkv_fp16_seqused_v2(
    cublasHandle_t handle,
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* logits,
    __half* out,
    int S, int S_kv_max, int NH, int HD,
    const int* seqused_k,
    float attn_scale,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);

    float zero = 0.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        S_kv_max, S * NH, HD,
        &attn_scale,
        K, CUDA_R_16F, HD,
        Q, CUDA_R_16F, HD,
        &zero,
        logits, CUDA_R_16F, S_kv_max,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    softmax_fp16_seqused_kernel<<<S * NH, SMV2_WARP_SIZE, 0, stream>>>(
        logits, S * NH, S_kv_max, seqused_k);

    float one = 1.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, S * NH, S_kv_max,
        &one,
        V, CUDA_R_16F, HD,
        logits, CUDA_R_16F, S_kv_max,
        &zero,
        out, CUDA_R_16F, HD,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

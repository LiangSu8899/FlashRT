// ============================================================================
//  FlashRT — MHA attention without the -inf logits pre-fill.
//
//  The plain ``attention_mha_{fp16,bf16}`` kernels softmax over the padded
//  logits width, so callers must pre-fill the whole (NH, max_q, max_kv)
//  scratch with -inf every invocation — a full DRAM sweep per layer. These
//  variants run a column-masked softmax that reads and writes only the
//  valid S_kv columns (row stride = padded width); the PV GEMM already
//  uses k = S_kv, so the padding is never read anywhere and the pre-fill
//  disappears.
//
//  Additive: new symbols only.
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define SMM_WARP 32
#define SMM_MAX_COLS 1024
#define SMM_ITERS (SMM_MAX_COLS / SMM_WARP)

namespace {

template <typename T>
__device__ __forceinline__ float to_f(T v);
template <>
__device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_f(float v);
template <>
__device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// Row-wise softmax over the first ``cols_valid`` of each ``cols_pad``-wide
// row. Padding columns are neither read nor written.
template <typename T>
__global__ void softmax_masked_kernel(T* data, int rows, int cols_pad,
                                      int cols_valid) {
    const int lane = threadIdx.x % SMM_WARP;
    const int row = blockIdx.x;
    if (row >= rows) return;

    T* src = data + (long)row * cols_pad;

    float reg[SMM_ITERS];
    float mx = -1e30f;
    #pragma unroll
    for (int it = 0; it < SMM_ITERS; ++it) {
        const int c = it * SMM_WARP + lane;
        if (c < cols_valid) {
            reg[it] = to_f<T>(src[c]);
            mx = fmaxf(mx, reg[it]);
        } else {
            reg[it] = -1e30f;
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));

    float sum = 0.f;
    #pragma unroll
    for (int it = 0; it < SMM_ITERS; ++it) {
        const int c = it * SMM_WARP + lane;
        if (c < cols_valid) {
            reg[it] = __expf(reg[it] - mx);
            sum += reg[it];
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float inv = 1.0f / sum;

    #pragma unroll
    for (int it = 0; it < SMM_ITERS; ++it) {
        const int c = it * SMM_WARP + lane;
        if (c < cols_valid)
            src[c] = from_f<T>(reg[it] * inv);
    }
}

}  // namespace

extern "C" {

void attention_mha_fp16_masked(
    cublasHandle_t handle,
    const __half* Q, const __half* K, const __half* V,
    __half* logits, __half* out,
    int S_q, int S_kv, int NH, int HD,
    float attn_scale, cudaStream_t stream) {
    cublasSetStream(handle, stream);
    const int S_kv_pad = ((S_kv + 7) / 8) * 8;
    float zero = 0.0f, one = 1.0f;
    const long long strideC = (long long)S_q * S_kv_pad;

    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        S_kv, S_q, HD,
        &attn_scale,
        K, CUDA_R_16F, NH * HD, (long long)HD,
        Q, CUDA_R_16F, NH * HD, (long long)HD,
        &zero,
        logits, CUDA_R_16F, S_kv_pad, strideC,
        NH,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    softmax_masked_kernel<__half><<<NH * S_q, SMM_WARP, 0, stream>>>(
        logits, NH * S_q, S_kv_pad, S_kv);

    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, S_q, S_kv,
        &one,
        V, CUDA_R_16F, NH * HD, (long long)HD,
        logits, CUDA_R_16F, S_kv_pad, strideC,
        &zero,
        out, CUDA_R_16F, NH * HD, (long long)HD,
        NH,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

void attention_mha_bf16_masked(
    cublasHandle_t handle,
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* logits, __nv_bfloat16* out,
    int S_q, int S_kv, int NH, int HD,
    float attn_scale, int logits_kv_stride, int qkv_token_stride,
    cudaStream_t stream) {
    cublasSetStream(handle, stream);
    const int S_kv_pad = ((S_kv + 7) / 8) * 8;
    const int kv_stride = (logits_kv_stride > 0) ? logits_kv_stride : S_kv_pad;
    // Token stride (elements) of the Q/K/V sources. NH*HD for packed
    // per-site buffers; 3*NH*HD lets the fused-QKV GEMM output be read in
    // place (no split copies).
    const int tstride = (qkv_token_stride > 0) ? qkv_token_stride : NH * HD;
    float zero = 0.0f, one = 1.0f;
    const long long strideC = (long long)S_q * kv_stride;

    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        S_kv, S_q, HD,
        &attn_scale,
        K, CUDA_R_16BF, tstride, (long long)HD,
        Q, CUDA_R_16BF, tstride, (long long)HD,
        &zero,
        logits, CUDA_R_16BF, kv_stride, strideC,
        NH,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    softmax_masked_kernel<__nv_bfloat16><<<NH * S_q, SMM_WARP, 0, stream>>>(
        logits, NH * S_q, kv_stride, S_kv);

    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, S_q, S_kv,
        &one,
        V, CUDA_R_16BF, tstride, (long long)HD,
        logits, CUDA_R_16BF, kv_stride, strideC,
        &zero,
        out, CUDA_R_16BF, NH * HD, (long long)HD,
        NH,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

}  // extern "C"

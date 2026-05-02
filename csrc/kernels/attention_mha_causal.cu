// ================================================================
// FlashRT — Causal MHA batched cuBLAS attention (for Qwen3-VL LLM
//                                                   in GROOT N1.7)
//
// Same layout & GEMM strategy as ``attention_mha_fp16`` but applies a
// strict-upper-triangular mask between QK^T and softmax via the
// dedicated ``softmax_causal_fp16`` kernel. The N1.7 LLM is causal
// (HF ``Qwen3VLTextAttention.is_causal=True``) and the existing
// non-causal path drops cosine ~0.005 vs the fp32 reference.
//
// New kernel — does NOT modify ``attention_mha_fp16``.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "softmax.cuh"

void attention_mha_causal_fp16(
    cublasHandle_t handle,
    const __half* Q,         // (S_q, NH*HD) — head-interleaved
    const __half* K,         // (S_kv, NH*HD)
    const __half* V,         // (S_kv, NH*HD)
    __half* logits,          // (NH * S_q, S_kv_pad) scratch
    __half* out,             // (S_q, NH*HD)
    int S_q, int S_kv, int NH, int HD,
    float attn_scale,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);

    int S_kv_pad = ((S_kv + 7) / 8) * 8;
    float zero = 0.0f;
    long long strideC = (long long)S_q * S_kv_pad;

    // QK^T per head — same as attention_mha_fp16
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        S_kv, S_q, HD,
        &attn_scale,
        K, CUDA_R_16F, NH * HD,
        (long long)HD,
        Q, CUDA_R_16F, NH * HD,
        (long long)HD,
        &zero,
        logits, CUDA_R_16F, S_kv_pad,
        strideC,
        NH,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Causal-masked softmax: rows are (NH*S_q), per-row q = row % S_q,
    // mask cols j > q AND cols >= S_kv (pad columns).
    softmax_causal_fp16(logits, NH * S_q, S_kv_pad, S_q, S_kv, stream);

    // Attention @ V — same as attention_mha_fp16.
    float one = 1.0f;
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, S_q, S_kv,
        &one,
        V, CUDA_R_16F, NH * HD,
        (long long)HD,
        logits, CUDA_R_16F, S_kv_pad,
        strideC,
        &zero,
        out, CUDA_R_16F, NH * HD,
        (long long)HD,
        NH,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

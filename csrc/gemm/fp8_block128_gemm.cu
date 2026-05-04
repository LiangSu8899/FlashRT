// SPDX-License-Identifier: Apache-2.0
//
// FP8 block-128 GEMM — Path D implementation (dequant + cuBLASLt BF16
// GEMM). Header: fp8_block128_gemm.cuh.
//
// cuBLASLt setup mirrors csrc/kernels/decoder_fused.cu's
// fp8_gemm_descale_bf16out:
//   * both ops = N (no transpose at the cublasLt layer)
//   * pass weight as cublasLt's A_arg, activation as B_arg
//   * Adesc shape (N, K) ld=N, Bdesc shape (K, M) ld=K, Ddesc shape
//     (N, M) ld=N
// What we logically compute: D_rm[M,N] = A_rm[M,K] @ B_rm[N,K]^T.
//
// All-new file; no edits to the existing fp8_gemm_descale_bf16out
// path. Per-shape algo cache is local to this file (own static table)
// to avoid colliding with the existing g_lt_cache in decoder_fused.cu.

#include "fp8_block128_gemm.cuh"
#include "../quantize/fp8_block128_dequant.cuh"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace flash_rt {
namespace gemm {

namespace {

struct LtKey {
    int M, N, K;
    bool operator==(const LtKey& o) const {
        return M == o.M && N == o.N && K == o.K;
    }
};
struct LtKeyHash {
    size_t operator()(const LtKey& k) const noexcept {
        return (static_cast<size_t>(k.M) * 1315423911u)
             ^ (static_cast<size_t>(k.N) * 2654435761u)
             ^ static_cast<size_t>(k.K);
    }
};

struct CachedLt {
    cublasLtMatmulDesc_t   desc  = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr;
    cublasLtMatrixLayout_t Bdesc = nullptr;
    cublasLtMatrixLayout_t Ddesc = nullptr;
    cublasLtMatmulAlgo_t   algo{};
    bool                   has_algo = false;
};

cublasLtHandle_t                                   g_lt   = nullptr;
void*                                              g_ws   = nullptr;
constexpr size_t                                   g_ws_sz = 64ull << 20;
std::unordered_map<LtKey, CachedLt, LtKeyHash>     g_cache;
std::mutex                                         g_mu;

void ensure_init() {
    if (g_lt) return;
    cublasLtCreate(&g_lt);
    cudaMalloc(&g_ws, g_ws_sz);
}

CachedLt& get_cached(int M, int N, int K) {
    LtKey key{M, N, K};
    auto it = g_cache.find(key);
    if (it != g_cache.end()) return it->second;

    CachedLt cg;
    cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    // We want D_rm[M,N] = act_rm[M,K] @ w_rm[N,K]^T with weights and
    // activations both stored in their natural (X, K) row-major layout
    // (no pre-transpose). cuBLASLt computes col-major D_cm = opA(A) @
    // opB(B). Setting opA=T, opB=N with:
    //   * A = weight pointer (N,K)-row -> col-major view (K,N) ld=K
    //   * B = act    pointer (M,K)-row -> col-major view (K,M) ld=K
    //   * D = out    pointer (M,N)-row -> col-major view (N,M) ld=N
    // gives D_cm[i_N, i_M] = sum_k w_rm[i_N, k] * act_rm[i_M, k]
    // which is exactly D_rm[i_M, i_N].
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasOperation_t opT = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(cg.desc,
        CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(cg.desc,
        CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_16BF, K, N, K);
    cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_16BF, K, M, K);
    cublasLtMatrixLayoutCreate(&cg.Ddesc, CUDA_R_16BF, N, M, N);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws_sz = g_ws_sz;
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_sz, sizeof(ws_sz));

    cublasLtMatmulHeuristicResult_t result{};
    int returned = 0;
    auto status = cublasLtMatmulAlgoGetHeuristic(
        g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Ddesc, cg.Ddesc,
        pref, /*requestedAlgoCount=*/1, &result, &returned);
    cublasLtMatmulPreferenceDestroy(pref);

    if (status == CUBLAS_STATUS_SUCCESS && returned > 0) {
        cg.algo = result.algo;
        cg.has_algo = true;
    } else {
        std::fprintf(stderr,
            "[fp8_block128_gemm_descale_bf16out] heuristic FAILED for "
            "M=%d N=%d K=%d (status=%d returned=%d)\n",
            M, N, K, (int)status, returned);
    }
    auto inserted = g_cache.emplace(key, cg).first;
    return inserted->second;
}

}  // namespace

void fp8_block128_gemm_descale_bf16out(
    const void* A_fp8,
    const void* B_fp8,
    void*       D_bf16,
    int M, int N, int K,
    const float* act_block_scale,
    const float* w_block_scale,
    void* scratch_A_bf16,
    void* scratch_B_bf16,
    cudaStream_t stream)
{
    // 1. dequant FP8 weight (N, K) -> bf16 scratch_B_bf16 in place.
    flash_rt::quantize::fp8_block128_dequantize_to_bf16(
        B_fp8, w_block_scale, scratch_B_bf16, N, K, stream);

    // 2. dequant FP8 activation (M, K) -> bf16 scratch_A_bf16.
    flash_rt::quantize::fp8_per_token_block128_dequantize_to_bf16(
        A_fp8, act_block_scale, scratch_A_bf16, M, K, stream);

    // 3. cuBLASLt BF16 GEMM with the swap idiom (see decoder_fused.cu
    //    fp8_gemm_descale_bf16out for the same convention).
    {
        std::lock_guard<std::mutex> lk(g_mu);
        ensure_init();
    }
    auto& cg = get_cached(M, N, K);
    if (!cg.has_algo) {
        std::fprintf(stderr,
            "[fp8_block128_gemm_descale_bf16out] no algo for "
            "M=%d N=%d K=%d; output left undefined\n", M, N, K);
        return;
    }
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(
        g_lt, cg.desc,
        &alpha,
        scratch_B_bf16, cg.Adesc,    // weight as cublasLt-A
        scratch_A_bf16, cg.Bdesc,    // act    as cublasLt-B
        &beta,
        D_bf16, cg.Ddesc,
        D_bf16, cg.Ddesc,
        &cg.algo,
        g_ws, g_ws_sz,
        stream);
}

}  // namespace gemm
}  // namespace flash_rt

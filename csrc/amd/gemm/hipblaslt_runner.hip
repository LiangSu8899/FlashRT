#include "hipblaslt_runner.h"
#include <iostream>
#include <vector>

// ================================================================
// GemmRunner (AMD): hipBLASLt-based GEMM for BF16/FP8 on CDNA
// Direct C++ calls, zero Python overhead.
//
// Port of csrc/gemm/gemm_runner.cu (cuBLASLt). Math semantics are
// identical; the one structural difference is the layout encoding:
//
//   cuBLASLt side: descriptors carry CUBLASLT_ORDER_ROW, so the
//   row-major matrices are described directly and operands are
//   passed in FlashRT order (A first, B second).
//
//   hipBLASLt side: matmul supports column-major layouts only, so
//   every entry point here uses the standard col-major swap, which
//   is bit-for-bit the same computation:
//       D_row(M,N) = A_row(M,K) @ B_row(K,N)
//   ==  D_col(N,M) = B_col(N,K) @ A_col(K,M)
//   where X_col is the same memory as X_row reinterpreted col-major.
//   Consequently hipBLASLt operand 0 ("A" in the library API) is the
//   FlashRT B matrix and operand 1 ("B") is the FlashRT A matrix.
//   The CUDA file itself already uses this exact trick for the pi05
//   FP8 epilogue paths (gemm_runner.cu:1139-1141 create A_desc from
//   the B matrix and call cublasLtMatmul(..., B, e.A_desc, A,
//   e.B_desc, ...) at gemm_runner.cu:1155).
//
// FP8 descale semantics (mirrors gemm_runner.cu:1089-1092):
//   cuBLASLt applies *(A_SCALE_POINTER) to operand A and
//   *(B_SCALE_POINTER) to operand B as per-tensor FP32 multipliers on
//   the FP32 accumulator — NOT via alpha and NOT in the epilogue.
//   hipBLASLt's HIPBLASLT_MATMUL_DESC_A/B_SCALE_POINTER attributes
//   have the same contract. Because of the operand swap above, the
//   FlashRT d_scale_a (activations) is attached to hipBLASLt operand
//   "B" and d_scale_b (weights) to operand "A". The product
//   scale_a * scale_b * (A@B) is unchanged.
//
// FP8 dtype: OCP e4m3 (HIP_R_8F_E4M3) — gfx950 native. Never the
// fnuz variant. BF16 = HIP_R_16BF. Compute type = HIPBLAS_COMPUTE_32F
// with FP32 scale type, matching CUBLAS_COMPUTE_32F / CUDA_R_32F.
// ================================================================

GemmRunner::GemmRunner() {
    HIPBLASLT_CHECK(hipblasLtCreate(&handle_));
    workspace_size_ = 256 * 1024 * 1024;  // 256 MB workspace (enables split-K etc.)
    HIP_CHECK(hipMalloc(&workspace_, workspace_size_));
    // NOTE: the CUDA ctor also pre-allocates d_scale_a_/d_scale_b_ device
    // scalars for fp8_run (host-scale variant). fp8_run is not in the
    // ported subset, so that scratch is omitted here.
}

GemmRunner::~GemmRunner() {
    // Destroy cached descriptors
    for (auto& [key, entry] : gemm_cache_) {
        hipblasLtMatrixLayoutDestroy(entry.op0_desc);
        hipblasLtMatrixLayoutDestroy(entry.op1_desc);
        hipblasLtMatrixLayoutDestroy(entry.D_desc);
        hipblasLtMatmulDescDestroy(entry.matmul_desc);
    }
    gemm_cache_.clear();
    if (workspace_) hipFree(workspace_);
    if (handle_) hipblasLtDestroy(handle_);
}

// ================================================================
// Descriptor cache: create once, reuse across all calls
//
// Col-major swap layouts (op0 = FlashRT B, op1 = FlashRT A):
//   NN  D_row(M,N) = A_row(M,K) @ B_row(K,N)
//       op0 = B_col(N,K) ld=N, TRANSA=N
//       op1 = A_col(K,M) ld=K, TRANSB=N
//       D   = D_col(N,M) ld=N
//   NT  D_row(M,N) = A_row(M,K) @ B_row(N,K)^T
//       op0 = B_col(K,N) ld=K, TRANSA=T  → op(A)=(N,K)
//       op1 = A_col(K,M) ld=K, TRANSB=N  → op(B)=(K,M)
//       D   = D_col(N,M) ld=N
// ================================================================
GemmRunner::CachedGemm& GemmRunner::get_or_create_cached(GemmType type, int M, int N, int K) {
    GemmKey key{static_cast<int>(type), M, N, K};
    auto it = gemm_cache_.find(key);
    if (it != gemm_cache_.end()) return it->second;

    CachedGemm entry;
    hipblasOperation_t op_N = HIPBLAS_OP_N;
    hipblasOperation_t op_T = HIPBLAS_OP_T;

    if (type == FP8_NN_DEV) {
        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&entry.matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op0_desc, HIP_R_8F_E4M3, N, K, N));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op1_desc, HIP_R_8F_E4M3, K, M, K));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.D_desc, HIP_R_16BF, N, M, N));
    } else if (type == FP8_NT_DEV) {
        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&entry.matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_T, sizeof(op_T)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op0_desc, HIP_R_8F_E4M3, K, N, K));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op1_desc, HIP_R_8F_E4M3, K, M, K));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.D_desc, HIP_R_16BF, N, M, N));
    } else {
        // BF16_NN
        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&entry.matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op0_desc, HIP_R_16BF, N, K, N));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op1_desc, HIP_R_16BF, K, M, K));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.D_desc, HIP_R_16BF, N, M, N));
    }

    // Get heuristic top-1 algorithm as default
    hipblasLtMatmulPreference_t preference;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    hipblasLtMatmulHeuristicResult_t heuristic;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle_, entry.matmul_desc,
        entry.op0_desc, entry.op1_desc, entry.D_desc, entry.D_desc,
        preference, 1, &heuristic, &returned_results));
    hipblasLtMatmulPreferenceDestroy(preference);

    if (returned_results == 0) {
        throw std::runtime_error("hipBLASLt: no algorithm found for cached GEMM");
    }
    entry.algo = heuristic.algo;

    auto [inserted_it, _] = gemm_cache_.emplace(key, entry);
    return inserted_it->second;
}

// ================================================================
// Autotune: benchmark top-N algorithms and pick the fastest
// (mirrors gemm_runner.cu:148-227; op0/op1 are in hipBLASLt operand
// order, i.e. already swapped by the public wrappers)
// ================================================================
void GemmRunner::autotune_cached(CachedGemm& entry, void* op0, void* op1, void* D,
                                  float alpha, float beta, int num_algos,
                                  float* scale_op0, float* scale_op1) {
    // Update scale pointers if FP8
    if (scale_op0) {
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
            HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_op0, sizeof(scale_op0)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
            HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_op1, sizeof(scale_op1)));
    }

    hipblasLtMatmulPreference_t preference;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristics(num_algos);
    int returned_results = 0;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle_, entry.matmul_desc,
        entry.op0_desc, entry.op1_desc, entry.D_desc, entry.D_desc,
        preference, num_algos, heuristics.data(), &returned_results));
    hipblasLtMatmulPreferenceDestroy(preference);

    if (returned_results == 0) {
        std::cerr << "  autotune: no algorithms found, keeping default" << std::endl;
        return;
    }

    // Benchmark each algorithm
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    float best_ms = 1e9f;
    int best_idx = 0;
    const int warmup_iters = 3;
    const int bench_iters = 10;

    for (int i = 0; i < returned_results; ++i) {
        // Warmup
        bool algo_ok = true;
        for (int w = 0; w < warmup_iters; ++w) {
            hipblasStatus_t st = hipblasLtMatmul(handle_, entry.matmul_desc,
                &alpha, op0, entry.op0_desc, op1, entry.op1_desc,
                &beta, D, entry.D_desc, D, entry.D_desc,
                &heuristics[i].algo, workspace_, workspace_size_, 0);
            if (st != HIPBLAS_STATUS_SUCCESS) { algo_ok = false; break; }
        }
        if (!algo_ok) continue;
        HIP_CHECK(hipDeviceSynchronize());

        // Benchmark
        HIP_CHECK(hipEventRecord(start));
        for (int b = 0; b < bench_iters; ++b) {
            hipblasLtMatmul(handle_, entry.matmul_desc,
                &alpha, op0, entry.op0_desc, op1, entry.op1_desc,
                &beta, D, entry.D_desc, D, entry.D_desc,
                &heuristics[i].algo, workspace_, workspace_size_, 0);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        ms /= bench_iters;

        if (ms < best_ms) {
            best_ms = ms;
            best_idx = i;
        }
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    entry.algo = heuristics[best_idx].algo;
    std::cout << "  autotune: tested " << returned_results << " algos, best="
              << best_idx << " (" << best_ms * 1000.0f << " us)" << std::endl;
}

void GemmRunner::autotune_bf16_nn(void* A, void* B, void* D,
                                   int M, int N, int K, int num_algos) {
    auto& entry = get_or_create_cached(BF16_NN, M, N, K);
    autotune_cached(entry, B, A, D, 1.0f, 0.0f, num_algos);
}

void GemmRunner::autotune_fp8_nn_dev(void* A, void* B, void* D,
                                      int M, int N, int K,
                                      float* d_scale_a, float* d_scale_b,
                                      int num_algos) {
    auto& entry = get_or_create_cached(FP8_NN_DEV, M, N, K);
    // Operand swap: op0 = B (weights, scale_b), op1 = A (acts, scale_a)
    autotune_cached(entry, B, A, D, 1.0f, 0.0f, num_algos, d_scale_b, d_scale_a);
}

void GemmRunner::autotune_fp8_nt_dev(void* A, void* B, void* D,
                                      int M, int N, int K,
                                      float* d_scale_a, float* d_scale_b,
                                      int num_algos) {
    auto& entry = get_or_create_cached(FP8_NT_DEV, M, N, K);
    autotune_cached(entry, B, A, D, 1.0f, 0.0f, num_algos, d_scale_b, d_scale_a);
}

// ================================================================
//  BF16 inference run: D = A(M,K) @ B(N,K)^T  (no timing, no sync)
//  Mirrors gemm_runner.cu:670-720 (per-call descriptors, heuristic
//  top-1, alpha=1 beta=0).
// ================================================================
void GemmRunner::bf16_run(void* A, void* B, void* D,
                          int M, int N, int K, hipStream_t stream) {
    hipblasLtMatmulDesc_t matmul_desc;
    hipblasLtMatrixLayout_t op0_desc, op1_desc, D_desc;

    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t op_N = HIPBLAS_OP_N;
    hipblasOperation_t op_T = HIPBLAS_OP_T;

    // Col-major swap, NT: op0 = B_col(K,N) with OP_T, op1 = A_col(K,M) with OP_N
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_T, sizeof(op_T)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op0_desc, HIP_R_16BF, K, N, K));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op1_desc, HIP_R_16BF, K, M, K));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&D_desc, HIP_R_16BF, N, M, N));

    hipblasLtMatmulPreference_t preference;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    hipblasLtMatmulHeuristicResult_t heuristic;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        op0_desc, op1_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        hipblasLtMatmulPreferenceDestroy(preference);
        hipblasLtMatrixLayoutDestroy(op0_desc);
        hipblasLtMatrixLayoutDestroy(op1_desc);
        hipblasLtMatrixLayoutDestroy(D_desc);
        hipblasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("hipBLASLt bf16_run: no algorithm found");
    }

    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, matmul_desc,
        &alpha, B, op0_desc, A, op1_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace_, workspace_size_, stream));

    hipblasLtMatmulPreferenceDestroy(preference);
    hipblasLtMatrixLayoutDestroy(op0_desc);
    hipblasLtMatrixLayoutDestroy(op1_desc);
    hipblasLtMatrixLayoutDestroy(D_desc);
    hipblasLtMatmulDescDestroy(matmul_desc);
}

// ================================================================
//  BF16 no-transpose: D(M,N) = A(M,K) @ B(K,N)  (row-major)
//  B is NOT transposed. For Pi05 weights stored as (K, N).
//  Mirrors gemm_runner.cu:726-734 (cached descriptor + algo).
// ================================================================
void GemmRunner::bf16_nn(void* A, void* B, void* D,
                         int M, int N, int K, hipStream_t stream) {
    auto& entry = get_or_create_cached(BF16_NN, M, N, K);
    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, entry.matmul_desc,
        &alpha, B, entry.op0_desc, A, entry.op1_desc,
        &beta, D, entry.D_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

// ================================================================
//  BF16 no-transpose with BIAS epilogue: D(M,N) = A(M,K) @ B(K,N) + bias(N)
//  Uses EPILOGUE_BIAS to fuse bias addition into the GEMM.
//  With the col-major swap, D_col is (N,M), so the epilogue bias vector
//  (length = rows of D_col = N, broadcast across columns) lands exactly
//  as bias(N) broadcast over the M rows — same math as the CUDA
//  row-major-order version (gemm_runner.cu:769-825).
// ================================================================
void GemmRunner::bf16_nn_bias(void* A, void* B, void* D, void* bias,
                               int M, int N, int K, hipStream_t stream) {
    hipblasLtMatmulDesc_t matmul_desc;
    hipblasLtMatrixLayout_t op0_desc, op1_desc, D_desc;

    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t op_N = HIPBLAS_OP_N;

    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

    // Set BIAS epilogue
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    hipDataType bias_type = HIP_R_16BF;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));

    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op0_desc, HIP_R_16BF, N, K, N));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op1_desc, HIP_R_16BF, K, M, K));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&D_desc, HIP_R_16BF, N, M, N));

    hipblasLtMatmulPreference_t preference;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    hipblasLtMatmulHeuristicResult_t heuristic;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        op0_desc, op1_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        hipblasLtMatmulPreferenceDestroy(preference);
        hipblasLtMatrixLayoutDestroy(op0_desc);
        hipblasLtMatrixLayoutDestroy(op1_desc);
        hipblasLtMatrixLayoutDestroy(D_desc);
        hipblasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("hipBLASLt bf16_nn_bias: no algorithm found");
    }

    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, matmul_desc,
        &alpha, B, op0_desc, A, op1_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace_, workspace_size_, stream));

    hipblasLtMatmulPreferenceDestroy(preference);
    hipblasLtMatrixLayoutDestroy(op0_desc);
    hipblasLtMatrixLayoutDestroy(op1_desc);
    hipblasLtMatrixLayoutDestroy(D_desc);
    hipblasLtMatmulDescDestroy(matmul_desc);
}

// ================================================================
//  BF16 GEMM + BIAS + GELU epilogue: D(M,N) = GELU(A(M,K) @ B(K,N) + bias(N))
//  Fuses GEMM + bias + GELU into a single hipBLASLt launch.
//  GELU_BIAS in both libraries computes gelu(x + bias) with the tanh
//  approximation (mirrors gemm_runner.cu:831-887).
// ================================================================
void GemmRunner::bf16_nn_bias_gelu(void* A, void* B, void* D, void* bias,
                                    int M, int N, int K, hipStream_t stream) {
    hipblasLtMatmulDesc_t matmul_desc;
    hipblasLtMatrixLayout_t op0_desc, op1_desc, D_desc;

    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t op_N = HIPBLAS_OP_N;

    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

    // Set GELU_BIAS epilogue
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_GELU_BIAS;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    hipDataType bias_type = HIP_R_16BF;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));

    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op0_desc, HIP_R_16BF, N, K, N));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op1_desc, HIP_R_16BF, K, M, K));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&D_desc, HIP_R_16BF, N, M, N));

    hipblasLtMatmulPreference_t preference;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    hipblasLtMatmulHeuristicResult_t heuristic;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        op0_desc, op1_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        hipblasLtMatmulPreferenceDestroy(preference);
        hipblasLtMatrixLayoutDestroy(op0_desc);
        hipblasLtMatrixLayoutDestroy(op1_desc);
        hipblasLtMatrixLayoutDestroy(D_desc);
        hipblasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("hipBLASLt bf16_nn_bias_gelu: no algorithm found");
    }

    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, matmul_desc,
        &alpha, B, op0_desc, A, op1_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace_, workspace_size_, stream));

    hipblasLtMatmulPreferenceDestroy(preference);
    hipblasLtMatrixLayoutDestroy(op0_desc);
    hipblasLtMatrixLayoutDestroy(op1_desc);
    hipblasLtMatrixLayoutDestroy(D_desc);
    hipblasLtMatmulDescDestroy(matmul_desc);
}

// ================================================================
//  FP8 no-transpose: D_bf16 = A_fp8(M,K) @ B_fp8(K,N)
//  with device scale pointers (HIP Graph compatible).
//  Matches bf16_nn layout — B stored as (K,N) row-major, no transpose.
//  Mirrors gemm_runner.cu:1083-1099: descale enters via the library's
//  A/B_SCALE_POINTER accumulator scaling, alpha stays 1.0.
// ================================================================
void GemmRunner::fp8_nn_dev(void* A, void* B, void* D,
                             int M, int N, int K,
                             float* d_scale_a, float* d_scale_b,
                             hipStream_t stream) {
    auto& entry = get_or_create_cached(FP8_NN_DEV, M, N, K);
    // Update scale pointers per-call (different layers have different scales).
    // Operand swap: hipBLASLt operand A = FlashRT B → gets d_scale_b.
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
        HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_b, sizeof(d_scale_b)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
        HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_a, sizeof(d_scale_a)));

    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, entry.matmul_desc,
        &alpha, B, entry.op0_desc, A, entry.op1_desc,
        &beta, D, entry.D_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

// ================================================================
//  FP8 transpose-B path: D_bf16 = A_fp8(M,K) @ B_fp8(N,K)^T
//  with device scale pointers. B is stored as (N,K) row-major.
//  Mirrors gemm_runner.cu:1104-1119.
// ================================================================
void GemmRunner::fp8_nt_dev(void* A, void* B, void* D,
                             int M, int N, int K,
                             float* d_scale_a, float* d_scale_b,
                             hipStream_t stream) {
    auto& entry = get_or_create_cached(FP8_NT_DEV, M, N, K);
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
        HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_b, sizeof(d_scale_b)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
        HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_a, sizeof(d_scale_a)));

    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, entry.matmul_desc,
        &alpha, B, entry.op0_desc, A, entry.op1_desc,
        &beta, D, entry.D_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

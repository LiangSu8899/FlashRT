// FlashRT — Patch Embedding GEMM API (standalone .so)
//
// Compile:
//   nvcc -O3 -std=c++17 -arch=sm_110a --use_fast_math \
//        -shared -Xcompiler -fPIC patch_embed_api.cu -o libpatch_embed.so -lcublas
//
// C API:
//   patch_embed(patches, pe_w, pe_b, pos_emb, output, nv, S_per_view, D, K, stream)
//
// Computes: output = patches @ pe_w + pe_b + pos_emb
// All pointers are device (cudaMalloc). FP16 throughout.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>

static cublasHandle_t g_handle = nullptr;
static bool g_handle_init = false;

static void ensure_handle(cudaStream_t stream) {
    if (!g_handle_init) {
        cublasStatus_t st = cublasCreate(&g_handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[patch_embed] cublasCreate failed: %d\n", (int)st);
            return;
        }
        g_handle_init = true;
    }
    cublasSetStream(g_handle, stream);
}

// Kernel: add bias + positional embedding
// output[i,j] += bias[j] + pos_emb[i % S_per_view, j]
__global__ void add_bias_pos_k(half* output, const half* bias, const half* pos_emb,
                                 int rows, int cols, int S_per_view) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int i = idx / cols;
    int j = idx % cols;
    int pos_i = i % S_per_view;
    float v = __half2float(output[idx])
            + __half2float(bias[j])
            + __half2float(pos_emb[pos_i * cols + j]);
    output[idx] = __float2half(v);
}

extern "C" {

/// Patch embedding: output = patches @ pe_w + pe_b + pos_emb
///
/// @param patches   half* device, (S, K) where S = nv * S_per_view
/// @param pe_w      half* device, (K, D) — patch embedding weight (transposed for GEMM)
/// @param pe_b      half* device, (D,) — patch embedding bias
/// @param pos_emb   half* device, (S_per_view, D) — positional embedding
/// @param output    half* device, (S, D) — output buffer
/// @param S         total sequence length (nv * S_per_view)
/// @param S_per_view tokens per view (256)
/// @param D         embedding dimension (1152)
/// @param K         patch dimension (588 = 14*14*3)
/// @param stream    cudaStream_t
/// @return 0 on success
int patch_embed(void* patches, void* pe_w, void* pe_b, void* pos_emb,
                void* output, int S, int S_per_view, int D, int K,
                void* stream_ptr) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    ensure_handle(stream);

    // GEMM: output = patches @ pe_w
    // cublasGemmEx: C = alpha * A * B + beta * C
    // A = patches (S, K), B = pe_w (K, D), C = output (S, D)
    // cuBLAS uses column-major, compute: C^T = B^T * A^T
    // For row-major: use NN with swapped dims
    const __half alpha_h = __float2half(1.0f);
    const __half beta_h = __float2half(0.0f);

    // Row-major GEMM: C(S,D) = A(S,K) * B(K,D)
    // In cuBLAS column-major: C^T(D,S) = B^T(D,K) * A^T(K,S)
    cublasStatus_t stat = cublasGemmEx(
        g_handle,
        CUBLAS_OP_N,     // B^T is already (D,K) if B is (K,D) row-major → read as (D,K) col-major = N
        CUBLAS_OP_N,     // A^T is already (K,S) if A is (S,K) row-major → read as (K,S) col-major = N
        D, S, K,         // m, n, k (output is m×n = D×S in col-major)
        &alpha_h,
        pe_w, CUDA_R_16F, D,    // B: (D, K) col-major, ldb=D
        patches, CUDA_R_16F, K, // A: (K, S) col-major, lda=K
        &beta_h,
        output, CUDA_R_16F, D,  // C: (D, S) col-major, ldc=D
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        return (int)stat;
    }

    // Add bias + pos_emb
    int total = S * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_pos_k<<<blocks, threads, 0, stream>>>(
        (half*)output, (const half*)pe_b, (const half*)pos_emb,
        S, D, S_per_view);

    return 0;
}

}  // extern "C"

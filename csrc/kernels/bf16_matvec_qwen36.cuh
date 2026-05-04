// bf16 row-major matvec kernel for Qwen3.6.
//
// Computes out[n] = sum_k x[k] * W[n][k] for n in [0, N), bf16
// throughout, fp32 internal accumulation. Designed for the
// Qwen3.6 hot path where:
//   - in_proj_a / in_proj_b: K=5120, N=48
//   - lm_head (untied):     K=5120, N=vocab=248320
//
// Why a custom kernel: cuBLASLt's per-stream and per-graph-context
// heuristic chooses different GEMM algorithms with different bf16
// reduction orders, breaking CUDA Graph correctness. This kernel
// is purely deterministic (each thread sums in fixed K-order with
// fp32 fma) and stream-invariant (no cuBLAS handle / workspace).
//
// Shapes:
//   x   : (K,)        bf16
//   W   : (N, K)      bf16, row-major
//   out : (N,)        bf16
//
// Launch: ceil(N / 256) blocks of 256 threads. Each thread computes
// one output element.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt::kernels {

void bf16_matvec_qwen36_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* out,
    int N,
    int K,
    cudaStream_t stream);

}  // namespace flash_rt::kernels

// ============================================================================
//  FlashRT — NVFP4 GEMM with FP4 (e2m1) packed output + SFA tile-interleaved.
//  Used by the P1 split-GU FFN path: gate_proj and up_proj both produce FP4
//  buffers that the next-step silu_mul kernel consumes directly.
//
//  D[M, N/2]    = packed e2m1 NVFP4
//  D_SFA        = CUTLASS Sm1xxBlkScaledConfig SFA layout, UE4M3 scales
//  Epilogue     = LinCombBlockScaleFactor (proven pattern, examples 72b/79b)
//
//  Additive: does NOT modify existing cutlass_fp4_gemm.cu / variants.
// ============================================================================
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// FP4-out NVFP4 GEMM:  D = quant_block(X @ W^T)  (alpha=1, beta=0)
//
// A: NVFP4 packed activation [M, K/2]  + SFA  (CUTLASS layout)
// B: NVFP4 packed weight     [N, K/2]  + SFB  (CUTLASS layout, ColumnMajor effective)
// D: NVFP4 packed output     [M, N/2]  + SFD  (CUTLASS layout = SFA-shape for next GEMM)
//
// Returns 0 on success; nonzero CUTLASS Status code on error.
int cutlass_fp4_gemm_fp4out(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void*       D_packed,
    void*       D_SFD,            // = next-layer SFA
    int M, int N, int K,
    cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt

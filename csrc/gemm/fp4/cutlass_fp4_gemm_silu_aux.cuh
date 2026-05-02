// ============================================================================
//  FlashVLA — P1 NVFP4 GEMM with fused silu(aux) * acc → fp4 + SFA epilogue.
//
//  Kernel signature for the second leg of the split-GU FFN path:
//
//     gate_acc = X @ Wg^T           (separate normal NVFP4 GEMM; output fp16)
//     out_fp4  = NVFP4 GEMM(X, Wu) with epilogue:
//                  load gate_acc (aux fp16, RowMajor [M, N])
//                  silu(gate_acc) * acc   in fp32 registers
//                  → block-scale (SFVecSize=16, UE4M3 SF)
//                  → pack to e2m1
//                  → write packed[M, N/2] + SFA tile-interleaved
//
//  This eliminates the standalone F4 v2 / F4 v2+mul kernel (143 μs/layer)
//  by absorbing silu_mul + fp4 quant + SFA write into the GEMM epilogue.
//  Expected per-layer saving: -100 to -150 μs * 18 = -1.8 to -2.7 ms E2E.
//
//  Status: WIP scaffold (not yet wired into build). See
//  docs/v2/fp4_p1_day1_design.md for the EVT plan.
//
//  Additive: does NOT modify existing cutlass_fp4_gemm.cu / variants.
// ============================================================================
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// Run NVFP4 GEMM with fused silu_mul aux epilogue.
//
//   D[M, N/2] (fp4 packed)  =  blockscale(silu(aux_gate[M, N]) * acc)
//   SFA out                  =  per-16-element UE4M3 scales, CUTLASS SFA layout
//
// Inputs are NVFP4 quantized (A: act packed + SFA, B: weight packed + SFB).
// Aux gate is fp16 row-major [M, N] device buffer.
//
// Returns 0 on success, nonzero CUTLASS status code on error.
//
// NOTE (Day 1): function declared but not yet implemented. .cu file contains
// a WIP custom EVT that currently does NOT compile. See design doc.
int cutlass_fp4_gemm_silu_aux_fp4(
    void const* A_packed, void const* SFA,           // A operand (NVFP4)
    void const* B_packed, void const* SFB,           // B operand (NVFP4 weight)
    void const* aux_gate_fp16,                       // [M, N] fp16 gate
    void*       D_packed,                            // [M, N/2] fp4 out
    void*       D_SFA,                               // SFA tile-interleaved out
    int M, int N, int K,
    cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt

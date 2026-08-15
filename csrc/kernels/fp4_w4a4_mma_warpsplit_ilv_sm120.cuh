// SPDX-License-Identifier: Apache-2.0
//
// Interleaved-B warp-split-K NVFP4 W4A4 M=1 GEMV for sm_120, plus the
// bind-time B repack that produces its layout. Bit-exact vs the base
// warpsplit kernel (identical reduction order); serves the wide decode
// shapes at 89-91% of the achievable DRAM read roof. Additive.
#pragma once
#include <cuda_runtime.h>
namespace flash_rt {
namespace gemm {
// A_packed (K/2,), B_ilv = interleaved B from fp4_w4a4_repack_b_ilv_sm120,
// D_bf16 (N,). SFA (K/16,) and SFB (N, K/16) keep the base kernel's
// swizzled layouts. warps in {2,4,8}, stages in {3,4,6}. N%8==0, K%64==0,
// (K/64)%warps==0. Returns 0 on success.
int fp4_w4a4_mma_sm120_warpsplit_ilv_bf16out(
    const void* A_packed, const void* B_ilv, void* D_bf16, int N, int K,
    const void* SFA, const void* SFB, float alpha, int warps, int stages,
    cudaStream_t stream);
// dense packed B (N, K/2 row-major) -> interleaved layout
// [N/8][K/64][8 cols x 32B]. dst size equals src size (N*K/2 bytes).
int fp4_w4a4_repack_b_ilv_sm120(const void* B_packed, void* B_ilv, int N,
                                int K, cudaStream_t stream);
}  // namespace gemm
}  // namespace flash_rt

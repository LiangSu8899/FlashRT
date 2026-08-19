// SPDX-License-Identifier: Apache-2.0
//
// Multi-row (M<=16) interleaved-B warp-split-K NVFP4 W4A4 GEMM for sm_120
// — the spec-verify shapes (M = draft block + 1, and the shorter re-advance
// prefixes). The 16x8x64 block-scaled MMA atom already produces a 16-row
// output tile; the M=1 interleaved GEMV feeds it one live row and fifteen
// rows of zero ballast. This entry feeds up to 16 live rows through the
// same atom, same shared-memory plan, same accumulator budget: B stays the
// only DRAM-bound stream (A is M*K/2 bytes, L2-resident across the grid),
// so a multi-row call lands near single-GEMV cost. Bit-exact per row vs
// the M=1 interleaved kernel (identical reduction order). Additive.
#pragma once
#include <cuda_runtime.h>
namespace flash_rt {
namespace gemm {
// A_packed (M, K/2) row-major; B_ilv = interleaved B from
// fp4_w4a4_repack_b_ilv_sm120 (shared with the M=1 entry). D_bf16 (M, N)
// row-major. SFA: the production 512B-per-K64-tile block, row r's four
// scales at (tile*512 + r*16). SFB keeps the base swizzled layout.
// warps in {2,4,8}, stages in {3,4,6}. 1<=M<=16, N%8==0, K%64==0,
// (K/64)%warps==0. Returns 0 on success.
int fp4_w4a4_mma_sm120_warpsplit_ilv_mrows_bf16out(
    const void* A_packed, const void* B_ilv, void* D_bf16, int M, int N,
    int K, const void* SFA, const void* SFB, float alpha, int warps,
    int stages, cudaStream_t stream);
}  // namespace gemm
}  // namespace flash_rt

// SPDX-License-Identifier: Apache-2.0
//
// Tile-shape sweep harness for the production SM120a NVFP4 W4A16 GEMM
// (arch::Sm120, OpClassBlockScaledTensorOp, KernelTmaWarpSpecializedCooperative,
// BF16 out) — identical kernel family to fp4_w4a16_gemm_sm120_bf16out, varying
// ONLY the MMA TileShape.  Used to measure, apples-to-apples, the efficiency
// cost of the smaller per-accumulator tiles (e.g. 128x64) that a dual-
// accumulator SwiGLU mainloop would be forced into, and to tune the narrow-N
// qkv/o_proj shapes toward roofline.
//
// Variants (idx -> tile MxNxK, cluster 1x1x1):
//   0: 128x128x256   (production baseline; gate_up 98.9%)
//   1: 128x 64x256   (N-half, same K — dual per-accumulator tile)
//   2: 128x 64x128   (N-half, half K)
//   3:  64x128x256   (M-half alternative)
//   4: 256x128x256   (wide M)
//   5: 128x256x256   (wide N)
//   6:  64x 64x256   (both-half)
//   7: 128x128x128   (half K)

#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// Returns 0 on success, else (cutlass::Status | 0x{1,2,3}0000) like the variant
// harness, so the caller can tell can_implement / init / run failures apart.
int fp4_w4a16_tilesweep_sm120_bf16out(
    int variant,
    const void* A_packed, const void* B_packed, void* D_bf16,
    int M, int N, int K,
    const void* SFA, const void* SFB,
    float alpha, cudaStream_t stream);

int fp4_w4a16_tilesweep_sm120_num_variants();
const char* fp4_w4a16_tilesweep_sm120_name(int variant);

}  // namespace gemm
}  // namespace flash_rt

// SPDX-License-Identifier: Apache-2.0
//
// Even-column FP4 compaction for the SwiGLU epilogue-fold output. See the .cu
// for the full design. Gathers the low nibble of adjacent byte pairs of the
// duplicated [M, intermediate]-byte FP4 output into the packed [M, intermediate]
// (i.e. [M, intermediate/2]-byte) tensor the down GEMM consumes. The SFD tensor
// is passed straight through (already per-16-output, byte-identical to down's
// per-16 SFA).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

//   in_packed  : (M, intermediate)   uint8   dup'd FP4 (silu_mul in both nibbles)
//   out_packed : (M, intermediate/2) uint8   compacted NVFP4
void fp4_swiglu_even_col_compact(
    const void* in_packed, void* out_packed, int M, int inter,
    cudaStream_t stream);

// Vectorized (16B-load / 8B-store) variant of the above — same semantics, higher
// memory-bandwidth efficiency. Falls back to the scalar kernel if inter % 16 != 0.
void fp4_swiglu_even_col_compact_v2(
    const void* in_packed, void* out_packed, int M, int inter,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt

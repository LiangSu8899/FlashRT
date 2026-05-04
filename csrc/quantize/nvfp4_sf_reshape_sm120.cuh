// SPDX-License-Identifier: Apache-2.0
//
// Reshape linear (rows, K/16) FP8 e4m3 group-scale tensor into the
// CUTLASS Sm1xx blockscaled tile-interleaved layout that the SM120
// NVFP4 W4A16 GEMM kernel expects.
//
// This is a hand-coded version of the equivalent CUTLASS-template
// kernel in csrc/quantize/reshape_scales_sfa.cu. We reuse the same
// permutation formula already proven inside
// csrc/kernels/quantize.cu::quantize_bf16_to_nvfp4_swizzled_kernel
// (lines 408-422) — pure memory layout permutation, no CUTLASS
// types in the kernel body, so it compiles standalone on sm_120
// without the NVCC stub-gen bug we hit with the cute::Layout-based
// template kernel.
//
// Layout (verified against the SM120 GEMM by smoke test cos=1.0):
//   Super-atom = 128 rows × 64 SF-cols = 8192 bytes.
//   Within a super-atom, byte at logical (row r, sf-col c) is at
//   inner offset:
//       (r % 32) * 16  +  (r / 32) * 4  +  (c % 4)
//   Super-atom index in the global stream = rb * n_col_super + cb,
//   where rb = r / 128, cb = c / 4. Super-atom offset = idx * 512.
//
//   Total bytes = ceil(rows/128) * ceil((K/16)/4) * 8192.
//
// Same convention works for both SFA (M-side) and SFB (N-side) on
// SM120 BlockScaled GEMM with our group_size=16 / vector_size=16
// config — `is_sfb` flag is accepted for future-proofing but
// currently unused.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace fp4 {

// Required byte count for the swizzled SF buffer for given linear shape.
// rows = M (for SFA) or N (for SFB); D = K (the contracted dim).
//
// Each super-atom is (128 M-rows) x (4 K-blocks) = 512 entries × 1 byte
// (matches the offset formula in the kernel). Total super-atoms tile
// the (M_super, K_super) grid in row-major order.
inline int64_t nvfp4_sf_swizzled_bytes(int rows, int D) {
  int n_blocks    = D / 16;
  int n_row_super = (rows + 127) / 128;
  int n_col_super = (n_blocks + 3) / 4;
  return static_cast<int64_t>(n_row_super) * n_col_super * 512;
}

// Reshape linear (rows, D/16) e4m3 SF into CUTLASS Sm1xx swizzled layout.
//   src_linear : [rows, D/16] u8 (stored as fp8 e4m3) — HF natural
//   dst_swz    : pre-allocated buffer of nvfp4_sf_swizzled_bytes(rows, D)
//                bytes. Writer assumes zero-init for padded regions.
// Returns 0 on success.
int nvfp4_sf_linear_to_swizzled(
    const void* src_linear,
    void*       dst_swz,
    int rows, int D, bool is_sfb,
    cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt

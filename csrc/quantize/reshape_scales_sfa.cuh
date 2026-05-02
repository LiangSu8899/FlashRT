// ============================================================================
//  FlashVLA — Linear scale layout → CUTLASS Sm1xxBlockScaledConfig SFA/SFB.
//
//  Our dynamic quantize kernel produces scales in linear [N, D/16] row-major
//  layout. CUTLASS SM100 block-scaled GEMM wants scales in a tile-interleaved
//  layout described by `Sm1xxBlockScaledConfig<16>::tile_atom_to_shape_SFA`.
//
//  This module provides a CUDA kernel that permutes linear scales into the
//  CUTLASS layout, for both the M-side (SFA) and N-side (SFB) operands.
//  Runs once per activation, negligible cost vs GEMM.
// ============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace fp4 {

// Reorder linear [rows, D/16] block-scales (fp8 e4m3) into CUTLASS SFA layout
// for problem (M, _, K, 1). `rows == M`, `D == K`.
// dst_sfa must be pre-allocated with size == Sm1xxBlockScaledConfig<16>::
//    tile_atom_to_shape_SFA(M, K, 1).size() (filter_zeros-free).
// `is_sfb = true` flips to the N-side (SFB) semantics (problem N, K).
int reshape_linear_scales_to_sfa(
    const void* src_linear_fp8,    // [rows, D/16] row-major, fp8 e4m3 storage
    void* dst_sfa_fp8,             // CUTLASS SFA/SFB buffer
    int rows, int D,               // rows = M (for SFA) or N (for SFB); D = K
    bool is_sfb,
    cudaStream_t stream);

// Returns the number of BYTES needed to hold the CUTLASS SFA/SFB buffer for
// a given (rows, D, is_sfb) triple. Used by callers to allocate correctly.
int64_t sfa_size_bytes(int rows, int D, bool is_sfb);

}  // namespace fp4
}  // namespace flash_rt

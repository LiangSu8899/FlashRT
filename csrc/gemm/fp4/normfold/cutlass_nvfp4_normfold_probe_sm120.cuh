// SPDX-License-Identifier: Apache-2.0
//
// M-FULL-3a-i probe: instantiate the forked NormFold CollectiveMma at IDENTITY
// (no A-path edits) and run it as a plain NVFP4 W4A4 blockscaled GEMM. Output
// must be bit-identical to fp4_w4a16_gemm_sm120_bf16out — proving the forked
// collective is instantiable before any RMSNorm-into-A-load transform is added.
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// Same signature/semantics as fp4_w4a16_gemm_sm120_bf16out. Returns 0 on
// success, or the CUTLASS status (int) on can_implement/initialize/run failure.
int fp4_normfold_probe_sm120_bf16out(
    const void* A_packed, const void* B_packed, void* D_bf16,
    int M, int N, int K,
    const void* SFA, const void* SFB,
    float alpha, cudaStream_t stream);

// Variant-selectable form. variant 0 = <128,128,256> (production tile, identity
// anchor); variant 1 = <128,128,64> (the BLK_K=64 tile the bf16-A norm-fold
// requires for smem budget). Returns -99 on an unknown variant.
int fp4_normfold_probe_sm120_bf16out_v(
    int variant,
    const void* A_packed, const void* B_packed, void* D_bf16,
    int M, int N, int K,
    const void* SFA, const void* SFB,
    float alpha, cudaStream_t stream);

// bf16-A norm fold (M-FULL-3a-ii): A is bf16 (un-quantized), quantized to NVFP4 in
// the consumer; B/SFB are the usual fp4 weight + scales. No SFA input. <128,128,128>.
int fp4_normfold_bf16a_probe_sm120(
    const void* A_bf16, const void* B_packed, void* D_bf16,
    int M, int N, int K, const void* SFB, float alpha, cudaStream_t stream);

// PRODUCER-QUANT (PQ) norm fold: A is bf16; the producer TMAs it to a bf16 staging
// buffer and the consumer quantizes it (NATURAL layout) into fp4 sA + SFA, then the
// stock fp4 consumer reads it. Same args as the bf16a probe. <128,128,128>.
int fp4_normfold_pq_probe_sm120(
    const void* A_bf16, const void* B_packed, void* D_bf16,
    int M, int N, int K, const void* SFB, float alpha, cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt

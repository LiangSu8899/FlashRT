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

}  // namespace gemm
}  // namespace flash_rt

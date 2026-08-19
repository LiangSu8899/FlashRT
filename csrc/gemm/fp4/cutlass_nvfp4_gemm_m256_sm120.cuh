// SPDX-License-Identifier: Apache-2.0
//
// Large-M (M >= ~512) NVFP4 W4A4 GEMM tier for sm_120: 256x128x128
// cooperative tile, bf16 output. See the .cu for measurements and the
// CUTLASS >= 4.5 requirement. Additive.
#pragma once
#include <cstddef>
#include <cuda_runtime.h>
namespace flash_rt {
namespace gemm {
size_t nvfp4_gemm_m256_sm120_workspace_size(int M, int N, int K);
// A (M, K/2) row-major packed, B (N, K/2) column-major-K packed, SFA/SFB
// in the CUTLASS Sm1xx block-scaled layouts, D (M, N) bf16 row-major.
// workspace from nvfp4_gemm_m256_sm120_workspace_size. Returns 0 on
// success, 1 refused (can_implement), 2 initialize, 3 run.
int nvfp4_gemm_m256_sm120_bf16(const void* A_packed, const void* SFA,
                               const void* B_packed, const void* SFB,
                               void* D_bf16, int M, int N, int K,
                               float alpha, void* workspace,
                               cudaStream_t stream);
}  // namespace gemm
}  // namespace flash_rt

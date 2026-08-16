// SPDX-License-Identifier: Apache-2.0
//
// Batched 64x64 unit-lower-triangular inverse: X = inv(I + strict_tril(A))
// in fp32. Column-independent forward substitution — one lane per column,
// per-column history in registers (fully unrolled), the L tile broadcast
// from shared memory, no cross-lane synchronization. The identity/tril
// preparation folds into the kernel (the strict lower triangle is read
// straight from A, the unit diagonal is implicit), removing the eye/
// expand/tril materializations the host-side solve needed. Additive.
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// A: (B, 64, 64) fp32 row-major (only the strict lower triangle is
// read). X: (B, 64, 64) fp32. Returns 0 on success.
int batched_unit_ltri_inv64_f32(const void* A, void* X, int B,
                                cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt

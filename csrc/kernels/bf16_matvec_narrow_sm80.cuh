// SPDX-License-Identifier: Apache-2.0
//
// A bfloat16 matrix-vector product whose output is narrow and whose
// contraction is long.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// y(N) = W(N, K) * x(K), for small N and large K.
//
// The usual arrangement gives a warp an output row, which at a wide N fills
// the part and at a narrow one does not: sixty-four rows is sixty-four warps,
// and the launch then costs many times what its bytes are worth -- measured
// on a 4B model, a 64x2560 projection took longer than the 12288x2560 one
// beside it, reading fifty times fewer bytes.
//
// Here a block owns a row and its threads divide the contraction, so the work
// available is N * blockDim rather than N * 32 and a narrow projection is
// bounded by what it reads rather than by how little of the part it uses.
//
// Returns 0, or -1 if K is not a multiple of eight.
int bf16_matvec_narrow_bf16(
    const void* x, const void* weight, void* out,
    int N, int K, cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt

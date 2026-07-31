// SPDX-License-Identifier: Apache-2.0
//
// Weight-only 4-bit GEMV/GEMM over the int32-packed, group-scaled layout that
// quantized checkpoints are commonly published in.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// y(M,N) = x(M,K) * W(N,K)^T, with W read in the layout it was published in
// rather than converted to another one first. Converting would cost a second
// quantization -- one 4-bit grid resampled onto another -- for values that are
// already what the checkpoint author chose.
//
//   packed  (N, K/8)          int32, eight values per word
//   scale   (N, K/group)      bf16, one per group along K
//   w[n][c] = (((packed[n][c / 8] >> (4 * (c % 8))) & 0xF) - 8)
//             * scale[n][c / group]
//
// Two properties of the layout are worth stating because getting either wrong
// leaves products that are finite and plausible:
//
//   - the nibbles are offset binary, not two's complement, so the subtraction
//     is part of the format and not a normalization;
//   - value c lives in nibble c % 8 of word c / 8, so a word's eight values
//     are eight consecutive columns.
//
// ``group`` must be a multiple of 8 and divide K; 32, 64 and 128 are the sizes
// producers use. A group is therefore a whole number of words and a lane never
// straddles a scale boundary.
int w4a16_packed_matvec_bf16(
    const void* x, const void* packed, const void* scale, void* out,
    int N, int K, int group, cudaStream_t stream);

// The gate-and-up projection with its gate applied, in one pass.
//
// A pair of projections over one activation, concatenated along N, is followed
// by silu(first) * second. Split as a projection and an elementwise kernel,
// the whole 2*I-wide result goes out to memory and comes back to be reduced to
// I. Here a warp owns row r and row r + N/2 together and applies the gate in
// registers, so what reaches memory is the I values that are wanted -- and the
// elementwise kernel, which lives in a model-specific build tier, is not
// needed at all.
//
// ``N`` is the fused width and must be even; ``out`` holds N/2 values.
int w4a16_packed_matvec_gated_bf16(
    const void* x, const void* packed, const void* scale, void* out,
    int N, int K, int group, cudaStream_t stream);

// The same weight against M rows of activation. Delegates to the matvec at
// M=1 so the two cannot drift apart at the shape a decode step uses.
int w4a16_packed_gemm_bf16(
    const void* x, const void* packed, const void* scale, void* out,
    int M, int N, int K, int group, cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt

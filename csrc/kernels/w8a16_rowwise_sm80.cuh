// SPDX-License-Identifier: Apache-2.0
//
// Weight-only INT8 with one scale per output channel, and the row gather that
// reads the same table.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// y(1,N) = x(1,K) * W(N,K)^T, W held as int8 with a per-row half scale.
//
//   weight  (N, K)  int8
//   scale   (N,)    half
//   w[n][c] = weight[n][c] * scale[n]
//
// Symmetric and per-row, so there is no zero point and no group index: the
// scale is a property of the output channel and multiplies the whole dot
// product rather than any part of it.
int w8a16_rowwise_matvec_bf16(
    const void* x, const void* weight, const void* scale, void* out,
    int N, int K, cudaStream_t stream);

// Rows of the same table, gathered by index and widened.
//
// A tied output projection is the embedding table read the other way, so both
// live on one copy; this is the read that treats it as a table. An index
// outside the table writes zeros rather than reading whatever lies past it,
// because a sampled id and a table are two different things to be wrong about.
int int8_rowwise_gather_bf16(
    const void* ids, const void* weight, const void* scale, void* out,
    int rows, int table_rows, int K, cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt

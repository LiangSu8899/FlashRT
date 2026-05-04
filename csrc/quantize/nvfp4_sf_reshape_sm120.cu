// SPDX-License-Identifier: Apache-2.0
//
// Reshape linear NVFP4 group-scale to CUTLASS Sm1xx blockscaled layout.
// Header: nvfp4_sf_reshape_sm120.cuh

#include "nvfp4_sf_reshape_sm120.cuh"

#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

namespace {

// One thread per (row, k-block). Reads src_linear[row, kblk] and writes
// dst_swz at the CUTLASS swizzled offset.
__global__ void kernel_nvfp4_sf_linear_to_swizzled(
    const uint8_t* __restrict__ src_linear,
    uint8_t*       __restrict__ dst_swz,
    int rows, int n_blocks, int n_col_super)
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int blk = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || blk >= n_blocks) return;

  const int rb = row / 128;
  const int ri = row % 128;
  const int cb = blk / 4;
  const int ci = blk % 4;

  const int super_idx = rb * n_col_super + cb;
  const int inner_off = (ri % 32) * 16 + (ri / 32) * 4 + ci;
  const int dst_off   = super_idx * 512 + inner_off;

  dst_swz[dst_off] = src_linear[row * n_blocks + blk];
}

}  // namespace

int nvfp4_sf_linear_to_swizzled(
    const void* src_linear,
    void*       dst_swz,
    int rows, int D, bool /*is_sfb*/,
    cudaStream_t stream)
{
  const int n_blocks    = D / 16;
  const int n_col_super = (n_blocks + 3) / 4;

  // 32x8 thread block: 32 row-threads cover one M-inner band; 8 col-threads
  // a contiguous run of K-blocks. Grid covers all (row, blk).
  dim3 block(8, 32);
  dim3 grid((n_blocks + block.x - 1) / block.x,
            (rows + block.y - 1) / block.y);

  kernel_nvfp4_sf_linear_to_swizzled<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(src_linear),
      reinterpret_cast<uint8_t*>(dst_swz),
      rows, n_blocks, n_col_super);
  return 0;
}

}  // namespace fp4
}  // namespace flash_rt

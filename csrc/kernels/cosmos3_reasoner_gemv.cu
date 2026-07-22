// Cosmos3-Reasoner decode GEMV: M=1 W4A16 with a plain (non-swizzled) layout.
//
// Weights are quantized host-side to e2m1 codes packed two-per-byte
// ([N, K/2] u8, low nibble = even k) with a bf16 scale per 16-element block
// ([N, K/16]). At M=1 the GEMM is a dot product — pure weight-bandwidth work —
// so a SIMT FMA kernel with one warp per output row streams weights at DRAM
// speed without tensor cores (the tiled CUTLASS W4A16 path underfills SMs at
// these shapes).

#include "cosmos3_reasoner_gemv.cuh"

#include <cuda_bf16.h>

namespace flash_rt::kernels {
namespace {

__constant__ float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// One warp per output row. blockDim.x = 32 lanes, blockDim.y = rows/block.
// The activation vector is staged in shared memory once per block.
__global__ void reasoner_gemv_w4a16_kernel(
    const uint8_t* __restrict__ wp,       // [N, K/2]
    const __nv_bfloat16* __restrict__ ws, // [N, K/16]
    const __nv_bfloat16* __restrict__ a,  // [K]
    __nv_bfloat16* __restrict__ out,      // [N]
    int n_rows,
    int k)
{
  // Dynamic smem: [16] decode LUT then [k] activation floats. The LUT lives in
  // shared memory because divergent indexing into __constant__ serializes.
  extern __shared__ float sh[];
  float* sh_lut = sh;
  float* sh_a = sh + 16;
  const int tid = threadIdx.y * 32 + threadIdx.x;
  const int block_threads = blockDim.x * blockDim.y;
  if (tid < 16) {
    sh_lut[tid] = kE2M1[tid];
  }
  for (int i = tid; i < k; i += block_threads) {
    sh_a[i] = __bfloat162float(a[i]);
  }
  __syncthreads();

  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= n_rows) {
    return;
  }
  const int lane = threadIdx.x;
  const uint8_t* wrow = wp + (size_t)row * (k >> 1);
  const __nv_bfloat16* srow = ws + (size_t)row * (k >> 4);

  float acc = 0.0f;
  // Each lane handles one 16-element block per iteration: 8 packed bytes.
  const int n_blocks = k >> 4;
  for (int b = lane; b < n_blocks; b += 32) {
    const float scale = __bfloat162float(srow[b]);
    const uint2 packed = reinterpret_cast<const uint2*>(wrow + (b << 3))[0];
    const float* av = sh_a + (b << 4);
    float part = 0.0f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      uint32_t word = (j == 0) ? packed.x : packed.y;
#pragma unroll
      for (int byte = 0; byte < 4; ++byte) {
        const uint32_t v = (word >> (byte * 8)) & 0xffu;
        const int base = j * 8 + byte * 2;
        part += sh_lut[v & 0xfu] * av[base];
        part += sh_lut[v >> 4] * av[base + 1];
      }
    }
    acc += part * scale;
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    acc += __shfl_xor_sync(0xffffffffu, acc, off);
  }
  if (lane == 0) {
    out[row] = __float2bfloat16(acc);
  }
}

}  // namespace

void cosmos3_reasoner_gemv_w4a16_bf16(
    const uint8_t* w_packed,
    const __nv_bfloat16* w_scales,
    const __nv_bfloat16* a,
    __nv_bfloat16* out,
    int n_rows,
    int k,
    cudaStream_t stream)
{
  if (w_packed == nullptr || w_scales == nullptr || a == nullptr || out == nullptr) {
    return;
  }
  if (n_rows <= 0 || k <= 0 || (k & 15) != 0) {
    return;
  }
  constexpr int kRowsPerBlock = 8;
  dim3 block(32, kRowsPerBlock);
  const int blocks = (n_rows + kRowsPerBlock - 1) / kRowsPerBlock;
  const size_t smem = (size_t)(k + 16) * sizeof(float);
  reasoner_gemv_w4a16_kernel<<<blocks, block, smem, stream>>>(
      w_packed, w_scales, a, out, n_rows, k);
}

}  // namespace flash_rt::kernels

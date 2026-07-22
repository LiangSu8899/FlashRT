// Cosmos3-Reasoner decode GEMV: M=1 W4A16 with a plain (non-swizzled) layout.
//
// Weights are quantized host-side to e2m1 codes packed two-per-byte
// ([N, K/2] u8, low nibble = even k) with a bf16 scale per 16-element block
// ([N, K/16]). At M=1 the GEMM is a dot product — pure weight-bandwidth work —
// so a SIMT FMA kernel with one warp per output row streams weights at DRAM
// speed without tensor cores. The activation vector is read directly from
// global memory: it is a few KB shared by every block, so it lives in L1/L2
// and a shared-memory staging pass would only add traffic and a barrier.

#include "cosmos3_reasoner_gemv.cuh"

#include <cuda_bf16.h>

namespace flash_rt::kernels {
namespace {

__constant__ float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// One warp per output row. blockDim.x = 32 lanes, blockDim.y = rows/block.
__global__ void reasoner_gemv_w4a16_kernel(
    const uint8_t* __restrict__ wp,       // [N, K/2]
    const __nv_bfloat16* __restrict__ ws, // [N, K/16]
    const __nv_bfloat16* __restrict__ a,  // [K]
    __nv_bfloat16* __restrict__ out,      // [N]
    int n_rows,
    int k)
{
  // 16-entry nibble LUT in shared memory: divergent indexing into
  // __constant__ serializes, while 16 floats span 16 smem banks.
  __shared__ float sh_lut[16];
  const int tid = threadIdx.y * 32 + threadIdx.x;
  if (tid < 16) {
    sh_lut[tid] = kE2M1[tid];
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
    const __nv_bfloat162* av2 = reinterpret_cast<const __nv_bfloat162*>(a + (b << 4));
    float part = 0.0f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      uint32_t word = (j == 0) ? packed.x : packed.y;
#pragma unroll
      for (int byte = 0; byte < 4; ++byte) {
        const uint32_t v = (word >> (byte * 8)) & 0xffu;
        const __nv_bfloat162 a2 = av2[j * 4 + byte];
        part += sh_lut[v & 0xfu] * __bfloat162float(a2.x);
        part += sh_lut[v >> 4] * __bfloat162float(a2.y);
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
  reasoner_gemv_w4a16_kernel<<<blocks, block, 0, stream>>>(
      w_packed, w_scales, a, out, n_rows, k);
}

}  // namespace flash_rt::kernels

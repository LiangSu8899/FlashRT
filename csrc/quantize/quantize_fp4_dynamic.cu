// ============================================================================
//  FlashVLA — fp16 → NVFP4 dynamic quantize CUDA kernel impl.
//
//  Naming / format:
//    NVFP4 element = e2m1 (sign + 2 exp + 1 mantissa) stored as 4 bits.
//    Representable positive magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
//    Block scale = UE4M3 (unsigned e4m3 = e4m3_fn with sign=0 always)
//
//  Encoding table for positive magnitudes → 3-bit index:
//    index  value
//      0    0.0
//      1    0.5
//      2    1.0
//      3    1.5
//      4    2.0
//      5    3.0
//      6    4.0
//      7    6.0
//    sign bit is bit 3 (high bit of nibble).
//
//  Note: This kernel uses a linear scales layout [N, D/16]. CUTLASS SM100
//  block-scaled GEMM wants a tile-interleaved layout (Sm1xxBlkScaledConfig).
//  Transform is handled at integration time by a separate reshape step.
// ============================================================================
#include "quantize_fp4_dynamic.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace flash_rt {
namespace fp4 {

// ── Device helpers ────────────────────────────────────────────────────────
__device__ __forceinline__ uint8_t fp32_to_e2m1(float x) {
  // Sign + magnitude quantization to 4-bit e2m1.
  // Boundary semantics match pytorch torch.bucketize(..., right=False):
  //   boundaries[i-1] < x <= boundaries[i]  → use <= on upper bound.
  // Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
  // Mids:  0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
  uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
  float ax = fabsf(x);

  uint8_t mant;
  if      (ax <= 0.25f) mant = 0u;  // 0.0
  else if (ax <= 0.75f) mant = 1u;  // 0.5
  else if (ax <= 1.25f) mant = 2u;  // 1.0
  else if (ax <= 1.75f) mant = 3u;  // 1.5
  else if (ax <= 2.5f)  mant = 4u;  // 2.0
  else if (ax <= 3.5f)  mant = 5u;  // 3.0
  else if (ax <= 5.0f)  mant = 6u;  // 4.0
  else                  mant = 7u;  // 6.0
  return sign | mant;
}

__device__ __forceinline__ float e2m1_to_fp32(uint8_t nibble) {
  // Inverse of fp32_to_e2m1: 4-bit → fp32
  static constexpr float kMag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = kMag[nibble & 0x7u];
  return (nibble & 0x8u) ? -v : v;
}

// Quantize to UE4M3: positive-only fp8 e4m3 form.
// We use torch.float8_e4m3fn which stores sign; for "UE4M3" the value is
// always stored with sign=0. We rely on the host-side conversion
// `__nv_fp8_e4m3(positive_value)` doing the right thing; here on device we
// use the __nv_fp8_e4m3 narrow type constructor.
__device__ __forceinline__ __nv_fp8_e4m3 quantize_ue4m3(float x) {
  // x must be >= 0; we clamp to guard against FP noise near zero.
  float v = fmaxf(x, 0.f);
  return __nv_fp8_e4m3(v);  // Calls intrinsic rounding-to-nearest-even
}

__device__ __forceinline__ float dequantize_ue4m3(__nv_fp8_e4m3 s) {
  return static_cast<float>(s);
}

// ── Kernel: one thread per 16-element block ───────────────────────────────
// Grid: (ceil(D/16), N)   Block: (threads_per_block)
// Each thread handles one (row, block) pair.
__global__ void kernel_quantize_fp4(
    const __half* __restrict__ src,
    uint8_t* __restrict__ dst_packed,
    __nv_fp8_e4m3* __restrict__ dst_scales,
    int N, int D) {
  const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row       = blockIdx.y;
  const int n_blocks  = D / 16;
  if (row >= N || block_idx >= n_blocks) return;

  const int base = row * D + block_idx * 16;
  // 1) Load 16 fp16 elements, compute amax.
  float vals[16];
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    vals[i] = __half2float(src[base + i]);
    float a = fabsf(vals[i]);
    if (a > amax) amax = a;
  }
  // 2) Compute block scale: amax / 6, quantized to UE4M3.
  float desired = amax / 6.f;
  if (desired < 1e-12f) desired = 1e-12f;
  __nv_fp8_e4m3 bs_q = quantize_ue4m3(desired);
  float bs_dq        = dequantize_ue4m3(bs_q);
  dst_scales[row * n_blocks + block_idx] = bs_q;

  // 3) Quantize each element to e2m1 and pack pairs into bytes.
  const int out_base = row * (D / 2) + block_idx * 8;  // 16 elems → 8 bytes
  const float inv_bs = 1.f / bs_dq;
  #pragma unroll
  for (int p = 0; p < 8; ++p) {
    float v_lo = vals[2 * p    ] * inv_bs;
    float v_hi = vals[2 * p + 1] * inv_bs;
    uint8_t lo = fp32_to_e2m1(v_lo);
    uint8_t hi = fp32_to_e2m1(v_hi);
    dst_packed[out_base + p] = lo | (hi << 4);
  }
}

// Dequantize kernel — used for unit tests and as inverse during debugging.
__global__ void kernel_dequantize_fp4(
    const uint8_t* __restrict__ packed,
    const __nv_fp8_e4m3* __restrict__ scales,
    __half* __restrict__ dst,
    int N, int D) {
  const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row       = blockIdx.y;
  const int n_blocks  = D / 16;
  if (row >= N || block_idx >= n_blocks) return;

  const __nv_fp8_e4m3 bs = scales[row * n_blocks + block_idx];
  const float bs_dq      = dequantize_ue4m3(bs);
  const int in_base  = row * (D / 2) + block_idx * 8;
  const int out_base = row * D       + block_idx * 16;
  #pragma unroll
  for (int p = 0; p < 8; ++p) {
    uint8_t byte = packed[in_base + p];
    uint8_t lo   = byte & 0x0Fu;
    uint8_t hi   = (byte >> 4) & 0x0Fu;
    dst[out_base + 2 * p    ] = __float2half(e2m1_to_fp32(lo) * bs_dq);
    dst[out_base + 2 * p + 1] = __float2half(e2m1_to_fp32(hi) * bs_dq);
  }
}

// ── Host entry points ─────────────────────────────────────────────────────
int quantize_fp4_dynamic_fp16(
    const void* src_fp16, void* dst_fp4_packed, void* dst_block_scales,
    int N, int D, cudaStream_t stream) {
  if (D % 16 != 0) return -1;
  const int n_blocks_per_row = D / 16;
  // Pack blocks along x; rows along y.
  const int threads = 128;
  dim3 grid((n_blocks_per_row + threads - 1) / threads, N);
  dim3 block(threads);
  kernel_quantize_fp4<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(src_fp16),
      reinterpret_cast<uint8_t*>(dst_fp4_packed),
      reinterpret_cast<__nv_fp8_e4m3*>(dst_block_scales),
      N, D);
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int dequantize_fp4_to_fp16(
    const void* src_fp4_packed, const void* src_block_scales,
    void* dst_fp16, int N, int D, cudaStream_t stream) {
  if (D % 16 != 0) return -1;
  const int n_blocks_per_row = D / 16;
  const int threads = 128;
  dim3 grid((n_blocks_per_row + threads - 1) / threads, N);
  dim3 block(threads);
  kernel_dequantize_fp4<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(src_fp4_packed),
      reinterpret_cast<const __nv_fp8_e4m3*>(src_block_scales),
      reinterpret_cast<__half*>(dst_fp16),
      N, D);
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace fp4
}  // namespace flash_rt

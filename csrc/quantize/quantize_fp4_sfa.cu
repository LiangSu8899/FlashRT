// ============================================================================
//  Fused FP4 quantize + CUTLASS SFA/SFB tile-interleaved scale write.
//
//  Implementation = kernel_quantize_fp4 (quantize_fp4_dynamic.cu) with the
//  scale-store address replaced by the CUTLASS layout functor. Packed fp4
//  elements layout is UNCHANGED (still linear [N, D/2]), only the scale
//  byte goes to a different location.
// ============================================================================
#include "quantize_fp4_sfa.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(__CUDA_ARCH__)
#  include "cutlass/cutlass.h"
#  include "cutlass/detail/sm100_blockscaled_layout.hpp"
#  include "cute/tensor.hpp"
#  define FV_HAVE_CUTLASS 1
#else
#  define FV_HAVE_CUTLASS 0
#endif

namespace flash_rt {
namespace fp4 {

#if FV_HAVE_CUTLASS

using Cfg = cutlass::detail::Sm1xxBlockScaledConfig<16>;

// ── Device helpers (duplicated locally to stay additive — not linking against
//    quantize_fp4_dynamic.cu so we don't risk ODR issues). Identical logic. ──
__device__ __forceinline__ uint8_t fp32_to_e2m1_sfa(float x) {
    uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
    float ax = fabsf(x);
    uint8_t mant;
    if      (ax <= 0.25f) mant = 0u;
    else if (ax <= 0.75f) mant = 1u;
    else if (ax <= 1.25f) mant = 2u;
    else if (ax <= 1.75f) mant = 3u;
    else if (ax <= 2.5f)  mant = 4u;
    else if (ax <= 3.5f)  mant = 5u;
    else if (ax <= 5.0f)  mant = 6u;
    else                  mant = 7u;
    return sign | mant;
}

__device__ __forceinline__ __nv_fp8_e4m3 quantize_ue4m3_sfa(float x) {
    float v = fmaxf(x, 0.f);
    return __nv_fp8_e4m3(v);
}

__device__ __forceinline__ float dequantize_ue4m3_sfa(__nv_fp8_e4m3 s) {
    return static_cast<float>(s);
}

// ── Fused kernel ──
// One thread per (row, 16-element block). Scale byte goes to
// dst_sfa[layout(row, block_idx*16, 0)].
template <class LayoutSF>
__global__ void kernel_quantize_fp4_sfa(
    const __half* __restrict__ src,
    uint8_t* __restrict__ dst_packed,
    uint8_t* __restrict__ dst_sfa,   // raw byte view of the CUTLASS SFA/SFB buffer
    LayoutSF layout,
    int N, int D) {
  const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row       = blockIdx.y;
  const int n_blocks  = D / 16;
  if (row >= N || block_idx >= n_blocks) return;

  const int base = row * D + block_idx * 16;
  float vals[16];
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    vals[i] = __half2float(src[base + i]);
    float a = fabsf(vals[i]);
    if (a > amax) amax = a;
  }

  float desired = amax / 6.f;
  if (desired < 1e-12f) desired = 1e-12f;
  __nv_fp8_e4m3 bs_q = quantize_ue4m3_sfa(desired);
  float bs_dq        = dequantize_ue4m3_sfa(bs_q);

  // ── CORE FUSION: direct SFA tile-layout write ──
  // LayoutSF maps (row, k, L=0) → byte offset. k is the full-K coordinate;
  // SFVecSize=16 is baked in so any k in [block*16, block*16+15] hits the
  // same offset. Use block_idx*16 (same convention as reshape_scales_sfa.cu).
  int sfa_off = layout(row, block_idx * 16, 0);
  dst_sfa[sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);

  // Packed fp4 elements: layout unchanged.
  const int out_base = row * (D / 2) + block_idx * 8;
  const float inv_bs = 1.f / bs_dq;
  #pragma unroll
  for (int p = 0; p < 8; ++p) {
    float v_lo = vals[2 * p    ] * inv_bs;
    float v_hi = vals[2 * p + 1] * inv_bs;
    uint8_t lo = fp32_to_e2m1_sfa(v_lo);
    uint8_t hi = fp32_to_e2m1_sfa(v_hi);
    dst_packed[out_base + p] = lo | (hi << 4);
  }
}

#endif  // FV_HAVE_CUTLASS

int quantize_fp4_dynamic_sfa_fp16(
    const void* src_fp16, void* dst_packed, void* dst_sfa,
    int N, int D, bool is_sfb, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
  if (D % 16 != 0) return -1;
  const int n_blocks = D / 16;
  const int threads = 128;
  dim3 grid((n_blocks + threads - 1) / threads, N);
  dim3 block(threads);

  // Shape: SFA uses (M=N, 1, K=D, L=1); SFB uses (1, N=N, K=D, L=1).
  auto shape = cute::make_shape(
      is_sfb ? 1 : N,
      is_sfb ? N : 1,
      D, 1);

  if (is_sfb) {
    auto layout = Cfg::tile_atom_to_shape_SFB(shape);
    kernel_quantize_fp4_sfa<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(src_fp16),
        reinterpret_cast<uint8_t*>(dst_packed),
        reinterpret_cast<uint8_t*>(dst_sfa),
        layout, N, D);
  } else {
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    kernel_quantize_fp4_sfa<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(src_fp16),
        reinterpret_cast<uint8_t*>(dst_packed),
        reinterpret_cast<uint8_t*>(dst_sfa),
        layout, N, D);
  }
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
#else
  (void)src_fp16; (void)dst_packed; (void)dst_sfa;
  (void)N; (void)D; (void)is_sfb; (void)stream;
  return -2;
#endif
}

}  // namespace fp4
}  // namespace flash_rt

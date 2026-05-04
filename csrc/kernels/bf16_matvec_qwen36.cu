// bf16 row-major matvec for Qwen3.6 — stream-invariant, deterministic.
//
// One warp per output: 32 threads cooperatively dot-product
// x (K,) with W[n] (K,). Within a warp, lane t reads W[n][lane],
// W[n][32+lane], W[n][64+lane], ... -- adjacent lanes hit adjacent
// 16-bit slots, giving fully coalesced 64B memory transactions per
// warp instruction. Then a warp shuffle reduction folds the 32
// partial sums into one output.
//
// Block size = 256 threads = 8 warps = 8 outputs per block.
// Blocks = ceil(N / 8).

#include "bf16_matvec_qwen36.cuh"

namespace flash_rt::kernels {

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;  // 256

// Vectorized: each thread reads 8 bf16 = 16 bytes per iteration via
// int4. With 32 threads/warp, one iteration covers 256 bf16 of W.
// K_FIXED must be a multiple of 256 (5120 = 20 * 256, 4096 = 16 * 256).
template<int K_FIXED>
__global__ void bf16_matvec_warp_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ out,
    int N)
{
  __shared__ __nv_bfloat16 x_sh[K_FIXED];

  // Cooperative x load via int4 (16 bytes per iter per thread).
  const int4* x_i4 = reinterpret_cast<const int4*>(x);
  int4* x_sh_i4 = reinterpret_cast<int4*>(x_sh);
  const int K_int4 = K_FIXED / 8;  // bf16 elements per int4 = 8
  #pragma unroll 1
  for (int j = threadIdx.x; j < K_int4; j += kThreads) {
    x_sh_i4[j] = x_i4[j];
  }
  __syncthreads();

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * kWarpsPerBlock + warp_id;
  if (n >= N) return;

  const int4* w_row_i4 = reinterpret_cast<const int4*>(W + n * K_FIXED);

  float acc = 0.0f;
  // Each warp iter: 32 threads * 8 bf16 = 256 bf16 covered.
  // K_int4 = K_FIXED / 8. Per-thread iters = K_int4 / 32.
  #pragma unroll 1
  for (int i4 = lane; i4 < K_int4; i4 += 32) {
    int4 wv = w_row_i4[i4];
    int4 xv = x_sh_i4[i4];
    // Each int4 = 8 bf16 = 4 __nv_bfloat162.
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      __nv_bfloat162 wb = *reinterpret_cast<__nv_bfloat162*>(
          &(reinterpret_cast<int*>(&wv)[k]));
      __nv_bfloat162 xb = *reinterpret_cast<__nv_bfloat162*>(
          &(reinterpret_cast<int*>(&xv)[k]));
      float2 wf = __bfloat1622float2(wb);
      float2 xf = __bfloat1622float2(xb);
      acc = fmaf(xf.x, wf.x, acc);
      acc = fmaf(xf.y, wf.y, acc);
    }
  }

  // Warp reduction.
  #pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    acc += __shfl_xor_sync(0xffffffff, acc, off);
  }
  if (lane == 0) {
    out[n] = __float2bfloat16(acc);
  }
}

// Generic-K fallback (chunked smem). Same warp pattern.
__global__ void bf16_matvec_warp_kernel_generic(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ out,
    int N, int K)
{
  extern __shared__ __nv_bfloat16 x_sh[];
  const int K_chunk_max = 4096;

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * kWarpsPerBlock + warp_id;

  float acc = 0.0f;

  for (int k_off = 0; k_off < K; k_off += K_chunk_max) {
    const int chunk = min(K_chunk_max, K - k_off);
    for (int j = threadIdx.x; j < chunk; j += kThreads) {
      x_sh[j] = x[k_off + j];
    }
    __syncthreads();

    if (n < N) {
      const __nv_bfloat16* w_row = W + n * K + k_off;
      #pragma unroll 1
      for (int j = lane; j < chunk; j += 32) {
        float xv = static_cast<float>(x_sh[j]);
        float wv = static_cast<float>(w_row[j]);
        acc = fmaf(xv, wv, acc);
      }
    }
    __syncthreads();
  }

  if (n >= N) return;

  #pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    acc += __shfl_xor_sync(0xffffffff, acc, off);
  }
  if (lane == 0) {
    out[n] = __float2bfloat16(acc);
  }
}

}  // namespace

void bf16_matvec_qwen36_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* out,
    int N,
    int K,
    cudaStream_t stream)
{
  const int grid = (N + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (K == 5120) {
    bf16_matvec_warp_kernel<5120>
        <<<grid, kThreads, 0, stream>>>(x, W, out, N);
  } else if (K == 4096) {
    bf16_matvec_warp_kernel<4096>
        <<<grid, kThreads, 0, stream>>>(x, W, out, N);
  } else {
    const int smem_bytes = 4096 * sizeof(__nv_bfloat16);  // 8 KB
    bf16_matvec_warp_kernel_generic
        <<<grid, kThreads, smem_bytes, stream>>>(x, W, out, N, K);
  }
}

}  // namespace flash_rt::kernels

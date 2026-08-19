// SPDX-License-Identifier: Apache-2.0
//
// MMA rewrite of the WY K*K^T chunk kernel. The scalar kernel assigns one
// (i,j) pair per thread - a 128-wide fmaf chain with no smem reuse, half
// the threads writing only zeros, and the same K^T K dot recomputed for
// every v-head in a K-head group. Here one block owns one (chunk, k-head):
// the 64x128 K slab loads to shared memory once, eight warps produce the
// 64x64 Gram tile with wmma (bf16 in, f32 accumulate), and the epilogue
// applies each group member's beta_i * exp(gi - gj) scaling. Measured
// 32.8x on the production S=2048 shape (876.8 -> 26.7 us/layer); numerics
// sit in the bf16 reduction-order band (max rel ~2e-3) - the consumer's
// teacher-forced gate is the judge, as with the sibling mma_fla kernels.
// Any S (in-bounds guard + zero padding), covers short continuation and
// deep buckets. Additive: new file + new entry; same argument surface and
// A layout as qwen36_gdn_wy_kkt_b64_bf16.
#include "gated_delta_wy_kkt_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace flash_rt {
namespace kernels {
namespace wy_kkt_mma {
namespace {

constexpr int kHD = 128;
constexpr int kQHeads = 16;
constexpr int kVHeads = 48;
constexpr int kWyChunk = 64;

using namespace nvcuda;

constexpr int CH = kWyChunk;
constexpr int HD = kHD;

// grid: (num_kh, chunks); block: 256 (8 warps)
__global__ void qwen36_gdn_wy_kkt_b64_mma_kernel(
    const __nv_bfloat16* __restrict__ k16_l2,   // [S, KH, HD]
    const __nv_bfloat16* __restrict__ beta,     // [S, VH]
    const __nv_bfloat16* __restrict__ g_cumsum, // [S, VH]
    float* __restrict__ A,                      // [chunks, VH, CH, CH]
    int S, int num_kh, int num_vh, int group) {
  const int kh = blockIdx.x;
  const int chunk = blockIdx.y;
  const int s0 = chunk * CH;
  const int tid = threadIdx.x, warp = tid >> 5;

  __shared__ __align__(16) __nv_bfloat16 sK[CH][HD + 8];  // +8 pad 防冲突
  __shared__ __align__(16) float sD[CH][CH];

  // K 块装载: 行 si 超界补零 (S 非 64 倍时)
  for (int idx = tid * 8; idx < CH * HD; idx += blockDim.x * 8) {
    int r = idx / HD, c = idx % HD;
    int si = s0 + r;
    if (si < S) {
      *reinterpret_cast<int4*>(&sK[r][c]) = *reinterpret_cast<const int4*>(
          &k16_l2[(static_cast<size_t>(si) * num_kh + kh) * HD + c]);
    } else {
      int4 z{0, 0, 0, 0};
      *reinterpret_cast<int4*>(&sK[r][c]) = z;
    }
  }
  __syncthreads();

  // 8 warp: warp w 负责行块 (w>>1)*16, 列块 (w&1)*32 (2 个 16x16 tile)
  {
    const int rt = warp >> 1;        // 0..3 → 行 16 块
    const int ct = warp & 1;         // 0..1 → 列 32 半区
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    wmma::fill_fragment(acc[0], 0.f);
    wmma::fill_fragment(acc[1], 0.f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                   wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                   wmma::col_major> fb;
    for (int kk = 0; kk < HD; kk += 16) {
      wmma::load_matrix_sync(fa, &sK[rt * 16][kk], HD + 8);
      #pragma unroll
      for (int t = 0; t < 2; ++t) {
        // B = K^T: 列 j 块 = sK 行 (ct*32 + t*16) 作 col_major
        wmma::load_matrix_sync(fb, &sK[ct * 32 + t * 16][kk], HD + 8);
        wmma::mma_sync(acc[t], fa, fb, acc[t]);
      }
    }
    #pragma unroll
    for (int t = 0; t < 2; ++t)
      wmma::store_matrix_sync(&sD[rt * 16][ct * 32 + t * 16], acc[t],
                              CH, wmma::mem_row_major);
  }
  __syncthreads();

  // epilogue: 组内每 vh 独立缩放写出 (i>j 且界内, 否则 0)
  const int vh0 = kh * group;
  for (int g = 0; g < group; ++g) {
    const int vh = vh0 + g;
    const size_t base =
        ((static_cast<size_t>(chunk) * num_vh + vh) * CH) * CH;
    for (int idx = tid; idx < CH * CH; idx += blockDim.x) {
      const int i = idx >> 6, j = idx & 63;
      const int si = s0 + i, sj = s0 + j;
      float out = 0.f;
      if (i > j && si < S && sj < S) {
        const float bi = __bfloat162float(
            beta[static_cast<size_t>(si) * num_vh + vh]);
        const float gi = __bfloat162float(
            g_cumsum[static_cast<size_t>(si) * num_vh + vh]);
        const float gj = __bfloat162float(
            g_cumsum[static_cast<size_t>(sj) * num_vh + vh]);
        out = bi * sD[i][j] * __expf(gi - gj);
      }
      A[base + idx] = out;
    }
  }
}


}  // namespace

void run(const void* k16_l2, const void* beta, const void* g_cumsum,
         void* A, int S, cudaStream_t stream) {
  if (S <= 0) return;
  const int chunks = (S + kWyChunk - 1) / kWyChunk;
  qwen36_gdn_wy_kkt_b64_mma_kernel<<<dim3(kQHeads, chunks), 256, 0,
                                     stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k16_l2),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<float*>(A), S, kQHeads, kVHeads,
      kVHeads / kQHeads);
}

}  // namespace wy_kkt_mma
}  // namespace kernels
}  // namespace flash_rt

void qwen36_gdn_wy_kkt_b64_mma_bf16(
    const void* k16_l2,
    const void* beta,
    const void* g_cumsum,
    void*       A,
    int S,
    cudaStream_t stream)
{
  flash_rt::kernels::wy_kkt_mma::run(k16_l2, beta, g_cumsum, A, S, stream);
}

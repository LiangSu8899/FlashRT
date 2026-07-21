// ================================================================
//  flash_rt — BF16 Conv3d fprop, SM110 bring-up v0
//
//  Cosmos3-Edge Wan2.2 VAE steady CausalConv3d probe:
//    cache_x [B, T_cache=2, H, W, Ci] BF16
//    new_x   [B, T_new,     H, W, Ci] BF16
//    w       [Co, 3, 3, 3, Ci]        BF16
//    y       [B, T_new, H, W, Co]     BF16
//
//  This mirrors fp8_conv3d_v17's virtual cache concat and direct causal
//  output, but uses BF16 tensor cores. It is a probe kernel, not a default
//  production path.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

namespace flash_rt {
namespace conv {

constexpr int B3D_BLOCK_M = 128;
constexpr int B3D_BLOCK_N = 128;
constexpr int B3D_BLOCK_K = 16;
constexpr int B3D_N_ATOMS = B3D_BLOCK_N / 8;
constexpr int B3D_NUM_WARPS = 8;
constexpr int B3D_THREADS = B3D_NUM_WARPS * 32;
constexpr int B3D_STAGES = 2;
constexpr int B3D_SMEM_K_STRIDE_BYTES = 48;

__device__ __forceinline__
void b3d_mma_m16n8k16_bf16(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%0, %1, %2, %3};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__
const uint8_t* b3d_x_byte_ptr(const __nv_bfloat16* cache_x,
                              const __nv_bfloat16* new_x,
                              int m_global, int k_global,
                              int N, int T_cache, int T_new,
                              int H, int W, int Ci) {
  int K_total = 27 * Ci;
  int M_total = N * T_new * H * W;
  if (k_global >= K_total || m_global >= M_total) return nullptr;
  int spatial = T_new * H * W;
  int b_idx = m_global / spatial;
  int rem = m_global - b_idx * spatial;
  int t_out = rem / (H * W);
  rem -= t_out * (H * W);
  int h_out = rem / W;
  int w_out = rem - h_out * W;
  int q = k_global / Ci;
  int ci0 = k_global - q * Ci;
  int ks = q % 3; q /= 3;
  int kr = q % 3;
  int kt = q / 3;
  int d_in = t_out + kt;
  int h_in = h_out + kr - 1;
  int w_in = w_out + ks - 1;
  if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) return nullptr;
  if (d_in < T_cache) {
    int idx = (((b_idx * T_cache + d_in) * H + h_in) * W + w_in) * Ci + ci0;
    return reinterpret_cast<const uint8_t*>(&cache_x[idx]);
  }
  int d_new = d_in - T_cache;
  int idx = (((b_idx * T_new + d_new) * H + h_in) * W + w_in) * Ci + ci0;
  return reinterpret_cast<const uint8_t*>(&new_x[idx]);
}

__device__ __forceinline__
const uint8_t* b3d_w_byte_ptr(const __nv_bfloat16* w,
                              int co, int k_global, int Co, int Ci) {
  int K_total = 27 * Ci;
  if (co >= Co || k_global >= K_total) return nullptr;
  int q = k_global / Ci;
  int ci0 = k_global - q * Ci;
  int ks = q % 3; q /= 3;
  int kr = q % 3;
  int kt = q / 3;
  int idx = (((co * 3 + kt) * 3 + kr) * 3 + ks) * Ci + ci0;
  return reinterpret_cast<const uint8_t*>(&w[idx]);
}

__device__ __forceinline__
void b3d_cp_async_16(uint32_t smem_int_ptr, const uint8_t* src) {
  int src_bytes = (src == nullptr) ? 0 : 16;
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
    :: "r"(smem_int_ptr), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__
uint32_t b3d_to_smem_int(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ void __launch_bounds__(B3D_THREADS, 2)
bf16_conv3d_v0_kernel(
    const __nv_bfloat16* __restrict__ cache_x,
    const __nv_bfloat16* __restrict__ new_x,
    const __nv_bfloat16* __restrict__ w,
          __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ bias,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha,
    int M_tiles, int N_tiles)
{
  __shared__ __align__(16) uint8_t A_smem[B3D_STAGES][B3D_BLOCK_M * B3D_SMEM_K_STRIDE_BYTES];
  __shared__ __align__(16) uint8_t B_smem[B3D_STAGES][B3D_BLOCK_N * B3D_SMEM_K_STRIDE_BYTES];

  const int t = threadIdx.x;
  const int warp_id = t / 32;
  const int lane = t % 32;
  const int l = lane % 4;
  const int h = lane / 4;

  const int M_total = N * T_new * H * W;
  const int K_total = 27 * Ci;

  const int ld_row_a = t / 2;
  const int ld_k_elem_a = (t & 1) * 8;
  const int ld_row_b = t / 2;
  const int ld_k_elem_b = (t & 1) * 8;

  int tile_idx = blockIdx.x;
  int m_idx = tile_idx / N_tiles;
  int n_idx = tile_idx - m_idx * N_tiles;
  int m_base = m_idx * B3D_BLOCK_M;
  int co_base = n_idx * B3D_BLOCK_N;
  if (m_base >= M_total || co_base >= Co) return;

  float dA[B3D_N_ATOMS] = {0};
  float dB[B3D_N_ATOMS] = {0};
  float dC[B3D_N_ATOMS] = {0};
  float dD[B3D_N_ATOMS] = {0};

  auto issue_load = [&](int stage, int k_base) {
    {
      const uint8_t* src = b3d_x_byte_ptr(cache_x, new_x,
                                          m_base + ld_row_a,
                                          k_base + ld_k_elem_a,
                                          N, T_cache, T_new, H, W, Ci);
      uint32_t smem_int = b3d_to_smem_int(
          &A_smem[stage][ld_row_a * B3D_SMEM_K_STRIDE_BYTES + ld_k_elem_a * 2]);
      b3d_cp_async_16(smem_int, src);
    }
    {
      const uint8_t* src = b3d_w_byte_ptr(w, co_base + ld_row_b,
                                          k_base + ld_k_elem_b,
                                          Co, Ci);
      uint32_t smem_int = b3d_to_smem_int(
          &B_smem[stage][ld_row_b * B3D_SMEM_K_STRIDE_BYTES + ld_k_elem_b * 2]);
      b3d_cp_async_16(smem_int, src);
    }
  };

  issue_load(0, 0);
  asm volatile("cp.async.commit_group;\n" ::);

  int compute_stage = 0;
  for (int k_base = 0; k_base < K_total; k_base += B3D_BLOCK_K) {
    int next_stage = compute_stage ^ 1;
    int k_next = k_base + B3D_BLOCK_K;
    if (k_next < K_total) {
      issue_load(next_stage, k_next);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();

    const int warp_M_off = warp_id * 16;
    const int kA0 = 4 * l;
    const int kA2 = 4 * l + 16;
    int rA0 = warp_M_off + h;
    int rA1 = warp_M_off + h + 8;
    uint32_t A0 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA0 * B3D_SMEM_K_STRIDE_BYTES + kA0]);
    uint32_t A1 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA1 * B3D_SMEM_K_STRIDE_BYTES + kA0]);
    uint32_t A2 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA0 * B3D_SMEM_K_STRIDE_BYTES + kA2]);
    uint32_t A3 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA1 * B3D_SMEM_K_STRIDE_BYTES + kA2]);

    #pragma unroll
    for (int n_atom = 0; n_atom < B3D_N_ATOMS; ++n_atom) {
      int co_n = n_atom * 8 + h;
      uint32_t B0 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co_n * B3D_SMEM_K_STRIDE_BYTES + kA0]);
      uint32_t B1 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co_n * B3D_SMEM_K_STRIDE_BYTES + kA2]);
      b3d_mma_m16n8k16_bf16(
          dA[n_atom], dB[n_atom], dC[n_atom], dD[n_atom],
          A0, A1, A2, A3, B0, B1);
    }

    compute_stage = next_stage;
  }

  asm volatile("cp.async.wait_all;\n" ::);

  const int warp_M_off = warp_id * 16;
  #pragma unroll
  for (int n_atom = 0; n_atom < B3D_N_ATOMS; ++n_atom) {
    int co_pair = co_base + n_atom * 8 + 2 * l;
    int row0 = m_base + warp_M_off + h;
    int row1 = m_base + warp_M_off + h + 8;
    float b0 = 0.f, b1 = 0.f;
    if (bias != nullptr && co_pair < Co) {
      b0 = __bfloat162float(bias[co_pair]);
      if (co_pair + 1 < Co) b1 = __bfloat162float(bias[co_pair + 1]);
    }
    if (co_pair + 1 < Co) {
      __nv_bfloat162 packAB;
      packAB.x = __float2bfloat16(dA[n_atom] * alpha + b0);
      packAB.y = __float2bfloat16(dB[n_atom] * alpha + b1);
      __nv_bfloat162 packCD;
      packCD.x = __float2bfloat16(dC[n_atom] * alpha + b0);
      packCD.y = __float2bfloat16(dD[n_atom] * alpha + b1);
      if (row0 < M_total) {
        *reinterpret_cast<__nv_bfloat162*>(&y[row0 * Co + co_pair]) = packAB;
      }
      if (row1 < M_total) {
        *reinterpret_cast<__nv_bfloat162*>(&y[row1 * Co + co_pair]) = packCD;
      }
    } else {
      auto store = [&](int row, int co, float v, float bv) {
        if (row < M_total && co < Co) {
          y[row * Co + co] = __float2bfloat16(v * alpha + bv);
        }
      };
      store(row0, co_pair, dA[n_atom], b0);
      store(row0, co_pair + 1, dB[n_atom], b1);
      store(row1, co_pair, dC[n_atom], b0);
      store(row1, co_pair + 1, dD[n_atom], b1);
    }
  }
}

extern "C" int bf16_conv3d_v0_ndhwc_bf16out(
    const void* cache_x_bf16, const void* new_x_bf16,
    const void* w_bf16, void* y_bf16,
    const void* bias_bf16,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream)
{
  if (Ci % B3D_BLOCK_K != 0 || Co % 8 != 0) {
    std::fprintf(stderr,
        "[bf16_conv3d_v0] Ci%%%d (got %d) or Co%%8 (got %d) bad\n",
        B3D_BLOCK_K, Ci, Co);
    return -1;
  }
  if (T_cache != 2) {
    std::fprintf(stderr, "[bf16_conv3d_v0] T_cache must be 2 (got %d)\n", T_cache);
    return -3;
  }
  int M = N * T_new * H * W;
  int M_tiles = (M + B3D_BLOCK_M - 1) / B3D_BLOCK_M;
  int N_tiles = (Co + B3D_BLOCK_N - 1) / B3D_BLOCK_N;
  int total_tiles = M_tiles * N_tiles;

  bf16_conv3d_v0_kernel<<<dim3(total_tiles), dim3(B3D_THREADS), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(cache_x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(new_x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(w_bf16),
      reinterpret_cast<__nv_bfloat16*>(y_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      N, T_cache, T_new, H, W, Ci, Co, alpha,
      M_tiles, N_tiles);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[bf16_conv3d_v0] launch err: %s\n", cudaGetErrorString(e));
    return -2;
  }
  return 0;
}

}  // namespace conv
}  // namespace flash_rt

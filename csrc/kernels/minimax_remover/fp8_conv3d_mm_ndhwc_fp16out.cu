// ================================================================
// flash_rt — MiniMax-Remover FP8 Conv3d fprop, sm_120a.
//
// Implicit-GEMM FP8 e4m3 conv3d with on-the-fly im2col indexing.
// Tile: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, 8 warps, cp.async
// 2-stage.  Causal temporal cache via virtual dual-pointer concat.
//
// Output: fp16 (NDHWC / channels-last 3D physical layout).
// Bias:   fp16 per-channel.
// Alpha:  per-output-channel float vector (act_scale × w_scale[co]).
//
// Constraints: Ci % 32 == 0, T_cache == 2, kernel = 3×3×3, pad = 1,
// stride = dilation = 1, groups = 1.
// ================================================================

#include "fp8_conv3d_mm_ndhwc_fp16out.cuh"
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

constexpr int MM_BLOCK_M = 128;
constexpr int MM_BLOCK_N = 128;
constexpr int MM_BLOCK_K = 32;
constexpr int MM_N_ATOMS  = MM_BLOCK_N / 8;
constexpr int MM_NUM_WARPS = 8;
constexpr int MM_THREADS = MM_NUM_WARPS * 32;
constexpr int MM_STAGES = 2;
constexpr int MM_SMEM_K_STRIDE = 48;

__device__ __forceinline__
void mm_mma_m16n8k32_e4m3(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
  asm volatile(
    "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%0, %1, %2, %3};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1));
}

// Activation address: virtual concat of cache_x + new_x along T,
// causal output mapping d_in = t_out + kt.
// Spatial pad=1 (h_in/w_in OOB → nullptr → cp.async zeros smem).
__device__ __forceinline__
const uint8_t* mm_x_byte_ptr(const __nv_fp8_e4m3* cache_x,
                              const __nv_fp8_e4m3* new_x,
                              int m_global, int k_global,
                              int N, int T_cache, int T_new,
                              int H, int W, int Ci) {
  int K_total = 27 * Ci;
  int M_total = N * T_new * H * W;
  if (k_global >= K_total || m_global >= M_total) return nullptr;
  int spatial = T_new * H * W;
  int b_idx = m_global / spatial;
  int rem   = m_global - b_idx * spatial;
  int t_out = rem / (H * W);
  rem      -= t_out * (H * W);
  int h_out = rem / W;
  int w_out = rem - h_out * W;
  int q   = k_global / Ci;
  int ci0 = k_global % Ci;
  int ks  = q % 3; q /= 3;
  int kr  = q % 3;
  int kt  = q / 3;
  int d_in = t_out + kt;
  int h_in = h_out + kr - 1;
  int w_in = w_out + ks - 1;
  if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) return nullptr;
  if (d_in < T_cache) {
    int idx = (((b_idx * T_cache + d_in) * H + h_in) * W + w_in) * Ci + ci0;
    return reinterpret_cast<const uint8_t*>(&cache_x[idx]);
  } else {
    int d_new = d_in - T_cache;
    int idx = (((b_idx * T_new + d_new) * H + h_in) * W + w_in) * Ci + ci0;
    return reinterpret_cast<const uint8_t*>(&new_x[idx]);
  }
}

// Weight address: [Co, kT, kH, kW, Ci] contiguous.
__device__ __forceinline__
const uint8_t* mm_w_byte_ptr(const __nv_fp8_e4m3* w,
                              int co, int k_global, int Co, int Ci) {
  int K_total = 27 * Ci;
  if (co >= Co || k_global >= K_total) return nullptr;
  int q   = k_global / Ci;
  int ci0 = k_global % Ci;
  int ks  = q % 3; q /= 3;
  int kr  = q % 3;
  int kt  = q / 3;
  int idx = (((co * 3 + kt) * 3 + kr) * 3 + ks) * Ci + ci0;
  return reinterpret_cast<const uint8_t*>(&w[idx]);
}

__device__ __forceinline__
void mm_cp_async_16(uint32_t smem_int_ptr, const uint8_t* src) {
  int src_bytes = (src == nullptr) ? 0 : 16;
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
    :: "r"(smem_int_ptr), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__
uint32_t mm_to_smem_int(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ void __launch_bounds__(MM_THREADS, 2)
fp8_conv3d_mm_kernel(
    const __nv_fp8_e4m3* __restrict__ cache_x,
    const __nv_fp8_e4m3* __restrict__ new_x,
    const __nv_fp8_e4m3* __restrict__ w,
          __half* __restrict__ y,
    const __half* __restrict__ bias,
    const float* __restrict__ alpha_vec,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    int M_tiles, int N_tiles)
{
  __shared__ __align__(16) uint8_t A_smem[MM_STAGES][MM_BLOCK_M * MM_SMEM_K_STRIDE];
  __shared__ __align__(16) uint8_t B_smem[MM_STAGES][MM_BLOCK_N * MM_SMEM_K_STRIDE];

  const int t       = threadIdx.x;
  const int warp_id = t / 32;
  const int lane    = t % 32;
  const int l       = lane % 4;
  const int h       = lane / 4;

  const int M_total  = N * T_new * H * W;
  const int K_total  = 27 * Ci;

  const int ld_row_a   = t / 2;
  const int ld_k_off_a = (t & 1) * 16;
  const int ld_row_b   = t / 2;
  const int ld_k_off_b = (t & 1) * 16;

  {
    int tile_idx = blockIdx.x;
    int m_idx  = tile_idx / N_tiles;
    int n_idx  = tile_idx % N_tiles;
    int m_base = m_idx * MM_BLOCK_M;
    int co_base = n_idx * MM_BLOCK_N;

    if (m_base >= M_total || co_base >= Co) return;

    float dA[MM_N_ATOMS] = {0};
    float dB[MM_N_ATOMS] = {0};
    float dC[MM_N_ATOMS] = {0};
    float dD[MM_N_ATOMS] = {0};

    auto issue_load = [&](int stage, int k_base) {
      {
        const uint8_t* src = mm_x_byte_ptr(cache_x, new_x,
                                            m_base + ld_row_a,
                                            k_base + ld_k_off_a,
                                            N, T_cache, T_new, H, W, Ci);
        uint32_t smem_int = mm_to_smem_int(
            &A_smem[stage][ld_row_a * MM_SMEM_K_STRIDE + ld_k_off_a]);
        mm_cp_async_16(smem_int, src);
      }
      {
        const uint8_t* src = mm_w_byte_ptr(w, co_base + ld_row_b,
                                            k_base + ld_k_off_b,
                                            Co, Ci);
        uint32_t smem_int = mm_to_smem_int(
            &B_smem[stage][ld_row_b * MM_SMEM_K_STRIDE + ld_k_off_b]);
        mm_cp_async_16(smem_int, src);
      }
    };

    // Prologue
    issue_load(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);

    int compute_stage = 0;

    for (int k_base = 0; k_base < K_total; k_base += MM_BLOCK_K) {
      int next_stage = compute_stage ^ 1;
      int k_next = k_base + MM_BLOCK_K;

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
          &A_smem[compute_stage][rA0 * MM_SMEM_K_STRIDE + kA0]);
      uint32_t A1 = *reinterpret_cast<const uint32_t*>(
          &A_smem[compute_stage][rA1 * MM_SMEM_K_STRIDE + kA0]);
      uint32_t A2 = *reinterpret_cast<const uint32_t*>(
          &A_smem[compute_stage][rA0 * MM_SMEM_K_STRIDE + kA2]);
      uint32_t A3 = *reinterpret_cast<const uint32_t*>(
          &A_smem[compute_stage][rA1 * MM_SMEM_K_STRIDE + kA2]);

      #pragma unroll
      for (int n_atom = 0; n_atom < MM_N_ATOMS; ++n_atom) {
        int co_n = n_atom * 8 + h;
        uint32_t B0 = *reinterpret_cast<const uint32_t*>(
            &B_smem[compute_stage][co_n * MM_SMEM_K_STRIDE + kA0]);
        uint32_t B1 = *reinterpret_cast<const uint32_t*>(
            &B_smem[compute_stage][co_n * MM_SMEM_K_STRIDE + kA2]);
        mm_mma_m16n8k32_e4m3(
            dA[n_atom], dB[n_atom], dC[n_atom], dD[n_atom],
            A0, A1, A2, A3, B0, B1);
      }

      compute_stage = next_stage;
    }

    asm volatile("cp.async.wait_all;\n" ::);

    // Epilogue: per-channel alpha × fp32 accumulator + fp16 bias → fp16 store.
    // Output in NDHWC: y[b, t_out, h_out, w_out, co].
    const int warp_M_off = warp_id * 16;
    #pragma unroll
    for (int n_atom = 0; n_atom < MM_N_ATOMS; ++n_atom) {
      int co_pair = co_base + n_atom * 8 + 2 * l;
      int row0    = m_base + warp_M_off + h;
      int row1    = m_base + warp_M_off + h + 8;

      float a0 = 1.0f, a1 = 1.0f;
      if (alpha_vec != nullptr) {
        a0 = alpha_vec[co_pair];
        if (co_pair + 1 < Co) a1 = alpha_vec[co_pair + 1];
      }
      float b0 = 0.f, b1 = 0.f;
      if (bias != nullptr && co_pair < Co) {
        b0 = __half2float(bias[co_pair]);
        if (co_pair + 1 < Co) b1 = __half2float(bias[co_pair + 1]);
      }

      if (co_pair + 1 < Co) {
        __half2 packAB;
        packAB.x = __float2half_rn(dA[n_atom] * a0 + b0);
        packAB.y = __float2half_rn(dB[n_atom] * a1 + b1);
        __half2 packCD;
        packCD.x = __float2half_rn(dC[n_atom] * a0 + b0);
        packCD.y = __float2half_rn(dD[n_atom] * a1 + b1);
        if (row0 < M_total) {
          *reinterpret_cast<__half2*>(&y[row0 * Co + co_pair]) = packAB;
        }
        if (row1 < M_total) {
          *reinterpret_cast<__half2*>(&y[row1 * Co + co_pair]) = packCD;
        }
      } else {
        auto store = [&](int row, int co, float v, float av, float bv) {
          if (row < M_total && co < Co) {
            y[row * Co + co] = __float2half_rn(v * av + bv);
          }
        };
        store(row0, co_pair + 0, dA[n_atom], a0, b0);
        store(row0, co_pair + 1, dB[n_atom], a1, b1);
        store(row1, co_pair + 0, dC[n_atom], a0, b0);
        store(row1, co_pair + 1, dD[n_atom], a1, b1);
      }
    }
  }
}

int fp8_conv3d_mm_ndhwc_fp16out(
    const void* cache_x_fp8, const void* new_x_fp8,
    const void* w_fp8, void* y_fp16,
    const void* bias_fp16, const void* alpha_vec,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    cudaStream_t stream)
{
  if (Ci % MM_BLOCK_K != 0) {
    std::fprintf(stderr,
        "[fp8_conv3d_mm] Ci%%%d (got %d) — must be multiple of 32\n",
        MM_BLOCK_K, Ci);
    return -1;
  }
  if (T_cache != 2) {
    std::fprintf(stderr,
        "[fp8_conv3d_mm] T_cache must be 2 (got %d)\n", T_cache);
    return -3;
  }
  if (T_new < 1) {
    std::fprintf(stderr,
        "[fp8_conv3d_mm] T_new must be >= 1 (got %d)\n", T_new);
    return -4;
  }
  int M = N * T_new * H * W;
  int M_tiles = (M + MM_BLOCK_M - 1) / MM_BLOCK_M;
  int N_tiles = (Co + MM_BLOCK_N - 1) / MM_BLOCK_N;
  int total_tiles = M_tiles * N_tiles;

  dim3 grid(total_tiles);
  dim3 block(MM_THREADS);
  fp8_conv3d_mm_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(cache_x_fp8),
      reinterpret_cast<const __nv_fp8_e4m3*>(new_x_fp8),
      reinterpret_cast<const __nv_fp8_e4m3*>(w_fp8),
      reinterpret_cast<__half*>(y_fp16),
      reinterpret_cast<const __half*>(bias_fp16),
      reinterpret_cast<const float*>(alpha_vec),
      N, T_cache, T_new, H, W, Ci, Co,
      M_tiles, N_tiles);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[fp8_conv3d_mm] launch err: %s\n",
                 cudaGetErrorString(e));
    return -2;
  }
  return 0;
}

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt

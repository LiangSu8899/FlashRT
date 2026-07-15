// SPDX-License-Identifier: Apache-2.0
//
// Tensor-core INT4 (E0M3) W4A4 GEMV + quantizer for sm_120. See the
// header (int4_w4a4_mma_sm120.cuh) for the format contract and the
// mandatory OMMA bit-patch post-build step. Fragment layout notes
// live in fp4_w4a4_mma_sm120.cu — this TU uses the identical layout,
// expressed as raw PTX so the file stays free of CUTLASS includes
// (self-contained for the llama.cpp backend build).

#include "int4_w4a4_mma_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {

namespace {

// ── Block-scaled MMA (compiled e2m1, patched to E0M3) ─────────────
//
// Same operand order as cute SM120_16x8x64_TN_VS<..., VS=16>::fma.
// scale_vec::4X — each lane's sfa/sfb u32 carries the 4 UE4M3 bytes
// for its fragment rows/cols across the 4 K-groups of this K-tile.

__device__ __forceinline__ void int4_mma_16x8x64(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint32_t sfa, uint32_t sfb) {
  constexpr uint16_t bid = 0, tid = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
      ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
      "{%14},{%15,%16},{%17},{%18,%19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(sfa), "h"(bid), "h"(tid),
        "r"(sfb), "h"(bid), "h"(tid));
}

// ── UE4M3 helpers (same semantics as the NVFP4 quantizers) ────────

__device__ __forceinline__ uint8_t fp32_to_ue4m3_ceil(float v) {
  if (v <= 0.0f) return 0;
  if (v > 240.0f) return 0xFE;
  uint32_t bits = __float_as_uint(v);
  int float_exp = ((bits >> 23) & 0xFF) - 127;
  uint32_t frac = bits & 0x7FFFFF;
  int ue_exp = float_exp + 7;
  if (ue_exp <= 0) {
    float scaled = v * 512.0f;
    int m = (int)ceilf(scaled);
    if (m > 7) return (1 << 3) | 0;
    if (m < 1) m = 1;
    return (uint8_t)m;
  }
  if (ue_exp >= 15) return 0xFE;
  int m = (int)(frac >> 20);
  if (frac & 0xFFFFF) m++;
  if (m >= 8) { m = 0; ue_exp++; }
  if (ue_exp >= 15) return 0xFE;
  return (uint8_t)((ue_exp << 3) | m);
}

__device__ __forceinline__ float ue4m3_to_fp32(uint8_t v) {
  int e = (v >> 3) & 0xF;
  int m = v & 0x7;
  if (e == 0) return ldexpf((float)m / 8.0f, -6);
  return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

// ── INT4 (E0M3) element encode ────────────────────────────────────
// nibble = (sign<<3) | mag, value = (-1)^sign * mag, mag 0..7.
// Round-to-nearest-even on the integer grid, clamp to 7.

__device__ __forceinline__ uint8_t fp32_to_int4_nibble(float x) {
  float q = rintf(fabsf(x));
  int mag = (int)fminf(q, 7.0f);
  return (uint8_t)(((x < 0.0f) ? 8 : 0) | mag);
}

// ── SF swizzle addressing ─────────────────────────────────────────
// nvfp4_sf_linear_to_swizzled scheme (see fp4_w4a4_mma_sm120.cu):
// for row r, K-group b (16 elements each):
//   rb = r>>7, ri = r&127, cb = b>>2, ci = b&3
//   off = (rb * n_col_super + cb) * 512 + (ri&31)*16 + ((ri>>5)&3)*4 + ci
// n_col_super = ceil(K_groups / 4). One 64-K tile = 4 consecutive bytes.

__device__ __forceinline__ int sf_swizzled_off(
    int row, int kgroup, int n_col_super) {
  int rb = row >> 7, ri = row & 127;
  int cb = kgroup >> 2, ci = kgroup & 3;
  return (rb * n_col_super + cb) * 512 +
         (ri & 31) * 16 + ((ri >> 5) & 3) * 4 + ci;
}

// ── Quantizer kernel ──────────────────────────────────────────────
// One block per row; threads stride over the row's 16-element groups.

__global__ void int4_quantize_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ out_packed,
    uint8_t* __restrict__ out_sf,
    const float* __restrict__ global_scale,
    int rows, int K) {
  int row = blockIdx.x;
  if (row >= rows) return;
  const float g = *global_scale;
  const float inv_g = (g > 0.f) ? (1.f / g) : 0.f;
  const int K_groups = K / 16;
  const int n_col_super = (K_groups + 3) / 4;
  const __nv_bfloat16* xr = x + (long long)row * K;
  uint8_t* pr = out_packed + (long long)row * (K / 2);

  for (int gidx = threadIdx.x; gidx < K_groups; gidx += blockDim.x) {
    float v[16];
    float bmax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      v[i] = __bfloat162float(xr[gidx * 16 + i]);
      bmax = fmaxf(bmax, fabsf(v[i]));
    }
    uint8_t sf_byte = fp32_to_ue4m3_ceil((bmax / 7.0f) * inv_g);
    out_sf[sf_swizzled_off(row, gidx, n_col_super)] = sf_byte;
    float s = ue4m3_to_fp32(sf_byte) * g;
    float inv_s = (s > 0.f) ? (1.f / s) : 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint8_t lo = fp32_to_int4_nibble(v[2 * i] * inv_s);
      uint8_t hi = fp32_to_int4_nibble(v[2 * i + 1] * inv_s);
      pr[gidx * 8 + i] = (uint8_t)(lo | (hi << 4));
    }
  }
}

__global__ void int4_global_scale_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ scale_out,
    long long numel) {
  float local = 0.f;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
       i < numel; i += (long long)gridDim.x * blockDim.x) {
    local = fmaxf(local, fabsf(__bfloat162float(x[i])));
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, off));
  }
  if ((threadIdx.x & 31) == 0 && local > 0.f) {
    // amax / (INT4_MAX * UE4M3_MAX); atomicMax on non-negative floats
    // via int reinterpretation.
    atomicMax(reinterpret_cast<int*>(scale_out),
              __float_as_int(local / (7.0f * 448.0f)));
  }
}

// ── GEMV kernel (twin of fp4 full_n_kernel; layout notes there) ───

constexpr int INT4_WARPS_PER_BLOCK = 1;
constexpr int INT4_THREADS_PER_BLOCK = INT4_WARPS_PER_BLOCK * 32;
constexpr int INT4_COLS_PER_WARP = 8;
constexpr int INT4_COLS_PER_BLOCK =
    INT4_WARPS_PER_BLOCK * INT4_COLS_PER_WARP;

__device__ __forceinline__ uint32_t fast_load_a(
    const uint8_t* sA, int t0, int t1, int reg_idx) {
  int row_off = ((reg_idx & 1) ? (t1 + 8) : t1) * 32;
  int col_off = t0 * 4 + ((reg_idx >> 1) & 1) * 16;
  return *reinterpret_cast<const uint32_t*>(sA + row_off + col_off);
}

__device__ __forceinline__ uint32_t fast_load_b(
    const uint8_t* sB, int t0, int t1, int reg_idx) {
  int col_off = t0 * 4 + reg_idx * 16;
  return *reinterpret_cast<const uint32_t*>(sB + t1 * 32 + col_off);
}

__device__ __forceinline__ void cp_async_4(
    uint8_t* smem_dst, const uint8_t* gmem_src) {
  uint32_t smem_int = __cvta_generic_to_shared(smem_dst);
  asm volatile(
      "cp.async.ca.shared.global.L2::128B [%0], [%1], 4;\n"
      :: "r"(smem_int), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_1() {
  asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_0() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

__global__ void int4_full_n_kernel(
    const uint8_t* __restrict__ A_packed,    // (K/2,)
    const uint8_t* __restrict__ B_packed,    // (N, K/2)
    const uint8_t* __restrict__ SFA,         // swizzled, 1 row
    const uint8_t* __restrict__ SFB,         // swizzled, N rows
    __nv_bfloat16* __restrict__ D,           // (N,)
    float alpha,
    int N, int K) {
  __shared__ alignas(16) uint8_t s_A[2][16 * 32];
  __shared__ alignas(16) uint8_t s_SFA[2][16 * 4];
  __shared__ alignas(16) uint8_t s_B_all[2][INT4_WARPS_PER_BLOCK * 8 * 32];
  __shared__ alignas(16) uint8_t s_SFB_all[2][INT4_WARPS_PER_BLOCK * 8 * 4];

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;

  int block_n_off = blockIdx.x * INT4_COLS_PER_BLOCK;
  int my_n_off = block_n_off + warp * INT4_COLS_PER_WARP;
  if (my_n_off >= N) return;

  int t0 = lane & 3;
  int t1 = lane >> 2;
  int sfa_unique_row = (lane & 1) * 8 + (lane >> 2);
  int sfb_unique_col = lane >> 2;

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

  const int K_iters = K / 64;
  const int K_half = K / 2;

  // M=1 padding: zero rows 1..15 of A and SFA in both banks once.
  if (lane < 16) {
    int row = lane;
    if (row >= 1) {
      int4* a0_v = reinterpret_cast<int4*>(s_A[0]);
      int4* a1_v = reinterpret_cast<int4*>(s_A[1]);
      int4 z; z.x = 0; z.y = 0; z.z = 0; z.w = 0;
      a0_v[row * 2 + 0] = z; a0_v[row * 2 + 1] = z;
      a1_v[row * 2 + 0] = z; a1_v[row * 2 + 1] = z;
    }
  }
  if (lane < 4) {
    for (int i = 4 + lane; i < 64; i += 4) {
      s_SFA[0][i] = 0;
      s_SFA[1][i] = 0;
    }
  }

  const int K_blocks = K / 16;
  const int n_col_super = (K_blocks + 3) / 4;

  auto issue_async_load = [&](int buf, int kt) {
    int byte_off = kt * 32;
    if (lane < 8) {
      cp_async_4(s_A[buf] + lane * 4, A_packed + byte_off + lane * 4);
    }
    if (lane == 0) {
      cp_async_4(s_SFA[buf] + 0, SFA + kt * 512);
    }
    {
      uint8_t* my_s_B = s_B_all[buf] + warp * (8 * 32);
      for (int c = 0; c < 2; ++c) {
        int chunk = lane + c * 32;
        int col = chunk >> 3;
        int off = chunk & 7;
        cp_async_4(
            my_s_B + chunk * 4,
            B_packed + (long long)(my_n_off + col) * K_half +
                byte_off + off * 4);
      }
    }
    if (lane < 8) {
      uint8_t* my_s_SFB = s_SFB_all[buf] + warp * (8 * 4);
      int col = my_n_off + lane;
      int rb = col >> 7;
      int ri = col & 127;
      int super_idx = rb * n_col_super + kt;
      int inner_base = (ri & 31) * 16 + ((ri >> 5) & 3) * 4;
      cp_async_4(
          my_s_SFB + lane * 4,
          SFB + (long long)super_idx * 512 + inner_base);
    }
  };

  issue_async_load(0, 0);
  cp_async_commit_group();
  if (K_iters > 1) {
    issue_async_load(1, 1);
    cp_async_commit_group();
  }

  for (int kt = 0; kt < K_iters; ++kt) {
    int curr_buf = kt & 1;
    if (kt + 1 < K_iters) {
      cp_async_wait_1();
    } else {
      cp_async_wait_0();
    }
    __syncwarp();

    uint32_t a0 = fast_load_a(s_A[curr_buf], t0, t1, 0);
    uint32_t a1 = fast_load_a(s_A[curr_buf], t0, t1, 1);
    uint32_t a2 = fast_load_a(s_A[curr_buf], t0, t1, 2);
    uint32_t a3 = fast_load_a(s_A[curr_buf], t0, t1, 3);
    uint8_t* my_s_B = s_B_all[curr_buf] + warp * (8 * 32);
    uint8_t* my_s_SFB = s_SFB_all[curr_buf] + warp * (8 * 4);
    uint32_t b0 = fast_load_b(my_s_B, t0, t1, 0);
    uint32_t b1 = fast_load_b(my_s_B, t0, t1, 1);
    uint32_t sfa = *reinterpret_cast<const uint32_t*>(
        s_SFA[curr_buf] + sfa_unique_row * 4);
    uint32_t sfb = *reinterpret_cast<const uint32_t*>(
        my_s_SFB + sfb_unique_col * 4);

    float d0, d1, d2, d3;
    int4_mma_16x8x64(d0, d1, d2, d3,
                     a0, a1, a2, a3, b0, b1,
                     c0, c1, c2, c3, sfa, sfb);
    c0 = d0; c1 = d1; c2 = d2; c3 = d3;

    if (kt + 2 < K_iters) {
      issue_async_load(curr_buf, kt + 2);
      cp_async_commit_group();
    }
  }

  int q = lane >> 2;
  int r = lane & 3;
  if (q == 0) {
    int col0 = my_n_off + r * 2;
    int col1 = col0 + 1;
    if (col0 < N) D[col0] = __float2bfloat16(c0 * alpha);
    if (col1 < N) D[col1] = __float2bfloat16(c1 * alpha);
  }
}

// ── Codebook canary ───────────────────────────────────────────────
// A = nibble 0x7 broadcast, B = nibble 0x1 broadcast, SF = 1.0.
// Every output element = 64 * dec(0x7) * dec(0x1):
//   E0M3 (patched):   64 * 7 * 1   = 448
//   E2M1 (unpatched): 64 * 6 * 0.5 = 192
// One warp, one MMA; lane 0 writes the verdict.

__global__ void int4_codebook_canary_kernel(int* __restrict__ verdict) {
  uint32_t a = 0x77777777u, b = 0x11111111u, sf = 0x38383838u;
  float d0, d1, d2, d3;
  int4_mma_16x8x64(d0, d1, d2, d3, a, a, a, a, b, b,
                   0.f, 0.f, 0.f, 0.f, sf, sf);
  if (threadIdx.x == 0) {
    *verdict = (d0 == 448.0f) ? 0 : ((d0 == 192.0f) ? 1 : 2);
  }
}

}  // namespace

// ── Host dispatch ─────────────────────────────────────────────────

int int4_w4a4_mma_sm120_full_n_bf16out(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    int          N,
    int          K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream) {
  if (!A_packed || !B_packed || !D_bf16 || !SFA || !SFB) return 1;
  if (K <= 0 || (K % 64) != 0) return 2;
  if (N <= 0 || (N % INT4_COLS_PER_BLOCK) != 0) return 3;

  dim3 block(INT4_THREADS_PER_BLOCK);
  dim3 grid((N + INT4_COLS_PER_BLOCK - 1) / INT4_COLS_PER_BLOCK);
  int4_full_n_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(A_packed),
      reinterpret_cast<const uint8_t*>(B_packed),
      reinterpret_cast<const uint8_t*>(SFA),
      reinterpret_cast<const uint8_t*>(SFB),
      reinterpret_cast<__nv_bfloat16*>(D_bf16),
      alpha, N, K);
  return (cudaGetLastError() == cudaSuccess) ? 0 : 100;
}

int int4_quantize_bf16_sm120(
    const void*  x_bf16,
    void*        out_packed,
    void*        out_sf_swizzled,
    const void*  global_scale,
    int          rows,
    int          K,
    cudaStream_t stream) {
  if (!x_bf16 || !out_packed || !out_sf_swizzled || !global_scale) return 1;
  if (K <= 0 || (K % 64) != 0 || rows <= 0) return 2;
  dim3 block(128);
  dim3 grid(rows);
  int4_quantize_bf16_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<uint8_t*>(out_packed),
      reinterpret_cast<uint8_t*>(out_sf_swizzled),
      reinterpret_cast<const float*>(global_scale),
      rows, K);
  return (cudaGetLastError() == cudaSuccess) ? 0 : 100;
}

int int4_global_scale_bf16_sm120(
    const void*  x_bf16,
    void*        scale_out,
    long long    numel,
    cudaStream_t stream) {
  if (!x_bf16 || !scale_out || numel <= 0) return 1;
  int blocks = (int)((numel + 255) / 256);
  if (blocks > 1024) blocks = 1024;
  int4_global_scale_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<float*>(scale_out),
      numel);
  return (cudaGetLastError() == cudaSuccess) ? 0 : 100;
}

int int4_w4a4_sm120_codebook_canary(cudaStream_t stream) {
  int* d_verdict = nullptr;
  if (cudaMalloc(&d_verdict, sizeof(int)) != cudaSuccess) return -1;
  int4_codebook_canary_kernel<<<1, 32, 0, stream>>>(d_verdict);
  if (cudaGetLastError() != cudaSuccess) { cudaFree(d_verdict); return -1; }
  int verdict = -2;
  cudaError_t err = cudaMemcpyAsync(&verdict, d_verdict, sizeof(int),
                                    cudaMemcpyDeviceToHost, stream);
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  cudaFree(d_verdict);
  if (err != cudaSuccess) return -1;
  return verdict;
}

}  // namespace gemm
}  // namespace flash_rt

// ── extern "C" wrappers (ctypes / non-C++ backends, e.g. GGML) ────
//
// Prefixed `flashrt_int4_sm120_` so the unmangled global symbols do not
// collide with a host application (e.g. llama.cpp / GGML) when this TU is
// linked into or dlopen'd alongside it. The C++ entry points in the
// header keep the `flash_rt::gemm` namespace.

extern "C" {

int flashrt_int4_sm120_w4a4_full_n_bf16out(
    const void* A, const void* B, void* D, int N, int K,
    const void* SFA, const void* SFB, float alpha, void* stream) {
  return flash_rt::gemm::int4_w4a4_mma_sm120_full_n_bf16out(
      A, B, D, N, K, SFA, SFB, alpha,
      reinterpret_cast<cudaStream_t>(stream));
}

int flashrt_int4_sm120_quantize_bf16(
    const void* x, void* out_packed, void* out_sf,
    const void* global_scale, int rows, int K, void* stream) {
  return flash_rt::gemm::int4_quantize_bf16_sm120(
      x, out_packed, out_sf, global_scale, rows, K,
      reinterpret_cast<cudaStream_t>(stream));
}

int flashrt_int4_sm120_global_scale_bf16(
    const void* x, void* scale_out, long long numel, void* stream) {
  return flash_rt::gemm::int4_global_scale_bf16_sm120(
      x, scale_out, numel, reinterpret_cast<cudaStream_t>(stream));
}

int flashrt_int4_sm120_codebook_canary(void* stream) {
  return flash_rt::gemm::int4_w4a4_sm120_codebook_canary(
      reinterpret_cast<cudaStream_t>(stream));
}

}  // extern "C"

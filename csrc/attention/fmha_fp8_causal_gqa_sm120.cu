// SPDX-License-Identifier: Apache-2.0
//
// FP8 causal GQA FlashAttention, SM120a.
// Header: fmha_fp8_causal_gqa_sm120.cuh
//
// One CTA per (query-tile, query-head): 8 MMA warps over a 128-row
// query tile, iterating the causal KV range with the fp8 m16n8k32 MMA
// (fp32 accumulate) and online softmax. The design notes:
//   * K/V global loads are cp.async into raw staging buffers and
//     prefetched one KV tile ahead, overlapping the QK/softmax/PV
//     compute of the current tile.
//   * Q/K shared tiles use a 16-byte-chunk XOR swizzle so the MMA
//     fragment loads are bank-conflict free.
//   * The V tile is transposed into (d, k) order for the P*V B-operand;
//     its staging row stride is padded 128 -> 136 bytes, which breaks
//     the same-bank alignment of consecutive d-rows (the transpose
//     scatter would otherwise serialise 8-way).
//   * P is rescaled by a fixed power of two (PSCALE) before the e4m3
//     requantisation so the post-softmax values use the format's
//     resolution; the epilogue divides it back out together with the
//     softmax normaliser.

#include "fmha_fp8_causal_gqa_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace flash_rt {
namespace attention {
namespace {

constexpr int kHeadDim = 128;      // d
constexpr int kQHeads  = 32;       // Hq
constexpr int kKVHeads = 8;        // Hkv (GQA 4:1)
constexpr int kQTile   = 128;      // query rows per CTA
constexpr int kKTile   = 128;      // kv rows per inner step
constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kNBlk    = kKTile / 8;
constexpr int kDBlk    = kHeadDim / 8;
constexpr int kVtPad   = 136;      // padded Vt row stride (bank-conflict fix)
constexpr float kPScale = 64.f;    // P e4m3 requantisation scale

__device__ __forceinline__ void mma_fp8_k32(
    float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// XOR swizzle for a 128-byte-row fp8 tile: rotates the 16-byte chunk by
// (row & 7) so the 8 rows of an MMA fragment hit 8 different banks.
__device__ __forceinline__ int sw(int row, int col) {
  return row * kHeadDim + (((col >> 4) ^ (row & 7)) << 4) + (col & 15);
}

__device__ __forceinline__ void cp_async16(void* dst_smem, const void* src_gmem) {
  unsigned a = __cvta_generic_to_shared(dst_smem);
  asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" ::"r"(a), "l"(src_gmem));
}

// NVFP4 encode helpers (same formulas as the standalone quantize kernels,
// replicated so the epilogue emits byte-identical packed/SF output).
__device__ __forceinline__ uint8_t fp4_e2m1_encode(float v) {
  float a = fabsf(v);
  uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
  uint8_t mag = (uint8_t)((a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f)
                        + (a >= 1.75f) + (a >= 2.5f)  + (a >= 3.5f)
                        + (a >= 5.0f));
  return sign | mag;
}
__device__ __forceinline__ uint8_t ue4m3_ceil_encode(float v) {
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
__device__ __forceinline__ float ue4m3_decode(uint8_t v) {
  int e = (v >> 3) & 0xF;
  int m = v & 0x7;
  if (e == 0) return ldexpf((float)m / 8.0f, -6);
  return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

// Fp4Out=false: O written as bf16 (Lq, Hq, 128).
// Fp4Out=true : O emitted directly as NVFP4 (packed (Lq, Hq*128/2) u8 +
//   swizzled ue4m3 SF, the o_proj GEMM's A-operand format), skipping the
//   bf16 O round-trip and the standalone quantize launch. Values round
//   through bf16 in-register first, so packed/SF bytes are identical to
//   the [bf16 O write + quantize kernel] chain.
template <bool Fp4Out>
__global__ void __launch_bounds__(kThreads, 1)
fmha_fp8_kernel(const __nv_fp8_e4m3* __restrict__ Q,
                const __nv_fp8_e4m3* __restrict__ K,
                const __nv_fp8_e4m3* __restrict__ V,
                __nv_bfloat16* __restrict__ O,
                uint8_t* __restrict__ o_fp4,
                uint8_t* __restrict__ o_sf, float scale) {
  __shared__ __align__(16) uint8_t Qs[kQTile * kHeadDim], Ks[kKTile * kHeadDim],
      Vt[kHeadDim * kVtPad], Ps[kQTile * kKTile], Kraw[kKTile * kHeadDim],
      Vraw[kKTile * kHeadDim];
  const int qt = blockIdx.x, head = blockIdx.y, kvh = head / (kQHeads / kKVHeads);
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, g = lane >> 2, tt = lane & 3;
  const int q_base = qt * kQTile;

  for (int j = t; j < kQTile * kHeadDim / 16; j += kThreads) {
    int r = j / (kHeadDim / 16), c = j % (kHeadDim / 16);
    ((uint4*)Qs)[r * (kHeadDim / 16) + (c ^ (r & 7))] = *(const uint4*)&(
        (const uint8_t*)Q)[(size_t)(q_base + r) * kQHeads * kHeadDim + head * kHeadDim + c * 16];
  }
  const int qrow0 = warp * 16;
  float Oc[kDBlk][4];
  float mi0 = -1e30f, mi1 = -1e30f, li0 = 0, li1 = 0;
#pragma unroll
  for (int nb = 0; nb < kDBlk; nb++) { Oc[nb][0] = Oc[nb][1] = Oc[nb][2] = Oc[nb][3] = 0; }

  auto load_raw = [&](int kv) {
    int kb = kv * kKTile;
    for (int j = t; j < kKTile * kHeadDim / 16; j += kThreads) {
      int k = j / (kHeadDim / 16), c = j % (kHeadDim / 16);
      cp_async16(&Kraw[k * kHeadDim + c * 16],
                 &((const uint8_t*)K)[(size_t)(kb + k) * kKVHeads * kHeadDim + kvh * kHeadDim + c * 16]);
      cp_async16(&Vraw[k * kHeadDim + c * 16],
                 &((const uint8_t*)V)[(size_t)(kb + k) * kKVHeads * kHeadDim + kvh * kHeadDim + c * 16]);
    }
  };
  // Kraw -> swizzled Ks (16B chunks); Vraw -> transposed Vt via a 4x4
  // register transpose, stored as uint32 rows into the padded layout.
  auto stage_tiles = [&]() {
    for (int j = t; j < kKTile * kHeadDim / 16; j += kThreads) {
      int k = j / (kHeadDim / 16), c = j % (kHeadDim / 16);
      ((uint4*)Ks)[k * (kHeadDim / 16) + (c ^ (k & 7))] = *(const uint4*)&Kraw[k * kHeadDim + c * 16];
    }
    for (int j = t; j < (kKTile / 4) * (kHeadDim / 4); j += kThreads) {
      int kt = j / (kHeadDim / 4), dt = j % (kHeadDim / 4);
      int k0 = kt * 4, d0 = dt * 4;
      uint32_t r0 = *(const uint32_t*)&Vraw[(k0 + 0) * kHeadDim + d0];
      uint32_t r1 = *(const uint32_t*)&Vraw[(k0 + 1) * kHeadDim + d0];
      uint32_t r2 = *(const uint32_t*)&Vraw[(k0 + 2) * kHeadDim + d0];
      uint32_t r3 = *(const uint32_t*)&Vraw[(k0 + 3) * kHeadDim + d0];
      uint32_t c0 = (r0 & 0xFF) | ((r1 & 0xFF) << 8) | ((r2 & 0xFF) << 16) | ((r3 & 0xFF) << 24);
      uint32_t c1 = ((r0 >> 8) & 0xFF) | (((r1 >> 8) & 0xFF) << 8) | (((r2 >> 8) & 0xFF) << 16) | (((r3 >> 8) & 0xFF) << 24);
      uint32_t c2 = ((r0 >> 16) & 0xFF) | (((r1 >> 16) & 0xFF) << 8) | (((r2 >> 16) & 0xFF) << 16) | (((r3 >> 16) & 0xFF) << 24);
      uint32_t c3 = ((r0 >> 24) & 0xFF) | (((r1 >> 24) & 0xFF) << 8) | (((r2 >> 24) & 0xFF) << 16) | (((r3 >> 24) & 0xFF) << 24);
      *(uint32_t*)&Vt[(d0 + 0) * kVtPad + ((((k0 >> 4) ^ ((d0 + 0) & 7)) << 4) + (k0 & 15))] = c0;
      *(uint32_t*)&Vt[(d0 + 1) * kVtPad + ((((k0 >> 4) ^ ((d0 + 1) & 7)) << 4) + (k0 & 15))] = c1;
      *(uint32_t*)&Vt[(d0 + 2) * kVtPad + ((((k0 >> 4) ^ ((d0 + 2) & 7)) << 4) + (k0 & 15))] = c2;
      *(uint32_t*)&Vt[(d0 + 3) * kVtPad + ((((k0 >> 4) ^ ((d0 + 3) & 7)) << 4) + (k0 & 15))] = c3;
    }
  };

  load_raw(0);
  asm volatile("cp.async.commit_group;\n");
  for (int kv = 0; kv <= qt; ++kv) {
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();
    stage_tiles();
    __syncthreads();
    if (kv + 1 <= qt) { load_raw(kv + 1); asm volatile("cp.async.commit_group;\n"); }

    float C[kNBlk][4];
#pragma unroll
    for (int nb = 0; nb < kNBlk; nb++) { C[nb][0] = C[nb][1] = C[nb][2] = C[nb][3] = 0; }
#pragma unroll
    for (int ks = 0; ks < kHeadDim / 32; ks++) {
      int ko = ks * 32;
      uint32_t a0 = *(const uint32_t*)&Qs[sw(qrow0 + g, ko + tt * 4)];
      uint32_t a1 = *(const uint32_t*)&Qs[sw(qrow0 + g + 8, ko + tt * 4)];
      uint32_t a2 = *(const uint32_t*)&Qs[sw(qrow0 + g, ko + 16 + tt * 4)];
      uint32_t a3 = *(const uint32_t*)&Qs[sw(qrow0 + g + 8, ko + 16 + tt * 4)];
#pragma unroll
      for (int nb = 0; nb < kNBlk; nb++) {
        int n = nb * 8;
        uint32_t b0 = *(const uint32_t*)&Ks[sw(n + g, ko + tt * 4)];
        uint32_t b1 = *(const uint32_t*)&Ks[sw(n + g, ko + 16 + tt * 4)];
        mma_fp8_k32(C[nb], a0, a1, a2, a3, b0, b1);
      }
    }
    if (kv == qt) {  // diagonal tile: mask key column > query row
      int gr0 = qrow0 + g, gr1 = qrow0 + g + 8;
#pragma unroll
      for (int nb = 0; nb < kNBlk; nb++) {
        int c0 = nb * 8 + 2 * tt, c1 = c0 + 1;
        if (c0 > gr0) C[nb][0] = -1e30f;
        if (c1 > gr0) C[nb][1] = -1e30f;
        if (c0 > gr1) C[nb][2] = -1e30f;
        if (c1 > gr1) C[nb][3] = -1e30f;
      }
    }
    float mt0 = -1e30f, mt1 = -1e30f;
#pragma unroll
    for (int nb = 0; nb < kNBlk; nb++) {
      mt0 = fmaxf(mt0, fmaxf(C[nb][0], C[nb][1]));
      mt1 = fmaxf(mt1, fmaxf(C[nb][2], C[nb][3]));
    }
    mt0 *= scale; mt1 *= scale;
    mt0 = fmaxf(mt0, __shfl_xor_sync(~0u, mt0, 1)); mt0 = fmaxf(mt0, __shfl_xor_sync(~0u, mt0, 2));
    mt1 = fmaxf(mt1, __shfl_xor_sync(~0u, mt1, 1)); mt1 = fmaxf(mt1, __shfl_xor_sync(~0u, mt1, 2));
    float mn0 = fmaxf(mi0, mt0), mn1 = fmaxf(mi1, mt1);
    float corr0 = __expf(mi0 - mn0), corr1 = __expf(mi1 - mn1);
    float lt0 = 0, lt1 = 0;
#pragma unroll
    for (int nb = 0; nb < kNBlk; nb++) {
      C[nb][0] = __expf(C[nb][0] * scale - mn0); C[nb][1] = __expf(C[nb][1] * scale - mn0);
      lt0 += C[nb][0] + C[nb][1];
      C[nb][2] = __expf(C[nb][2] * scale - mn1); C[nb][3] = __expf(C[nb][3] * scale - mn1);
      lt1 += C[nb][2] + C[nb][3];
    }
    lt0 += __shfl_xor_sync(~0u, lt0, 1); lt0 += __shfl_xor_sync(~0u, lt0, 2);
    lt1 += __shfl_xor_sync(~0u, lt1, 1); lt1 += __shfl_xor_sync(~0u, lt1, 2);
    li0 = li0 * corr0 + lt0; li1 = li1 * corr1 + lt1; mi0 = mn0; mi1 = mn1;
#pragma unroll
    for (int nb = 0; nb < kDBlk; nb++) {
      Oc[nb][0] *= corr0; Oc[nb][1] *= corr0; Oc[nb][2] *= corr1; Oc[nb][3] *= corr1;
    }
    __syncthreads();
#pragma unroll
    for (int nb = 0; nb < kNBlk; nb++) {
      int n = nb * 8;
      Ps[sw(qrow0 + g, n + 2 * tt + 0)] = __nv_fp8_e4m3(C[nb][0] * kPScale).__x;
      Ps[sw(qrow0 + g, n + 2 * tt + 1)] = __nv_fp8_e4m3(C[nb][1] * kPScale).__x;
      Ps[sw(qrow0 + g + 8, n + 2 * tt + 0)] = __nv_fp8_e4m3(C[nb][2] * kPScale).__x;
      Ps[sw(qrow0 + g + 8, n + 2 * tt + 1)] = __nv_fp8_e4m3(C[nb][3] * kPScale).__x;
    }
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < kKTile / 32; ks++) {
      int ko = ks * 32;
      uint32_t a0 = *(const uint32_t*)&Ps[sw(qrow0 + g, ko + tt * 4)];
      uint32_t a1 = *(const uint32_t*)&Ps[sw(qrow0 + g + 8, ko + tt * 4)];
      uint32_t a2 = *(const uint32_t*)&Ps[sw(qrow0 + g, ko + 16 + tt * 4)];
      uint32_t a3 = *(const uint32_t*)&Ps[sw(qrow0 + g + 8, ko + 16 + tt * 4)];
#pragma unroll
      for (int nb = 0; nb < kDBlk; nb++) {
        int n = nb * 8;
        uint32_t b0 = *(const uint32_t*)&Vt[(n + g) * kVtPad + ((((ko + tt * 4) >> 4) ^ ((n + g) & 7)) << 4) + ((ko + tt * 4) & 15)];
        uint32_t b1 = *(const uint32_t*)&Vt[(n + g) * kVtPad + ((((ko + 16 + tt * 4) >> 4) ^ ((n + g) & 7)) << 4) + ((ko + 16 + tt * 4) & 15)];
        mma_fp8_k32(Oc[nb], a0, a1, a2, a3, b0, b1);
      }
    }
  }
  float ip0 = 1.f / (li0 * kPScale), ip1 = 1.f / (li1 * kPScale);
  if constexpr (!Fp4Out) {
#pragma unroll
    for (int nb = 0; nb < kDBlk; nb++) {
      int d = nb * 8;
      size_t r0 = (size_t)(q_base + qrow0 + g) * kQHeads * kHeadDim + head * kHeadDim + d;
      size_t r1 = (size_t)(q_base + qrow0 + g + 8) * kQHeads * kHeadDim + head * kHeadDim + d;
      O[r0 + 2 * tt + 0] = __float2bfloat16(Oc[nb][0] * ip0);
      O[r0 + 2 * tt + 1] = __float2bfloat16(Oc[nb][1] * ip0);
      O[r1 + 2 * tt + 0] = __float2bfloat16(Oc[nb][2] * ip1);
      O[r1 + 2 * tt + 1] = __float2bfloat16(Oc[nb][3] * ip1);
    }
  } else {
    // Each 16-element quant block b of this head's 128 output columns lives
    // in one lane quad (same g, tt=0..3): nb=2b holds block columns
    // 2tt/2tt+1, nb=2b+1 holds 8+2tt/8+2tt+1. amax and the 8 packed bytes
    // assemble with two quad shuffles each; the tt==0 lane stores the
    // 8-byte block and its swizzled SF byte.
    constexpr int kK = kQHeads * kHeadDim;          // o_proj K = 4096
    constexpr int kNColBlocks = (kK / 16 + 3) / 4;  // SF swizzle col groups
#pragma unroll
    for (int half = 0; half < 2; ++half) {          // rows m0, m1
      const int row = q_base + qrow0 + g + 8 * half;
      const float ip = half ? ip1 : ip0;
      const int rb = row / 128, ri = row % 128;
#pragma unroll
      for (int b = 0; b < kHeadDim / 16; ++b) {
        float v0 = __bfloat162float(__float2bfloat16(Oc[2 * b][2 * half + 0] * ip));
        float v1 = __bfloat162float(__float2bfloat16(Oc[2 * b][2 * half + 1] * ip));
        float v2 = __bfloat162float(__float2bfloat16(Oc[2 * b + 1][2 * half + 0] * ip));
        float v3 = __bfloat162float(__float2bfloat16(Oc[2 * b + 1][2 * half + 1] * ip));
        float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, 2));
        uint8_t ue = ue4m3_ceil_encode(amax / 6.0f);
        float fscale = ue4m3_decode(ue);
        float inv = (fscale > 0.f) ? (1.f / fscale) : 0.f;
        // lane tt contributes packed byte tt (from nb=2b) and byte 4+tt.
        uint32_t lo = (uint32_t)((fp4_e2m1_encode(v1 * inv) << 4)
                                 | (fp4_e2m1_encode(v0 * inv) & 0xF)) << (8 * tt);
        uint32_t hi = (uint32_t)((fp4_e2m1_encode(v3 * inv) << 4)
                                 | (fp4_e2m1_encode(v2 * inv) & 0xF)) << (8 * tt);
        lo |= __shfl_xor_sync(~0u, lo, 1); lo |= __shfl_xor_sync(~0u, lo, 2);
        hi |= __shfl_xor_sync(~0u, hi, 1); hi |= __shfl_xor_sync(~0u, hi, 2);
        if (tt == 0) {
          uint2 pk = make_uint2(lo, hi);
          *reinterpret_cast<uint2*>(
              &o_fp4[(size_t)row * (kK / 2) + head * (kHeadDim / 2) + 8 * b]) = pk;
          const int gb = head * (kHeadDim / 16) + b;   // global col block
          const int cb = gb / 4, ci = gb % 4;
          o_sf[(size_t)(rb * kNColBlocks + cb) * 512
               + (ri % 32) * 16 + (ri / 32) * 4 + ci] = ue;
        }
      }
    }
  }
}

}  // namespace

int fmha_fp8_causal_gqa_nhd_d128(
    const void* q_fp8, const void* k_fp8, const void* v_fp8, void* out_bf16,
    int Lq, int Lk, int num_q_heads, int num_kv_heads,
    float softmax_scale, cudaStream_t stream) {
  if (Lq != Lk || Lq <= 0 || (Lq % kQTile) != 0) return 1;
  if (num_q_heads != kQHeads || num_kv_heads != kKVHeads) return 1;
  dim3 grid(Lq / kQTile, kQHeads);
  fmha_fp8_kernel<false><<<grid, kThreads, 0, stream>>>(
      (const __nv_fp8_e4m3*)q_fp8, (const __nv_fp8_e4m3*)k_fp8,
      (const __nv_fp8_e4m3*)v_fp8, (__nv_bfloat16*)out_bf16,
      nullptr, nullptr, softmax_scale);
  return 0;
}

int fmha_fp8_causal_gqa_nhd_d128_fp4out(
    const void* q_fp8, const void* k_fp8, const void* v_fp8,
    void* out_fp4, void* out_sf,
    int Lq, int Lk, int num_q_heads, int num_kv_heads,
    float softmax_scale, cudaStream_t stream) {
  if (Lq != Lk || Lq <= 0 || (Lq % kQTile) != 0) return 1;
  if (num_q_heads != kQHeads || num_kv_heads != kKVHeads) return 1;
  dim3 grid(Lq / kQTile, kQHeads);
  fmha_fp8_kernel<true><<<grid, kThreads, 0, stream>>>(
      (const __nv_fp8_e4m3*)q_fp8, (const __nv_fp8_e4m3*)k_fp8,
      (const __nv_fp8_e4m3*)v_fp8, nullptr,
      (uint8_t*)out_fp4, (uint8_t*)out_sf, softmax_scale);
  return 0;
}

}  // namespace attention
}  // namespace flash_rt

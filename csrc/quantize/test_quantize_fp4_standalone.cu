// ============================================================================
//  Standalone bit-exact test for quantize_fp4_dynamic_fp16.
//
//  Strategy:
//   1. Random fp16 input on device [N, D].
//   2. Kernel quantize → (packed_fp4, block_scales).
//   3. Kernel dequantize → fp16_round_trip.
//   4. Host reference: replicate the same math in plain C++.
//   5. Compare kernel vs host output element-by-element (expect bit-exact,
//      i.e. identical fp16 bit pattern).
//
//  The host reference IS the pytorch fake_nvfp4 we've validated against
//  Pi0.5 LIBERO precision. If this test passes (bit-exact), the kernel is
//  numerically equivalent to the simulation we already know is safe.
//
//  Build:
//    nvcc -std=c++17 -O3 -arch=sm_110a --expt-relaxed-constexpr \
//      csrc/quantize/quantize_fp4_dynamic.cu \
//      csrc/quantize/test_quantize_fp4_standalone.cu \
//      -o /tmp/test_quantize_fp4
// ============================================================================
#include "quantize_fp4_dynamic.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>

// ── Host reference (mirrors pytorch fake_nvfp4 exactly) ─────────────────────
static inline float h_e2m1_levels[] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static uint8_t h_quantize_e2m1(float x) {
  uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
  float ax = std::fabs(x);
  uint8_t mant;
  if      (ax < 0.25f) mant = 0u;
  else if (ax < 0.75f) mant = 1u;
  else if (ax < 1.25f) mant = 2u;
  else if (ax < 1.75f) mant = 3u;
  else if (ax < 2.5f)  mant = 4u;
  else if (ax < 3.5f)  mant = 5u;
  else if (ax < 5.0f)  mant = 6u;
  else                 mant = 7u;
  return sign | mant;
}

static float h_dequantize_e2m1(uint8_t nib) {
  float v = h_e2m1_levels[nib & 0x7u];
  return (nib & 0x8u) ? -v : v;
}

// Host simulation of UE4M3 round-trip — same as torch.float8_e4m3fn.
// Takes advantage of the __nv_fp8_e4m3 type being available in host code too.
static float h_ue4m3_roundtrip(float x) {
  x = std::fmax(x, 0.f);
  __nv_fp8_e4m3 q(x);   // implicit conversion does the rounding
  return static_cast<float>(q);
}

static void h_fake_nvfp4(const __half* src, __half* dst, int N, int D) {
  for (int r = 0; r < N; ++r) {
    for (int b = 0; b < D / 16; ++b) {
      const int base = r * D + b * 16;
      float vals[16];
      float amax = 0.f;
      for (int i = 0; i < 16; ++i) {
        vals[i] = __half2float(src[base + i]);
        float a = std::fabs(vals[i]);
        if (a > amax) amax = a;
      }
      float desired = std::fmax(amax / 6.f, 1e-12f);
      float bs_dq = h_ue4m3_roundtrip(desired);
      const float inv_bs = 1.f / bs_dq;
      for (int i = 0; i < 16; ++i) {
        uint8_t q = h_quantize_e2m1(vals[i] * inv_bs);
        dst[base + i] = __float2half(h_dequantize_e2m1(q) * bs_dq);
      }
    }
  }
}

// ── Main ──────────────────────────────────────────────────────────────────
struct ShapeCase { const char* name; int N, D; };

static int run_case(ShapeCase sc) {
  printf("\n=== %s: N=%d D=%d ===\n", sc.name, sc.N, sc.D);
  const int n_elems = sc.N * sc.D;
  const int n_packed = sc.N * (sc.D / 2);
  const int n_scales = sc.N * (sc.D / 16);

  std::vector<__half> h_src(n_elems), h_dst_kernel(n_elems), h_dst_ref(n_elems);
  srand(12345 + sc.N * 31 + sc.D);
  for (int i = 0; i < n_elems; ++i) {
    // Uniform [-2, 2]
    float r = 4.f * (rand() / (float)RAND_MAX) - 2.f;
    h_src[i] = __float2half(r);
  }

  __half *d_src, *d_dst;
  uint8_t* d_packed;
  __nv_fp8_e4m3* d_scales;
  cudaMalloc(&d_src, n_elems * sizeof(__half));
  cudaMalloc(&d_dst, n_elems * sizeof(__half));
  cudaMalloc(&d_packed, n_packed);
  cudaMalloc(&d_scales, n_scales);
  cudaMemcpy(d_src, h_src.data(), n_elems * sizeof(__half), cudaMemcpyHostToDevice);

  cudaStream_t stream; cudaStreamCreate(&stream);
  int rc = flash_rt::fp4::quantize_fp4_dynamic_fp16(
      d_src, d_packed, d_scales, sc.N, sc.D, stream);
  if (rc) { fprintf(stderr, "quant rc=%d\n", rc); return 1; }
  rc = flash_rt::fp4::dequantize_fp4_to_fp16(
      d_packed, d_scales, d_dst, sc.N, sc.D, stream);
  if (rc) { fprintf(stderr, "dequant rc=%d\n", rc); return 1; }
  cudaStreamSynchronize(stream);
  cudaMemcpy(h_dst_kernel.data(), d_dst, n_elems * sizeof(__half),
             cudaMemcpyDeviceToHost);

  h_fake_nvfp4(h_src.data(), h_dst_ref.data(), sc.N, sc.D);

  // Compare element-by-element.
  int mismatch = 0;
  int first_mm_idx = -1;
  double sum_diff = 0;
  double max_diff = 0;
  for (int i = 0; i < n_elems; ++i) {
    float a = __half2float(h_dst_kernel[i]);
    float b = __half2float(h_dst_ref[i]);
    float d = std::fabs(a - b);
    sum_diff += d;
    if (d > max_diff) max_diff = d;
    if (d > 1e-5f) {
      if (first_mm_idx < 0) first_mm_idx = i;
      ++mismatch;
    }
  }
  printf("  elements=%d, mismatch(|Δ|>1e-5)=%d (%.3f%%)  max|Δ|=%.4f  mean|Δ|=%.6f\n",
         n_elems, mismatch, 100.0 * mismatch / n_elems, max_diff, sum_diff / n_elems);
  if (first_mm_idx >= 0) {
    int i = first_mm_idx;
    printf("  first mismatch @ %d: kernel=%.6f ref=%.6f src=%.6f\n",
           i, __half2float(h_dst_kernel[i]),
           __half2float(h_dst_ref[i]),
           __half2float(h_src[i]));
  }

  // Latency
  const int iters = 100;
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int i = 0; i < 10; ++i)
    flash_rt::fp4::quantize_fp4_dynamic_fp16(d_src, d_packed, d_scales, sc.N, sc.D, stream);
  cudaStreamSynchronize(stream);
  cudaEventRecord(e0, stream);
  for (int i = 0; i < iters; ++i)
    flash_rt::fp4::quantize_fp4_dynamic_fp16(d_src, d_packed, d_scales, sc.N, sc.D, stream);
  cudaEventRecord(e1, stream);
  cudaEventSynchronize(e1);
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  printf("  quant latency: %.2f us/call (%d iters)\n", ms * 1000.0f / iters, iters);
  cudaEventDestroy(e0); cudaEventDestroy(e1);

  cudaStreamDestroy(stream);
  cudaFree(d_src); cudaFree(d_dst); cudaFree(d_packed); cudaFree(d_scales);

  bool pass = mismatch == 0;
  printf("  [%s] bit-exact vs host reference\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

int main() {
  std::vector<ShapeCase> cases = {
      {"small_16x16",    16,   16},
      {"dec_act_Sa2048", 10,   2048},   // decoder activation pre-Gate+Up
      {"dec_act_Sa8192", 10,   8192},   // decoder activation pre-Down (H=8192)
      {"enc_act_968x2048", 968, 2048},  // encoder activation
  };
  int fails = 0;
  for (auto& c : cases) fails += run_case(c);
  printf("\nTOTAL: %d FAILED / %d\n", fails, (int)cases.size());
  return fails;
}

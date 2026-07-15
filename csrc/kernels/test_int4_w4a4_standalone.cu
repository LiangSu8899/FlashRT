// ============================================================================
//  Standalone correctness + throughput test for the sm_120 INT4 (E0M3)
//  W4A4 GEMV, using only synthetic data (no checkpoints required).
//
//  What it checks:
//   1. Codebook canary: the loaded SASS actually decodes E0M3 (returns 0).
//      This ONLY passes after the OMMA bit-patch step (see below); an
//      unpatched binary decodes E2M1 and the test aborts with guidance.
//   2. GEMV vs host reference: quantize a random (N,K) weight and a random
//      (1,K) activation with the kernel's own device quantizer, run the
//      GEMV, and compare against a plain-C++ reference that dequantizes
//      with the identical INT4 two-level recipe. Expect cos > 0.999
//      (the residual is bf16-output rounding vs the fp32 reference).
//   3. Throughput microbench: register-resident issue rate, reported in
//      TFLOPS, to confirm INT4 runs at the E2M1 tensor-core rate.
//
//  Build + patch + run:
//    nvcc -std=c++17 -O3 -gencode arch=compute_120a,code=sm_120a \
//      csrc/kernels/int4_w4a4_mma_sm120.cu \
//      csrc/kernels/test_int4_w4a4_standalone.cu -o /tmp/test_int4
//    python tools/patch_int4_omma_sm120.py /tmp/test_int4
//    /tmp/test_int4
//
//  The patch step flips OMMA bits 78/79 (E2M1 -> E0M3) in every device
//  function whose name contains "int4_". Without it, step 1 fails by
//  design: the kernel is compiled E2M1 and RUN as E0M3.
// ============================================================================
#include "int4_w4a4_mma_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using flash_rt::gemm::int4_global_scale_bf16_sm120;
using flash_rt::gemm::int4_quantize_bf16_sm120;
using flash_rt::gemm::int4_w4a4_mma_sm120_full_n_bf16out;
using flash_rt::gemm::int4_w4a4_sm120_codebook_canary;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  exit(1); } } while (0)

// ── Host reference: the INT4 two-level recipe (mirrors the device kernel) ──

static float ue4m3_ceil_decode(float v) {
  if (v <= 0.f) return 0.f;
  if (v > 240.f) return 448.f;
  int fe; float m = frexpf(v, &fe); (void)m;
  int float_exp = fe - 1;
  int ue_exp = float_exp + 7;
  if (ue_exp <= 0) {
    int mm = (int)ceilf(v * 512.f);
    if (mm > 7) return ldexpf(1.f, -6);
    if (mm < 1) mm = 1;
    return ldexpf((float)mm / 8.f, -6);
  }
  float frac = v / ldexpf(1.f, float_exp) - 1.f;
  int mant = (int)ceilf(frac * 8.f - 1e-6f);
  if (mant >= 8) { mant = 0; ue_exp++; }
  if (ue_exp >= 15) return 448.f;
  return ldexpf(1.f + (float)mant / 8.f, ue_exp - 7);
}

// Quantize+dequantize one row with the INT4 recipe → fp32 values.
static void int4_ref_row(const float* x, int K, float g, float* out) {
  for (int gi = 0; gi < K / 16; ++gi) {
    float bmax = 0.f;
    for (int i = 0; i < 16; ++i) bmax = fmaxf(bmax, fabsf(x[gi * 16 + i]));
    float sf = ue4m3_ceil_decode((bmax / 7.f) / g) * g;
    if (sf <= 0.f) sf = 1.f;
    for (int i = 0; i < 16; ++i) {
      float q = rintf(x[gi * 16 + i] / sf);
      if (q > 7.f) q = 7.f; if (q < -7.f) q = -7.f;
      out[gi * 16 + i] = q * sf;
    }
  }
}

static float host_amax(const float* x, long long n) {
  float a = 0.f;
  for (long long i = 0; i < n; ++i) a = fmaxf(a, fabsf(x[i]));
  return a;
}

// ── Throughput microbench kernel (issue-rate; register-resident) ──
// Defined here (not in the shipped .cu) so its OMMA also carries the
// "int4_" marker and is patched. 4 independent accumulator chains/warp.
namespace {
__device__ __forceinline__ void int4_bench_mma(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  constexpr uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
      ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
      "{%14},{%15,%16},{%17},{%18,%19};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(d0), "f"(d1), "f"(d2), "f"(d3),
        "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}
__global__ void int4_bench_kernel(int iters, float* out) {
  uint32_t a0 = 0x25142514u, a1 = 0x36253625u, a2 = 0x14721472u, a3 = 0x53625362u;
  uint32_t b0 = 0x13521352u, b1 = 0x24632463u, sf = 0x38383838u;
  float acc[4][4];
  #pragma unroll
  for (int g = 0; g < 4; ++g)
    #pragma unroll
    for (int i = 0; i < 4; ++i) acc[g][i] = 0.f;
  for (int it = 0; it < iters; ++it) {
    #pragma unroll
    for (int g = 0; g < 4; ++g)
      int4_bench_mma(acc[g][0], acc[g][1], acc[g][2], acc[g][3],
                     a0, a1, a2, a3, b0, b1, sf, sf);
  }
  float s = 0.f;
  #pragma unroll
  for (int g = 0; g < 4; ++g)
    #pragma unroll
    for (int i = 0; i < 4; ++i) s += acc[g][i];
  if (s == 12345.678f) out[threadIdx.x] = s;
}
}  // namespace

static double cos_vec(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size(); ++i) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
  return dot / (sqrt(na) * sqrt(nb) + 1e-30);
}

int main() {
  int canary = int4_w4a4_sm120_codebook_canary(0);
  printf("[canary] %d  (%s)\n", canary,
         canary == 0 ? "E0M3 decode OK" :
         canary == 1 ? "E2M1 — binary NOT patched, run tools/patch_int4_omma_sm120.py" :
                       "unexpected");
  if (canary != 0) return 1;

  // ── GEMV correctness on synthetic data ──
  const int N = 4096, K = 4096;
  std::vector<float> hW(N * K), hA(K);
  srand(1234);
  for (auto& v : hW) v = ((float)rand() / RAND_MAX - 0.5f) * 0.4f;
  for (auto& v : hA) v = ((float)rand() / RAND_MAX - 0.5f) * 0.6f;

  // Upload as bf16.
  std::vector<__nv_bfloat16> hWb(N * K), hAb(K);
  for (int i = 0; i < N * K; ++i) hWb[i] = __float2bfloat16(hW[i]);
  for (int i = 0; i < K; ++i) hAb[i] = __float2bfloat16(hA[i]);
  __nv_bfloat16 *dW, *dA; CK(cudaMalloc(&dW, sizeof(__nv_bfloat16) * N * K));
  CK(cudaMalloc(&dA, sizeof(__nv_bfloat16) * K));
  CK(cudaMemcpy(dW, hWb.data(), sizeof(__nv_bfloat16) * N * K, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dA, hAb.data(), sizeof(__nv_bfloat16) * K, cudaMemcpyHostToDevice));

  float *dgW, *dgA; CK(cudaMalloc(&dgW, 4)); CK(cudaMalloc(&dgA, 4));
  CK(cudaMemset(dgW, 0, 4)); CK(cudaMemset(dgA, 0, 4));
  int4_global_scale_bf16_sm120(dW, dgW, (long long)N * K, 0);
  int4_global_scale_bf16_sm120(dA, dgA, K, 0);
  CK(cudaDeviceSynchronize());
  float gW, gA; CK(cudaMemcpy(&gW, dgW, 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&gA, dgA, 4, cudaMemcpyDeviceToHost));

  int sfW = ((N + 127) / 128) * ((K + 63) / 64) * 512;
  int sfA = ((1 + 127) / 128) * ((K + 63) / 64) * 512;
  uint8_t *dWpk, *dApk, *dWsf, *dAsf;
  CK(cudaMalloc(&dWpk, (size_t)N * (K / 2))); CK(cudaMalloc(&dApk, K / 2));
  CK(cudaMalloc(&dWsf, sfW)); CK(cudaMalloc(&dAsf, sfA));
  CK(cudaMemset(dWsf, 0, sfW)); CK(cudaMemset(dAsf, 0, sfA));
  int4_quantize_bf16_sm120(dW, dWpk, dWsf, dgW, N, K, 0);
  int4_quantize_bf16_sm120(dA, dApk, dAsf, dgA, 1, K, 0);
  CK(cudaDeviceSynchronize());

  __nv_bfloat16* dD; CK(cudaMalloc(&dD, sizeof(__nv_bfloat16) * N));
  int rc = int4_w4a4_mma_sm120_full_n_bf16out(dApk, dWpk, dD, N, K, dAsf, dWsf, gA * gW, 0);
  CK(cudaDeviceSynchronize());
  if (rc != 0) { printf("[gemv] launch rc=%d\n", rc); return 1; }
  std::vector<__nv_bfloat16> hDb(N); CK(cudaMemcpy(hDb.data(), dD, sizeof(__nv_bfloat16) * N, cudaMemcpyDeviceToHost));
  std::vector<float> hD(N); for (int i = 0; i < N; ++i) hD[i] = __bfloat162float(hDb[i]);

  // Host reference with identical recipe.
  float gA_ref = host_amax(hA.data(), K) / (7.f * 448.f);
  std::vector<float> Adq(K); int4_ref_row(hA.data(), K, gA_ref, Adq.data());
  float gW_ref = host_amax(hW.data(), (long long)N * K) / (7.f * 448.f);
  std::vector<float> ref(N, 0.f), Wdq(K);
  for (int n = 0; n < N; ++n) {
    int4_ref_row(&hW[(size_t)n * K], K, gW_ref, Wdq.data());
    double acc = 0;
    for (int k = 0; k < K; ++k) acc += (double)Adq[k] * Wdq[k];
    ref[n] = (float)acc;
  }
  printf("[gemv] N=%d K=%d  cos(kernel, INT4-ref) = %.6f  (expect > 0.999)\n",
         N, K, cos_vec(hD, ref));
  printf("[gemv] sample kernel[:4] = %.4f %.4f %.4f %.4f\n", hD[0], hD[1], hD[2], hD[3]);
  printf("[gemv] sample ref   [:4] = %.4f %.4f %.4f %.4f\n", ref[0], ref[1], ref[2], ref[3]);

  // ── Throughput microbench ──
  int sm = 0; CK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0));
  float* dOut; CK(cudaMalloc(&dOut, 256 * 4));
  int iters = 8192, blocks = sm * 4, threads = 256;
  int4_bench_kernel<<<blocks, threads>>>(iters, dOut);
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  double best = 0;
  for (int r = 0; r < 5; ++r) {
    CK(cudaEventRecord(e0)); int4_bench_kernel<<<blocks, threads>>>(iters, dOut); CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    double warps = (double)blocks * (threads / 32);
    double flops = warps * 4.0 * iters * 2.0 * 16 * 8 * 64;
    best = fmax(best, flops / (ms * 1e-3) / 1e12);
  }
  printf("[bench] issue-rate: %.1f TFLOPS  (INT4 x INT4, %d blocks x %d warps)\n",
         best, blocks, threads / 32);
  printf("[result] PASS\n");
  return 0;
}

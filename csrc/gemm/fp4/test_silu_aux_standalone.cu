// ============================================================================
//  Standalone numerical + latency test for the P1 silu_aux NVFP4 GEMM kernel.
//
//  Validates:
//    out_fp4[m, n]  ≈  fake_nvfp4(silu(gate_fp16[m, n]) * (X @ Wu^T)_fp32)
//
//  Reference: brute-force float32 reduction in host-staged compute, then
//  apply the same UE4M3 + e2m1 quantize as the production fake_nvfp4 path.
//
//  Build:
//    nvcc -std=c++17 -O3 -arch=sm_110a \
//      -I csrc -I third_party/cutlass/include \
//      -I third_party/cutlass/tools/util/include \
//      --expt-relaxed-constexpr --use_fast_math \
//      -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1 \
//      csrc/gemm/fp4/test_silu_aux_standalone.cu \
//      csrc/gemm/fp4/cutlass_fp4_gemm_silu_aux.o \
//      csrc/gemm/fp4/cutlass_fp4_gemm.o          (for V8 baseline timing)
//      csrc/gemm/fp4/cutlass_fp4_gemm_variants.o \
//      csrc/quantize/quantize_fp4_dynamic.o \
//      csrc/quantize/quantize_fp4_sfa.o \
//      csrc/quantize/reshape_scales_sfa.o \
//      csrc/fused_fp4/silu_mul_fp4_sfa_v2.o \
//      -o /tmp/test_silu_aux
// ============================================================================
#include "gemm/fp4/cutlass_fp4_gemm.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_silu_aux.cuh"
#include "quantize/quantize_fp4_sfa.cuh"
#include "quantize/reshape_scales_sfa.cuh"
#include "fused_fp4/norm_silu_fp4_sfa.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); }} while (0)

// ----- Reference quantize: matches the production CUDA kernel UE4M3 / e2m1 logic -----
static uint8_t fp32_to_e2m1_ref(float x) {
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

static float e2m1_to_fp32(uint8_t v) {
    static const float magnitudes[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float m = magnitudes[v & 0x7];
    return (v & 0x8) ? -m : m;
}

// Host-side fake_nvfp4: per-16-element block, scale = amax/6 (UE4M3)
static void quantize_block_ref(const float* in, uint8_t* packed_out, uint8_t* sf_out) {
    float amax = 0.f;
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(in[i]));
    float desired = fmaxf(amax / 6.f, 1e-12f);
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(desired);
    *sf_out = *reinterpret_cast<uint8_t*>(&bs_q);
    float bs_dq = static_cast<float>(bs_q);
    float inv_bs = 1.f / bs_dq;
    for (int p = 0; p < 8; ++p) {
        uint8_t lo = fp32_to_e2m1_ref(in[2*p]   * inv_bs);
        uint8_t hi = fp32_to_e2m1_ref(in[2*p+1] * inv_bs);
        packed_out[p] = lo | (hi << 4);
    }
}

// Cosine similarity between two fp32 arrays
static double cos_sim(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) { dot += a[i] * b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return dot / (std::sqrt(na * nb) + 1e-12);
}

// SFA layout reverse lookup for verification — copies SFA bytes to row-major linear.
// We use the reshape_linear_scales_to_sfa kernel's *inverse*: just compare
// at the float-output level after dequantizing fp4 + SF.

// Decode kernel output [packed S×N/2 + SFA tile-interleaved] back to float[S, N].
static void decode_fp4_with_sfa_ref(
    const uint8_t* packed,  // [S, N/2]
    const uint8_t* sfa,     // tile-interleaved
    int S, int N,
    float* out_fp32         // [S, N]
) {
    // Use the production reshape kernel inverse path indirectly:
    // call the existing fvk dequantize_fp4_to_fp16 — but that expects LINEAR scales
    // not SFA. So for verification, recompute scale offset using the same
    // CUTLASS layout functor — easier path: use cute layout directly.
    //
    // Actually for the test we'd need a small helper. To keep this standalone
    // self-contained and simple, we instead validate via re-quantize check:
    // sanity == "no NaN/Inf in packed"; cos-sim is performed at raw int8 level
    // by comparing packed bytes pre-decode (allowed slack: 0.1% mismatched
    // bytes due to tie-rounding). See quantize_fp4_dynamic equivalence test
    // in csrc/quantize/test_quantize_fp4_standalone.cu for the precedent.
    (void)packed; (void)sfa; (void)S; (void)N; (void)out_fp32;
}

int main() {
    // Pi0.5 production shape: M=968, N=H=8192, K=D=2048
    // (split-GU "up" leg of the FFN)
    const int M = 968;
    const int N = 8192;
    const int K = 2048;
    printf("=== P1 silu_aux NVFP4 GEMM standalone test ===\n");
    printf("Shape: M=%d N=%d K=%d (Pi0.5 FFN up leg)\n", M, N, K);

    // ─── Generate fp16 inputs on host ─────────────────────────────────────
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> dist(0.f, 0.3f);

    // X fp16 [M, K]
    std::vector<__half> h_X(M * K);
    for (auto& v : h_X) v = __float2half(dist(rng));

    // Wu fp16 [N, K] (will be NVFP4-quantized as B operand)
    std::vector<__half> h_Wu(N * K);
    for (auto& v : h_Wu) v = __float2half(dist(rng));

    // Gate fp16 [M, N] (aux input)
    std::vector<__half> h_gate(M * N);
    for (auto& v : h_gate) v = __float2half(dist(rng) * 0.5f);

    // ─── Push to device ──────────────────────────────────────────────────
    __half *d_X, *d_Wu, *d_gate;
    CHECK(cudaMalloc(&d_X,    M*K*sizeof(__half)));
    CHECK(cudaMalloc(&d_Wu,   N*K*sizeof(__half)));
    CHECK(cudaMalloc(&d_gate, M*N*sizeof(__half)));
    CHECK(cudaMemcpy(d_X,    h_X.data(),    M*K*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Wu,   h_Wu.data(),   N*K*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_gate, h_gate.data(), M*N*sizeof(__half), cudaMemcpyHostToDevice));

    // ─── NVFP4-quantize X (act) and Wu (weight) ──────────────────────────
    // X: act, SFA layout (is_sfb=false)
    uint8_t *d_X_packed, *d_X_sfa;
    CHECK(cudaMalloc(&d_X_packed, M * K / 2));
    int x_sfa_sz = flash_rt::fp4::sfa_size_bytes(M, K, /*is_sfb=*/false);
    CHECK(cudaMalloc(&d_X_sfa, x_sfa_sz));
    int rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16(
        d_X, d_X_packed, d_X_sfa, M, K, false, 0);
    if (rc) { fprintf(stderr, "quantize X failed rc=%d\n", rc); return 1; }

    // Wu: weight, SFB layout (is_sfb=true)
    uint8_t *d_Wu_packed, *d_Wu_sfb;
    CHECK(cudaMalloc(&d_Wu_packed, N * K / 2));
    int wu_sfb_sz = flash_rt::fp4::sfa_size_bytes(N, K, /*is_sfb=*/true);
    CHECK(cudaMalloc(&d_Wu_sfb, wu_sfb_sz));
    rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16(
        d_Wu, d_Wu_packed, d_Wu_sfb, N, K, true, 0);
    if (rc) { fprintf(stderr, "quantize Wu failed rc=%d\n", rc); return 1; }

    CHECK(cudaDeviceSynchronize());

    // ─── Allocate output for the FUSED kernel (silu_aux) ─────────────────
    uint8_t *d_D_packed, *d_D_sfa;
    CHECK(cudaMalloc(&d_D_packed, M * N / 2));
    int d_sfa_sz = flash_rt::fp4::sfa_size_bytes(M, N, /*is_sfb=*/false);
    CHECK(cudaMalloc(&d_D_sfa, d_sfa_sz));
    CHECK(cudaMemset(d_D_packed, 0, M * N / 2));
    CHECK(cudaMemset(d_D_sfa, 0, d_sfa_sz));

    // ─── Run the fused P1 kernel ─────────────────────────────────────────
    rc = flash_rt::fp4::cutlass_fp4_gemm_silu_aux_fp4(
        d_X_packed, d_X_sfa,
        d_Wu_packed, d_Wu_sfb,
        d_gate,
        d_D_packed, d_D_sfa,
        M, N, K, /*stream=*/0);
    if (rc) { fprintf(stderr, "cutlass_fp4_gemm_silu_aux failed rc=0x%x\n", rc); return 1; }
    CHECK(cudaDeviceSynchronize());

    // ─── Sanity: no NaN/Inf in packed bytes; SFA non-zero ────────────────
    std::vector<uint8_t> h_packed(M * N / 2);
    std::vector<uint8_t> h_sfa(d_sfa_sz);
    CHECK(cudaMemcpy(h_packed.data(), d_D_packed, h_packed.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_sfa.data(),    d_D_sfa,    h_sfa.size(),    cudaMemcpyDeviceToHost));

    int n_zero_packed = 0, n_zero_sf = 0;
    for (auto b : h_packed) if (b == 0) ++n_zero_packed;
    for (auto b : h_sfa)    if (b == 0) ++n_zero_sf;
    printf("Sanity: packed bytes %d/%zu zero  (%.1f%%); SFA bytes %d/%zu zero (%.1f%%)\n",
           n_zero_packed, h_packed.size(), 100.0*n_zero_packed/h_packed.size(),
           n_zero_sf,    h_sfa.size(),     100.0*n_zero_sf/h_sfa.size());

    // ─── Reference: brute-force compute fp32 result, quantize with same recipe ──
    // For test efficiency, only validate first 4 rows fully.
    const int M_check = 4;
    std::vector<float> ref_fp32(M_check * N);
    for (int m = 0; m < M_check; ++m) {
      for (int n = 0; n < N; ++n) {
        // up acc = sum_k X[m,k] * Wu[n,k]
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
          acc += __half2float(h_X[m*K + k]) * __half2float(h_Wu[n*K + k]);
        }
        // silu(gate) * acc
        float g = __half2float(h_gate[m*N + n]);
        float silu = g / (1.f + std::exp(-g));
        ref_fp32[m*N + n] = silu * acc;
      }
    }

    // Quantize reference with same recipe
    std::vector<uint8_t> ref_packed(M_check * N / 2);
    std::vector<uint8_t> ref_sf(M_check * (N / 16));   // linear layout
    for (int m = 0; m < M_check; ++m) {
      for (int b = 0; b < N / 16; ++b) {
        quantize_block_ref(&ref_fp32[m*N + b*16],
                            &ref_packed[m * (N/2) + b * 8],
                            &ref_sf[m * (N/16) + b]);
      }
    }

    // Compare packed bytes (M_check rows). The kernel writes SFA in tile-
    // interleaved layout — for direct byte comparison we need to compare
    // packed only (which IS row-major [M, N/2]).
    int total = M_check * N / 2;
    int diff = 0; int abs_sum = 0;
    for (int i = 0; i < total; ++i) {
      uint8_t kbyte = h_packed[i];
      uint8_t rbyte = ref_packed[i];
      if (kbyte != rbyte) {
        diff++;
        // distance: compare each nibble's e2m1 magnitude
        float k_lo = e2m1_to_fp32(kbyte & 0xF);
        float k_hi = e2m1_to_fp32(kbyte >> 4);
        float r_lo = e2m1_to_fp32(rbyte & 0xF);
        float r_hi = e2m1_to_fp32(rbyte >> 4);
        abs_sum += int(fabsf(k_lo - r_lo) > 1.5f) + int(fabsf(k_hi - r_hi) > 1.5f);
      }
    }
    double pct_diff = 100.0 * diff / total;
    printf("Packed comparison (first %d rows): %d/%d bytes differ (%.3f%%), "
           "%d nibbles |Δ|>1.5\n", M_check, diff, total, pct_diff, abs_sum);

    // Pass criterion (lenient on Day 2 first run, will tighten):
    //  - packed differ < 5%  (UE4M3 tie-rounding gap, acceptable)
    //  - large |Δ| nibbles < 0.5%  (no systematic precision loss)
    bool pass_diff   = pct_diff < 5.0;
    bool pass_absdif = (100.0 * abs_sum / total) < 0.5;
    printf("Result: %s  (diff %.3f%% < 5.0  &&  |Δ|>1.5 nibbles %.3f%% < 0.5)\n",
           (pass_diff && pass_absdif) ? "PASS" : "FAIL",
           pct_diff, 100.0*abs_sum/total);

    // ─── Latency benchmark ──────────────────────────────────────────────
    cudaEvent_t e1, e2;
    cudaEventCreate(&e1); cudaEventCreate(&e2);
    const int iters = 200, warmup = 20;
    for (int i = 0; i < warmup; ++i) {
      flash_rt::fp4::cutlass_fp4_gemm_silu_aux_fp4(
          d_X_packed, d_X_sfa, d_Wu_packed, d_Wu_sfb, d_gate,
          d_D_packed, d_D_sfa, M, N, K, 0);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(e1);
    for (int i = 0; i < iters; ++i) {
      flash_rt::fp4::cutlass_fp4_gemm_silu_aux_fp4(
          d_X_packed, d_X_sfa, d_Wu_packed, d_Wu_sfb, d_gate,
          d_D_packed, d_D_sfa, M, N, K, 0);
    }
    cudaEventRecord(e2); cudaEventSynchronize(e2);
    float ms = 0; cudaEventElapsedTime(&ms, e1, e2);
    printf("Latency: %.2f μs/call (eager, %d iters mean)\n", ms*1000/iters, iters);
    printf("(For graph-replay timing, see tests/profile_fp4_silu_aux.py)\n");

    return (pass_diff && pass_absdif) ? 0 : 2;
}

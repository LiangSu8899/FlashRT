// ============================================================================
//  Standalone sanity + latency test for cutlass_fp4_gemm_fp4out.
//
//  Validates:
//    1. Kernel runs to completion at production shape (no illegal access).
//    2. Output packed bytes are not all zero / NaN-pattern.
//    3. Output packed bytes vs the existing 2-kernel reference path
//       (cutlass_fp4_sq_fp16 → quantize_fp4_dynamic_sfa_fp16) have
//       <5% mismatched bytes (UE4M3 tie rounding gap, acceptable —
//       same precedent as Phase 4 standalone tests).
//    4. Latency vs the existing 2-kernel path (saving = -fp16 write/read
//       round-trip).
// ============================================================================
#include "gemm/fp4/cutlass_fp4_gemm.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_fp4out.cuh"
#include "quantize/quantize_fp4_sfa.cuh"
#include "quantize/reshape_scales_sfa.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); }} while (0)

int main() {
    // Half-N shape for the split GU/proj GEMM:  M=968 N=H=8192 K=D=2048
    const int M = 968, N = 8192, K = 2048;
    printf("=== fp4out NVFP4 GEMM standalone test ===\n");
    printf("Shape: M=%d N=%d K=%d (Pi0.5 split-GU half)\n", M, N, K);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s, SM%d%d\n", prop.name, prop.major, prop.minor);

    // ─ Allocate & init ─
    std::vector<__half> h_X(M*K), h_W(N*K);
    for (auto& v : h_X) v = __float2half((float)((rand() % 100) - 50) / 200.f);
    for (auto& v : h_W) v = __float2half((float)((rand() % 100) - 50) / 200.f);
    __half *d_X, *d_W;
    CHECK(cudaMalloc(&d_X, M*K*sizeof(__half)));
    CHECK(cudaMalloc(&d_W, N*K*sizeof(__half)));
    CHECK(cudaMemcpy(d_X, h_X.data(), M*K*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_W, h_W.data(), N*K*sizeof(__half), cudaMemcpyHostToDevice));

    uint8_t *X_p, *X_sfa, *W_p, *W_sfb;
    CHECK(cudaMalloc(&X_p,   M*K/2));
    CHECK(cudaMalloc(&W_p,   N*K/2));
    int x_sfa_sz = flash_rt::fp4::sfa_size_bytes(M, K, false);
    int w_sfb_sz = flash_rt::fp4::sfa_size_bytes(N, K, true);
    CHECK(cudaMalloc(&X_sfa, x_sfa_sz));
    CHECK(cudaMalloc(&W_sfb, w_sfb_sz));
    flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16(d_X, X_p, X_sfa, M, K, false, 0);
    flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16(d_W, W_p, W_sfb, N, K, true,  0);
    cudaDeviceSynchronize();

    // ─ Path A: new fp4out GEMM ─
    uint8_t *D_p, *D_sfd;
    int d_sfd_sz = flash_rt::fp4::sfa_size_bytes(M, N, false);
    CHECK(cudaMalloc(&D_p,   M*N/2));
    CHECK(cudaMalloc(&D_sfd, d_sfd_sz));
    CHECK(cudaMemset(D_p,   0, M*N/2));
    CHECK(cudaMemset(D_sfd, 0, d_sfd_sz));
    // norm_constant must be valid (not null) — allocate small device fp32 = 1.0
    float* d_norm = nullptr;
    CHECK(cudaMalloc(&d_norm, sizeof(float)));
    float h_norm = 1.0f;
    CHECK(cudaMemcpy(d_norm, &h_norm, sizeof(float), cudaMemcpyHostToDevice));
    int rc = flash_rt::fp4::cutlass_fp4_gemm_fp4out(
        X_p, X_sfa, W_p, W_sfb, D_p, D_sfd, M, N, K, 0);
    CHECK(cudaDeviceSynchronize());
    if (rc) { fprintf(stderr, "cutlass_fp4_gemm_fp4out failed rc=0x%x\n", rc); return 1; }
    printf("[A] fp4out GEMM: rc=0\n");

    // ─ Path B: reference 2-kernel = fp16-out GEMM + quant_fp4 ─
    __half* D_fp16;
    uint8_t *D_p_ref, *D_sfa_ref;
    CHECK(cudaMalloc(&D_fp16, M*N*sizeof(__half)));
    CHECK(cudaMalloc(&D_p_ref, M*N/2));
    CHECK(cudaMalloc(&D_sfa_ref, d_sfd_sz));
    rc = flash_rt::fp4::cutlass_fp4_sq_fp16(X_p, X_sfa, W_p, W_sfb, D_fp16, M, N, K, 1.0f, 0.0f, 0);
    CHECK(cudaDeviceSynchronize());
    if (rc) { fprintf(stderr, "fp16 ref GEMM failed rc=0x%x\n", rc); return 1; }
    rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16(D_fp16, D_p_ref, D_sfa_ref, M, N, false, 0);
    CHECK(cudaDeviceSynchronize());
    if (rc) { fprintf(stderr, "ref quantize_fp4 failed rc=%d\n", rc); return 1; }
    printf("[B] reference fp16 GEMM + fp4 quant: rc=0\n");

    // ─ Compare packed bytes ─
    std::vector<uint8_t> hA(M*N/2), hB(M*N/2);
    CHECK(cudaMemcpy(hA.data(), D_p,     hA.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hB.data(), D_p_ref, hB.size(), cudaMemcpyDeviceToHost));
    int diff = 0, nz_a = 0, nz_b = 0;
    for (size_t i = 0; i < hA.size(); ++i) {
        if (hA[i] != hB[i]) ++diff;
        if (hA[i] == 0) ++nz_a;
        if (hB[i] == 0) ++nz_b;
    }
    double pct = 100.0 * diff / hA.size();
    printf("Packed mismatch: %d/%zu (%.3f%%); zero-bytes A=%d B=%d\n",
           diff, hA.size(), pct, nz_a, nz_b);

    // ─ Compare SFA bytes ─
    std::vector<uint8_t> sA(d_sfd_sz), sB(d_sfd_sz);
    CHECK(cudaMemcpy(sA.data(), D_sfd,     sA.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sB.data(), D_sfa_ref, sB.size(), cudaMemcpyDeviceToHost));
    int sfa_diff = 0;
    for (size_t i = 0; i < sA.size(); ++i) if (sA[i] != sB[i]) ++sfa_diff;
    double sfa_pct = 100.0 * sfa_diff / sA.size();
    printf("SFA mismatch:    %d/%zu (%.3f%%)\n", sfa_diff, sA.size(), sfa_pct);

    bool pass = pct < 5.0 && nz_a < (int)hA.size() / 2 && sfa_pct < 5.0;
    printf("Verdict: %s  (packed diff %.3f%% < 5  &&  SFA diff %.3f%% < 5)\n",
           pass ? "PASS" : "FAIL", pct, sfa_pct);

    // ─ Latency: fp4out vs (fp16 GEMM + quant) ─
    cudaEvent_t e1, e2;
    cudaEventCreate(&e1); cudaEventCreate(&e2);
    const int iters = 200, warmup = 20;

    for (int i = 0; i < warmup; ++i) {
        flash_rt::fp4::cutlass_fp4_gemm_fp4out(X_p, X_sfa, W_p, W_sfb, D_p, D_sfd, M, N, K, 0);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(e1);
    for (int i = 0; i < iters; ++i) {
        flash_rt::fp4::cutlass_fp4_gemm_fp4out(X_p, X_sfa, W_p, W_sfb, D_p, D_sfd, M, N, K, 0);
    }
    cudaEventRecord(e2); cudaEventSynchronize(e2);
    float ms_a = 0; cudaEventElapsedTime(&ms_a, e1, e2);
    float us_a = ms_a * 1000 / iters;

    for (int i = 0; i < warmup; ++i) {
        flash_rt::fp4::cutlass_fp4_sq_fp16(X_p, X_sfa, W_p, W_sfb, D_fp16, M, N, K, 1.0f, 0.0f, 0);
        flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16(D_fp16, D_p_ref, D_sfa_ref, M, N, false, 0);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(e1);
    for (int i = 0; i < iters; ++i) {
        flash_rt::fp4::cutlass_fp4_sq_fp16(X_p, X_sfa, W_p, W_sfb, D_fp16, M, N, K, 1.0f, 0.0f, 0);
        flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16(D_fp16, D_p_ref, D_sfa_ref, M, N, false, 0);
    }
    cudaEventRecord(e2); cudaEventSynchronize(e2);
    float ms_b = 0; cudaEventElapsedTime(&ms_b, e1, e2);
    float us_b = ms_b * 1000 / iters;

    printf("\nLatency (eager, %d iters mean):\n", iters);
    printf("  [A] fp4out GEMM (1 launch)         : %7.2f μs\n", us_a);
    printf("  [B] fp16 GEMM + quant (2 launches) : %7.2f μs\n", us_b);
    printf("  Δ A vs B                            : %+7.2f μs\n", us_a - us_b);

    return pass ? 0 : 2;
}

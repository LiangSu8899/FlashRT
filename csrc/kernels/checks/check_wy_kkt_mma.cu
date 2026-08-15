// WY K*KT: scalar reference (transcribed from the production kernel)
// vs the MMA entry - numeric band + speed.
// Build: nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 \
//   check_wy_kkt_mma.cu -o check_wy_kkt_mma
#include "../gated_delta_wy_kkt_mma.cu"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){printf("err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
constexpr int kWyChunk = 64, kHD = 128;

__global__ void kkt_v1_kernel(
    const __nv_bfloat16* __restrict__ k16_l2, const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ g_cumsum, float* __restrict__ A,
    int S, int num_k_heads, int num_v_heads, int head_group_size) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= kWyChunk * kWyChunk) return;
  const int i = pair / kWyChunk, j = pair - i * kWyChunk;
  const int vh = blockIdx.y, chunk = blockIdx.z;
  const int si = chunk * kWyChunk + i, sj = chunk * kWyChunk + j;
  const size_t a_off = (((static_cast<size_t>(chunk) * num_v_heads + vh) * kWyChunk + i) * kWyChunk + j);
  if (i <= j || si >= S || sj >= S) { A[a_off] = 0.0f; return; }
  const int kh = vh / head_group_size;
  const size_t ki = (static_cast<size_t>(si) * num_k_heads + kh) * kHD;
  const size_t kj = (static_cast<size_t>(sj) * num_k_heads + kh) * kHD;
  float dot = 0.0f;
  #pragma unroll 16
  for (int d = 0; d < kHD; ++d)
    dot = fmaf((float)k16_l2[ki + d], (float)k16_l2[kj + d], dot);
  const float bi = (float)beta[(size_t)si * num_v_heads + vh];
  const float gi = (float)g_cumsum[(size_t)si * num_v_heads + vh];
  const float gj = (float)g_cumsum[(size_t)sj * num_v_heads + vh];
  A[a_off] = bi * dot * __expf(gi - gj);
}

int main() {
  int S = 2048, KH = 16, VH = 48, GRP = 3;
  int chunks = S / 64;
  size_t nK = (size_t)S * KH * kHD, nBG = (size_t)S * VH, nA = (size_t)chunks * VH * 64 * 64;
  std::vector<uint16_t> hK(nK), hB(nBG), hG(nBG);
  srand(11);
  auto rb = [&](float scale){ float f = ((rand()%2000)-1000)/1000.0f*scale; __nv_bfloat16 b=__float2bfloat16(f); return *reinterpret_cast<uint16_t*>(&b); };
  for (auto& x : hK) x = rb(1.0f);
  for (auto& x : hB) x = rb(0.9f);
  for (auto& x : hG) x = rb(2.0f);
  __nv_bfloat16 *K, *B, *G; float *A1, *A2;
  CK(cudaMalloc(&K, nK*2)); CK(cudaMalloc(&B, nBG*2)); CK(cudaMalloc(&G, nBG*2));
  CK(cudaMalloc(&A1, nA*4)); CK(cudaMalloc(&A2, nA*4));
  cudaMemcpy(K, hK.data(), nK*2, cudaMemcpyHostToDevice);
  cudaMemcpy(B, hB.data(), nBG*2, cudaMemcpyHostToDevice);
  cudaMemcpy(G, hG.data(), nBG*2, cudaMemcpyHostToDevice);
  dim3 g1((64*64+255)/256, VH, chunks);
  kkt_v1_kernel<<<g1, 256>>>(K, B, G, A1, S, KH, VH, GRP);
  qwen36_gdn_wy_kkt_b64_mma_bf16(K, B, G, A2, S, 0);
  CK(cudaDeviceSynchronize());
  std::vector<float> o1(nA), o2(nA);
  cudaMemcpy(o1.data(), A1, nA*4, cudaMemcpyDeviceToHost);
  cudaMemcpy(o2.data(), A2, nA*4, cudaMemcpyDeviceToHost);
  double maxrel = 0, maxabs = 0; size_t bad = 0;
  for (size_t i = 0; i < nA; ++i) {
    double d = fabs((double)o1[i] - o2[i]);
    maxabs = fmax(maxabs, d);
    double den = fmax(fabs((double)o1[i]), 1e-3);
    maxrel = fmax(maxrel, d / den);
    if (d / den > 1e-2 && d > 1e-3) ++bad;
  }
  printf("numeric: maxabs=%.3e maxrel=%.3e bad=%zu/%zu\n", maxabs, maxrel, bad, nA);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (auto which : {1, 2}) {
    cudaEventRecord(e0);
    for (int it = 0; it < 30; ++it) {
      if (which == 1) kkt_v1_kernel<<<g1, 256>>>(K, B, G, A1, S, KH, VH, GRP);
      else qwen36_gdn_wy_kkt_b64_mma_bf16(K, B, G, A2, S, 0);
    }
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 30;
    printf("v%d: %8.1f us/layer  (x48 layers = %.1f ms)\n", which, ms*1e3, ms*48);
  }
  return 0;
}

// Large-M NVFP4 GEMM tier: TFLOPS receipt on the prefill shape family.
// Build: nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 \
//   --expt-relaxed-constexpr -I <cutlass>/include \
//   -I <cutlass>/tools/util/include -I ../../csrc/gemm/fp4 \
//   bench_gemm_m256.cu -o bench_gemm_m256   (CUTLASS >= 4.5)
#include "../../gemm/fp4/cutlass_nvfp4_gemm_m256_sm120.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){printf("err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
int main() {
  const int M = 2044;
  int shapes[][2] = {{17408,5120},{5120,17408},{12288,5120},{16384,5120}};
  uint8_t *A,*B,*SFA,*SFB,*D;
  CK(cudaMalloc(&A,(size_t)M*17408/2)); CK(cudaMalloc(&B,(size_t)17408ull*17408/2));
  CK(cudaMalloc(&SFA,(size_t)M*17408/8)); CK(cudaMalloc(&SFB,(size_t)17408ull*17408/8));
  CK(cudaMalloc(&D,(size_t)M*17408*2));
  std::vector<uint8_t> h(1<<24); for (auto& x : h) x = rand() & 0xff;
  cudaMemcpy(A,h.data(),std::min((size_t)M*17408/2,h.size()),cudaMemcpyHostToDevice);
  cudaMemset(SFA,0x3f,(size_t)M*17408/8); cudaMemset(SFB,0x3f,(size_t)17408ull*17408/8);
  for (auto& sh : shapes) {
    int N = sh[0], K = sh[1];
    size_t ws = flash_rt::gemm::nvfp4_gemm_m256_sm120_workspace_size(M,N,K);
    void* wk; CK(cudaMalloc(&wk, ws ? ws : 4));
    int rc = flash_rt::gemm::nvfp4_gemm_m256_sm120_bf16(A,SFA,B,SFB,D,M,N,K,1.f,wk,0);
    CK(cudaDeviceSynchronize());
    if (rc) { printf("N=%d K=%d rc=%d\n",N,K,rc); cudaFree(wk); continue; }
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i=0;i<20;++i) flash_rt::gemm::nvfp4_gemm_m256_sm120_bf16(A,SFA,B,SFB,D,M,N,K,1.f,wk,0);
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms,e0,e1); ms/=20;
    printf("M=%d N=%5d K=%5d: %7.0f TFLOPS (roof ~2020)\n", M,N,K, 2.0*M*N*K/ms/1e9);
    cudaFree(wk);
  }
  return 0;
}

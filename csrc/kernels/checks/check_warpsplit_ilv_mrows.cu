#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
// Multi-row interleaved GEMM bit-exactness: random packed data, the
// multi-row kernel's M rows compared bit-for-bit against M independent
// M=1 interleaved GEMV calls (row r's scales staged at the production
// 512B-block row offset for the multi-row read, at block offset 0 for
// the per-row reference). Rows are independent under the shared MMA atom
// and the reduction order is identical, so any difference is a defect.
// Build: nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 \
//   --expt-relaxed-constexpr -I <cutlass>/include \
//   check_warpsplit_ilv_mrows.cu ../fp4_w4a4_mma_warpsplit_ilv_sm120.cu \
//   ../fp4_w4a4_mma_warpsplit_ilv_mrows_sm120.cu -o check_warpsplit_ilv_mrows
#include "../fp4_w4a4_mma_warpsplit_ilv_sm120.cuh"
#include "../fp4_w4a4_mma_warpsplit_ilv_mrows_sm120.cuh"
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){printf("err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
int main() {
  int fails = 0;
  int shapes[][2] = {{17408,5120},{5120,17408},{12288,5120},{1024,5120},{5120,6144},{16384,5120}};
  for (auto& sh : shapes) {
    int N = sh[0], K = sh[1], KH = K/2, KI = K/64;
    size_t bBytes=(size_t)N*KH, sfbBytes=((size_t)((N+127)/128))*((K/16+3)/4)*512+4096;
    std::vector<uint8_t> hB(bBytes), hSFB(sfbBytes), hBi(bBytes);
    srand(N ^ K);
    for (auto& x : hB) x = rand() & 0xff;
    for (auto& x : hSFB) x = 0x30 + (rand() & 0xf);
    for (int g = 0; g < N/8; ++g)
      for (int kt = 0; kt < KI; ++kt)
        for (int col = 0; col < 8; ++col)
          memcpy(&hBi[(size_t)g*KH*8 + (size_t)kt*256 + col*32],
                 &hB[(size_t)(g*8+col)*KH + (size_t)kt*32], 32);
    uint8_t *Bi,*SFB; CK(cudaMalloc(&Bi,bBytes)); CK(cudaMalloc(&SFB,sfbBytes));
    cudaMemcpy(Bi,hBi.data(),bBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(SFB,hSFB.data(),sfbBytes,cudaMemcpyHostToDevice);
    for (int M : {1, 2, 4, 7, 8, 16}) {
      std::vector<uint8_t> hA((size_t)M*KH), hS((size_t)M*KI*4);
      for (auto& x : hA) x = rand() & 0xff;
      for (auto& x : hS) x = 0x30 + (rand() & 0xf);
      std::vector<uint8_t> hSFAm((size_t)KI*512+4096, 0);
      for (int r = 0; r < M; ++r)
        for (int kt = 0; kt < KI; ++kt)
          memcpy(&hSFAm[(size_t)kt*512 + r*16], &hS[((size_t)r*KI+kt)*4], 4);
      uint8_t *A,*SFAm,*SFA1; __nv_bfloat16 *Dm,*D1;
      CK(cudaMalloc(&A,hA.size())); CK(cudaMalloc(&SFAm,hSFAm.size()));
      CK(cudaMalloc(&SFA1,(size_t)KI*512+4096));
      CK(cudaMalloc(&Dm,(size_t)M*N*2)); CK(cudaMalloc(&D1,N*2));
      cudaMemcpy(A,hA.data(),hA.size(),cudaMemcpyHostToDevice);
      cudaMemcpy(SFAm,hSFAm.data(),hSFAm.size(),cudaMemcpyHostToDevice);
      for (int w : {2, 4}) {
        if (KI % w) continue;
        cudaMemset(Dm,0,(size_t)M*N*2);
        int rcm = flash_rt::gemm::fp4_w4a4_mma_sm120_warpsplit_ilv_mrows_bf16out(
            A,Bi,Dm,M,N,K,SFAm,SFB,1.f,w,3,0);
        CK(cudaDeviceSynchronize());
        std::vector<uint16_t> om((size_t)M*N), o1(N);
        cudaMemcpy(om.data(),Dm,(size_t)M*N*2,cudaMemcpyDeviceToHost);
        int diff = 0, rc1sum = 0;
        for (int r = 0; r < M; ++r) {
          std::vector<uint8_t> hSFA1((size_t)KI*512+4096, 0);
          for (int kt = 0; kt < KI; ++kt)
            memcpy(&hSFA1[(size_t)kt*512], &hS[((size_t)r*KI+kt)*4], 4);
          cudaMemcpy(SFA1,hSFA1.data(),hSFA1.size(),cudaMemcpyHostToDevice);
          cudaMemset(D1,0,N*2);
          rc1sum += flash_rt::gemm::fp4_w4a4_mma_sm120_warpsplit_ilv_bf16out(
              A + (size_t)r*KH,Bi,D1,N,K,SFA1,SFB,1.f,w,3,0);
          CK(cudaDeviceSynchronize());
          cudaMemcpy(o1.data(),D1,N*2,cudaMemcpyDeviceToHost);
          for (int i = 0; i < N; ++i) diff += (om[(size_t)r*N+i] != o1[i]);
        }
        printf("N=%d K=%d M=%d w%d: rc=%d/%d bit-diff=%d %s\n",
               N,K,M,w,rcm,rc1sum,diff, diff? "FAIL":"OK");
        fails += (diff != 0) + rcm + rc1sum;
      }
      cudaFree(A);cudaFree(SFAm);cudaFree(SFA1);cudaFree(Dm);cudaFree(D1);
    }
    cudaFree(Bi);cudaFree(SFB);
  }
  printf(fails ? "CHECK FAILED\n" : "ALL BIT-EXACT\n");
  return fails != 0;
}

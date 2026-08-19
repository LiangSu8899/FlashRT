#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
// Interleaved-B GEMV bit-exactness: random packed data, host-side repack
// reference, base vs ilv outputs compared bit-for-bit across shapes and
// warp configs; the device repack entry is checked against the host
// reference bytes as well.
// Build: nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 \
//   -I <cutlass>/include check_warpsplit_ilv.cu -o check_warpsplit_ilv
#include "../fp4_w4a4_mma_warpsplit_sm120.cuh"
#include "../fp4_w4a4_mma_warpsplit_ilv_sm120.cuh"
#include <cstdio>
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
    std::vector<uint8_t> hB(bBytes), hSFB(sfbBytes), hA(KH), hSFA((size_t)KI*512+4096), hBi(bBytes);
    srand(N ^ K);
    for (auto& x : hB) x = rand() & 0xff;
    for (auto& x : hSFB) x = 0x30 + (rand() & 0xf);
    for (auto& x : hA) x = rand() & 0xff;
    for (auto& x : hSFA) x = 0x30 + (rand() & 0xf);
    // host repack: Bi[g*KH*8 + kt*256 + col*32 + off*4 .. +4] = B[(g*8+col)*KH + kt*32 + off*4]
    for (int g = 0; g < N/8; ++g)
      for (int kt = 0; kt < KI; ++kt)
        for (int col = 0; col < 8; ++col)
          memcpy(&hBi[(size_t)g*KH*8 + (size_t)kt*256 + col*32],
                 &hB[(size_t)(g*8+col)*KH + (size_t)kt*32], 32);
    uint8_t *A,*SFA,*B,*Bi,*Bi_dev,*SFB; __nv_bfloat16 *D1,*D2;
    CK(cudaMalloc(&A,KH)); CK(cudaMalloc(&SFA,hSFA.size())); CK(cudaMalloc(&B,bBytes));
    CK(cudaMalloc(&Bi,bBytes)); CK(cudaMalloc(&Bi_dev,bBytes)); CK(cudaMalloc(&SFB,sfbBytes));
    CK(cudaMalloc(&D1,N*2)); CK(cudaMalloc(&D2,N*2));
    cudaMemcpy(A,hA.data(),KH,cudaMemcpyHostToDevice);
    cudaMemcpy(SFA,hSFA.data(),hSFA.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(B,hB.data(),bBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(Bi,hBi.data(),bBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(SFB,hSFB.data(),sfbBytes,cudaMemcpyHostToDevice);
    flash_rt::gemm::fp4_w4a4_repack_b_ilv_sm120(B, Bi_dev, N, K, 0);
    CK(cudaDeviceSynchronize());
    { std::vector<uint8_t> dv(bBytes);
      cudaMemcpy(dv.data(), Bi_dev, bBytes, cudaMemcpyDeviceToHost);
      size_t rd = 0; for (size_t i2 = 0; i2 < bBytes; ++i2) rd += (dv[i2] != hBi[i2]);
      printf("N=%d K=%d device-repack vs host: byte-diff=%zu %s\n", N, K, rd, rd? "FAIL":"OK");
      fails += (rd != 0); }
    for (int w : {2, 4, 8}) {
      if (KI % w) continue;
      cudaMemset(D1,0,N*2); cudaMemset(D2,0,N*2);
      int r1 = flash_rt::gemm::fp4_w4a4_mma_sm120_warpsplit_bf16out(A,B,D1,N,K,SFA,SFB,1.f,w,3,0);
      int r2 = flash_rt::gemm::fp4_w4a4_mma_sm120_warpsplit_ilv_bf16out(A,Bi,D2,N,K,SFA,SFB,1.f,w,3,0);
      CK(cudaDeviceSynchronize());
      std::vector<uint16_t> o1(N), o2(N);
      cudaMemcpy(o1.data(),D1,N*2,cudaMemcpyDeviceToHost);
      cudaMemcpy(o2.data(),D2,N*2,cudaMemcpyDeviceToHost);
      int diff = 0; for (int i = 0; i < N; ++i) diff += (o1[i] != o2[i]);
      printf("N=%d K=%d w%d: rc=%d/%d bit-diff=%d %s\n", N,K,w,r1,r2,diff, diff? "FAIL":"OK");
      fails += (diff != 0) + r1 + r2;
    }
    cudaFree(A);cudaFree(SFA);cudaFree(B);cudaFree(Bi);cudaFree(SFB);cudaFree(D1);cudaFree(D2);
  }
  printf(fails ? "CHECK FAILED\n" : "ALL BIT-EXACT\n");
  return fails != 0;
}

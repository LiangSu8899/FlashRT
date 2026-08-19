// SPDX-License-Identifier: Apache-2.0
//
// Batched 64x64 unit-lower-triangular inverse. See header for the
// contract; same forward-substitution recurrence the batched cuBLAS
// solve runs, so the fp32 error class matches the path it replaces.
#include "kernels/batched_unit_ltri_inv64.cuh"

namespace flash_rt {
namespace kernels {
namespace {

template <int MPB>  // matrices per block; blockDim.x = 64*MPB
__global__ void unit_ltri_inv64_kernel(const float* __restrict__ A,
                                       float* __restrict__ X, int B) {
  __shared__ float Ls[MPB][64][65];
  const int s = threadIdx.x >> 6;
  const int c = threadIdx.x & 63;
  const int mi = blockIdx.x * MPB + s;
  if (mi >= B) return;
  const float* Ab = A + (size_t)mi * 4096;
  for (int idx = c; idx < 4096; idx += 64) {
    const int i = idx >> 6, j = idx & 63;
    Ls[s][i][j] = (j < i) ? Ab[idx] : 0.f;
  }
  __syncwarp();

  float x[64];
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    float acc = (i == c) ? 1.f : 0.f;
    #pragma unroll
    for (int j = 0; j < i; ++j)
      acc -= Ls[s][i][j] * x[j];
    x[i] = acc;
  }

  float* Xb = X + (size_t)mi * 4096;
  #pragma unroll
  for (int i = 0; i < 64; ++i)
    Xb[i * 64 + c] = x[i];
}

}  // namespace

int batched_unit_ltri_inv64_f32(const void* A, void* X, int B,
                                cudaStream_t stream) {
  if (!A || !X || B <= 0) return 1;
  constexpr int MPB = 2;
  const int blocks = (B + MPB - 1) / MPB;
  unit_ltri_inv64_kernel<MPB><<<blocks, MPB * 64, 0, stream>>>(
      reinterpret_cast<const float*>(A),
      reinterpret_cast<float*>(X), B);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace kernels
}  // namespace flash_rt

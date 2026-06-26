// SPDX-License-Identifier: Apache-2.0
//
// Even-column FP4 compaction for the SwiGLU epilogue-fold output.
//
// The fused SwiGLU GEMM (cutlass_nvfp4_dual_gemm_silu_fp4out) runs on the
// interleaved gate|up weight at N=2*intermediate and writes silu(gate)*up
// DUPLICATED into both columns of each pair: packed byte j holds the same FP4
// value in both nibbles (= silu_mul[j]). With OutputSFVectorSize=32 the SFD is
// already per-16-OUTPUT-col and byte-identical to the down GEMM's per-16 SFA, so
// only the FP4 DATA needs compacting to [M, intermediate]:
//
//   out_byte[m, k] = (in_byte[m, 2k] & 0xF) | ((in_byte[m, 2k+1] & 0xF) << 4)
//
// i.e. gather the low nibble (the silu_mul value) of two adjacent input bytes.
// Memory-bound, trivial; ~3 us at the MLP shape. The SFD tensor is passed
// straight through to the down GEMM unchanged.

#include "fp4_swiglu_compact_sm120.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {
// One thread per output byte. in: [M, inter] bytes (dup'd nibbles).
// out: [M, inter/2] bytes (2 silu values packed).
__global__ void swiglu_even_col_compact_kernel(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
    int M, int inter)
{
    const int out_cols = inter >> 1;                 // bytes per output row
    const long total = (long)M * out_cols;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long)gridDim.x * blockDim.x) {
        int row = idx / out_cols;
        int k   = idx - (long)row * out_cols;
        const uint8_t* irow = in + (long)row * inter;
        uint8_t a = irow[2 * k];
        uint8_t b = irow[2 * k + 1];
        out[(long)row * out_cols + k] = (a & 0x0F) | ((b & 0x0F) << 4);
    }
}
}  // namespace

void fp4_swiglu_even_col_compact(
    const void* in_packed, void* out_packed, int M, int inter,
    cudaStream_t stream)
{
    const long total = (long)M * (inter >> 1);
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    swiglu_even_col_compact_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(in_packed),
        reinterpret_cast<uint8_t*>(out_packed), M, inter);
}

}  // namespace kernels
}  // namespace flash_rt

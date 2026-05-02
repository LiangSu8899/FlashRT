// ============================================================================
//  FlashVLA — fp16-output RMSNorm kernels (additive).
//  Copied from csrc/kernels/norm.cu {rms_norm_fp8_noweight_kernel,
//  res_rms_fp8_noweight_kernel}, with the descale division and fp8 cast
//  stripped. See .cuh for rationale.
// ============================================================================
#include "fused_fp16/rms_norm_noweight_fp16.cuh"

#include <cuda_fp16.h>

namespace flash_rt {
namespace fused_fp16 {

// Matches RMS_NW_* in norm.cu — keeps SASS shape identical for the shared
// reduction / cache path.
#define RMS_NW_FP16_THREADS 256
#define RMS_NW_FP16_D_MAX 2048
#define RMS_NW_FP16_ELEMS_PER_THREAD (RMS_NW_FP16_D_MAX / RMS_NW_FP16_THREADS)  // 8

__global__ void rms_norm_noweight_fp16_kernel(const __half* in, __half* out,
                                               int R, int C) {
    int r = blockIdx.x; if (r >= R) return;
    const __half* row = in + r * C;
    __half* orow = out + r * C;

    const __half2* row2 = reinterpret_cast<const __half2*>(row);
    __half2* orow2 = reinterpret_cast<__half2*>(orow);
    int C2 = C / 2;

    float cache[RMS_NW_FP16_ELEMS_PER_THREAD];
    float ssq = 0;
    #pragma unroll
    for (int it = 0; it < RMS_NW_FP16_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < C2) {
            __half2 v2 = row2[c2];
            cache[it*2]   = __half2float(v2.x);
            cache[it*2+1] = __half2float(v2.y);
            ssq += cache[it*2]*cache[it*2] + cache[it*2+1]*cache[it*2+1];
        } else {
            cache[it*2] = 0; cache[it*2+1] = 0;
        }
    }

    __shared__ float sh[16];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    if (!lane) sh[wid] = ssq; __syncthreads();
    if (!wid) { ssq = (lane < (blockDim.x/32)) ? sh[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o); }
    __syncthreads(); if (!threadIdx.x) sh[0] = ssq; __syncthreads();

    float scale = __frsqrt_rn(sh[0] / C + 1e-6f);  // no descale division

    #pragma unroll
    for (int it = 0; it < RMS_NW_FP16_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < C2) {
            float v0 = cache[it*2]   * scale;
            float v1 = cache[it*2+1] * scale;
            orow2[c2] = __halves2half2(__float2half(v0), __float2half(v1));
        }
    }
}

void rms_norm_noweight_fp16(const __half* x, __half* out,
                             int seq_len, int dim,
                             cudaStream_t stream) {
    rms_norm_noweight_fp16_kernel<<<seq_len, RMS_NW_FP16_THREADS, 0, stream>>>(
        x, out, seq_len, dim);
}

__global__ void res_rms_noweight_fp16_kernel(__half* residual, const __half* x,
                                              __half* out, int D) {
    int r = blockIdx.x;
    __half* res_row = residual + r * D;
    const __half* x_row = x + r * D;
    __half* orow = out + r * D;
    int D2 = D / 2;

    __half2* res2w = reinterpret_cast<__half2*>(res_row);
    const __half2* res2 = reinterpret_cast<const __half2*>(res_row);
    const __half2* x2 = reinterpret_cast<const __half2*>(x_row);
    __half2* orow2 = reinterpret_cast<__half2*>(orow);

    float cache[RMS_NW_FP16_ELEMS_PER_THREAD];
    float ssq = 0;
    #pragma unroll
    for (int it = 0; it < RMS_NW_FP16_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < D2) {
            __half2 rv2 = res2[c2], xv2 = x2[c2];
            float r0 = __half2float(rv2.x) + __half2float(xv2.x);
            float r1 = __half2float(rv2.y) + __half2float(xv2.y);
            cache[it*2]   = r0;
            cache[it*2+1] = r1;
            res2w[c2] = __halves2half2(__float2half(r0), __float2half(r1));
            ssq += r0*r0 + r1*r1;
        } else {
            cache[it*2] = 0; cache[it*2+1] = 0;
        }
    }

    __shared__ float sh[16];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    if (!lane) sh[wid] = ssq; __syncthreads();
    if (!wid) { ssq = (lane < (blockDim.x/32)) ? sh[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o); }
    __syncthreads(); if (!threadIdx.x) sh[0] = ssq; __syncthreads();

    float scale = __frsqrt_rn(sh[0] / D + 1e-6f);  // no descale division

    #pragma unroll
    for (int it = 0; it < RMS_NW_FP16_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < D2) {
            float v0 = cache[it*2]   * scale;
            float v1 = cache[it*2+1] * scale;
            orow2[c2] = __halves2half2(__float2half(v0), __float2half(v1));
        }
    }
}

void residual_add_rms_norm_noweight_fp16(__half* residual, const __half* x,
                                          __half* out,
                                          int seq_len, int dim,
                                          cudaStream_t stream) {
    res_rms_noweight_fp16_kernel<<<seq_len, RMS_NW_FP16_THREADS, 0, stream>>>(
        residual, x, out, dim);
}

}  // namespace fused_fp16
}  // namespace flash_rt

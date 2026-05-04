// SPDX-License-Identifier: Apache-2.0
//
// BF16 weight -> NVFP4 (swizzled SF + per-tensor global_scale).
// See header for design notes.

#include "bf16_weight_to_nvfp4_swizzled.cuh"

namespace flash_rt {
namespace quantize {

namespace {

// ---- helpers (same as fp8_block128_to_nvfp4_swizzled.cu) ----

__device__ __forceinline__ uint8_t fp32_to_e2m1(float v) {
    uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    float a = fabsf(v);
    uint8_t mag;
    if      (a < 0.25f)  mag = 0;
    else if (a < 0.75f)  mag = 1;
    else if (a < 1.25f)  mag = 2;
    else if (a < 1.75f)  mag = 3;
    else if (a < 2.5f)   mag = 4;
    else if (a < 3.5f)   mag = 5;
    else if (a < 5.0f)   mag = 6;
    else                 mag = 7;
    return sign | mag;
}

__device__ __forceinline__ uint8_t fp32_to_ue4m3_ceil(float v) {
    if (v <= 0.0f) return 0;
    if (v > 240.0f) return 0xFE;
    uint32_t bits = __float_as_uint(v);
    int float_exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t frac = bits & 0x7FFFFF;
    int ue_exp = float_exp + 7;
    if (ue_exp <= 0) {
        float scaled = v * 512.0f;
        int m = (int)ceilf(scaled);
        if (m > 7) return (1 << 3) | 0;
        if (m < 1) m = 1;
        return (uint8_t)m;
    }
    if (ue_exp >= 15) return 0xFE;
    int m = (int)(frac >> 20);
    if (frac & 0xFFFFF) m++;
    if (m >= 8) { m = 0; ue_exp++; }
    if (ue_exp >= 15) return 0xFE;
    return (uint8_t)((ue_exp << 3) | m);
}

__device__ __forceinline__ float ue4m3_to_fp32(uint8_t v) {
    int e = (v >> 3) & 0xF;
    int m = v & 0x7;
    if (e == 0) return ldexpf((float)m / 8.0f, -6);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

// ---- kernels ----

__global__ void bf16_weight_global_amax_kernel(
    const __nv_bfloat16* __restrict__ w_bf16,
    float* __restrict__ global_amax,
    int N, int K) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const size_t row_off = (size_t)row * K;

    float thread_max = 0.f;
    for (int col = threadIdx.x; col < K; col += blockDim.x) {
        const float v = __bfloat162float(w_bf16[row_off + col]);
        const float aw = fabsf(v);
        if (aw > thread_max) thread_max = aw;
    }

    __shared__ float smem[32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        thread_max = fmaxf(thread_max,
            __shfl_xor_sync(0xffffffff, thread_max, off));
    }
    if (lane == 0) smem[wid] = thread_max;
    __syncthreads();
    if (wid == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        thread_max = (lane < n_warps) ? smem[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            thread_max = fmaxf(thread_max,
                __shfl_xor_sync(0xffffffff, thread_max, off));
        }
        if (lane == 0) {
            atomicMax(reinterpret_cast<int*>(global_amax),
                      __float_as_int(thread_max));
        }
    }
}

__global__ void compute_global_scale_bf16_kernel(
    const float* global_amax,
    float* out_global_scale) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const float a = *global_amax;
        *out_global_scale = (a > 0.f) ? (a / 2688.f) : 1.f;
    }
}

__global__ void bf16_weight_quantize_pass2_kernel(
    const __nv_bfloat16* __restrict__ w_bf16,
    const float* __restrict__ global_scale_ptr,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ sf_swz,
    int N, int K,
    int num_blocks_per_row, int n_col_super) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const float gscale = *global_scale_ptr;
    const float inv_g = (gscale > 0.f) ? (1.f / gscale) : 0.f;

    const size_t row_in_off = (size_t)row * K;
    const size_t row_out_off = (size_t)row * (K / 2);
    const int rb = row / 128;
    const int ri = row % 128;

    for (int b = threadIdx.x; b < num_blocks_per_row; b += blockDim.x) {
        const int col0 = b * 16;

        float w_f32[16];
        float block_amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float w = __bfloat162float(w_bf16[row_in_off + col0 + i]);
            w_f32[i] = w;
            const float aw = fabsf(w);
            if (aw > block_amax) block_amax = aw;
        }

        const float ue4m3_value = (block_amax / 6.f) * inv_g;
        const uint8_t sf_byte = fp32_to_ue4m3_ceil(ue4m3_value);

        const float sf_decoded = ue4m3_to_fp32(sf_byte);
        const float block_scale_abs = sf_decoded * gscale;
        const float inv_block_scale = (block_scale_abs > 0.f)
            ? (1.f / block_scale_abs) : 0.f;

        uint8_t* packed_row = packed + row_out_off;
        #pragma unroll
        for (int i = 0; i < 16; i += 2) {
            const float v0 = w_f32[i] * inv_block_scale;
            const float v1 = w_f32[i + 1] * inv_block_scale;
            const uint8_t lo = fp32_to_e2m1(v0);
            const uint8_t hi = fp32_to_e2m1(v1);
            packed_row[(col0 + i) >> 1] = (hi << 4) | (lo & 0x0F);
        }

        const int cb = b / 4;
        const int ci = b % 4;
        const int out_idx = (rb * n_col_super + cb) * 512
                          + (ri % 32) * 16
                          + (ri / 32) * 4
                          + ci;
        sf_swz[out_idx] = sf_byte;
    }
}

}  // namespace

void bf16_weight_to_nvfp4_swizzled(
    const __nv_bfloat16* w_bf16,
    uint8_t* nvfp4_packed,
    uint8_t* nvfp4_sf_swizzled,
    float* scratch_global_amax,
    float* out_global_scale,
    int N, int K,
    cudaStream_t stream) {
    cudaMemsetAsync(scratch_global_amax, 0, sizeof(float), stream);

    {
        dim3 grid(N);
        dim3 block(256);
        bf16_weight_global_amax_kernel<<<grid, block, 0, stream>>>(
            w_bf16, scratch_global_amax, N, K);
    }

    compute_global_scale_bf16_kernel<<<1, 1, 0, stream>>>(
        scratch_global_amax, out_global_scale);

    const int num_blocks_per_row = K / 16;
    const int n_col_super = (num_blocks_per_row + 3) / 4;
    {
        dim3 grid(N);
        dim3 block(256);
        bf16_weight_quantize_pass2_kernel<<<grid, block, 0, stream>>>(
            w_bf16, out_global_scale,
            nvfp4_packed, nvfp4_sf_swizzled,
            N, K, num_blocks_per_row, n_col_super);
    }
}

}  // namespace quantize
}  // namespace flash_rt

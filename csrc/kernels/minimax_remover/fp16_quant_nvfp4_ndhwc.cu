// FlashRT — MiniMax-Remover WanVAE NVFP4 fused quantization kernels (sm_120a).
//
// See fp16_quant_nvfp4_ndhwc.cuh for interface docs.
//
// Design adapted from motus_bf16_rms_silu_quant_nvfp4_sm120.cu with:
//   - fp16 input (not bf16) — eliminates the fp16→bf16 cast.
//   - kThreadsY=6 (not 8) — supports WanVAE channels 96/192/384.
//   - SiLU in fp32 (not bf16-rounded) — preserves fp16 mantissa precision.
//   - Single-pass quant: norm+silu+block-scale+quant in one walk over xcache,
//     no intermediate buffer (lower register pressure than motus variant).

#include "fp16_quant_nvfp4_ndhwc.cuh"

#include <cstdint>
#include <cstdio>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

namespace {

constexpr int kThreadsX = 32;
constexpr int kThreadsY = 6;
constexpr int kThreads  = kThreadsX * kThreadsY;   // 192
constexpr int kWBlock   = kThreadsX;                // 32
constexpr int kPadFp4   = 4;
constexpr int kMaxHalf2 = 64;   // covers c_per_y up to 128

__device__ __forceinline__ uint8_t fp32_to_e2m1(float v) {
    uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    float a = fabsf(v);
    uint8_t mag;
    if      (a < 0.25f) mag = 0;
    else if (a < 0.75f) mag = 1;
    else if (a < 1.25f) mag = 2;
    else if (a < 1.75f) mag = 3;
    else if (a < 2.5f)  mag = 4;
    else if (a < 3.5f)  mag = 5;
    else if (a < 5.0f)  mag = 6;
    else                mag = 7;
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

__device__ __forceinline__ float silu_f32(float x) {
    return x * (1.0f / (1.0f + __expf(-x)));
}

template<bool kChannelsLast, bool kApplyNorm, bool kApplyBias>
__global__ void quant_nvfp4_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const __half* __restrict__ bias,
    uint8_t*      __restrict__ y_fp4,
    uint8_t*      __restrict__ y_sf,
    int B, int C, int T, int H, int W,
    int W_blocks_per_row, float eps)
{
    extern __shared__ __align__(16) char sm_buf[];
    // Strides must be multiples of 4 for uint32 read/write alignment.
    const int sm_fp4_stride = ((C / 2) + 3) & ~3;
    const int sm_sf_stride  = ((C / 16) + 3) & ~3;
    uint8_t* sm_fp4 = reinterpret_cast<uint8_t*>(sm_buf);
    uint8_t* sm_sf  = sm_fp4 + (size_t)kWBlock * sm_fp4_stride;
    float*   sm_red = reinterpret_cast<float*>(
                        sm_sf + (size_t)kWBlock * sm_sf_stride);

    const int wb   = blockIdx.x % W_blocks_per_row;
    const int rest = blockIdx.x / W_blocks_per_row;
    const int hwt  = T * H;
    const int b    = rest / hwt;
    const int rh   = rest - b * hwt;
    const int t    = rh / H;
    const int h    = rh - t * H;
    if (b >= B) return;

    const int w_start = wb * kWBlock;
    const int tx = threadIdx.x & 31;
    const int ty = threadIdx.x >> 5;
    const int my_w = w_start + tx;
    const bool active = (my_w < W);

    const int c_per_y    = (C + kThreadsY - 1) / kThreadsY;
    const int my_c_start = ty * c_per_y;
    const int my_c_end   = min(my_c_start + c_per_y, C);
    const int my_n_c     = my_c_end - my_c_start;
    const int my_n_pair  = (my_n_c + 1) >> 1;

    const long long stride_C = (long long)T * H * W;
    const long long row_off  = (long long)t * H * W + (long long)h * W;
    const long long b_off    = (long long)b * (long long)C * stride_C;
    // Channels-last: physical NDHWC, channel is innermost (coalesced)
    const long long cl_row   = ((long long)t * H + h) * W * C + (long long)my_w * C;
    const long long cl_b_off = (long long)b * stride_C * C;

    // ── Pass 1: read x → register cache (half2) + sum_sq for RMS ──
    __half2 xcache[kMaxHalf2];
    float sum_sq = 0.f;
    if (active) {
        #pragma unroll 1
        for (int p = 0; p < my_n_pair; ++p) {
            int c0 = my_c_start + (p << 1);
            int c1 = c0 + 1;
            __half v0, v1;
            if (kChannelsLast) {
                // NDHWC physical: channel is innermost — coalesced reads
                v0 = x[cl_b_off + cl_row + c0];
                v1 = (c1 < my_c_end) ? x[cl_b_off + cl_row + c1] : __float2half(0.f);
            } else {
                v0 = x[b_off + (long long)c0 * stride_C + row_off + my_w];
                v1 = (c1 < my_c_end)
                    ? x[b_off + (long long)c1 * stride_C + row_off + my_w]
                    : __float2half(0.f);
            }
            xcache[p] = __halves2half2(v0, v1);
            if (kApplyNorm) {
                float f0 = __half2float(v0);
                float f1 = __half2float(v1);
                sum_sq = fmaf(f0, f0, sum_sq);
                if (c1 < my_c_end) sum_sq = fmaf(f1, f1, sum_sq);
            }
        }
    }

    // ── RMS reduction (only when norm is applied) ──
    float inv_rms = 0.f;
    if (kApplyNorm) {
        sm_red[ty * kThreadsX + tx] = active ? sum_sq : 0.f;
        __syncthreads();
        float total = 0.f;
        #pragma unroll
        for (int yi = 0; yi < kThreadsY; ++yi)
            total += sm_red[yi * kThreadsX + tx];
        inv_rms = active ? rsqrtf(total / static_cast<float>(C) + eps) : 0.f;
    }

    // ── Pass 2: norm(+bias)·SiLU → per-16-block FP4 quant → smem ──
    if (active) {
        const int blocks_per_thread = my_n_c / 16;
        #pragma unroll 1
        for (int blk = 0; blk < blocks_per_thread; ++blk) {
            // Compute norm+silu for this 16-element block, find max_abs
            float vals[16];
            float mx = 0.f;
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                int c0 = my_c_start + blk * 16 + p * 2;
                int c1 = c0 + 1;
                __half2 vp = xcache[blk * 8 + p];
                float f0, f1;
                if (kApplyNorm) {
                    float xv0 = __half2float(__low2half(vp));
                    float gv0 = __half2float(gamma[c0]);
                    float n0  = xv0 * inv_rms * gv0;
                    if (kApplyBias) n0 += __half2float(bias[c0]);
                    f0 = silu_f32(n0);
                    float xv1 = __half2float(__high2half(vp));
                    float gv1 = __half2float(gamma[c1]);
                    float n1 = xv1 * inv_rms * gv1;
                    if (kApplyBias) n1 += __half2float(bias[c1]);
                    f1 = silu_f32(n1);
                } else {
                    f0 = __half2float(__low2half(vp));
                    f1 = __half2float(__high2half(vp));
                }
                vals[p * 2]     = f0;
                vals[p * 2 + 1] = f1;
                mx = fmaxf(mx, fmaxf(fabsf(f0), fabsf(f1)));
            }
            // Block scale + quantize
            float sf_f = mx / 6.0f;
            uint8_t sf_byte = fp32_to_ue4m3_ceil(sf_f);
            float sf_dec = ue4m3_to_fp32(sf_byte);
            float inv_sf = (sf_dec > 0.f) ? (1.0f / sf_dec) : 0.f;
            int sf_idx = (my_c_start / 16) + blk;
            sm_sf[tx * sm_sf_stride + sf_idx] = sf_byte;
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                uint8_t lo = fp32_to_e2m1(vals[p * 2]     * inv_sf);
                uint8_t hi = fp32_to_e2m1(vals[p * 2 + 1] * inv_sf);
                int byte_idx = (my_c_start / 2) + (blk * 8) + p;
                sm_fp4[tx * sm_fp4_stride + byte_idx] = (hi << 4) | (lo & 0xF);
            }
        }
    }
    __syncthreads();

    // ── Pass 3: coalesced uint32 global writes ──
    const long long y_base_fp4 = ((long long)b * T * H * W
                                + (long long)t * H * W
                                + (long long)h * W
                                + w_start) * (long long)(C / 2);
    const long long y_base_sf  = ((long long)b * T * H * W
                                + (long long)t * H * W
                                + (long long)h * W
                                + w_start) * (long long)(C / 16);

    const int fp4_words_per_row = C / 8;
    const int fp4_total_words   = kWBlock * fp4_words_per_row;
    const int tid = threadIdx.x;
    #pragma unroll 1
    for (int idx = tid; idx < fp4_total_words; idx += kThreads) {
        int w_off = idx / fp4_words_per_row;
        int wd    = idx - w_off * fp4_words_per_row;
        if (w_start + w_off < W) {
            uint32_t pack = *reinterpret_cast<const uint32_t*>(
                &sm_fp4[w_off * sm_fp4_stride + (wd << 2)]);
            *reinterpret_cast<uint32_t*>(
                &y_fp4[y_base_fp4 + (long long)w_off * (long long)(C / 2)
                                  + (long long)(wd << 2)]) = pack;
        }
    }
    {
        const int sf_bytes_per_row = C / 16;
        const int sf_words_per_row = sf_bytes_per_row / 4;
        const int sf_remainder     = sf_bytes_per_row & 3;
        // uint32-aligned portion
        if (sf_words_per_row > 0) {
            const int sf_total_words = kWBlock * sf_words_per_row;
            #pragma unroll 1
            for (int idx = tid; idx < sf_total_words; idx += kThreads) {
                int w_off = idx / sf_words_per_row;
                int wd    = idx - w_off * sf_words_per_row;
                if (w_start + w_off < W) {
                    uint32_t pack = *reinterpret_cast<const uint32_t*>(
                        &sm_sf[w_off * sm_sf_stride + (wd << 2)]);
                    *reinterpret_cast<uint32_t*>(
                        &y_sf[y_base_sf + (long long)w_off * (long long)sf_bytes_per_row
                                        + (long long)(wd << 2)]) = pack;
                }
            }
        }
        // Scalar remainder (handles C/16 not divisible by 4, e.g. C=96 → 6 bytes)
        if (sf_remainder > 0) {
            const int base = sf_words_per_row * 4;
            const int rem_total = kWBlock * sf_remainder;
            #pragma unroll 1
            for (int idx = tid; idx < rem_total; idx += kThreads) {
                int w_off = idx / sf_remainder;
                int bd    = idx - w_off * sf_remainder;
                if (w_start + w_off < W) {
                    y_sf[y_base_sf + (long long)w_off * (long long)sf_bytes_per_row
                                 + (long long)base + bd] =
                        sm_sf[w_off * sm_sf_stride + base + bd];
                }
            }
        }
    }
}

inline int validate_and_smem(int B, int C, int T, int H, int W, size_t* smem) {
    if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
    if ((C % 96) != 0)   return -2;
    if (C > 768)         return -3;
    if ((C / 16) < 4)    return -4;
    *smem = (size_t)kWBlock * (((C / 2) + 3) & ~3)
          + (size_t)kWBlock * (((C / 16) + 3) & ~3)
          + (size_t)kThreadsX * kThreadsY * 4;
    return 0;
}

}  // namespace

extern "C" int fp16_rms_silu_quant_nvfp4_ndhwc(
    const void*  x_fp16,
    const void*  gamma_fp16,
    const void*  bias_fp16,
    void*        y_fp4,
    void*        y_sf,
    int B, int C, int T, int H, int W,
    float eps, cudaStream_t stream)
{
    size_t smem;
    int rc = validate_and_smem(B, C, T, H, W, &smem);
    if (rc != 0) return rc;

    const int W_blocks = (W + kWBlock - 1) / kWBlock;
    const long long n_ctas = (long long)B * T * H * (long long)W_blocks;
    if (n_ctas > (long long)INT32_MAX) return -5;

    dim3 grid(static_cast<unsigned>(n_ctas));
    dim3 block(kThreads);

    if (bias_fp16) {
        quant_nvfp4_kernel<false, true, true><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(x_fp16),
            reinterpret_cast<const __half*>(gamma_fp16),
            reinterpret_cast<const __half*>(bias_fp16),
            reinterpret_cast<uint8_t*>(y_fp4),
            reinterpret_cast<uint8_t*>(y_sf),
            B, C, T, H, W, W_blocks, eps);
    } else {
        quant_nvfp4_kernel<false, true, false><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(x_fp16),
            reinterpret_cast<const __half*>(gamma_fp16),
            nullptr,
            reinterpret_cast<uint8_t*>(y_fp4),
            reinterpret_cast<uint8_t*>(y_sf),
            B, C, T, H, W, W_blocks, eps);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[fp16_rms_silu_quant_nvfp4] launch err: %s\n",
                     cudaGetErrorString(e));
        return -10;
    }
    return 0;
}

extern "C" int fp16_quant_nvfp4_ndhwc(
    const void*  x_fp16,
    void*        y_fp4,
    void*        y_sf,
    int B, int C, int T, int H, int W,
    cudaStream_t stream)
{
    size_t smem;
    int rc = validate_and_smem(B, C, T, H, W, &smem);
    if (rc != 0) return rc;

    const int W_blocks = (W + kWBlock - 1) / kWBlock;
    const long long n_ctas = (long long)B * T * H * (long long)W_blocks;
    if (n_ctas > (long long)INT32_MAX) return -5;

    dim3 grid(static_cast<unsigned>(n_ctas));
    dim3 block(kThreads);

    quant_nvfp4_kernel<false, false, false><<<grid, block, smem, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        nullptr, nullptr,
        reinterpret_cast<uint8_t*>(y_fp4),
        reinterpret_cast<uint8_t*>(y_sf),
        B, C, T, H, W, W_blocks, 0.f);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[fp16_quant_nvfp4] launch err: %s\n",
                     cudaGetErrorString(e));
        return -10;
    }
    return 0;
}

// ── Channels-last input variants (eliminates contiguous() copy) ──

extern "C" int fp16_rms_silu_quant_nvfp4_cl_ndhwc(
    const void*  x_fp16,     // channels-last 3D [B,C,T,H,W] physical NDHWC
    const void*  gamma_fp16,
    const void*  bias_fp16,
    void*        y_fp4,
    void*        y_sf,
    int B, int C, int T, int H, int W,
    float eps, cudaStream_t stream)
{
    size_t smem;
    int rc = validate_and_smem(B, C, T, H, W, &smem);
    if (rc != 0) return rc;
    const int W_blocks = (W + kWBlock - 1) / kWBlock;
    const long long n_ctas = (long long)B * T * H * (long long)W_blocks;
    if (n_ctas > (long long)INT32_MAX) return -5;
    dim3 grid(static_cast<unsigned>(n_ctas));
    dim3 block(kThreads);
    if (bias_fp16) {
        quant_nvfp4_kernel<true, true, true><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(x_fp16),
            reinterpret_cast<const __half*>(gamma_fp16),
            reinterpret_cast<const __half*>(bias_fp16),
            reinterpret_cast<uint8_t*>(y_fp4), reinterpret_cast<uint8_t*>(y_sf),
            B, C, T, H, W, W_blocks, eps);
    } else {
        quant_nvfp4_kernel<true, true, false><<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(x_fp16),
            reinterpret_cast<const __half*>(gamma_fp16), nullptr,
            reinterpret_cast<uint8_t*>(y_fp4), reinterpret_cast<uint8_t*>(y_sf),
            B, C, T, H, W, W_blocks, eps);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "[rms_silu_quant_cl] err: %s\n", cudaGetErrorString(e)); return -10; }
    return 0;
}

extern "C" int fp16_quant_nvfp4_cl_ndhwc(
    const void*  x_fp16,     // channels-last 3D [B,C,T,H,W] physical NDHWC
    void*        y_fp4,
    void*        y_sf,
    int B, int C, int T, int H, int W,
    cudaStream_t stream)
{
    size_t smem;
    int rc = validate_and_smem(B, C, T, H, W, &smem);
    if (rc != 0) return rc;

    const int W_blocks = (W + kWBlock - 1) / kWBlock;
    const long long n_ctas = (long long)B * T * H * (long long)W_blocks;
    if (n_ctas > (long long)INT32_MAX) return -5;

    dim3 grid(static_cast<unsigned>(n_ctas));
    dim3 block(kThreads);

    quant_nvfp4_kernel<true, false, false><<<grid, block, smem, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        nullptr, nullptr,
        reinterpret_cast<uint8_t*>(y_fp4),
        reinterpret_cast<uint8_t*>(y_sf),
        B, C, T, H, W, W_blocks, 0.f);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[fp16_quant_nvfp4_cl] launch err: %s\n",
                     cudaGetErrorString(e));
        return -10;
    }
    return 0;
}

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt

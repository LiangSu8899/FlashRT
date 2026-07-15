// ================================================================
// flash_rt_minimax_remover — fused FFN epilogue kernels.
//
// Replaces the 3-kernel sequence in the MiniMax-Remover transformer
// FFN path:
//   add_bias_fp16  (read fp16 [M,N] + write fp16 [M,N])
//   gelu_inplace   (read fp16 [M,N] + write fp16 [M,N])
//   quantize_fp8   (read fp16 [M,N] + write fp8  [M,N])
// with a single pass that reads the GEMM's raw fp16 output ONCE and
// writes fp8 ONCE — eliminating ~3 full-tensor memory round-trips.
//
// The output fp8 tensor is the pre-quantised input of the NEXT FP8
// Linear (net.2), which skips its own activation quantise step.
//
// Precision: all arithmetic (bias add + tanh-gelu) is done in fp32
// before the fp8 cast, so the result is actually more accurate than
// the original path (which rounds to fp16 twice along the way).
// ================================================================
#include "fp16_bias_gelu_quant_fp8.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

namespace {

__device__ __forceinline__ float gelu_tanh(float x) {
    // Matches gelu_inplace_fp16 / gelu_tanh_nvfp4 exactly:
    //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr float kSqrt2Pi = 0.7978845608f;
    float t = tanhf(kSqrt2Pi * (x + 0.044715f * x * x * x));
    return x * 0.5f * (1.0f + t);
}

// ── Fused: bias + tanh-gelu + quantise → fp8 e4m3 ──
// 4 elements / thread, vectorised fp16 reads, packed fp8 writes.
// Requires N % 4 == 0 (all MiniMax-Remover Linears satisfy this:
// inner_dim=13824, dim=5120, etc.).
__global__ void bias_gelu_quant_fp16_fp8_kernel(
    const __half* __restrict__ gemm_out,  // [M*N] raw GEMM output (no bias)
    const __half* __restrict__ bias,       // [N]
    __nv_fp8_e4m3* __restrict__ out,       // [M*N]
    const float* __restrict__ d_scale,     // act_scale of the NEXT linear
    int M, int N)
{
    const int n_total = M * N;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n_total) return;

    const float inv_scale = 1.0f / fmaxf(*d_scale, 1e-12f);
    const int col = i % N;  // N%4==0 ⇒ col is a multiple of 4, no row-cross

    // Load 4 fp16 GEMM-output values.
    const __half2* in2 = reinterpret_cast<const __half2*>(gemm_out + i);
    __half2 vA = in2[0];
    __half2 vB = in2[1];
    float fv[4] = {
        __half2float(vA.x), __half2float(vA.y),
        __half2float(vB.x), __half2float(vB.y)
    };

    // Load 4 fp16 bias values (sequential within the row).
    const __half2* bias2 = reinterpret_cast<const __half2*>(bias + col);
    __half2 bA = bias2[0];
    __half2 bB = bias2[1];
    float bv[4] = {
        __half2float(bA.x), __half2float(bA.y),
        __half2float(bB.x), __half2float(bB.y)
    };

    // bias + gelu(tanh) + quantise, all in fp32.
    __nv_fp8_e4m3 fp8_pack[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float v = fv[j] + bv[j];
        float g = gelu_tanh(v);
        g = fminf(fmaxf(g * inv_scale, -448.0f), 448.0f);
        fp8_pack[j] = __nv_fp8_e4m3(g);
    }
    *reinterpret_cast<uint32_t*>(out + i) =
        *reinterpret_cast<const uint32_t*>(fp8_pack);
}

// ── Fused: bias + identity + quantise → fp8 e4m3 ──
// Used for Linear→Linear chains with NO activation in between
// (kept for completeness / future attention QKV fusion).
__global__ void bias_quant_fp16_fp8_kernel(
    const __half* __restrict__ gemm_out,
    const __half* __restrict__ bias,
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ d_scale,
    int M, int N)
{
    const int n_total = M * N;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n_total) return;

    const float inv_scale = 1.0f / fmaxf(*d_scale, 1e-12f);
    const int col = i % N;

    const __half2* in2 = reinterpret_cast<const __half2*>(gemm_out + i);
    __half2 vA = in2[0];
    __half2 vB = in2[1];

    const __half2* bias2 = reinterpret_cast<const __half2*>(bias + col);
    __half2 bA = bias2[0];
    __half2 bB = bias2[1];

    __nv_fp8_e4m3 fp8_pack[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float v = (j == 0 ? __half2float(vA.x) :
                   j == 1 ? __half2float(vA.y) :
                   j == 2 ? __half2float(vB.x) :
                            __half2float(vB.y))
              +   (j == 0 ? __half2float(bA.x) :
                   j == 1 ? __half2float(bA.y) :
                   j == 2 ? __half2float(bB.x) :
                            __half2float(bB.y));
        v = fminf(fmaxf(v * inv_scale, -448.0f), 448.0f);
        fp8_pack[j] = __nv_fp8_e4m3(v);
    }
    *reinterpret_cast<uint32_t*>(out + i) =
        *reinterpret_cast<const uint32_t*>(fp8_pack);
}

}  // anonymous namespace

int bias_gelu_quant_fp16_fp8(
    const void* gemm_out, const void* bias,
    void* out, const float* d_scale,
    int M, int N, cudaStream_t stream)
{
    if (M <= 0 || N <= 0 || (N & 3) != 0) return -1;
    const int n_total = M * N;
    const int threads = 256;
    const int blocks = (n_total / 4 + threads - 1) / threads;
    bias_gelu_quant_fp16_fp8_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(gemm_out),
        reinterpret_cast<const __half*>(bias),
        reinterpret_cast<__nv_fp8_e4m3*>(out),
        d_scale, M, N);
    return 0;
}

int bias_quant_fp16_fp8(
    const void* gemm_out, const void* bias,
    void* out, const float* d_scale,
    int M, int N, cudaStream_t stream)
{
    if (M <= 0 || N <= 0 || (N & 3) != 0) return -1;
    const int n_total = M * N;
    const int threads = 256;
    const int blocks = (n_total / 4 + threads - 1) / threads;
    bias_quant_fp16_fp8_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(gemm_out),
        reinterpret_cast<const __half*>(bias),
        reinterpret_cast<__nv_fp8_e4m3*>(out),
        d_scale, M, N);
    return 0;
}

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt

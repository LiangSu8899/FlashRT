// ================================================================
// FlashRT — Qwen3 RoPE kernel (rotate_half style)
// Applies rotary position embedding in Qwen3/LLaMA format:
//   out[..., :HD/2] = x[..., :HD/2] * cos - x[..., HD/2:] * sin
//   out[..., HD/2:] = x[..., HD/2:] * cos + x[..., :HD/2] * sin
//
// Input layout: x = [S, NH*HD] contiguous (head-interleaved)
// cos/sin tables: [S, HD] (broadcast over heads)
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void rope_rotate_half_fp16_kernel(
    __half* __restrict__ x,           // [S, NH*HD] — modified in-place
    const __half* __restrict__ cos_t,  // [S, HD]
    const __half* __restrict__ sin_t,  // [S, HD]
    int S, int NH, int HD)
{
    // Each thread handles a PAIR (d, d+half) to avoid in-place data dependency
    int half_hd = HD / 2;
    int total_pairs = S * NH * half_hd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int d = idx % half_hd;
    int remainder = idx / half_hd;
    int h = remainder % NH;
    int s = remainder / NH;

    int rope_idx = s * HD + d;  // cos/sin use first half_hd entries per position
    float c = __half2float(cos_t[rope_idx]);
    float si = __half2float(sin_t[rope_idx]);

    int base = s * NH * HD + h * HD;
    // Read both halves FIRST (before writing)
    float x_lo = __half2float(x[base + d]);           // x[..., d]
    float x_hi = __half2float(x[base + d + half_hd]); // x[..., d + HD/2]

    // rotate_half: [-x_hi, x_lo] → out_lo = x_lo * cos - x_hi * sin
    //                              → out_hi = x_hi * cos + x_lo * sin
    x[base + d]           = __float2half(x_lo * c - x_hi * si);
    x[base + d + half_hd] = __float2half(x_hi * c + x_lo * si);
}

void rope_rotate_half_fp16(
    __half* x,                   // [S, NH*HD] — in-place
    const __half* cos_table,     // [S, HD]
    const __half* sin_table,     // [S, HD]
    int S, int NH, int HD,
    cudaStream_t stream)
{
    int total_pairs = S * NH * (HD / 2);
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    rope_rotate_half_fp16_kernel<<<blocks, threads, 0, stream>>>(
        x, cos_table, sin_table, S, NH, HD);
}

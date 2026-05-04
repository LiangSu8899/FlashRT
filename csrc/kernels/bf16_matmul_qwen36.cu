// bf16 row-major matmul (small-M) — see header for design notes.
// Mirrors the warp-per-output pattern of bf16_matvec_qwen36 and
// extends it across M rows by launching M-many (n-tile) blocks.

#include "bf16_matmul_qwen36.cuh"

namespace flash_rt::kernels {

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;  // 256

// Vectorized: each thread reads 8 bf16 = 16 bytes per iter via int4.
// Block grid: (ceil(N/8), M). Each block handles one M row × 8 N elements.
// W is shared across M rows (read once per (n-tile) block; M blocks load
// the same w_row for the same n-tile, so the L2 cache absorbs reuse).
template<int K_FIXED>
__global__ void bf16_matmul_warp_kernel(
    const __nv_bfloat16* __restrict__ x,        // (M, K)
    const __nv_bfloat16* __restrict__ W,        // (N, K)
    __nv_bfloat16* __restrict__ out,            // (M, N)
    int M, int N) {
    __shared__ __nv_bfloat16 x_sh[K_FIXED];

    const int m = blockIdx.y;
    if (m >= M) return;

    // Cooperative load of x[m, :] into smem.
    const int4* x_i4 = reinterpret_cast<const int4*>(x + m * K_FIXED);
    int4* x_sh_i4 = reinterpret_cast<int4*>(x_sh);
    const int K_int4 = K_FIXED / 8;
    #pragma unroll 1
    for (int j = threadIdx.x; j < K_int4; j += kThreads) {
        x_sh_i4[j] = x_i4[j];
    }
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * kWarpsPerBlock + warp_id;
    if (n >= N) return;

    const int4* w_row_i4 = reinterpret_cast<const int4*>(W + n * K_FIXED);

    float acc = 0.0f;
    #pragma unroll 1
    for (int i4 = lane; i4 < K_int4; i4 += 32) {
        int4 wv = w_row_i4[i4];
        int4 xv = x_sh_i4[i4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            __nv_bfloat162 wb = *reinterpret_cast<__nv_bfloat162*>(
                &(reinterpret_cast<int*>(&wv)[k]));
            __nv_bfloat162 xb = *reinterpret_cast<__nv_bfloat162*>(
                &(reinterpret_cast<int*>(&xv)[k]));
            float2 wf = __bfloat1622float2(wb);
            float2 xf = __bfloat1622float2(xb);
            acc = fmaf(xf.x, wf.x, acc);
            acc = fmaf(xf.y, wf.y, acc);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
        out[m * N + n] = __float2bfloat16(acc);
    }
}

// Generic-K fallback (chunked smem). Same warp pattern.
__global__ void bf16_matmul_warp_kernel_generic(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ out,
    int M, int N, int K) {
    extern __shared__ __nv_bfloat16 x_sh[];
    const int K_chunk_max = 4096;

    const int m = blockIdx.y;
    if (m >= M) return;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * kWarpsPerBlock + warp_id;

    float acc = 0.0f;

    for (int k_off = 0; k_off < K; k_off += K_chunk_max) {
        const int chunk = min(K_chunk_max, K - k_off);
        for (int j = threadIdx.x; j < chunk; j += kThreads) {
            x_sh[j] = x[m * K + k_off + j];
        }
        __syncthreads();

        if (n < N) {
            const __nv_bfloat16* w_row = W + n * K + k_off;
            #pragma unroll 1
            for (int j = lane; j < chunk; j += 32) {
                float xv = static_cast<float>(x_sh[j]);
                float wv = static_cast<float>(w_row[j]);
                acc = fmaf(xv, wv, acc);
            }
        }
        __syncthreads();
    }

    if (n >= N) return;

    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
        out[m * N + n] = __float2bfloat16(acc);
    }
}

}  // namespace

void bf16_matmul_qwen36_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* out,
    int M, int N, int K,
    cudaStream_t stream) {
    dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock, M);
    // Specialization set must match the bf16_matvec sibling so the
    // M=1 reference and the M=K test produce bit-identical reductions
    // (different chunking → different fma order → bf16 drift). matvec
    // specializes K=5120 and K=4096; everything else (incl. K=6144 for
    // lin_K out_proj) falls to the generic chunked path.
    if (K == 5120) {
        bf16_matmul_warp_kernel<5120>
            <<<grid, kThreads, 0, stream>>>(x, W, out, M, N);
    } else if (K == 4096) {
        bf16_matmul_warp_kernel<4096>
            <<<grid, kThreads, 0, stream>>>(x, W, out, M, N);
    } else {
        const int smem_bytes = 4096 * sizeof(__nv_bfloat16);
        bf16_matmul_warp_kernel_generic
            <<<grid, kThreads, smem_bytes, stream>>>(x, W, out, M, N, K);
    }
}

}  // namespace flash_rt::kernels

// DiT bf16 attention path — Phase 5a-2 (R0 pure additions).
// Mirrors softmax_fp16 + attention_mha_fp16 but operates on __nv_bfloat16,
// enabling the production DiT path to run at the ckpt's native dtype
// without an fp16 round-trip.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

// ────────────────────────────────────────────────────────────────────
// Softmax bf16 — mirror of softmax_fp16_kernel (1 warp per row).
// ────────────────────────────────────────────────────────────────────
#define SM_WARP_SIZE 32
#define SM_MAX_COLS 1024
#define SM_ITERS (SM_MAX_COLS / SM_WARP_SIZE)

__global__ void softmax_bf16_kernel(__nv_bfloat16* data, int rows, int cols) {
    int lane = threadIdx.x % SM_WARP_SIZE;
    int row = blockIdx.x;
    if (row >= rows) return;

    __nv_bfloat16* src = data + row * cols;
    int cols2 = cols / 2;
    __nv_bfloat162* src2 = reinterpret_cast<__nv_bfloat162*>(src);

    float reg[SM_ITERS];
    float mx = -1e30f;

    #pragma unroll
    for (int it = 0; it < SM_ITERS / 2; it++) {
        int c2 = it * SM_WARP_SIZE + lane;
        if (c2 < cols2) {
            __nv_bfloat162 v2 = src2[c2];
            reg[it*2]   = __bfloat162float(v2.x);
            reg[it*2+1] = __bfloat162float(v2.y);
            mx = fmaxf(mx, fmaxf(reg[it*2], reg[it*2+1]));
        } else {
            reg[it*2]   = -1e30f;
            reg[it*2+1] = -1e30f;
        }
    }
    if ((cols & 1) && lane == 0) {
        float v = __bfloat162float(src[cols-1]);
        reg[SM_ITERS-1] = v;
        mx = fmaxf(mx, v);
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));

    float sm = 0;
    #pragma unroll
    for (int it = 0; it < SM_ITERS; it++) {
        reg[it] = __expf(reg[it] - mx);
        sm += reg[it];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sm += __shfl_xor_sync(0xffffffff, sm, o);

    float inv = 1.f / (sm + 1e-8f);
    #pragma unroll
    for (int it = 0; it < SM_ITERS / 2; it++) {
        int c2 = it * SM_WARP_SIZE + lane;
        if (c2 < cols2) {
            __nv_bfloat162 v2;
            v2.x = __float2bfloat16(reg[it*2]   * inv);
            v2.y = __float2bfloat16(reg[it*2+1] * inv);
            src2[c2] = v2;
        }
    }
    if ((cols & 1) && lane == 0) {
        src[cols-1] = __float2bfloat16(reg[SM_ITERS-1] * inv);
    }
}

void softmax_bf16(__nv_bfloat16* data, int rows, int cols, cudaStream_t stream) {
    softmax_bf16_kernel<<<rows, SM_WARP_SIZE, 0, stream>>>(data, rows, cols);
}

// ────────────────────────────────────────────────────────────────────
// gpu_fill_neginf_bf16 — DiT softmax pre-fill (matches fp16 contract).
// ────────────────────────────────────────────────────────────────────
__global__ void gpu_fill_neginf_bf16_kernel(__nv_bfloat16* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16(-1e30f);
}

void gpu_fill_neginf_bf16(__nv_bfloat16* x, int n, cudaStream_t stream) {
    gpu_fill_neginf_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, n);
}

// ────────────────────────────────────────────────────────────────────
// attention_mha_bf16 — MHA via cuBLAS strided-batched bf16 GEMMs +
// softmax_bf16. Direct mirror of attention_mha_fp16 with one additional
// parameter:
//
//   logits_kv_stride — leading dim of the caller-provided logits buffer
//   along the kv axis (i.e. each head occupies ``S_q * logits_kv_stride``
//   bf16 cells, row-major). Pass 0 to fall back to the legacy contract
//   (``S_kv_pad = round_up(S_kv, 8)``); must otherwise be >= S_kv_pad and
//   >= max kv length the caller intends to feed across calls that share
//   this buffer (otherwise GEMM head batching writes past head boundaries).
//
// The pre-fill -inf contract still applies up to ``logits_kv_stride``;
// callers should fill the *whole* head slab so softmax pads correctly.
// ────────────────────────────────────────────────────────────────────
void attention_mha_bf16(
    cublasHandle_t handle,
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* logits,
    __nv_bfloat16* out,
    int S_q, int S_kv, int NH, int HD,
    float attn_scale,
    int logits_kv_stride,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);

    int S_kv_pad = ((S_kv + 7) / 8) * 8;
    int kv_stride = (logits_kv_stride > 0) ? logits_kv_stride : S_kv_pad;
    float zero = 0.0f;
    long long strideC = (long long)S_q * kv_stride;

    // QK^T per head → logits [NH, S_q, kv_stride] (caller buffer layout)
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        S_kv, S_q, HD,
        &attn_scale,
        K, CUDA_R_16BF, NH * HD, (long long)HD,
        Q, CUDA_R_16BF, NH * HD, (long long)HD,
        &zero,
        logits, CUDA_R_16BF, kv_stride, strideC,
        NH,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Row-wise softmax over (NH * S_q) rows of width kv_stride. The
    // caller has pre-filled cols [S_kv, kv_stride) with -inf so they
    // contribute zero weight.
    softmax_bf16(logits, NH * S_q, kv_stride, stream);

    // PV per head → out [S_q, NH*HD]. k-dim uses kv_stride (the actual
    // row length we softmax'd over) — the trailing logits cols are 0
    // post-softmax (from -inf input), so they read out as no-op.
    float one = 1.0f;
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, S_q, S_kv,
        &one,
        V, CUDA_R_16BF, NH * HD, (long long)HD,
        logits, CUDA_R_16BF, kv_stride, strideC,
        &zero,
        out, CUDA_R_16BF, NH * HD, (long long)HD,
        NH,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ================================================================
// FlashRT AMD — QKV split kernels (CDNA wave64)
// QKV split, fused QKV split + RoPE, and the devpos variant
// (runtime device K/V-cache row offset for fixed-shape prompt mode).
// Mirrors csrc/kernels/rope.cu host-wrapper signatures exactly so
// the pipeline layer stays portable text across platforms.
// ================================================================

#include "common_hip.h"

// ── QKV Split ──
// Pure memcpy by column region: qkv (seq, q_dim+k_dim+v_dim) row-major
// scatters into Q (seq, q_dim), K (seq, k_dim), V (seq, v_dim).
template<typename T>
__global__ void qkv_split_kernel_t(const T* __restrict__ qkv,
                                   T* __restrict__ Q,
                                   T* __restrict__ K,
                                   T* __restrict__ V,
                                   int seq, int q_dim, int k_dim, int v_dim) {
    int qkv_dim = q_dim + k_dim + v_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * qkv_dim;
    if (idx < total) {
        int row = idx / qkv_dim;
        int col = idx % qkv_dim;
        T val = qkv[idx];
        if (col < q_dim) {
            Q[row * q_dim + col] = val;
        } else if (col < q_dim + k_dim) {
            K[row * k_dim + (col - q_dim)] = val;
        } else {
            V[row * v_dim + (col - q_dim - k_dim)] = val;
        }
    }
}

template __global__ void qkv_split_kernel_t<__half>(
    const __half*, __half*, __half*, __half*, int, int, int, int);
template __global__ void qkv_split_kernel_t<__hip_bfloat16>(
    const __hip_bfloat16*, __hip_bfloat16*, __hip_bfloat16*, __hip_bfloat16*,
    int, int, int, int);

void qkv_split(const __hip_bfloat16* qkv,
               __hip_bfloat16* Q, __hip_bfloat16* K, __hip_bfloat16* V,
               int seq, int q_dim, int k_dim, int v_dim,
               hipStream_t stream) {
    int total = seq * (q_dim + k_dim + v_dim);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_kernel_t<__hip_bfloat16><<<blocks, threads, 0, stream>>>(
        qkv, Q, K, V, seq, q_dim, k_dim, v_dim);
}

// ── Fused QKV Split + RoPE ──
// Interleaved-pair rotation: within each head, elements (2d, 2d+1) form a
// rotation pair. rope_weights layout: (seq, head_dim) with interleaved
// (cos, sin) per pair — rope_weights[row, 2p] = cos, [row, 2p+1] = sin.
// Q has q_dim/head_dim heads (all rotated); K is single-head (k_dim wide,
// rotated with the same per-row weights); V is copied through.
__global__ void qkv_split_rope_kernel(
    const __hip_bfloat16* __restrict__ qkv,
    const __hip_bfloat16* __restrict__ rope_weights,
    __hip_bfloat16* __restrict__ Q,
    __hip_bfloat16* __restrict__ K,
    __hip_bfloat16* __restrict__ V,
    int seq, int q_dim, int k_dim, int v_dim, int head_dim) {
    int qkv_dim = q_dim + k_dim + v_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * qkv_dim;
    if (idx >= total) return;

    int row = idx / qkv_dim;
    int col = idx % qkv_dim;

    if (col < q_dim) {
        int q_col = col;
        int head = q_col / head_dim;
        int d_in_head = q_col % head_dim;
        int pair = d_in_head / 2;
        int is_odd = d_in_head & 1;

        int pair_base_qkv = row * qkv_dim + head * head_dim + pair * 2;
        float x0 = __bfloat162float(qkv[pair_base_qkv]);
        float x1 = __bfloat162float(qkv[pair_base_qkv + 1]);

        int rope_base = row * head_dim + pair * 2;
        float c = __bfloat162float(rope_weights[rope_base]);
        float s = __bfloat162float(rope_weights[rope_base + 1]);

        int out_idx = row * q_dim + q_col;
        if (is_odd == 0) {
            Q[out_idx] = __float2bfloat16(x0 * c - x1 * s);
        } else {
            Q[out_idx] = __float2bfloat16(x1 * c + x0 * s);
        }
    } else if (col < q_dim + k_dim) {
        int k_col = col - q_dim;
        int pair = k_col / 2;
        int is_odd = k_col & 1;

        int pair_base_qkv = row * qkv_dim + q_dim + pair * 2;
        float x0 = __bfloat162float(qkv[pair_base_qkv]);
        float x1 = __bfloat162float(qkv[pair_base_qkv + 1]);

        int rope_base = row * head_dim + pair * 2;
        float c = __bfloat162float(rope_weights[rope_base]);
        float s = __bfloat162float(rope_weights[rope_base + 1]);

        int out_idx = row * k_dim + k_col;
        if (is_odd == 0) {
            K[out_idx] = __float2bfloat16(x0 * c - x1 * s);
        } else {
            K[out_idx] = __float2bfloat16(x1 * c + x0 * s);
        }
    } else {
        int v_col = col - q_dim - k_dim;
        V[row * v_dim + v_col] = qkv[idx];
    }
}

void qkv_split_rope(const __hip_bfloat16* qkv,
                    const __hip_bfloat16* rope_weights,
                    __hip_bfloat16* Q, __hip_bfloat16* K, __hip_bfloat16* V,
                    int seq, int q_dim, int k_dim, int v_dim, int head_dim,
                    hipStream_t stream) {
    int total = seq * (q_dim + k_dim + v_dim);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_rope_kernel<<<blocks, threads, 0, stream>>>(
        qkv, rope_weights, Q, K, V, seq, q_dim, k_dim, v_dim, head_dim);
}

// ── Fused QKV Split + RoPE with a RUNTIME (device) K/V-cache row offset ──
// Identical math to qkv_split_rope, but the K/V destination row is shifted by
// ``devpos[0]`` read at launch time. Q is still written contiguous (rows 0..seq).
// This lets a single fixed-shape HIP graph append the decoder's K/V right
// after a variable-length valid prefix (devpos = valid prefix length), so the
// cross-attention can be one contiguous (seqused) call instead of recapturing
// per prompt length. K/V are the cache BASE pointers (row 0), not pre-offset.
__global__ void qkv_split_rope_devpos_kernel(
    const __hip_bfloat16* __restrict__ qkv,
    const __hip_bfloat16* __restrict__ rope_weights,
    __hip_bfloat16* __restrict__ Q,
    __hip_bfloat16* __restrict__ K,
    __hip_bfloat16* __restrict__ V,
    const int* __restrict__ devpos,
    int seq, int q_dim, int k_dim, int v_dim, int head_dim) {
    int qkv_dim = q_dim + k_dim + v_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * qkv_dim;
    if (idx >= total) return;

    int row = idx / qkv_dim;
    int col = idx % qkv_dim;
    int pos = devpos[0];  // runtime K/V-cache row offset (valid prefix length)

    if (col < q_dim) {
        int q_col = col;
        int head = q_col / head_dim;
        int d_in_head = q_col % head_dim;
        int pair = d_in_head / 2;
        int is_odd = d_in_head & 1;
        int pair_base_qkv = row * qkv_dim + head * head_dim + pair * 2;
        float x0 = __bfloat162float(qkv[pair_base_qkv]);
        float x1 = __bfloat162float(qkv[pair_base_qkv + 1]);
        int rope_base = row * head_dim + pair * 2;
        float c = __bfloat162float(rope_weights[rope_base]);
        float s = __bfloat162float(rope_weights[rope_base + 1]);
        int out_idx = row * q_dim + q_col;
        if (is_odd == 0) Q[out_idx] = __float2bfloat16(x0 * c - x1 * s);
        else             Q[out_idx] = __float2bfloat16(x1 * c + x0 * s);
    } else if (col < q_dim + k_dim) {
        int k_col = col - q_dim;
        int pair = k_col / 2;
        int is_odd = k_col & 1;
        int pair_base_qkv = row * qkv_dim + q_dim + pair * 2;
        float x0 = __bfloat162float(qkv[pair_base_qkv]);
        float x1 = __bfloat162float(qkv[pair_base_qkv + 1]);
        int rope_base = row * head_dim + pair * 2;
        float c = __bfloat162float(rope_weights[rope_base]);
        float s = __bfloat162float(rope_weights[rope_base + 1]);
        int out_idx = (pos + row) * k_dim + k_col;
        if (is_odd == 0) K[out_idx] = __float2bfloat16(x0 * c - x1 * s);
        else             K[out_idx] = __float2bfloat16(x1 * c + x0 * s);
    } else {
        int v_col = col - q_dim - k_dim;
        V[(pos + row) * v_dim + v_col] = qkv[idx];
    }
}

void qkv_split_rope_devpos(const __hip_bfloat16* qkv,
                           const __hip_bfloat16* rope_weights,
                           __hip_bfloat16* Q, __hip_bfloat16* K, __hip_bfloat16* V,
                           const int* devpos,
                           int seq, int q_dim, int k_dim, int v_dim,
                           int head_dim, hipStream_t stream) {
    int total = seq * (q_dim + k_dim + v_dim);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_rope_devpos_kernel<<<blocks, threads, 0, stream>>>(
        qkv, rope_weights, Q, K, V, devpos, seq, q_dim, k_dim, v_dim, head_dim);
}

// ================================================================
// FlashRT AMD — Elementwise kernels (dtype-generic, CDNA wave64)
// Residual add, gate multiply, bias residual, broadcast bias add.
// Mirrors csrc/kernels/elementwise.cu (add_bias_bf16 comes from
// csrc/kernels/dit_bf16.cu) host-wrapper signatures exactly.
// ================================================================

#include "common_hip.h"

// ── Gate Multiply + Residual: residual += x * gate ──
template<typename T>
__global__ void gate_mul_res_kernel(T* __restrict__ residual,
                                    const T* __restrict__ x,
                                    const T* __restrict__ gate, int n) {
    using T2 = typename packed2<T>::type;
    T2* res2 = reinterpret_cast<T2*>(residual);
    const T2* x2 = reinterpret_cast<const T2*>(x);
    const T2* g2 = reinterpret_cast<const T2*>(gate);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        T2 rv = res2[idx], xv = x2[idx], gv = g2[idx];
        float r0 = to_f32(rv.x) + to_f32(xv.x) * to_f32(gv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y) * to_f32(gv.y);
        res2[idx] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
    }
}

template __global__ void gate_mul_res_kernel<__half>(__half*, const __half*, const __half*, int);
template __global__ void gate_mul_res_kernel<__hip_bfloat16>(__hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*, int);

void gate_mul_residual(__hip_bfloat16* residual, const __hip_bfloat16* x,
                       const __hip_bfloat16* gate, int n, hipStream_t stream) {
    int n2 = n >> 1;
    gate_mul_res_kernel<__hip_bfloat16><<<(n2 + 255) / 256, 256, 0, stream>>>(residual, x, gate, n);
}

// ── Bias + Residual: residual[row,:] += x[row,:] + bias ──
template<typename T>
__global__ void bias_res_kernel(T* __restrict__ residual,
                                const T* __restrict__ x,
                                const T* __restrict__ bias,
                                int dim) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    int dim2 = dim >> 1;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], bv = b2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x) + to_f32(bv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y) + to_f32(bv.y);
        res2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
    }
}

template __global__ void bias_res_kernel<__half>(__half*, const __half*, const __half*, int);
template __global__ void bias_res_kernel<__hip_bfloat16>(__hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*, int);

void bias_residual(__hip_bfloat16* residual, const __hip_bfloat16* x,
                   const __hip_bfloat16* bias, int seq_len, int dim,
                   hipStream_t stream) {
    bias_res_kernel<__hip_bfloat16><<<seq_len, 256, 0, stream>>>(residual, x, bias, dim);
}

// ── Residual Add: residual += x ──
template<typename T>
__global__ void res_add_kernel(T* __restrict__ residual,
                               const T* __restrict__ x, int n) {
    using T2 = typename packed2<T>::type;
    T2* res2 = reinterpret_cast<T2*>(residual);
    const T2* x2 = reinterpret_cast<const T2*>(x);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        T2 rv = res2[idx], xv = x2[idx];
        res2[idx] = make_packed2<T>(
            from_f32<T>(to_f32(rv.x) + to_f32(xv.x)),
            from_f32<T>(to_f32(rv.y) + to_f32(xv.y)));
    }
}

template __global__ void res_add_kernel<__half>(__half*, const __half*, int);
template __global__ void res_add_kernel<__hip_bfloat16>(__hip_bfloat16*, const __hip_bfloat16*, int);

void residual_add(__hip_bfloat16* residual, const __hip_bfloat16* x, int n,
                  hipStream_t stream) {
    int n2 = n >> 1;
    res_add_kernel<__hip_bfloat16><<<(n2 + 255) / 256, 256, 0, stream>>>(residual, x, n);
}

// ── Broadcast bias add: x[i] += b[i % D] — BF16 ──
__global__ void add_bias_bf16_kernel(__hip_bfloat16* x, const __hip_bfloat16* b,
                                     int S, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < S * D) {
        x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(b[i % D]));
    }
}

void add_bias_bf16(__hip_bfloat16* x, const __hip_bfloat16* b,
                   int S, int D, hipStream_t stream) {
    add_bias_bf16_kernel<<<(S * D + 255) / 256, 256, 0, stream>>>(x, b, S, D);
}

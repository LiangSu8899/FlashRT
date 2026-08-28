// ================================================================
// FlashRT AMD — Normalization kernels (dtype-generic, CDNA wave64)
// RMSNorm; LayerNorm and AdaRMSNorm follow in later commits.
// Mirrors csrc/kernels/norm.cu host-wrapper signatures exactly so
// the pipeline layer stays portable text across platforms.
// ================================================================

#include "common_hip.h"

// ── RMSNorm ──
template<typename T>
__global__ void rms_norm_kernel(const T* __restrict__ x,
                                const T* __restrict__ weight,
                                T* __restrict__ out,
                                int dim, float eps) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + (size_t)row * dim);
    T2* out2 = reinterpret_cast<T2*>(out + (size_t)row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        local_sum += v0 * v0 + v1 * v1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i];
        float v0 = to_f32(xv.x) * rms * to_f32(wv.x);
        float v1 = to_f32(xv.y) * rms * to_f32(wv.y);
        out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
    }
}

template __global__ void rms_norm_kernel<__half>(const __half*, const __half*, __half*, int, float);
template __global__ void rms_norm_kernel<__hip_bfloat16>(const __hip_bfloat16*, const __hip_bfloat16*, __hip_bfloat16*, int, float);

void rms_norm(const __hip_bfloat16* x, const __hip_bfloat16* weight,
              __hip_bfloat16* out, int seq_len, int dim, float eps,
              hipStream_t stream) {
    rms_norm_kernel<__hip_bfloat16><<<seq_len, 256, 4 * sizeof(float), stream>>>(x, weight, out, dim, eps);
}

void rms_norm_fp16(const __half* x, const __half* weight,
                   __half* out, int seq_len, int dim, float eps,
                   hipStream_t stream) {
    rms_norm_kernel<__half><<<seq_len, 256, 4 * sizeof(float), stream>>>(x, weight, out, dim, eps);
}

void rms_norm_inplace(const __hip_bfloat16* weight,
                      __hip_bfloat16* x, int seq_len, int dim, float eps,
                      hipStream_t stream) {
    rms_norm_kernel<__hip_bfloat16><<<seq_len, 256, 4 * sizeof(float), stream>>>(x, weight, x, dim, eps);
}

// ================================================================
// FlashVLA — Normalization kernels (dtype-generic)
// RMSNorm, LayerNorm, AdaRMSNorm
// Supports: __half (FP16), __nv_bfloat16 (BF16) via templates
// ================================================================

#include "norm.cuh"
#include "common.cuh"

// ── RMSNorm ──
template<typename T>
__global__ void rms_norm_kernel(const T* __restrict__ x,
                                const T* __restrict__ weight,
                                T* __restrict__ out,
                                int dim, float eps) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
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

// Explicit instantiation
FVK_KERNEL_INSTANTIATE(__global__ void rms_norm_kernel<__half>(const __half*, const __half*, __half*, int, float))
FVK_KERNEL_INSTANTIATE(__global__ void rms_norm_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, float))
void rms_norm(const __nv_bfloat16* x, const __nv_bfloat16* weight,
              __nv_bfloat16* out, int seq_len, int dim, float eps,
              cudaStream_t stream) {
    rms_norm_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, weight, out, dim, eps);
}
void rms_norm_fp16(const __half* x, const __half* weight,
                    __half* out, int seq_len, int dim, float eps,
                    cudaStream_t stream) {
    rms_norm_kernel<__half><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, weight, out, dim, eps);
}

void rms_norm_inplace(const __nv_bfloat16* weight,
                      __nv_bfloat16* x, int seq_len, int dim, float eps,
                      cudaStream_t stream) {
    rms_norm_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, weight, x, dim, eps);
}

// ── LayerNorm ──
template<typename T>
__global__ void layer_norm_kernel(const T* __restrict__ x,
                                  const T* __restrict__ weight,
                                  const T* __restrict__ bias,
                                  T* __restrict__ out,
                                  int dim, float eps) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        local_sum += to_f32(val.x) + to_f32(val.y);
    }
    float mean = block_reduce_sum(local_sum, shared) / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float d0 = to_f32(val.x) - mean, d1 = to_f32(val.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i], bv = b2[i];
        float v0 = (to_f32(xv.x) - mean) * inv_std * to_f32(wv.x) + to_f32(bv.x);
        float v1 = (to_f32(xv.y) - mean) * inv_std * to_f32(wv.y) + to_f32(bv.y);
        out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void layer_norm_kernel<__half>(const __half*, const __half*, const __half*, __half*, int, float))
FVK_KERNEL_INSTANTIATE(__global__ void layer_norm_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, float))
void layer_norm(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                const __nv_bfloat16* bias, __nv_bfloat16* out,
                int seq_len, int dim, float eps, cudaStream_t stream) {
    layer_norm_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, weight, bias, out, dim, eps);
}
void layer_norm_fp16(const __half* x, const __half* weight,
                      const __half* bias, __half* out,
                      int seq_len, int dim, float eps, cudaStream_t stream) {
    layer_norm_kernel<__half><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, weight, bias, out, dim, eps);
}

// ── LayerNorm → FP8 (fused, matches pi05 fused_layernorm_fp8) ──
// FP16 specialization: verbatim production fused_layernorm_fp8
__global__ void layer_norm_fp8_kernel_fp16(const __half* in, __nv_fp8_e4m3* out, const __half* gamma,
                                    const __half* beta, int R, int C) {
    int r=blockIdx.x; if(r>=R)return;
    const __half*row=in+r*C; __nv_fp8_e4m3*orow=out+r*C;
    float sum=0;
    for(int i=threadIdx.x;i<C;i+=blockDim.x) sum+=__half2float(row[i]);
    __shared__ float sh[32]; int l=threadIdx.x%32,w=threadIdx.x/32;
    for(int o=16;o>0;o>>=1) sum+=__shfl_xor_sync(0xffffffff,sum,o);
    if(!l)sh[w]=sum;__syncthreads();
    if(!w){sum=(l<(blockDim.x+31)/32)?sh[l]:0;for(int o=16;o>0;o>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,o);}
    __syncthreads();if(!threadIdx.x)sh[0]=sum;__syncthreads();
    float mean=sh[0]/C;
    float var=0;
    for(int i=threadIdx.x;i<C;i+=blockDim.x){float v=__half2float(row[i])-mean;var+=v*v;}
    for(int o=16;o>0;o>>=1)var+=__shfl_xor_sync(0xffffffff,var,o);
    if(!l)sh[w]=var;__syncthreads();
    if(!w){var=(l<(blockDim.x+31)/32)?sh[l]:0;for(int o=16;o>0;o>>=1)var+=__shfl_xor_sync(0xffffffff,var,o);}
    __syncthreads();if(!threadIdx.x)sh[0]=var;__syncthreads();
    float rstd=rsqrtf(sh[0]/C+1e-6f);
    for(int i=threadIdx.x;i<C;i+=blockDim.x){
        float v=(__half2float(row[i])-mean)*rstd;
        float normed = v*__half2float(gamma[i])+__half2float(beta[i]);
        orow[i]=__nv_fp8_e4m3(normed);
    }
}

// BF16 generic version
template<typename T>
__global__ void layer_norm_fp8_kernel(const T* __restrict__ x,
                                       __nv_fp8_e4m3* __restrict__ out,
                                       const T* __restrict__ gamma,
                                       const T* __restrict__ beta,
                                       int dim, float eps) {
    int row = blockIdx.x;
    const T* x_row = x + row * dim;
    __nv_fp8_e4m3* o_row = out + row * dim;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        local_sum += to_f32(x_row[i]);
    float mean = block_reduce_sum(local_sum, shared) / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = to_f32(x_row[i]) - mean;
        local_var += d * d;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float normed = (to_f32(x_row[i]) - mean) * inv_std * to_f32(gamma[i]) + to_f32(beta[i]);
        o_row[i] = __nv_fp8_e4m3(normed);
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void layer_norm_fp8_kernel<__nv_bfloat16>(const __nv_bfloat16*, __nv_fp8_e4m3*, const __nv_bfloat16*, const __nv_bfloat16*, int, float))
void layer_norm_fp8(const __half* x, __nv_fp8_e4m3* out,
                     const __half* gamma, const __half* beta,
                     int seq_len, int dim, float eps, cudaStream_t stream) {
    // Use production-verbatim FP16 kernel (no __restrict__, fixed shared[32])
    layer_norm_fp8_kernel_fp16<<<seq_len, 256, 0, stream>>>(x, out, gamma, beta, seq_len, dim);
}
void layer_norm_fp8_bf16(const __nv_bfloat16* x, __nv_fp8_e4m3* out,
                          const __nv_bfloat16* gamma, const __nv_bfloat16* beta,
                          int seq_len, int dim, float eps, cudaStream_t stream) {
    layer_norm_fp8_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, out, gamma, beta, dim, eps);
}

// ── AdaRMSNorm + Style ──
template<typename T>
__global__ void ada_rms_norm_style_kernel(
    const T* __restrict__ x, const T* __restrict__ weight,
    const T* __restrict__ style, T* __restrict__ out, T* __restrict__ gate_out,
    int dim, float eps) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    T2* gate2 = reinterpret_cast<T2*>(gate_out + row * dim);
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
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(xv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(xv.y) * rms * to_f32(wv.y);
        out2[i] = make_packed2<T>(
            from_f32<T>(n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)),
            from_f32<T>(n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)));
        gate2[i] = gv;
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void ada_rms_norm_style_kernel<__half>(const __half*, const __half*, const __half*, __half*, __half*, int, float))
FVK_KERNEL_INSTANTIATE(__global__ void ada_rms_norm_style_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, float))
void ada_rms_norm_style(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                        const __nv_bfloat16* style,
                        __nv_bfloat16* out, __nv_bfloat16* gate_out,
                        int seq_len, int dim, float eps, cudaStream_t stream) {
    ada_rms_norm_style_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, weight, style, out, gate_out, dim, eps);
}

// ── RMSNorm → FP8 ──
template<typename T>
__global__ void rms_norm_fp8_kernel(const T* __restrict__ x,
                                     const T* __restrict__ weight,
                                     __nv_fp8_e4m3* __restrict__ out,
                                     int dim, float eps,
                                     const float* __restrict__ d_scale) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    __nv_fp8_e4m3* out_row = out + row * dim;
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        local_sum += v0 * v0 + v1 * v1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i];
        float v0 = to_f32(xv.x) * rms * to_f32(wv.x) * inv_scale;
        float v1 = to_f32(xv.y) * rms * to_f32(wv.y) * inv_scale;
        out_row[2*i]   = __nv_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
        out_row[2*i+1] = __nv_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void rms_norm_fp8_kernel<__half>(const __half*, const __half*, __nv_fp8_e4m3*, int, float, const float*))
FVK_KERNEL_INSTANTIATE(__global__ void rms_norm_fp8_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_fp8_e4m3*, int, float, const float*))
void rms_norm_fp8(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                   __nv_fp8_e4m3* out, int seq_len, int dim, float eps,
                   const float* d_scale, cudaStream_t stream) {
    rms_norm_fp8_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, weight, out, dim, eps, d_scale);
}
void rms_norm_fp8_fp16(const __half* x, const __half* weight,
                        __nv_fp8_e4m3* out, int seq_len, int dim, float eps,
                        const float* d_scale, cudaStream_t stream) {
    rms_norm_fp8_kernel<__half><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, weight, out, dim, eps, d_scale);
}

// ── AdaRMSNorm + Style → FP8 ──
template<typename T>
__global__ void ada_rms_norm_style_fp8_kernel(
    const T* __restrict__ x, const T* __restrict__ weight,
    const T* __restrict__ style, __nv_fp8_e4m3* __restrict__ out, T* __restrict__ gate_out,
    int dim, float eps, const float* __restrict__ d_scale) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    __nv_fp8_e4m3* out_row = out + row * dim;
    T2* gate2 = reinterpret_cast<T2*>(gate_out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        local_sum += v0 * v0 + v1 * v1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(xv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(xv.y) * rms * to_f32(wv.y);
        float val0 = (n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)) * inv_scale;
        float val1 = (n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)) * inv_scale;
        out_row[2*i]   = __nv_fp8_e4m3(fminf(fmaxf(val0, -448.0f), 448.0f));
        out_row[2*i+1] = __nv_fp8_e4m3(fminf(fmaxf(val1, -448.0f), 448.0f));
        gate2[i] = gv;
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void ada_rms_norm_style_fp8_kernel<__half>(const __half*, const __half*, const __half*, __nv_fp8_e4m3*, __half*, int, float, const float*))
FVK_KERNEL_INSTANTIATE(__global__ void ada_rms_norm_style_fp8_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_fp8_e4m3*, __nv_bfloat16*, int, float, const float*))
void ada_rms_norm_style_fp8(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                             const __nv_bfloat16* style,
                             __nv_fp8_e4m3* out, __nv_bfloat16* gate_out,
                             int seq_len, int dim, float eps,
                             const float* d_scale, cudaStream_t stream) {
    ada_rms_norm_style_fp8_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, weight, style, out, gate_out, dim, eps, d_scale);
}

// ── Residual Add + RMSNorm → FP8 ──
template<typename T>
__global__ void residual_add_rms_norm_fp8_kernel(
    T* __restrict__ residual, const T* __restrict__ x,
    const T* __restrict__ weight, __nv_fp8_e4m3* __restrict__ out,
    int dim, float eps, const float* __restrict__ d_scale) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    __nv_fp8_e4m3* out_row = out + row * dim;
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y);
        res2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
        local_sum += r0 * r0 + r1 * r1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        float v0 = to_f32(rv.x) * rms * to_f32(wv.x) * inv_scale;
        float v1 = to_f32(rv.y) * rms * to_f32(wv.y) * inv_scale;
        out_row[2*i]   = __nv_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
        out_row[2*i+1] = __nv_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void residual_add_rms_norm_fp8_kernel<__half>(__half*, const __half*, const __half*, __nv_fp8_e4m3*, int, float, const float*))
FVK_KERNEL_INSTANTIATE(__global__ void residual_add_rms_norm_fp8_kernel<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_fp8_e4m3*, int, float, const float*))
void residual_add_rms_norm_fp8(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                const __nv_bfloat16* weight, __nv_fp8_e4m3* out,
                                int seq_len, int dim, float eps,
                                const float* d_scale, cudaStream_t stream) {
    residual_add_rms_norm_fp8_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, weight, out, dim, eps, d_scale);
}
void residual_add_rms_norm_fp8_fp16(__half* residual, const __half* x,
                                     const __half* weight, __nv_fp8_e4m3* out,
                                     int seq_len, int dim, float eps,
                                     const float* d_scale, cudaStream_t stream) {
    residual_add_rms_norm_fp8_kernel<__half><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, weight, out, dim, eps, d_scale);
}

// ── Residual Add + RMSNorm → T (same dtype output) ──
template<typename T>
__global__ void residual_add_rms_norm_kernel(
    T* __restrict__ residual, const T* __restrict__ x,
    const T* __restrict__ weight, T* __restrict__ out,
    int dim, float eps) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y);
        res2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
        local_sum += r0 * r0 + r1 * r1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        float v0 = to_f32(rv.x) * rms * to_f32(wv.x);
        float v1 = to_f32(rv.y) * rms * to_f32(wv.y);
        out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void residual_add_rms_norm_kernel<__half>(__half*, const __half*, const __half*, __half*, int, float))
FVK_KERNEL_INSTANTIATE(__global__ void residual_add_rms_norm_kernel<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, float))
void residual_add_rms_norm(__nv_bfloat16* residual, const __nv_bfloat16* x,
                            const __nv_bfloat16* weight, __nv_bfloat16* out,
                            int seq_len, int dim, float eps,
                            cudaStream_t stream) {
    residual_add_rms_norm_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, weight, out, dim, eps);
}


// ================================================================
// Production-exact kernels (no weight, no d_scale variant)
// ================================================================

// RMSNorm → FP8 (no weight, no d_scale). Bit-exact reference variant.
__global__ void plain_rms_fp8_kernel(const __half* __restrict__ in,
                                      __nv_fp8_e4m3* __restrict__ out,
                                      int C) {
    int r = blockIdx.x;
    const __half* row = in + r * C;
    __nv_fp8_e4m3* orow = out + r * C;

    __shared__ float sh[32];
    float ssq = 0;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = __half2float(row[i]);
        ssq += v * v;
    }
    // Warp reduce
    for (int o = 16; o > 0; o >>= 1)
        ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    int l = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (!l) sh[w] = ssq;
    __syncthreads();
    if (!w) {
        ssq = (l < (blockDim.x + 31) / 32) ? sh[l] : 0;
        for (int o = 16; o > 0; o >>= 1)
            ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    }
    __syncthreads();
    if (!threadIdx.x) sh[0] = ssq;
    __syncthreads();
    float rms = rsqrtf(sh[0] / C + 1e-6f);

    for (int i = threadIdx.x; i < C; i += blockDim.x)
        orow[i] = __nv_fp8_e4m3(__half2float(row[i]) * rms);
}

void plain_rms_fp8_fp16(const __half* x, __nv_fp8_e4m3* out,
                         int seq_len, int dim, cudaStream_t stream) {
    plain_rms_fp8_kernel<<<seq_len, 256, 0, stream>>>(x, out, dim);
}

// Residual add + RMSNorm → FP8 (no weight, no d_scale). Identical to pi05 res_rms_fp8_k.
__global__ void plain_res_rms_fp8_kernel(__half* __restrict__ residual,
                                          const __half* __restrict__ x,
                                          __nv_fp8_e4m3* __restrict__ out,
                                          int D) {
    int row = blockIdx.x;
    __half2* res2 = reinterpret_cast<__half2*>(residual + row * D);
    const __half2* x2 = reinterpret_cast<const __half2*>(x + row * D);
    __nv_fp8_e4m3* out_row = out + row * D;
    int D2 = D >> 1;

    extern __shared__ float sh[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        __half2 rv = res2[i], xv = x2[i];
        float r0 = __half2float(rv.x) + __half2float(xv.x);
        float r1 = __half2float(rv.y) + __half2float(xv.y);
        res2[i] = __halves2half2(__float2half(r0), __float2half(r1));
        local_sum += r0 * r0 + r1 * r1;
    }
    // Block reduce
    float val = local_sum;
    for (int o = 16; o > 0; o >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (!lane) sh[wid] = val;
    __syncthreads();
    int nw = blockDim.x >> 5;
    val = (threadIdx.x < nw) ? sh[threadIdx.x] : 0;
    if (!wid) {
        for (int o = 16; o > 0; o >>= 1)
            val += __shfl_xor_sync(0xffffffff, val, o);
    }
    if (!threadIdx.x) sh[0] = val;
    __syncthreads();
    float rms = rsqrtf(sh[0] / D + 1e-6f);

    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        __half2 rv = res2[i];
        float v0 = __half2float(rv.x) * rms;
        float v1 = __half2float(rv.y) * rms;
        out_row[2 * i]     = __nv_fp8_e4m3(v0);
        out_row[2 * i + 1] = __nv_fp8_e4m3(v1);
    }
}

void plain_res_rms_fp8_fp16(__half* residual, const __half* x,
                             __nv_fp8_e4m3* out, int seq_len, int dim,
                             cudaStream_t stream) {
    plain_res_rms_fp8_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, out, dim);
}

// ── RMSNorm → FP8 with d_scale, no weight (norm weight baked into GEMM weights) ──
// Verbatim copy of production rms_norm_fp8_static_k.
// NOT "equivalent" — literally the same source to get identical SASS.

#define RMS_NW_THREADS 256
#define RMS_NW_D_MAX 2048
#define RMS_NW_ELEMS_PER_THREAD (RMS_NW_D_MAX / RMS_NW_THREADS)  // 8

__global__ void rms_norm_fp8_noweight_kernel(const __half* in, __nv_fp8_e4m3* out, int R, int C,
                                       const float* descale_ptr) {
    int r = blockIdx.x; if (r >= R) return;
    const __half* row = in + r * C;
    __nv_fp8_e4m3* orow = out + r * C;

    const __half2* row2 = reinterpret_cast<const __half2*>(row);
    int C2 = C / 2;

    float cache[RMS_NW_ELEMS_PER_THREAD];  // 8 floats
    float ssq = 0;
    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
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

    float scale = __frsqrt_rn(sh[0] / C + 1e-6f) / fmaxf(*descale_ptr, 1e-12f);

    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < C2) {
            int c = c2 * 2;
            __nv_fp8_e4m3 pair[2];
            pair[0] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2]   * scale, -448.f), 448.f));
            pair[1] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2+1] * scale, -448.f), 448.f));
            *reinterpret_cast<uint16_t*>(orow + c) = *reinterpret_cast<uint16_t*>(pair);
        }
    }
}

void rms_norm_fp8_noweight_fp16(const __half* x, __nv_fp8_e4m3* out,
                                 int seq_len, int dim,
                                 const float* d_scale, cudaStream_t stream) {
    rms_norm_fp8_noweight_kernel<<<seq_len, 256, 0, stream>>>(x, out, seq_len, dim, d_scale);
}

// ── Residual + RMSNorm → FP8 with d_scale, no weight ──
// Verbatim copy of production res_rms_fp8_static_k.

__global__ void res_rms_fp8_noweight_kernel(__half* residual, const __half* x, __nv_fp8_e4m3* out, int D,
                                      const float* descale_ptr) {
    int r = blockIdx.x;
    __half* res_row = residual + r * D;
    const __half* x_row = x + r * D;
    __nv_fp8_e4m3* orow = out + r * D;
    int D2 = D / 2;

    __half2* res2w = reinterpret_cast<__half2*>(res_row);
    const __half2* res2 = reinterpret_cast<const __half2*>(res_row);
    const __half2* x2 = reinterpret_cast<const __half2*>(x_row);

    float cache[RMS_NW_ELEMS_PER_THREAD];  // 8 floats
    float ssq = 0;
    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
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

    float scale = __frsqrt_rn(sh[0] / D + 1e-6f) / fmaxf(*descale_ptr, 1e-12f);

    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < D2) {
            int c = c2 * 2;
            __nv_fp8_e4m3 pair[2];
            pair[0] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2]   * scale, -448.f), 448.f));
            pair[1] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2+1] * scale, -448.f), 448.f));
            *reinterpret_cast<uint16_t*>(orow + c) = *reinterpret_cast<uint16_t*>(pair);
        }
    }
}

void residual_add_rms_norm_fp8_noweight_fp16(__half* residual, const __half* x,
                                               __nv_fp8_e4m3* out,
                                               int seq_len, int dim,
                                               const float* d_scale, cudaStream_t stream) {
    res_rms_fp8_noweight_kernel<<<seq_len, 256, 0, stream>>>(
        residual, x, out, dim, d_scale);
}

// ── BF16 noweight variants ──
// For models with activations exceeding FP16 range (>65504).
// BF16 residual stream can store up to 3.4e38.

__global__ void rms_norm_fp8_noweight_bf16_kernel(const __nv_bfloat16* in, __nv_fp8_e4m3* out, int R, int C,
                                                    const float* descale_ptr) {
    int r = blockIdx.x; if (r >= R) return;
    const __nv_bfloat16* row = in + r * C;
    __nv_fp8_e4m3* orow = out + r * C;
    const __nv_bfloat162* row2 = reinterpret_cast<const __nv_bfloat162*>(row);
    int C2 = C / 2;
    float cache[RMS_NW_ELEMS_PER_THREAD];
    float ssq = 0;
    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < C2) {
            __nv_bfloat162 v2 = row2[c2];
            cache[it*2]   = __bfloat162float(v2.x);
            cache[it*2+1] = __bfloat162float(v2.y);
            ssq += cache[it*2]*cache[it*2] + cache[it*2+1]*cache[it*2+1];
        } else { cache[it*2] = 0; cache[it*2+1] = 0; }
    }
    __shared__ float sh[16];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    if (!lane) sh[wid] = ssq; __syncthreads();
    if (!wid) { ssq = (lane < (blockDim.x/32)) ? sh[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o); }
    __syncthreads(); if (!threadIdx.x) sh[0] = ssq; __syncthreads();
    float scale = __frsqrt_rn(sh[0] / C + 1e-6f) / fmaxf(*descale_ptr, 1e-12f);
    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < C2) {
            int c = c2 * 2;
            __nv_fp8_e4m3 pair[2];
            pair[0] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2]   * scale, -448.f), 448.f));
            pair[1] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2+1] * scale, -448.f), 448.f));
            *reinterpret_cast<uint16_t*>(orow + c) = *reinterpret_cast<uint16_t*>(pair);
        }
    }
}

void rms_norm_fp8_noweight_bf16(const __nv_bfloat16* x, __nv_fp8_e4m3* out,
                                 int seq_len, int dim,
                                 const float* d_scale, cudaStream_t stream) {
    rms_norm_fp8_noweight_bf16_kernel<<<seq_len, 256, 0, stream>>>(x, out, seq_len, dim, d_scale);
}

__global__ void res_rms_fp8_noweight_bf16_kernel(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                                   __nv_fp8_e4m3* out, int D,
                                                   const float* descale_ptr) {
    int r = blockIdx.x;
    __nv_bfloat16* res_row = residual + r * D;
    const __nv_bfloat16* x_row = x + r * D;
    __nv_fp8_e4m3* orow = out + r * D;
    int D2 = D / 2;
    __nv_bfloat162* res2w = reinterpret_cast<__nv_bfloat162*>(res_row);
    const __nv_bfloat162* res2 = reinterpret_cast<const __nv_bfloat162*>(res_row);
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x_row);
    float cache[RMS_NW_ELEMS_PER_THREAD];
    float ssq = 0;
    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < D2) {
            __nv_bfloat162 rv2 = res2[c2], xv2 = x2[c2];
            float r0 = __bfloat162float(rv2.x) + __bfloat162float(xv2.x);
            float r1 = __bfloat162float(rv2.y) + __bfloat162float(xv2.y);
            cache[it*2]   = r0;
            cache[it*2+1] = r1;
            res2w[c2] = __halves2bfloat162(__float2bfloat16(r0), __float2bfloat16(r1));
            ssq += r0*r0 + r1*r1;
        } else { cache[it*2] = 0; cache[it*2+1] = 0; }
    }
    __shared__ float sh[16];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    if (!lane) sh[wid] = ssq; __syncthreads();
    if (!wid) { ssq = (lane < (blockDim.x/32)) ? sh[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o); }
    __syncthreads(); if (!threadIdx.x) sh[0] = ssq; __syncthreads();
    float scale = __frsqrt_rn(sh[0] / D + 1e-6f) / fmaxf(*descale_ptr, 1e-12f);
    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < D2) {
            int c = c2 * 2;
            __nv_fp8_e4m3 pair[2];
            pair[0] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2]   * scale, -448.f), 448.f));
            pair[1] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2+1] * scale, -448.f), 448.f));
            *reinterpret_cast<uint16_t*>(orow + c) = *reinterpret_cast<uint16_t*>(pair);
        }
    }
}

void residual_add_rms_norm_fp8_noweight_bf16(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                               __nv_fp8_e4m3* out,
                                               int seq_len, int dim,
                                               const float* d_scale, cudaStream_t stream) {
    res_rms_fp8_noweight_bf16_kernel<<<seq_len, 256, 0, stream>>>(
        residual, x, out, dim, d_scale);
}

// Cast FP16 → FP8 (no scale). Identical to pi05 cast_fp16_fp8_k.
__global__ void cast_fp16_fp8_kernel(const __half* __restrict__ in,
                                      __nv_fp8_e4m3* __restrict__ out,
                                      int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = __nv_fp8_e4m3(__half2float(in[idx]));
}

void cast_fp16_fp8(const __half* input, __nv_fp8_e4m3* output,
                    int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cast_fp16_fp8_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

// ================================================================
// LayerNorm without affine parameters (for DiT AdaLayerNorm / norm3 / norm_out)
// out = (x - mean) / sqrt(var + eps)
// No weight/bias — elementwise_affine=False
// ================================================================

__global__ void layer_norm_no_affine_fp16_kernel(const __half* __restrict__ x,
                                                   __half* __restrict__ out,
                                                   int dim, float eps) {
    int row = blockIdx.x;
    const __half2* x2 = reinterpret_cast<const __half2*>(x + row * dim);
    __half2* out2 = reinterpret_cast<__half2*>(out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 val = x2[i];
        local_sum += __half2float(val.x) + __half2float(val.y);
    }
    // Block reduce for mean
    float val = local_sum;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) { val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o); }
    __syncthreads();
    if (!threadIdx.x) shared[0] = val;
    __syncthreads();
    float mean = shared[0] / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 v = x2[i];
        float d0 = __half2float(v.x) - mean, d1 = __half2float(v.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    val = local_var;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) { val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o); }
    __syncthreads();
    if (!threadIdx.x) shared[0] = val;
    __syncthreads();
    float inv_std = rsqrtf(shared[0] / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 xv = x2[i];
        float v0 = (__half2float(xv.x) - mean) * inv_std;
        float v1 = (__half2float(xv.y) - mean) * inv_std;
        out2[i] = __halves2half2(__float2half(v0), __float2half(v1));
    }
}

void layer_norm_no_affine_fp16(const __half* x, __half* out,
                                int seq_len, int dim, float eps,
                                cudaStream_t stream) {
    layer_norm_no_affine_fp16_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, out, dim, eps);
}

// ================================================================
// Fused AdaLayerNorm FP16 (for DiT)
// out = LN(x, no_affine) * (1 + scale) + shift
// scale, shift: [dim] per-layer precomputed from timestep
// ================================================================

__global__ void ada_layer_norm_fp16_kernel(const __half* __restrict__ x,
                                            const __half* __restrict__ scale,
                                            const __half* __restrict__ shift,
                                            __half* __restrict__ out,
                                            int dim, float eps) {
    int row = blockIdx.x;
    const __half2* x2 = reinterpret_cast<const __half2*>(x + row * dim);
    const __half2* sc2 = reinterpret_cast<const __half2*>(scale);
    const __half2* sh2 = reinterpret_cast<const __half2*>(shift);
    __half2* out2 = reinterpret_cast<__half2*>(out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 val = x2[i];
        local_sum += __half2float(val.x) + __half2float(val.y);
    }
    float val = local_sum;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) { val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o); }
    __syncthreads(); if (!threadIdx.x) shared[0] = val; __syncthreads();
    float mean = shared[0] / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 v = x2[i];
        float d0 = __half2float(v.x) - mean, d1 = __half2float(v.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    val = local_var;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) { val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o); }
    __syncthreads(); if (!threadIdx.x) shared[0] = val; __syncthreads();
    float inv_std = rsqrtf(shared[0] / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 xv = x2[i], sv = sc2[i], hv = sh2[i];
        float n0 = (__half2float(xv.x) - mean) * inv_std;
        float n1 = (__half2float(xv.y) - mean) * inv_std;
        float v0 = n0 * (1.0f + __half2float(sv.x)) + __half2float(hv.x);
        float v1 = n1 * (1.0f + __half2float(sv.y)) + __half2float(hv.y);
        out2[i] = __halves2half2(__float2half(v0), __float2half(v1));
    }
}

void ada_layer_norm_fp16(const __half* x, const __half* scale, const __half* shift,
                          __half* out, int seq_len, int dim, float eps,
                          cudaStream_t stream) {
    ada_layer_norm_fp16_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, scale, shift, out, dim, eps);
}

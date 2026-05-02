// ================================================================
// FlashVLA — Elementwise kernels (dtype-generic)
// Residual add, gate multiply, bias residual
// Supports: __half (FP16), __nv_bfloat16 (BF16) via templates
// ================================================================

#include "elementwise.cuh"
#include "common.cuh"

// ── Gate Multiply + Residual ──
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

FVK_KERNEL_INSTANTIATE(__global__ void gate_mul_res_kernel<__half>(__half*, const __half*, const __half*, int))
FVK_KERNEL_INSTANTIATE(__global__ void gate_mul_res_kernel<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int))
void gate_mul_residual(__nv_bfloat16* residual, const __nv_bfloat16* x,
                       const __nv_bfloat16* gate, int n, cudaStream_t stream) {
    int n2 = n >> 1;
    gate_mul_res_kernel<__nv_bfloat16><<<(n2 + 255) / 256, 256, 0, stream>>>(residual, x, gate, n);
}
void gate_mul_residual_fp16(__half* residual, const __half* x,
                            const __half* gate, int n, cudaStream_t stream) {
    int n2 = n >> 1;
    gate_mul_res_kernel<__half><<<(n2 + 255) / 256, 256, 0, stream>>>(residual, x, gate, n);
}

// ── Bias + Residual ──
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

FVK_KERNEL_INSTANTIATE(__global__ void bias_res_kernel<__half>(__half*, const __half*, const __half*, int))
FVK_KERNEL_INSTANTIATE(__global__ void bias_res_kernel<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int))
void bias_residual(__nv_bfloat16* residual, const __nv_bfloat16* x,
                   const __nv_bfloat16* bias, int seq_len, int dim,
                   cudaStream_t stream) {
    bias_res_kernel<__nv_bfloat16><<<seq_len, 256, 0, stream>>>(residual, x, bias, dim);
}
void bias_residual_fp16(__half* residual, const __half* x,
                        const __half* bias, int seq_len, int dim,
                        cudaStream_t stream) {
    bias_res_kernel<__half><<<seq_len, 256, 0, stream>>>(residual, x, bias, dim);
}

// ── Residual Add ──
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

FVK_KERNEL_INSTANTIATE(__global__ void res_add_kernel<__half>(__half*, const __half*, int))
FVK_KERNEL_INSTANTIATE(__global__ void res_add_kernel<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, int))
void residual_add(__nv_bfloat16* residual, const __nv_bfloat16* x, int n,
                  cudaStream_t stream) {
    int n2 = n >> 1;
    res_add_kernel<__nv_bfloat16><<<(n2 + 255) / 256, 256, 0, stream>>>(residual, x, n);
}
void residual_add_fp16(__half* residual, const __half* x, int n,
                       cudaStream_t stream) {
    int n2 = n >> 1;
    res_add_kernel<__half><<<(n2 + 255) / 256, 256, 0, stream>>>(residual, x, n);
}

// ── Classifier-Free Guidance combine ──
// In-place: residual[i] += v_uncond[i] + beta * (v_cond[i] - v_uncond[i])
template<typename T>
__global__ void cfg_combine_kernel(T* __restrict__ residual,
                                   const T* __restrict__ v_cond,
                                   const T* __restrict__ v_uncond,
                                   float beta, int n) {
    using T2 = typename packed2<T>::type;
    T2* res2 = reinterpret_cast<T2*>(residual);
    const T2* vc2 = reinterpret_cast<const T2*>(v_cond);
    const T2* vu2 = reinterpret_cast<const T2*>(v_uncond);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        T2 rv = res2[idx], vc = vc2[idx], vu = vu2[idx];
        float rx = to_f32(rv.x), ry = to_f32(rv.y);
        float vcx = to_f32(vc.x), vcy = to_f32(vc.y);
        float vux = to_f32(vu.x), vuy = to_f32(vu.y);
        float gx = vux + beta * (vcx - vux);
        float gy = vuy + beta * (vcy - vuy);
        res2[idx] = make_packed2<T>(
            from_f32<T>(rx + gx),
            from_f32<T>(ry + gy));
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void cfg_combine_kernel<__half>(
    __half*, const __half*, const __half*, float, int))
FVK_KERNEL_INSTANTIATE(__global__ void cfg_combine_kernel<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, float, int))
void cfg_combine_into_residual(__nv_bfloat16* residual,
                               const __nv_bfloat16* v_cond,
                               const __nv_bfloat16* v_uncond,
                               float beta, int n,
                               cudaStream_t stream) {
    int n2 = n >> 1;
    cfg_combine_kernel<__nv_bfloat16><<<(n2 + 255) / 256, 256, 0, stream>>>(
        residual, v_cond, v_uncond, beta, n);
}

void cfg_combine_into_residual_fp16(__half* residual,
                                    const __half* v_cond,
                                    const __half* v_uncond,
                                    float beta, int n,
                                    cudaStream_t stream) {
    int n2 = n >> 1;
    cfg_combine_kernel<__half><<<(n2 + 255) / 256, 256, 0, stream>>>(
        residual, v_cond, v_uncond, beta, n);
}

// ================================================================
// GPU memory/copy ops for CUDA Graph compatibility (DiT pipeline)
// These replace PyTorch .copy_()/.fill_()/.half() which don't
// submit to the correct CUDA stream during graph capture.
// ================================================================

void gpu_copy_async(void* dst, const void* src, size_t nbytes, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream);
}

// Fill FP16 buffer with large negative (for softmax masking in attention)
__global__ void fill_neginf_fp16_kernel(__half* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = __float2half(-1e30f);
}
void gpu_fill_neginf_fp16(__half* dst, int n, cudaStream_t stream) {
    fill_neginf_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(dst, n);
}

// Strided copy: src[rows, src_cols] col_offset:col_offset+dst_cols → dst[rows, dst_cols]
// For QKV split: [Sa, 3D] → [Sa, D] at offsets 0, D, 2D
__global__ void strided_copy_fp16_kernel(const __half* src, __half* dst,
                                          int rows, int dst_cols, int src_stride, int col_offset) {
    int r = blockIdx.x;
    int c = threadIdx.x;
    for (int cc = c; cc < dst_cols; cc += blockDim.x) {
        dst[r * dst_cols + cc] = src[r * src_stride + col_offset + cc];
    }
}
void gpu_strided_copy_fp16(const __half* src, __half* dst,
                            int rows, int dst_cols, int src_stride, int col_offset,
                            cudaStream_t stream) {
    strided_copy_fp16_kernel<<<rows, min(256, dst_cols), 0, stream>>>(
        src, dst, rows, dst_cols, src_stride, col_offset);
}

// Cast FP32 → FP16
__global__ void cast_fp32_fp16_kernel(const float* src, __half* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}
void gpu_cast_fp32_to_fp16(const float* src, __half* dst, int n, cudaStream_t stream) {
    cast_fp32_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, n);
}

// Euler step: actions_fp32[0:T*D] += dt * velocity_fp16[offset:offset+T*D]
__global__ void euler_step_kernel(float* actions, const __half* velocity,
                                   float dt, int n, int vel_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) actions[idx] += dt * __half2float(velocity[vel_offset + idx]);
}
// ── GQA KV repeat interleave (8→16 heads) ──
// src: [S, NH_src * HD], dst: [S, NH_dst * HD] where NH_dst = NH_src * repeat
// Each src head is copied `repeat` times to consecutive dst heads
__global__ void repeat_interleave_heads_kernel(
    const __half* src, __half* dst,
    int S, int NH_src, int HD, int repeat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int NH_dst = NH_src * repeat;
    int total = S * NH_dst * HD;
    if (idx >= total) return;

    int d = idx % HD;
    int remainder = idx / HD;
    int h_dst = remainder % NH_dst;
    int s = remainder / NH_dst;

    int h_src = h_dst / repeat;
    dst[s * NH_dst * HD + h_dst * HD + d] = src[s * NH_src * HD + h_src * HD + d];
}

void gpu_repeat_interleave_heads(const __half* src, __half* dst,
                                  int S, int NH_src, int HD, int repeat,
                                  cudaStream_t stream) {
    int NH_dst = NH_src * repeat;
    int total = S * NH_dst * HD;
    repeat_interleave_heads_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        src, dst, S, NH_src, HD, repeat);
}

void gpu_euler_step(float* actions, const __half* velocity,
                     int T, int action_dim, float dt, int vel_elem_offset,
                     cudaStream_t stream) {
    int n = T * action_dim;
    euler_step_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        actions, velocity, dt, n, vel_elem_offset);
}

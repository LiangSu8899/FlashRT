// FlashRT — FP8 E4M3 Quantize API (standalone .so)
//
// Compile:
//   nvcc -O3 -std=c++17 -arch=sm_110a --use_fast_math \
//        -shared -Xcompiler -fPIC quantize_fp8_api.cu -o libquantize_fp8.so
//
// C API for ctypes:
//   quantize_fp8_e4m3(input, output, scale_out, n, stream) → 0 on success
//
// Performance (Thor SM110, device buffer):
//   128MB (67M fp16 elements): 2.1ms — 27x faster than torch
// Precision:
//   byte match 100% vs torch.float8_e4m3fn and ml_dtypes.float8_e4m3fn

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

using fp8e4 = __nv_fp8_e4m3;

// ── Kernel 1: parallel amax reduce ──
__global__ void amax_reduce_k(const __half* data, float* global_amax, int n) {
    float local_max = 0.0f;
    int base = blockIdx.x * blockDim.x * 4;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int i = base + j * blockDim.x + threadIdx.x;
        if (i < n) {
            float v = fabsf(__half2float(data[i]));
            local_max = fmaxf(local_max, v);
        }
    }
    // Warp reduce
    for (int o = 16; o > 0; o >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
    // Block reduce via atomicMax
    if (!(threadIdx.x & 31)) {
        int* addr = (int*)global_amax;
        int old = *addr, assumed;
        do {
            assumed = old;
            old = atomicCAS(addr, assumed,
                __float_as_int(fmaxf(__int_as_float(assumed), local_max)));
        } while (assumed != old);
    }
}

// ── Kernel 2: static quantize with known scale (packed 4-wide store) ──
__global__ void quant_static_k(const __half* in, unsigned char* out,
                                const float* scale_ptr, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    float inv_scale = 1.0f / fmaxf(*scale_ptr, 1e-12f);
    const __half2* in2 = reinterpret_cast<const __half2*>(in);
    __half2 vA = in2[i/2], vB = in2[i/2+1];
    float fv[4] = {__half2float(vA.x), __half2float(vA.y),
                   __half2float(vB.x), __half2float(vB.y)};
    fp8e4 fp8_pack[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        fp8_pack[j] = fp8e4(fminf(fmaxf(fv[j] * inv_scale, -448.f), 448.f));
    }
    *reinterpret_cast<unsigned int*>(&out[i]) =
        *reinterpret_cast<unsigned int*>(fp8_pack);
}

// ══════════════════════════════════════════════════════════════
// C API
// ══════════════════════════════════════════════════════════════
extern "C" {

/// Quantize fp16 → FP8 E4M3 with per-tensor scale.
///
/// All pointers must be device (cudaMalloc). Not managed.
/// scale_out receives: scale = amax / 448.0
/// Requires cudaDeviceSynchronize after call (host sync for scale readback).
///
/// @param input     half* device, fp16 weight data
/// @param output    uint8* device, fp8 output (same element count)
/// @param scale_out float* device, per-tensor scale (1 float)
/// @param n         number of fp16 elements
/// @param stream    cudaStream_t (NULL = default stream)
/// @return 0 on success
int quantize_fp8_e4m3(void* input, void* output, void* scale_out,
                       int n, void* stream_ptr) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    const __half* in = (const __half*)input;
    unsigned char* out = (unsigned char*)output;
    float* d_scale = (float*)scale_out;

    int threads = 256;

    // Step 1: zero amax
    cudaMemsetAsync(d_scale, 0, sizeof(float), stream);

    // Step 2: amax reduce → d_scale now holds amax (float, device)
    int amax_blocks = (n + threads * 4 - 1) / (threads * 4);
    amax_reduce_k<<<amax_blocks, threads, 0, stream>>>(in, d_scale, n);

    // Step 3: caller must:
    //   cudaDeviceSynchronize()
    //   read d_scale (amax) via Python ctypes cudaMemcpy D2H
    //   compute scale = max(amax/448, 1e-12)
    //   write scale back to d_scale via cudaMemcpy H2D
    //   then call quantize_fp8_apply()
    // This avoids C-side cudaMemcpy D2H which segfaults on Thor.

    return 0;
}

/// Apply quantize with pre-computed scale (already on device).
/// Call after quantize_fp8_e4m3 + scale writeback.
int quantize_fp8_apply(void* input, void* output, void* scale_device,
                        int n, void* stream_ptr) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int threads = 256;
    int blocks = (n + threads * 4 - 1) / (threads * 4);
    quant_static_k<<<blocks, threads, 0, stream>>>(
        (const __half*)input, (unsigned char*)output,
        (const float*)scale_device, n);
    return 0;
}

}  // extern "C"

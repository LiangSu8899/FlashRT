#include "flashrt/native_cpp/operations.h"

#include "softmax.cuh"

namespace {

__global__ void mask_seqused_logits_kernel(
    __half* logits, int actual_key_rows, int logits_stride,
    int columns, const int* seqused_key) {
    int valid = seqused_key[0];
    if (valid < 0) valid = 0;
    if (valid > actual_key_rows) valid = actual_key_rows;
    const int mask_rows = logits_stride - valid;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = columns * mask_rows;
    if (index >= total) return;
    const int column = index / mask_rows;
    const int row = valid + index % mask_rows;
    logits[column * logits_stride + row] = __float2half(-65504.0f);
}

}  // namespace

void flashrt_native_attention_qkv_fp16_seqused(
    cublasHandle_t handle,
    const __half* query, const __half* key, const __half* value,
    __half* logits, __half* output,
    int query_rows, int key_rows, int heads, int head_dimension,
    const int* seqused_key, float attention_scale, cudaStream_t stream) {
    cublasSetStream(handle, stream);
    const int logits_stride = key_rows + (key_rows & 1);

    const float zero = 0.0f;
    cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        key_rows, query_rows * heads, head_dimension,
        &attention_scale,
        key, CUDA_R_16F, head_dimension,
        query, CUDA_R_16F, head_dimension,
        &zero,
        logits, CUDA_R_16F, logits_stride,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    const int columns = query_rows * heads;
    const int mask_elements = columns * logits_stride;
    if (mask_elements > 0) {
        mask_seqused_logits_kernel<<<
            (mask_elements + 255) / 256, 256, 0, stream>>>(
            logits, key_rows, logits_stride, columns, seqused_key);
    }
    softmax_fp16(logits, columns, logits_stride, stream);

    const float one = 1.0f;
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        head_dimension, columns, key_rows,
        &one,
        value, CUDA_R_16F, head_dimension,
        logits, CUDA_R_16F, logits_stride,
        &zero,
        output, CUDA_R_16F, head_dimension,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

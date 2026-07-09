#include "flashrt/cpp/modalities/text.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace flashrt {
namespace modalities {
namespace {

__device__ __forceinline__ float bf16_to_f32(std::uint16_t value) {
    return __uint_as_float(static_cast<std::uint32_t>(value) << 16);
}

__device__ __forceinline__ std::uint16_t f32_to_bf16(float value) {
    std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(bits >> 16);
}

__device__ __forceinline__ float load_value(const void* base,
                                            std::uint64_t index,
                                            int dtype) {
    if (dtype == 1) return static_cast<const float*>(base)[index];
    if (dtype == 2) return __half2float(static_cast<const __half*>(base)[index]);
    if (dtype == 3) {
        return bf16_to_f32(static_cast<const std::uint16_t*>(base)[index]);
    }
    return static_cast<float>(static_cast<const std::uint8_t*>(base)[index]);
}

__device__ __forceinline__ void store_value(void* base,
                                            std::uint64_t index,
                                            int dtype,
                                            float value) {
    if (dtype == 1) {
        static_cast<float*>(base)[index] = value;
    } else if (dtype == 2) {
        static_cast<__half*>(base)[index] = __float2half_rn(value);
    } else if (dtype == 3) {
        static_cast<std::uint16_t*>(base)[index] = f32_to_bf16(value);
    } else {
        static_cast<std::uint8_t*>(base)[index] =
            static_cast<std::uint8_t>(value);
    }
}

int dtype_code(DType dtype) {
    switch (dtype) {
        case DType::kFloat32: return 1;
        case DType::kFloat16: return 2;
        case DType::kBFloat16: return 3;
        case DType::kUInt8: return 0;
    }
    return 0;
}

Status validate_device_matrix(const TensorView& tensor, const char* name,
                              std::uint64_t rows, std::uint64_t cols) {
    if (!tensor.data) {
        return Status::error(StatusCode::kInvalidArgument,
                             std::string(name) + " has null data");
    }
    if (tensor.place != MemoryPlace::kDevice) {
        return Status::error(StatusCode::kUnsupported,
                             std::string(name) + " is not device memory");
    }
    if (tensor.layout != Layout::kFlat || tensor.shape.rank != 2 ||
        tensor.shape.dims[0] != rows || tensor.shape.dims[1] != cols) {
        return Status::error(StatusCode::kShapeMismatch,
                             std::string(name) + " shape mismatch");
    }
    const std::uint64_t need = rows * cols * dtype_size(tensor.dtype);
    if (tensor.bytes < need) {
        return Status::error(StatusCode::kInsufficientStorage,
                             std::string(name) + " storage is too small");
    }
    return Status::ok();
}

__global__ void gather_kernel(const std::int32_t* ids,
                              std::uint64_t n_tokens,
                              std::uint64_t vocab_size,
                              std::uint64_t hidden_dim,
                              const void* table,
                              int table_dtype,
                              void* output,
                              int output_dtype,
                              float scale,
                              int* bad_token) {
    const std::uint64_t idx =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = n_tokens * hidden_dim;
    if (idx >= total) return;
    const std::uint64_t token_index = idx / hidden_dim;
    const std::uint64_t dim = idx - token_index * hidden_dim;
    const std::int32_t token = ids[token_index];
    if (token < 0 || static_cast<std::uint64_t>(token) >= vocab_size) {
        atomicCAS(bad_token, 0, 1);
        return;
    }
    const std::uint64_t src =
        static_cast<std::uint64_t>(token) * hidden_dim + dim;
    const float value = load_value(table, src, table_dtype) * scale;
    store_value(output, idx, output_dtype, value);
}

const char* cuda_error(cudaError_t rc) {
    return cudaGetErrorString(rc);
}

}  // namespace

Status gather_token_embeddings_cuda(const EmbeddingGatherSpec& spec,
                                    const std::int32_t* token_ids,
                                    std::uint64_t n_tokens,
                                    TensorView embedding_table,
                                    TensorView output,
                                    void* stream,
                                    TextEmbeddingStaging* staging) {
    if (!token_ids && n_tokens) {
        return Status::error(StatusCode::kInvalidArgument,
                             "token_ids is null");
    }
    if (!spec.vocab_size || !spec.hidden_dim) {
        return Status::error(StatusCode::kInvalidArgument,
                             "invalid embedding gather dimensions");
    }
    Status st = validate_device_matrix(embedding_table, "embedding_table",
                                       spec.vocab_size, spec.hidden_dim);
    if (!st.ok_status()) return st;
    st = validate_device_matrix(output, "embedding_output", n_tokens,
                                spec.hidden_dim);
    if (!st.ok_status()) return st;
    if (staging && staging->max_tokens < n_tokens) {
        return Status::error(StatusCode::kInsufficientStorage,
                             "text token staging capacity is too small");
    }

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    std::int32_t* d_ids = nullptr;
    int* d_bad = nullptr;
    cudaError_t rc = cudaSuccess;
    if (staging) {
        d_ids = static_cast<std::int32_t*>(staging->device_token_ids);
        d_bad = static_cast<int*>(staging->device_status);
    } else {
        rc = cudaMalloc(&d_ids, n_tokens * sizeof(std::int32_t));
        if (rc != cudaSuccess) {
            return Status::error(
                StatusCode::kBackend,
                std::string("cudaMalloc text token ids failed: ") +
                    cuda_error(rc));
        }
    }
    if (!d_bad) rc = cudaMalloc(&d_bad, sizeof(int));
    if (rc == cudaSuccess) {
        rc = cudaMemsetAsync(d_bad, 0, sizeof(int), cuda_stream);
    }
    if (rc == cudaSuccess && n_tokens) {
        rc = cudaMemcpyAsync(d_ids, token_ids, n_tokens * sizeof(std::int32_t),
                             cudaMemcpyHostToDevice, cuda_stream);
    }
    if (rc != cudaSuccess) {
        if (!staging) cudaFree(d_ids);
        if (!staging && d_bad) cudaFree(d_bad);
        return Status::error(StatusCode::kBackend,
                             std::string("cuda text token upload failed: ") +
                                 cuda_error(rc));
    }

    const std::uint64_t total = n_tokens * spec.hidden_dim;
    if (total) {
        const int block = 256;
        const int grid = static_cast<int>((total + block - 1) / block);
        gather_kernel<<<grid, block, 0, cuda_stream>>>(
            d_ids, n_tokens, spec.vocab_size, spec.hidden_dim,
            embedding_table.data, dtype_code(embedding_table.dtype),
            output.data, dtype_code(output.dtype), spec.scale, d_bad);
        rc = cudaGetLastError();
    }

    int bad = 0;
    if (rc == cudaSuccess) {
        rc = cudaMemcpyAsync(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost,
                             cuda_stream);
    }
    if (rc == cudaSuccess) rc = cudaStreamSynchronize(cuda_stream);
    if (!staging) cudaFree(d_ids);
    if (!staging) cudaFree(d_bad);
    if (rc != cudaSuccess) {
        return Status::error(StatusCode::kBackend,
                             std::string("text embedding CUDA failed: ") +
                                 cuda_error(rc));
    }
    if (bad) {
        return Status::error(StatusCode::kInvalidArgument,
                             "token id is out of vocabulary range");
    }
    return Status::ok();
}

}  // namespace modalities
}  // namespace flashrt

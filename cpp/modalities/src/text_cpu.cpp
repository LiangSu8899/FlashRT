#include "flashrt/cpp/modalities/text.h"

#ifdef FLASHRT_CPP_WITH_CUDA_STAGING
#include <cuda_runtime_api.h>
#endif

#include <string>
#include <vector>

namespace flashrt {
namespace modalities {

#ifdef FLASHRT_CPP_WITH_CUDA_KERNELS
Status gather_token_embeddings_cuda(const EmbeddingGatherSpec& spec,
                                    const std::int32_t* token_ids,
                                    std::uint64_t n_tokens,
                                    TensorView embedding_table,
                                    TensorView output,
                                    void* stream,
                                    TextEmbeddingStaging* staging);
#endif

namespace {

float load_scalar(const void* base, std::uint64_t index, DType dtype) {
    switch (dtype) {
        case DType::kFloat32:
            return static_cast<const float*>(base)[index];
        case DType::kBFloat16:
            return bfloat16_to_float(
                static_cast<const std::uint16_t*>(base)[index]);
        case DType::kFloat16:
            return float16_to_float(
                static_cast<const std::uint16_t*>(base)[index]);
        case DType::kUInt8:
            return static_cast<float>(
                static_cast<const std::uint8_t*>(base)[index]);
    }
    return 0.0f;
}

void store_scalar(void* base, std::uint64_t index, DType dtype, float value) {
    switch (dtype) {
        case DType::kFloat32:
            static_cast<float*>(base)[index] = value;
            break;
        case DType::kBFloat16:
            static_cast<std::uint16_t*>(base)[index] = float_to_bfloat16(value);
            break;
        case DType::kFloat16:
            static_cast<std::uint16_t*>(base)[index] = float_to_float16(value);
            break;
        case DType::kUInt8:
            static_cast<std::uint8_t*>(base)[index] =
                static_cast<std::uint8_t>(value);
            break;
    }
}

Status validate_matrix(const TensorView& tensor, const char* name,
                       std::uint64_t rows, std::uint64_t cols) {
    Status st = validate_host_tensor(tensor, name);
    if (!st.ok_status()) return st;
    if (tensor.layout != Layout::kFlat || tensor.shape.rank != 2 ||
        tensor.shape.dims[0] != rows || tensor.shape.dims[1] != cols) {
        return Status::error(StatusCode::kShapeMismatch,
                             std::string(name) + " shape mismatch");
    }
    return Status::ok();
}

}  // namespace

#ifdef FLASHRT_CPP_WITH_CUDA_STAGING
Status text_embedding_staging_create(TextEmbeddingStaging* out,
                                     std::uint64_t max_tokens) {
    if (!out || !max_tokens) {
        return Status::error(StatusCode::kInvalidArgument,
                             "invalid text embedding staging capacity");
    }
    *out = TextEmbeddingStaging{};
    cudaError_t rc = cudaMalloc(&out->device_token_ids,
                                max_tokens * sizeof(std::int32_t));
    if (rc != cudaSuccess) {
        return Status::error(
            StatusCode::kBackend,
            std::string("text token staging cudaMalloc failed: ") +
                cudaGetErrorString(rc));
    }
    rc = cudaMalloc(&out->device_status, sizeof(int));
    if (rc != cudaSuccess) {
        cudaFree(out->device_token_ids);
        *out = TextEmbeddingStaging{};
        return Status::error(
            StatusCode::kBackend,
            std::string("text status staging cudaMalloc failed: ") +
                cudaGetErrorString(rc));
    }
    out->max_tokens = max_tokens;
    return Status::ok();
}

void text_embedding_staging_destroy(TextEmbeddingStaging* s) {
    if (!s) return;
    if (s->device_token_ids) cudaFree(s->device_token_ids);
    if (s->device_status) cudaFree(s->device_status);
    *s = TextEmbeddingStaging{};
}
#else
Status text_embedding_staging_create(TextEmbeddingStaging* out,
                                     std::uint64_t) {
    if (out) *out = TextEmbeddingStaging{};
    return Status::error(StatusCode::kUnsupported,
                         "text embedding staging requires the CUDA build");
}

void text_embedding_staging_destroy(TextEmbeddingStaging* s) {
    if (s) *s = TextEmbeddingStaging{};
}
#endif

Status gather_token_embeddings_cpu(const EmbeddingGatherSpec& spec,
                                   const std::int32_t* token_ids,
                                   std::uint64_t n_tokens,
                                   TensorView embedding_table,
                                   TensorView output) {
    if (!token_ids && n_tokens) {
        return Status::error(StatusCode::kInvalidArgument,
                             "token_ids is null");
    }
    if (!spec.vocab_size || !spec.hidden_dim) {
        return Status::error(StatusCode::kInvalidArgument,
                             "invalid embedding gather dimensions");
    }
    Status st = validate_matrix(embedding_table, "embedding_table",
                                spec.vocab_size, spec.hidden_dim);
    if (!st.ok_status()) return st;
    st = validate_matrix(output, "embedding_output", n_tokens,
                         spec.hidden_dim);
    if (!st.ok_status()) return st;

    for (std::uint64_t t = 0; t < n_tokens; ++t) {
        const std::int32_t token = token_ids[t];
        if (token < 0 ||
            static_cast<std::uint64_t>(token) >= spec.vocab_size) {
            return Status::error(StatusCode::kInvalidArgument,
                                 "token id is out of vocabulary range");
        }
        const std::uint64_t src_base =
            static_cast<std::uint64_t>(token) * spec.hidden_dim;
        const std::uint64_t dst_base = t * spec.hidden_dim;
        for (std::uint64_t d = 0; d < spec.hidden_dim; ++d) {
            const float value = load_scalar(
                embedding_table.data, src_base + d, embedding_table.dtype);
            store_scalar(output.data, dst_base + d, output.dtype,
                         value * spec.scale);
        }
    }
    return Status::ok();
}

Status gather_token_embeddings(const EmbeddingGatherSpec& spec,
                               const std::int32_t* token_ids,
                               std::uint64_t n_tokens,
                               TensorView embedding_table,
                               TensorView output,
                               void* stream,
                               TextEmbeddingStaging* staging) {
    if (output.place == MemoryPlace::kHost ||
        output.place == MemoryPlace::kHostPinned) {
        (void)stream;
        (void)staging;
        return gather_token_embeddings_cpu(spec, token_ids, n_tokens,
                                           embedding_table, output);
    }
    if (output.place != MemoryPlace::kDevice ||
        embedding_table.place != MemoryPlace::kDevice) {
        return Status::error(StatusCode::kUnsupported,
                             "device text embedding requires device tensors");
    }
#ifndef FLASHRT_CPP_WITH_CUDA_STAGING
    (void)stream;
    (void)staging;
    return Status::error(StatusCode::kUnsupported,
                         "device text embedding was not enabled at build time");
#else
    if (!token_ids && n_tokens) {
        return Status::error(StatusCode::kInvalidArgument,
                             "token_ids is null");
    }
    if (staging && staging->max_tokens < n_tokens) {
        return Status::error(StatusCode::kInsufficientStorage,
                             "text token staging capacity is too small");
    }
#ifdef FLASHRT_CPP_WITH_CUDA_KERNELS
    return gather_token_embeddings_cuda(spec, token_ids, n_tokens,
                                        embedding_table, output, stream,
                                        staging);
#else
    std::vector<std::uint8_t> host_bytes(
        static_cast<std::size_t>(n_tokens * spec.hidden_dim *
                                 dtype_size(output.dtype)));
    TensorView host_output{host_bytes.data(),
                           static_cast<std::uint64_t>(host_bytes.size()),
                           output.dtype, MemoryPlace::kHost, output.layout,
                           Shape{n_tokens, spec.hidden_dim}};
    TensorView host_table = embedding_table;
    if (embedding_table.place != MemoryPlace::kHost &&
        embedding_table.place != MemoryPlace::kHostPinned) {
        return Status::error(StatusCode::kUnsupported,
                             "CUDA kernel build is required for device embedding tables");
    }
    Status st = gather_token_embeddings_cpu(spec, token_ids, n_tokens,
                                            host_table, host_output);
    if (!st.ok_status()) return st;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t rc = cudaMemcpyAsync(output.data, host_bytes.data(),
                                     host_bytes.size(), cudaMemcpyHostToDevice,
                                     cuda_stream);
    if (rc == cudaSuccess) rc = cudaStreamSynchronize(cuda_stream);
    if (rc != cudaSuccess) {
        return Status::error(StatusCode::kBackend,
                             std::string("cuda H2D text embedding failed: ") +
                                 cudaGetErrorString(rc));
    }
    return Status::ok();
#endif
#endif
}

}  // namespace modalities
}  // namespace flashrt

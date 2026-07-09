#include "flashrt/cpp/modalities/text.h"

#include <string>

namespace flashrt {
namespace modalities {
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

}  // namespace modalities
}  // namespace flashrt

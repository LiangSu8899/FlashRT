#include "flashrt/cpp/models/pi05/model/prompt_embed.h"

#include "flashrt/cpp/models/pi05/model/prompt_format.h"

#ifdef FLASHRT_CPP_WITH_CUDA_STAGING
#include <cuda_runtime_api.h>
#endif

#include <cstring>
#include <limits>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

bool checked_multiply(std::uint64_t lhs, std::uint64_t rhs,
                      std::uint64_t* out) {
    if (!out ||
        (rhs && lhs > std::numeric_limits<std::uint64_t>::max() / rhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

modalities::Status validate_output_capacity(
    const PromptEmbeddingSpec& spec,
    const modalities::TensorView& output,
    std::uint64_t* required_bytes) {
    if (!spec.vocab_size || !spec.hidden_dim || !spec.max_tokens) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "invalid prompt embedding dimensions");
    }
    if (!output.data) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "prompt_embedding has null data");
    }
    if (output.place != modalities::MemoryPlace::kHost &&
        output.place != modalities::MemoryPlace::kHostPinned &&
        output.place != modalities::MemoryPlace::kDevice) {
        return modalities::Status::error(
            modalities::StatusCode::kUnsupported,
            "prompt_embedding memory place is unsupported");
    }
    if (output.layout != modalities::Layout::kFlat ||
        output.shape.rank != 2 ||
        output.shape.dims[0] != spec.max_tokens ||
        output.shape.dims[1] != spec.hidden_dim) {
        return modalities::Status::error(
            modalities::StatusCode::kShapeMismatch,
            "prompt_embedding shape mismatch");
    }
    std::uint64_t elements = 0;
    const std::uint64_t element_bytes = modalities::dtype_size(output.dtype);
    if (!element_bytes ||
        !checked_multiply(spec.max_tokens, spec.hidden_dim, &elements) ||
        !checked_multiply(elements, element_bytes, required_bytes)) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "prompt_embedding byte size overflows");
    }
    if (output.bytes < *required_bytes) {
        return modalities::Status::error(
            modalities::StatusCode::kInsufficientStorage,
            "prompt_embedding storage is too small");
    }
    return modalities::Status::ok();
}

modalities::Status zero_prompt_output(const modalities::TensorView& output,
                                      std::uint64_t bytes,
                                      void* stream) {
    if (output.place == modalities::MemoryPlace::kHost ||
        output.place == modalities::MemoryPlace::kHostPinned) {
        std::memset(output.data, 0, static_cast<std::size_t>(bytes));
        return modalities::Status::ok();
    }
#ifndef FLASHRT_CPP_WITH_CUDA_STAGING
    (void)stream;
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "device prompt zeroing requires the CUDA build");
#else
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t rc = cudaMemsetAsync(output.data, 0, bytes, cuda_stream);
    if (rc == cudaSuccess) rc = cudaStreamSynchronize(cuda_stream);
    if (rc != cudaSuccess) {
        return modalities::Status::error(
            modalities::StatusCode::kBackend,
            std::string("cuda prompt zeroing failed: ") +
                cudaGetErrorString(rc));
    }
    return modalities::Status::ok();
#endif
}

}  // namespace

modalities::Status embed_prompt(
    modalities::SentencePieceTokenizer& tokenizer,
    const PromptEmbeddingSpec& spec,
    const std::string& prompt,
    const float* state,
    std::uint64_t n_state,
    modalities::TensorView embedding_table,
    modalities::TensorView output,
    std::vector<std::int32_t>* token_ids,
    std::uint64_t* prompt_len,
    void* stream,
    modalities::TextEmbeddingStaging* staging,
    std::string* formatted_workspace) {
    if (!token_ids || !prompt_len || (!state && n_state)) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "prompt embedding arguments are invalid");
    }
    token_ids->clear();
    *prompt_len = 0;
    std::uint64_t output_bytes = 0;
    auto st = validate_output_capacity(spec, output, &output_bytes);
    if (!st.ok_status()) return st;
    if (!tokenizer.loaded()) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "SentencePiece model is not loaded");
    }

    modalities::SentencePieceEncodeOptions options;
    options.add_bos = true;
    options.max_tokens = spec.max_tokens;
    std::string local;
    std::string* formatted = formatted_workspace ? formatted_workspace
                                                 : &local;
    if (state) {
        format_state_prompt_into(prompt, state, n_state, formatted);
        st = tokenizer.encode(*formatted, options, token_ids);
    } else {
        format_state_prompt_into(prompt, nullptr, 0, formatted);
        st = tokenizer.encode(*formatted, options, token_ids);
        if (st.ok_status() && spec.no_state_suffix_token_id >= 0) {
            token_ids->push_back(spec.no_state_suffix_token_id);
        }
    }
    if (!st.ok_status()) return st;
    if (token_ids->size() > spec.max_tokens) {
        return modalities::Status::error(
            modalities::StatusCode::kShapeMismatch,
            "prompt token count exceeds max_tokens");
    }

    if (spec.zero_pad_output) {
        st = zero_prompt_output(output, output_bytes, stream);
        if (!st.ok_status()) return st;
    }
    modalities::TensorView prefix = output;
    prefix.shape = modalities::Shape{
        static_cast<std::uint64_t>(token_ids->size()), spec.hidden_dim};
    prefix.bytes = static_cast<std::uint64_t>(token_ids->size()) *
                   spec.hidden_dim * modalities::dtype_size(output.dtype);

    modalities::EmbeddingGatherSpec gather{spec.vocab_size, spec.hidden_dim,
                                           spec.scale};
    st = modalities::gather_token_embeddings(
        gather, token_ids->data(), token_ids->size(), embedding_table, prefix,
        stream, staging);
    if (!st.ok_status()) return st;
    *prompt_len = static_cast<std::uint64_t>(token_ids->size());
    return modalities::Status::ok();
}

modalities::Status embed_prompt_cpu(
    modalities::SentencePieceTokenizer& tokenizer,
    const PromptEmbeddingSpec& spec,
    const std::string& prompt,
    const float* state,
    std::uint64_t n_state,
    modalities::TensorView embedding_table,
    modalities::TensorView output,
    std::vector<std::int32_t>* token_ids,
    std::uint64_t* prompt_len) {
    return embed_prompt(tokenizer, spec, prompt, state, n_state,
                        embedding_table, output, token_ids, prompt_len);
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

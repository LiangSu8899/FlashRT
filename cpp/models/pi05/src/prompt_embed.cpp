#include "flashrt/cpp/models/pi05/prompt_embed.h"

#include "flashrt/cpp/models/pi05/prompt_format.h"

#include <cstring>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status validate_output_capacity(
        const PromptEmbeddingSpec& spec,
        const modalities::TensorView& output) {
    if (!spec.vocab_size || !spec.hidden_dim || !spec.max_tokens) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "invalid prompt embedding dimensions");
    }
    auto st = modalities::validate_host_tensor(output, "prompt_embedding");
    if (!st.ok_status()) return st;
    if (output.layout != modalities::Layout::kFlat ||
        output.shape.rank != 2 ||
        output.shape.dims[0] != spec.max_tokens ||
        output.shape.dims[1] != spec.hidden_dim) {
        return modalities::Status::error(
            modalities::StatusCode::kShapeMismatch,
            "prompt_embedding shape mismatch");
    }
    return modalities::Status::ok();
}

}  // namespace

modalities::Status embed_prompt_cpu(
        const modalities::SentencePieceTokenizer& tokenizer,
        const PromptEmbeddingSpec& spec,
        const std::string& prompt,
        const float* state,
        std::uint64_t n_state,
        modalities::TensorView embedding_table,
        modalities::TensorView output,
        std::vector<std::int32_t>* token_ids,
        std::uint64_t* prompt_len) {
    if (!token_ids || !prompt_len) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "prompt embedding outputs are null");
    }
    token_ids->clear();
    *prompt_len = 0;
    auto st = validate_output_capacity(spec, output);
    if (!st.ok_status()) return st;
    if (!tokenizer.loaded()) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "SentencePiece model is not loaded");
    }

    modalities::SentencePieceEncodeOptions options;
    options.add_bos = true;
    if (state) {
        const std::string formatted = format_state_prompt(prompt, state,
                                                          n_state);
        st = tokenizer.encode(formatted, options, token_ids);
    } else {
        st = tokenizer.encode(prompt, options, token_ids);
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
        std::memset(output.data, 0, static_cast<std::size_t>(output.bytes));
    }
    modalities::TensorView prefix = output;
    prefix.shape = modalities::Shape{static_cast<std::uint64_t>(
                                         token_ids->size()),
                                     spec.hidden_dim};
    prefix.bytes = static_cast<std::uint64_t>(token_ids->size()) *
                   spec.hidden_dim * modalities::dtype_size(output.dtype);

    modalities::EmbeddingGatherSpec gather{spec.vocab_size, spec.hidden_dim,
                                           spec.scale};
    st = modalities::gather_token_embeddings_cpu(
        gather, token_ids->data(), token_ids->size(), embedding_table, prefix);
    if (!st.ok_status()) return st;
    *prompt_len = static_cast<std::uint64_t>(token_ids->size());
    return modalities::Status::ok();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

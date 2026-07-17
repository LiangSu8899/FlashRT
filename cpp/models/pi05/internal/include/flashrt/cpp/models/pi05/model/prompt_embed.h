#ifndef FLASHRT_CPP_MODELS_PI05_PROMPT_EMBED_H
#define FLASHRT_CPP_MODELS_PI05_PROMPT_EMBED_H

#include "flashrt/cpp/modalities/text.h"
#include "flashrt/cpp/modalities/tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

struct PromptEmbeddingSpec {
    std::uint64_t vocab_size = 0;
    std::uint64_t hidden_dim = 0;
    std::uint64_t max_tokens = 0;
    float scale = 1.0f;
    std::int32_t no_state_suffix_token_id = 108;
    bool zero_pad_output = true;
};

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
    void* stream = nullptr,
    modalities::TextEmbeddingStaging* staging = nullptr,
    std::string* formatted_workspace = nullptr);

modalities::Status embed_prompt_cpu(
    modalities::SentencePieceTokenizer& tokenizer,
    const PromptEmbeddingSpec& spec,
    const std::string& prompt,
    const float* state,
    std::uint64_t n_state,
    modalities::TensorView embedding_table,
    modalities::TensorView output,
    std::vector<std::int32_t>* token_ids,
    std::uint64_t* prompt_len);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_PROMPT_EMBED_H

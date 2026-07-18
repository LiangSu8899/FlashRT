#ifndef FLASHRT_MODALITIES_TOKENIZER_H
#define FLASHRT_MODALITIES_TOKENIZER_H

#include "flashrt/cpp/modalities/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace flashrt {
namespace modalities {

struct SentencePieceEncodeOptions {
    bool add_bos = false;
    bool add_eos = false;
    bool pad_to_max_tokens = false;
    std::uint64_t max_tokens = 0;
    std::int32_t pad_id = 0;
};

class SentencePieceTokenizer final {
public:
    SentencePieceTokenizer();
    ~SentencePieceTokenizer();

    SentencePieceTokenizer(SentencePieceTokenizer&&) noexcept;
    SentencePieceTokenizer& operator=(SentencePieceTokenizer&&) noexcept;

    SentencePieceTokenizer(const SentencePieceTokenizer&) = delete;
    SentencePieceTokenizer& operator=(const SentencePieceTokenizer&) = delete;

    Status load_model(const std::string& model_path);
    Status encode(const std::string& text,
                  const SentencePieceEncodeOptions& options,
                  std::vector<std::int32_t>* token_ids);
    void reserve(std::uint64_t max_tokens);
    std::uint64_t workspace_capacity() const;

    std::int32_t bos_id() const;
    std::int32_t eos_id() const;
    std::int32_t unk_id() const;
    std::int32_t pad_id() const;
    std::uint64_t vocab_size() const;
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace modalities
}  // namespace flashrt

#endif  // FLASHRT_MODALITIES_TOKENIZER_H

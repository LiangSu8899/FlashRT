#include "flashrt/cpp/modalities/tokenizer.h"

namespace flashrt {
namespace modalities {

struct SentencePieceTokenizer::Impl {};

SentencePieceTokenizer::SentencePieceTokenizer()
    : impl_(new Impl()) {}

SentencePieceTokenizer::~SentencePieceTokenizer() = default;

SentencePieceTokenizer::SentencePieceTokenizer(
    SentencePieceTokenizer&&) noexcept = default;

SentencePieceTokenizer& SentencePieceTokenizer::operator=(
    SentencePieceTokenizer&&) noexcept = default;

Status SentencePieceTokenizer::load_model(const std::string& model_path) {
    (void)model_path;
    return Status::error(
        StatusCode::kUnsupported,
        "native SentencePiece support is not enabled in this build");
}

Status SentencePieceTokenizer::encode(
        const std::string& text,
        const SentencePieceEncodeOptions& options,
        std::vector<std::int32_t>* token_ids) {
    (void)text;
    (void)options;
    if (token_ids) token_ids->clear();
    return Status::error(
        StatusCode::kUnsupported,
        "native SentencePiece support is not enabled in this build");
}

void SentencePieceTokenizer::reserve(std::uint64_t max_tokens) {
    (void)max_tokens;
}

std::uint64_t SentencePieceTokenizer::workspace_capacity() const { return 0; }

std::int32_t SentencePieceTokenizer::bos_id() const { return -1; }
std::int32_t SentencePieceTokenizer::eos_id() const { return -1; }
std::int32_t SentencePieceTokenizer::unk_id() const { return -1; }
std::int32_t SentencePieceTokenizer::pad_id() const { return -1; }
std::uint64_t SentencePieceTokenizer::vocab_size() const { return 0; }
bool SentencePieceTokenizer::loaded() const { return false; }

}  // namespace modalities
}  // namespace flashrt

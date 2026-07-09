#include "flashrt/cpp/modalities/tokenizer.h"

#include <sentencepiece_processor.h>

#include <limits>
#include <utility>

namespace flashrt {
namespace modalities {

struct SentencePieceTokenizer::Impl {
    sentencepiece::SentencePieceProcessor processor;
    bool loaded = false;
};

SentencePieceTokenizer::SentencePieceTokenizer()
    : impl_(new Impl()) {}

SentencePieceTokenizer::~SentencePieceTokenizer() = default;

SentencePieceTokenizer::SentencePieceTokenizer(
    SentencePieceTokenizer&&) noexcept = default;

SentencePieceTokenizer& SentencePieceTokenizer::operator=(
    SentencePieceTokenizer&&) noexcept = default;

Status SentencePieceTokenizer::load_model(const std::string& model_path) {
    auto status = impl_->processor.Load(model_path);
    if (!status.ok()) {
        impl_->loaded = false;
        return Status::error(StatusCode::kNotFound, status.ToString());
    }
    impl_->loaded = true;
    return Status::ok();
}

Status SentencePieceTokenizer::encode(
        const std::string& text,
        const SentencePieceEncodeOptions& options,
        std::vector<std::int32_t>* token_ids) const {
    if (!token_ids) {
        return Status::error(StatusCode::kInvalidArgument,
                             "token_ids output is null");
    }
    token_ids->clear();
    if (!impl_->loaded) {
        return Status::error(StatusCode::kInvalidArgument,
                             "SentencePiece model is not loaded");
    }

    std::vector<int> encoded;
    auto status = impl_->processor.Encode(text, &encoded);
    if (!status.ok()) {
        return Status::error(StatusCode::kBackend, status.ToString());
    }
    const std::uint64_t extra =
        (options.add_bos ? 1u : 0u) + (options.add_eos ? 1u : 0u);
    if (encoded.size() + extra >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        return Status::error(StatusCode::kInsufficientStorage,
                             "encoded token sequence is too large");
    }

    if (options.add_bos) {
        const int bos = impl_->processor.bos_id();
        if (bos < 0) {
            return Status::error(StatusCode::kInvalidArgument,
                                 "tokenizer has no BOS id");
        }
        token_ids->push_back(static_cast<std::int32_t>(bos));
    }
    token_ids->reserve(encoded.size() + extra);
    for (int id : encoded) {
        token_ids->push_back(static_cast<std::int32_t>(id));
    }
    if (options.add_eos) {
        const int eos = impl_->processor.eos_id();
        if (eos < 0) {
            return Status::error(StatusCode::kInvalidArgument,
                                 "tokenizer has no EOS id");
        }
        token_ids->push_back(static_cast<std::int32_t>(eos));
    }

    if (options.max_tokens) {
        if (token_ids->size() > options.max_tokens) {
            return Status::error(StatusCode::kShapeMismatch,
                                 "encoded token sequence exceeds max_tokens");
        }
        if (options.pad_to_max_tokens) {
            token_ids->resize(options.max_tokens, options.pad_id);
        }
    } else if (options.pad_to_max_tokens) {
        return Status::error(StatusCode::kInvalidArgument,
                             "pad_to_max_tokens requires max_tokens");
    }
    return Status::ok();
}

std::int32_t SentencePieceTokenizer::bos_id() const {
    return impl_->loaded ? impl_->processor.bos_id() : -1;
}

std::int32_t SentencePieceTokenizer::eos_id() const {
    return impl_->loaded ? impl_->processor.eos_id() : -1;
}

std::int32_t SentencePieceTokenizer::unk_id() const {
    return impl_->loaded ? impl_->processor.unk_id() : -1;
}

std::int32_t SentencePieceTokenizer::pad_id() const {
    return impl_->loaded ? impl_->processor.pad_id() : -1;
}

std::uint64_t SentencePieceTokenizer::vocab_size() const {
    return impl_->loaded
               ? static_cast<std::uint64_t>(impl_->processor.GetPieceSize())
               : 0;
}

bool SentencePieceTokenizer::loaded() const {
    return impl_->loaded;
}

}  // namespace modalities
}  // namespace flashrt

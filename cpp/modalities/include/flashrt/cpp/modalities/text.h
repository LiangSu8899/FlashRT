#ifndef FLASHRT_MODALITIES_TEXT_H
#define FLASHRT_MODALITIES_TEXT_H

#include "flashrt/cpp/modalities/types.h"

#include <cstdint>

namespace flashrt {
namespace modalities {

struct EmbeddingGatherSpec {
    std::uint64_t vocab_size = 0;
    std::uint64_t hidden_dim = 0;
    float scale = 1.0f;
};

Status gather_token_embeddings_cpu(const EmbeddingGatherSpec& spec,
                                   const std::int32_t* token_ids,
                                   std::uint64_t n_tokens,
                                   TensorView embedding_table,
                                   TensorView output);

}  // namespace modalities
}  // namespace flashrt

#endif  // FLASHRT_MODALITIES_TEXT_H

#include "flashrt/cpp/models/pi05/prompt_embed.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using flashrt::modalities::DType;
using flashrt::modalities::Layout;
using flashrt::modalities::MemoryPlace;
using flashrt::modalities::SentencePieceTokenizer;
using flashrt::modalities::Shape;
using flashrt::modalities::StatusCode;
using flashrt::modalities::TensorView;
using flashrt::models::pi05::PromptEmbeddingSpec;
using flashrt::models::pi05::embed_prompt_cpu;

namespace {

std::string tokenizer_model_path() {
    const char* env = std::getenv("FLASH_RT_PALIGEMMA_TOKENIZER");
    return env ? std::string(env) : std::string();
}

void test_requires_loaded_tokenizer() {
    SentencePieceTokenizer tokenizer;
    std::vector<float> table(8, 0.0f);
    std::vector<float> out(8, 0.0f);
    std::vector<std::int32_t> ids;
    std::uint64_t prompt_len = 0;

    TensorView src{table.data(), static_cast<std::uint64_t>(table.size() * 4),
                   DType::kFloat32, MemoryPlace::kHost, Layout::kFlat,
                   Shape{2, 4}};
    TensorView dst{out.data(), static_cast<std::uint64_t>(out.size() * 4),
                   DType::kFloat32, MemoryPlace::kHost, Layout::kFlat,
                   Shape{2, 4}};
    PromptEmbeddingSpec spec{2, 4, 2, 1.0f};
    auto st = embed_prompt_cpu(tokenizer, spec, "pick", nullptr, 0, src, dst,
                               &ids, &prompt_len);
    assert(!st.ok_status());
    assert(st.code == StatusCode::kInvalidArgument);
}

void test_paligemma_prompt_embedding_when_configured() {
#ifdef FLASHRT_CPP_HAS_SENTENCEPIECE
    const std::string path = tokenizer_model_path();
    if (path.empty()) {
        std::cout << "SKIP - FLASH_RT_PALIGEMMA_TOKENIZER not set\n";
        return;
    }
    SentencePieceTokenizer tokenizer;
    auto st = tokenizer.load_model(path);
    assert(st.ok_status());

    constexpr std::uint64_t vocab = 257152;
    constexpr std::uint64_t hidden = 2;
    constexpr std::uint64_t max_tokens = 32;
    std::vector<float> table(vocab * hidden);
    for (std::uint64_t i = 0; i < vocab; ++i) {
        table[i * hidden + 0] = static_cast<float>(i);
        table[i * hidden + 1] = -static_cast<float>(i);
    }
    std::vector<float> out(max_tokens * hidden, 7.0f);
    TensorView src{table.data(), static_cast<std::uint64_t>(table.size() * 4),
                   DType::kFloat32, MemoryPlace::kHost, Layout::kFlat,
                   Shape{vocab, hidden}};
    TensorView dst{out.data(), static_cast<std::uint64_t>(out.size() * 4),
                   DType::kFloat32, MemoryPlace::kHost, Layout::kFlat,
                   Shape{max_tokens, hidden}};

    const float state[] = {0.0f, 1.0f, -1.0f};
    PromptEmbeddingSpec spec{vocab, hidden, max_tokens, 0.5f};
    std::vector<std::int32_t> ids;
    std::uint64_t prompt_len = 0;
    st = embed_prompt_cpu(tokenizer, spec, "pick_up_cube", state, 3, src, dst,
                          &ids, &prompt_len);
    assert(st.ok_status());
    const std::vector<std::int32_t> expected_ids = {
        2, 7071, 235292, 4788, 908, 28660, 235269, 3040, 235292,
        235248, 235274, 235284, 235321, 235248, 235284, 235308,
        235308, 235248, 235276, 235289, 108, 4022, 235292, 235248,
    };
    assert(ids == expected_ids);
    assert(prompt_len == expected_ids.size());
    for (std::uint64_t i = 0; i < prompt_len; ++i) {
        const float id = static_cast<float>(expected_ids[i]);
        assert(std::fabs(out[i * hidden + 0] - id * 0.5f) < 0.001f);
        assert(std::fabs(out[i * hidden + 1] + id * 0.5f) < 0.001f);
    }
    for (std::uint64_t i = prompt_len * hidden; i < out.size(); ++i) {
        assert(out[i] == 0.0f);
    }
#endif
}

}  // namespace

int main() {
    test_requires_loaded_tokenizer();
    test_paligemma_prompt_embedding_when_configured();
    std::cout << "PASS - Pi05 prompt embedding\n";
    return 0;
}

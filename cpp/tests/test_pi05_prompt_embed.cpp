#include "flashrt/cpp/models/pi05/model/prompt_embed.h"

#ifdef FLASHRT_CPP_WITH_CUDA_STAGING
#include <cuda_runtime_api.h>
#endif

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
using flashrt::modalities::TextEmbeddingStaging;
using flashrt::models::pi05::PromptEmbeddingSpec;
using flashrt::models::pi05::embed_prompt;
using flashrt::models::pi05::embed_prompt_cpu;

namespace {

#ifdef FLASHRT_CPP_HAS_SENTENCEPIECE
std::string tokenizer_model_path() {
    const char* env = std::getenv("FLASH_RT_PALIGEMMA_TOKENIZER");
    return env ? std::string(env) : std::string();
}
#endif

#if defined(FLASHRT_CPP_HAS_SENTENCEPIECE) && \
    defined(FLASHRT_CPP_WITH_CUDA_STAGING) && \
    defined(FLASHRT_CPP_TEST_WITH_CUDA_KERNELS)
bool has_cuda_device() {
#ifdef FLASHRT_CPP_WITH_CUDA_STAGING
    int n = 0;
    cudaError_t rc = cudaGetDeviceCount(&n);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return n > 0;
#else
    return false;
#endif
}
#endif

void test_rejects_invalid_contract() {
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

    spec.max_tokens = UINT64_MAX;
    dst.shape = Shape{UINT64_MAX, 4};
    st = embed_prompt_cpu(tokenizer, spec, "pick", nullptr, 0, src, dst,
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
        table[i * hidden] = static_cast<float>(i);
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
    ids.reserve(max_tokens + 1);
    tokenizer.reserve(max_tokens);
    std::string formatted;
    formatted.reserve(512);
    std::uint64_t prompt_len = 0;
    st = embed_prompt(tokenizer, spec, "pick_up_cube", state, 3, src, dst,
                      &ids, &prompt_len, nullptr, nullptr, &formatted);
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
        assert(std::fabs(out[i * hidden] - id * 0.5f) < 0.001f);
        assert(std::fabs(out[i * hidden + 1] + id * 0.5f) < 0.001f);
    }
    for (std::uint64_t i = prompt_len * hidden; i < out.size(); ++i) {
        assert(out[i] == 0.0f);
    }

    const std::vector<std::int32_t> clean_ids = {
        2, 18075, 908, 28660, 108};
    st = embed_prompt(tokenizer, spec, "  pick_up\ncube  ", nullptr, 0, src,
                      dst, &ids, &prompt_len, nullptr, nullptr, &formatted);
    assert(st.ok_status());
    assert(ids == clean_ids);
    assert(prompt_len == clean_ids.size());

    const std::size_t id_capacity = ids.capacity();
    const std::size_t formatted_capacity = formatted.capacity();
    const std::uint64_t tokenizer_capacity = tokenizer.workspace_capacity();
    for (int round = 0; round < 1000; ++round) {
        st = embed_prompt(tokenizer, spec, "pick_up_cube", state, 3, src, dst,
                          &ids, &prompt_len, nullptr, nullptr, &formatted);
        assert(st.ok_status());
        assert(ids.capacity() == id_capacity);
        assert(formatted.capacity() == formatted_capacity);
        assert(tokenizer.workspace_capacity() == tokenizer_capacity);
    }
#endif
}

void test_paligemma_prompt_embedding_device_when_configured() {
#if defined(FLASHRT_CPP_HAS_SENTENCEPIECE) && \
    defined(FLASHRT_CPP_WITH_CUDA_STAGING) && \
    defined(FLASHRT_CPP_TEST_WITH_CUDA_KERNELS)
    const std::string path = tokenizer_model_path();
    if (path.empty() || !has_cuda_device()) {
        std::cout << "SKIP - tokenizer or CUDA device not available\n";
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
        table[i * hidden] = static_cast<float>(i);
        table[i * hidden + 1] = -static_cast<float>(i);
    }
    void* d_table = nullptr;
    void* d_out = nullptr;
    assert(cudaMalloc(&d_table, table.size() * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_out, max_tokens * hidden * sizeof(float)) ==
           cudaSuccess);
    assert(cudaMemcpy(d_table, table.data(), table.size() * sizeof(float),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    TensorView src{d_table, static_cast<std::uint64_t>(table.size() * 4),
                   DType::kFloat32, MemoryPlace::kDevice, Layout::kFlat,
                   Shape{vocab, hidden}};
    TensorView dst{d_out,
                   static_cast<std::uint64_t>(max_tokens * hidden * 4),
                   DType::kFloat32, MemoryPlace::kDevice, Layout::kFlat,
                   Shape{max_tokens, hidden}};
    TextEmbeddingStaging staging;
    st = flashrt::modalities::text_embedding_staging_create(&staging,
                                                            max_tokens);
    assert(st.ok_status());
    std::vector<std::int32_t> ids;
    std::uint64_t prompt_len = 0;
    PromptEmbeddingSpec spec{vocab, hidden, max_tokens, 0.5f};
    st = embed_prompt(tokenizer, spec, "pick up cube", nullptr, 0, src, dst,
                      &ids, &prompt_len, nullptr, &staging);
    assert(st.ok_status());
    std::vector<float> out(max_tokens * hidden, 1.0f);
    assert(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(prompt_len == ids.size());
    assert(ids[0] == 2);
    assert(std::fabs(out[0] - 1.0f) < 0.001f);
    assert(std::fabs(out[1] + 1.0f) < 0.001f);
    assert(out[prompt_len * hidden] == 0.0f);
    flashrt::modalities::text_embedding_staging_destroy(&staging);
    cudaFree(d_out);
    cudaFree(d_table);
#endif
}

}  // namespace

int main() {
    test_rejects_invalid_contract();
    test_paligemma_prompt_embedding_when_configured();
    test_paligemma_prompt_embedding_device_when_configured();
    std::cout << "PASS - Pi05 prompt embedding\n";
    return 0;
}

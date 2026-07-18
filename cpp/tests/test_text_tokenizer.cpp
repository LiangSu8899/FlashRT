#include "flashrt/cpp/modalities/tokenizer.h"

#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using flashrt::modalities::SentencePieceEncodeOptions;
using flashrt::modalities::SentencePieceTokenizer;
using flashrt::modalities::StatusCode;

namespace {

#ifdef FLASHRT_CPP_HAS_SENTENCEPIECE
std::string tokenizer_model_path() {
    const char* env = std::getenv("FLASH_RT_PALIGEMMA_TOKENIZER");
    return env ? std::string(env) : std::string();
}
#endif

void test_unavailable_build_reports_unsupported() {
    SentencePieceTokenizer tokenizer;
#ifndef FLASHRT_CPP_HAS_SENTENCEPIECE
    auto st = tokenizer.load_model("missing.model");
    assert(!st.ok_status());
    assert(st.code == StatusCode::kUnsupported);
#else
    (void)tokenizer;
#endif
}

void test_paligemma_token_exact_when_configured() {
#ifdef FLASHRT_CPP_HAS_SENTENCEPIECE
    const std::string path = tokenizer_model_path();
    if (path.empty()) {
        std::cout << "SKIP - FLASH_RT_PALIGEMMA_TOKENIZER not set\n";
        return;
    }
    SentencePieceTokenizer tokenizer;
    auto st = tokenizer.load_model(path);
    assert(st.ok_status());
    assert(tokenizer.loaded());
    assert(tokenizer.vocab_size() == 257152);
    assert(tokenizer.bos_id() == 2);
    assert(tokenizer.eos_id() == 1);
    assert(tokenizer.unk_id() == 3);
    assert(tokenizer.pad_id() == 0);

    std::vector<std::int32_t> ids;
    SentencePieceEncodeOptions options;
    options.add_bos = true;
    st = tokenizer.encode(
        "Task: pick up cube, State: 0 128 255;\nAction: ",
        options, &ids);
    assert(st.ok_status());
    const std::vector<std::int32_t> expected = {
        2, 7071, 235292, 4788, 908, 28660, 235269, 3040, 235292,
        235248, 235276, 235248, 235274, 235284, 235321, 235248,
        235284, 235308, 235308, 235289, 108, 4022, 235292, 235248,
    };
    assert(ids == expected);

    options.max_tokens = expected.size() + 2;
    options.pad_to_max_tokens = true;
    st = tokenizer.encode(
        "Task: pick up cube, State: 0 128 255;\nAction: ",
        options, &ids);
    assert(st.ok_status());
    assert(ids.size() == expected.size() + 2);
    assert(ids[ids.size() - 1] == 0);
#endif
}

}  // namespace

int main() {
    test_unavailable_build_reports_unsupported();
    test_paligemma_token_exact_when_configured();
    std::cout << "PASS - text tokenizer contracts\n";
    return 0;
}

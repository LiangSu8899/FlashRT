#include "flashrt/cpp/modalities/tokenizer.h"
#include "flashrt/cpp/models/pi05/prompt_format.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kCorpusMagic = 0x50303554u;
constexpr std::uint32_t kOutputMagic = 0x50303549u;

template <typename T>
bool read_value(std::ifstream& input, T* value) {
    return static_cast<bool>(input.read(
        reinterpret_cast<char*>(value), sizeof(T)));
}

template <typename T>
bool write_value(std::ofstream& output, const T& value) {
    return static_cast<bool>(output.write(
        reinterpret_cast<const char*>(&value), sizeof(T)));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: pi05_tokenizer_corpus_probe TOKENIZER CORPUS OUT\n";
        return 2;
    }
    flashrt::modalities::SentencePieceTokenizer tokenizer;
    auto status = tokenizer.load_model(argv[1]);
    if (!status.ok_status()) {
        std::cerr << "tokenizer load failed: " << status.message << '\n';
        return 1;
    }
    tokenizer.reserve(200);
    std::ifstream input(argv[2], std::ios::binary);
    std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
    std::uint32_t magic = 0;
    std::uint32_t records = 0;
    if (!input || !output || !read_value(input, &magic) ||
        !read_value(input, &records) || magic != kCorpusMagic ||
        !write_value(output, kOutputMagic) || !write_value(output, records)) {
        std::cerr << "invalid tokenizer corpus header\n";
        return 1;
    }
    flashrt::modalities::SentencePieceEncodeOptions options;
    options.add_bos = true;
    options.max_tokens = 200;
    std::string task;
    std::string formatted;
    std::vector<float> state;
    std::vector<std::int32_t> ids;
    task.reserve(512);
    formatted.reserve(1024);
    state.reserve(32);
    ids.reserve(200);
    for (std::uint32_t record = 0; record < records; ++record) {
        std::uint32_t task_bytes = 0;
        std::uint32_t state_count = 0;
        if (!read_value(input, &task_bytes) ||
            !read_value(input, &state_count) || task_bytes > 4096 ||
            state_count > 1024) {
            std::cerr << "invalid tokenizer corpus record\n";
            return 1;
        }
        task.resize(task_bytes);
        state.resize(state_count);
        if ((task_bytes && !input.read(task.data(), task_bytes)) ||
            (state_count && !input.read(
                reinterpret_cast<char*>(state.data()),
                static_cast<std::streamsize>(state_count * sizeof(float))))) {
            std::cerr << "truncated tokenizer corpus record\n";
            return 1;
        }
        flashrt::models::pi05::format_state_prompt_into(
            task, state.data(), state.size(), &formatted);
        status = tokenizer.encode(formatted, options, &ids);
        if (!status.ok_status()) {
            std::cerr << "tokenization failed at record " << record << ": "
                      << status.message << '\n';
            return 1;
        }
        const std::uint32_t count = static_cast<std::uint32_t>(ids.size());
        if (!write_value(output, count) ||
            (count && !output.write(
                reinterpret_cast<const char*>(ids.data()),
                static_cast<std::streamsize>(count * sizeof(std::int32_t))))) {
            std::cerr << "tokenizer output write failed\n";
            return 1;
        }
    }
    char trailing = 0;
    if (input.read(&trailing, 1)) {
        std::cerr << "tokenizer corpus has trailing bytes\n";
        return 1;
    }
    std::cout << "PASS " << records << " tokenized prompt/state records\n";
    return 0;
}

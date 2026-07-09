#include "flashrt/cpp/models/pi05/native_workspace.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::uint64_t fnv1a(const std::vector<std::uint16_t>& values) {
    std::uint64_t hash = 14695981039346656037ull;
    const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
    for (std::size_t i = 0; i < values.size() * sizeof(std::uint16_t); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

std::vector<std::uint16_t> download(
    const flashrt::models::pi05::NativeWorkspaceBuffer& buffer) {
    std::vector<std::uint16_t> values(
        frt_buffer_bytes(buffer.buffer) / sizeof(std::uint16_t));
    if (cudaMemcpy(values.data(), frt_buffer_dptr(buffer.buffer),
                   values.size() * sizeof(std::uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return {};
    }
    return values;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: pi05_native_rope_probe VIEWS MAX_PROMPT CHUNK "
                     "POOL PROMPT\n";
        return 2;
    }
    flashrt::models::pi05::NativeWorkspaceConfig config;
    config.num_views = std::stoi(argv[1]);
    config.max_prompt_tokens = std::stoi(argv[2]);
    config.chunk_size = std::stoi(argv[3]);
    config.vision_pool_factor = std::stoi(argv[4]);
    const int prompt = std::stoi(argv[5]);
    frt_ctx ctx = frt_ctx_create();
    if (!ctx) return 1;
    flashrt::models::pi05::NativeWorkspace workspace(ctx);
    if (!workspace.allocate(config).ok_status() ||
        !workspace.update_decoder_rope(prompt).ok_status()) {
        frt_ctx_destroy(ctx);
        return 1;
    }
    const std::vector<std::uint16_t> encoder =
        download(*workspace.find("encoder_rope_weights"));
    const std::vector<std::uint16_t> decoder =
        download(*workspace.find("decoder_rope_weights"));
    std::cout << "encoder_shape=" << workspace.encoder_sequence() << ",256"
              << " encoder_fnv=" << std::hex << std::setw(16)
              << std::setfill('0') << fnv1a(encoder)
              << " decoder_shape=" << std::dec << config.chunk_size << ",256"
              << " decoder_fnv=" << std::hex << std::setw(16)
              << fnv1a(decoder) << '\n';
    frt_ctx_destroy(ctx);
    return 0;
}

#include "checkpoint_identity.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace flashrt::providers::llama_cpp {
namespace {

thread_local std::string g_runtime_open_error;

constexpr std::array<uint32_t, 64> kSha256Round = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

uint32_t rotate_right(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32u - bits));
}

struct Sha256 {
    std::array<uint32_t, 8> state = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    std::array<uint8_t, 64> block{};
    uint64_t bytes = 0;
    size_t used = 0;

    void transform(const uint8_t* input) {
        uint32_t words[64];
        for (size_t i = 0; i < 16; ++i) {
            words[i] = (static_cast<uint32_t>(input[i * 4]) << 24) |
                       (static_cast<uint32_t>(input[i * 4 + 1]) << 16) |
                       (static_cast<uint32_t>(input[i * 4 + 2]) << 8) |
                       static_cast<uint32_t>(input[i * 4 + 3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t s0 = rotate_right(words[i - 15], 7) ^
                                rotate_right(words[i - 15], 18) ^
                                (words[i - 15] >> 3);
            const uint32_t s1 = rotate_right(words[i - 2], 17) ^
                                rotate_right(words[i - 2], 19) ^
                                (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
        for (size_t i = 0; i < 64; ++i) {
            const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                                  rotate_right(e, 25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t temp1 = h + sum1 + choice + kSha256Round[i] + words[i];
            const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                                  rotate_right(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = sum0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    void update(const void* data, size_t size) {
        const auto* input = static_cast<const uint8_t*>(data);
        bytes += size;
        while (size > 0) {
            const size_t take = std::min(size, block.size() - used);
            std::copy(input, input + take, block.begin() + used);
            input += take;
            size -= take;
            used += take;
            if (used == block.size()) {
                transform(block.data());
                used = 0;
            }
        }
    }

    std::array<uint8_t, 32> finish() {
        const uint64_t bit_count = bytes * 8u;
        const uint8_t one = 0x80;
        update(&one, 1);
        const uint8_t zero = 0;
        while (used != 56) update(&zero, 1);
        uint8_t length[8];
        for (size_t i = 0; i < 8; ++i) {
            length[7 - i] = static_cast<uint8_t>(bit_count >> (i * 8));
        }
        update(length, sizeof(length));
        std::array<uint8_t, 32> digest{};
        for (size_t i = 0; i < state.size(); ++i) {
            digest[i * 4] = static_cast<uint8_t>(state[i] >> 24);
            digest[i * 4 + 1] = static_cast<uint8_t>(state[i] >> 16);
            digest[i * 4 + 2] = static_cast<uint8_t>(state[i] >> 8);
            digest[i * 4 + 3] = static_cast<uint8_t>(state[i]);
        }
        return digest;
    }
};

}  // namespace

bool checkpoint_identity(const char* path, std::string* identity,
                         std::string* error) {
    if (!path || !*path || !identity) {
        if (error) *error = "invalid checkpoint identity arguments";
        return false;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (error) *error = std::string("failed to open checkpoint for identity: ") + path;
        return false;
    }
    Sha256 hash;
    std::vector<char> chunk(1024 * 1024);
    while (file) {
        file.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const std::streamsize read = file.gcount();
        if (read > 0) hash.update(chunk.data(), static_cast<size_t>(read));
    }
    if (!file.eof()) {
        if (error) *error = std::string("failed to read checkpoint for identity: ") + path;
        return false;
    }
    const auto digest = hash.finish();
    char hex[65];
    for (size_t i = 0; i < digest.size(); ++i) {
        std::snprintf(hex + i * 2, 3, "%02x", digest[i]);
    }
    hex[64] = '\0';
    *identity = hex;
    if (error) error->clear();
    return true;
}

void clear_runtime_open_error() {
    g_runtime_open_error.clear();
}

void set_runtime_open_error(const std::string& error) {
    g_runtime_open_error = error;
}

const char* runtime_open_error() {
    return g_runtime_open_error.c_str();
}

}  // namespace flashrt::providers::llama_cpp

extern "C" const char* frt_llama_cpp_runtime_open_error(void) {
    return flashrt::providers::llama_cpp::runtime_open_error();
}

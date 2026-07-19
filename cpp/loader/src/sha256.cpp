#include "flashrt/cpp/loader/sha256.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#ifdef FLASHRT_CPP_HAS_OPENSSL
#include <openssl/evp.h>
#endif

namespace flashrt {
namespace loader {
namespace {

constexpr std::uint32_t kRound[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

std::uint32_t rotate(std::uint32_t value, unsigned bits) {
    return (value >> bits) | (value << (32 - bits));
}

class Sha256 {
public:
    void update(const unsigned char* data, std::size_t bytes) {
        total_bytes_ += bytes;
        while (bytes) {
            const std::size_t count =
                std::min(bytes, block_.size() - block_bytes_);
            std::copy(data, data + count, block_.begin() + block_bytes_);
            block_bytes_ += count;
            data += count;
            bytes -= count;
            if (block_bytes_ == block_.size()) {
                transform(block_.data());
                block_bytes_ = 0;
            }
        }
    }

    std::array<unsigned char, 32> finish() {
        const std::uint64_t bits = total_bytes_ * 8;
        block_[block_bytes_++] = 0x80;
        if (block_bytes_ > 56) {
            std::fill(block_.begin() + block_bytes_, block_.end(), 0);
            transform(block_.data());
            block_bytes_ = 0;
        }
        std::fill(block_.begin() + block_bytes_, block_.begin() + 56, 0);
        for (int i = 0; i < 8; ++i) {
            block_[63 - i] = static_cast<unsigned char>(bits >> (8 * i));
        }
        transform(block_.data());
        std::array<unsigned char, 32> output{};
        for (std::size_t i = 0; i < state_.size(); ++i) {
            output[4 * i] = static_cast<unsigned char>(state_[i] >> 24);
            output[4 * i + 1] = static_cast<unsigned char>(state_[i] >> 16);
            output[4 * i + 2] = static_cast<unsigned char>(state_[i] >> 8);
            output[4 * i + 3] = static_cast<unsigned char>(state_[i]);
        }
        return output;
    }

private:
    void transform(const unsigned char* data) {
        std::uint32_t words[64]{};
        for (int i = 0; i < 16; ++i) {
            words[i] = (static_cast<std::uint32_t>(data[4 * i]) << 24) |
                       (static_cast<std::uint32_t>(data[4 * i + 1]) << 16) |
                       (static_cast<std::uint32_t>(data[4 * i + 2]) << 8) |
                       static_cast<std::uint32_t>(data[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotate(words[i - 15], 7) ^
                                     rotate(words[i - 15], 18) ^
                                     (words[i - 15] >> 3);
            const std::uint32_t s1 = rotate(words[i - 2], 17) ^
                                     rotate(words[i - 2], 19) ^
                                     (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t sum1 = rotate(e, 6) ^ rotate(e, 11) ^
                                       rotate(e, 25);
            const std::uint32_t choose = (e & f) ^ (~e & g);
            const std::uint32_t t1 = h + sum1 + choose + kRound[i] + words[i];
            const std::uint32_t sum0 = rotate(a, 2) ^ rotate(a, 13) ^
                                       rotate(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_ = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    std::array<unsigned char, 64> block_{};
    std::size_t block_bytes_ = 0;
    std::uint64_t total_bytes_ = 0;
};

struct FileIdentity {
    std::uint64_t device = 0;
    std::uint64_t inode = 0;
    std::uint64_t bytes = 0;
    std::int64_t mtime_seconds = 0;
    std::int64_t mtime_nanoseconds = 0;
    std::int64_t ctime_seconds = 0;
    std::int64_t ctime_nanoseconds = 0;
};

bool file_identity(const std::string& path, FileIdentity* out) {
    if (!out) return false;
    struct stat value {};
    if (::stat(path.c_str(), &value) != 0 || !S_ISREG(value.st_mode) ||
        value.st_size < 0) {
        return false;
    }
    out->device = static_cast<std::uint64_t>(value.st_dev);
    out->inode = static_cast<std::uint64_t>(value.st_ino);
    out->bytes = static_cast<std::uint64_t>(value.st_size);
    out->mtime_seconds = value.st_mtim.tv_sec;
    out->mtime_nanoseconds = value.st_mtim.tv_nsec;
    out->ctime_seconds = value.st_ctim.tv_sec;
    out->ctime_nanoseconds = value.st_ctim.tv_nsec;
    return true;
}

bool same_identity(const FileIdentity& left, const FileIdentity& right) {
    return left.device == right.device && left.inode == right.inode &&
           left.bytes == right.bytes &&
           left.mtime_seconds == right.mtime_seconds &&
           left.mtime_nanoseconds == right.mtime_nanoseconds &&
           left.ctime_seconds == right.ctime_seconds &&
           left.ctime_nanoseconds == right.ctime_nanoseconds;
}

bool valid_hex_digest(const std::string& value) {
    if (value.size() != 64) return false;
    return std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

std::string cache_path(const std::string& path) {
    return path + ".flashrt.sha256";
}

bool read_cache(const std::string& path, const FileIdentity& identity,
                std::string* digest) {
    std::ifstream input(cache_path(path));
    std::string magic;
    FileIdentity cached;
    std::string value;
    if (!(input >> magic >> cached.device >> cached.inode >> cached.bytes >>
          cached.mtime_seconds >> cached.mtime_nanoseconds >>
          cached.ctime_seconds >> cached.ctime_nanoseconds >> value) ||
        magic != "flashrt-sha256-v1" || !same_identity(identity, cached) ||
        !valid_hex_digest(value)) {
        return false;
    }
    if (digest) *digest = std::move(value);
    return true;
}

void write_cache(const std::string& path, const FileIdentity& identity,
                 const std::string& digest) {
    static std::atomic<std::uint64_t> sequence{0};
    const std::string destination = cache_path(path);
    const std::string temporary =
        destination + ".tmp." + std::to_string(::getpid()) + "." +
        std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
    {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output) return;
        output << "flashrt-sha256-v1 " << identity.device << ' '
               << identity.inode << ' ' << identity.bytes << ' '
               << identity.mtime_seconds << ' '
               << identity.mtime_nanoseconds << ' '
               << identity.ctime_seconds << ' '
               << identity.ctime_nanoseconds << ' ' << digest << '\n';
        if (!output) {
            output.close();
            std::remove(temporary.c_str());
            return;
        }
    }
    if (std::rename(temporary.c_str(), destination.c_str()) != 0) {
        std::remove(temporary.c_str());
    }
}

}  // namespace

bool sha256_file(const std::string& path, std::string* hex,
                 std::string* error) {
    if (!hex) {
        if (error) *error = "SHA-256 output is null";
        return false;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (error) *error = "unable to open file for SHA-256: " + path;
        return false;
    }
#ifdef FLASHRT_CPP_HAS_OPENSSL
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context || EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1) {
        if (context) EVP_MD_CTX_free(context);
        if (error) *error = "unable to initialize SHA-256";
        return false;
    }
#else
    Sha256 hash;
#endif
    std::vector<unsigned char> buffer(4 << 20);
    while (file) {
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        const std::streamsize count = file.gcount();
        if (count > 0) {
#ifdef FLASHRT_CPP_HAS_OPENSSL
            if (EVP_DigestUpdate(context, buffer.data(),
                                 static_cast<std::size_t>(count)) != 1) {
                EVP_MD_CTX_free(context);
                if (error) *error = "failed while hashing file: " + path;
                return false;
            }
#else
            hash.update(buffer.data(), static_cast<std::size_t>(count));
#endif
        }
    }
    if (!file.eof()) {
#ifdef FLASHRT_CPP_HAS_OPENSSL
        EVP_MD_CTX_free(context);
#endif
        if (error) *error = "failed while reading file for SHA-256: " + path;
        return false;
    }
#ifdef FLASHRT_CPP_HAS_OPENSSL
    std::array<unsigned char, 32> digest{};
    unsigned int digest_bytes = 0;
    if (EVP_DigestFinal_ex(context, digest.data(), &digest_bytes) != 1 ||
        digest_bytes != digest.size()) {
        EVP_MD_CTX_free(context);
        if (error) *error = "unable to finalize SHA-256";
        return false;
    }
    EVP_MD_CTX_free(context);
#else
    const auto digest = hash.finish();
#endif
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned char byte : digest) output << std::setw(2) << unsigned(byte);
    *hex = output.str();
    if (error) error->clear();
    return true;
}

bool sha256_file_cached(const std::string& path, std::string* hex,
                        bool* cache_hit, std::string* error) {
    if (cache_hit) *cache_hit = false;
    if (!hex) {
        if (error) *error = "SHA-256 output is null";
        return false;
    }
    FileIdentity before;
    if (!file_identity(path, &before)) {
        if (error) *error = "unable to stat file for SHA-256: " + path;
        return false;
    }
    if (read_cache(path, before, hex)) {
        if (cache_hit) *cache_hit = true;
        if (error) error->clear();
        return true;
    }
    std::string digest;
    if (!sha256_file(path, &digest, error)) return false;
    FileIdentity after;
    if (!file_identity(path, &after) || !same_identity(before, after)) {
        if (error) *error = "file changed while hashing: " + path;
        return false;
    }
    write_cache(path, after, digest);
    *hex = std::move(digest);
    if (error) error->clear();
    return true;
}

}  // namespace loader
}  // namespace flashrt

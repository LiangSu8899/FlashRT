#include "flashrt/cpp/loader/sha256.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

int main() {
    char path[] = "/tmp/flashrt_sha256_XXXXXX";
    const int fd = ::mkstemp(path);
    assert(fd >= 0);
    ::close(fd);
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file << "abc";
        assert(file.good());
    }
    std::string digest;
    std::string error;
    assert(flashrt::loader::sha256_file(path, &digest, &error));
    assert(digest ==
           "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert(error.empty());
    ::unlink(path);
    assert(!flashrt::loader::sha256_file(path, &digest, &error));
    assert(!error.empty());
    std::printf("PASS - SHA-256 file hashing\n");
    return 0;
}

#include "flashrt/cpp/loader/sha256.h"

#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/stat.h>
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

    bool cache_hit = true;
    assert(flashrt::loader::sha256_file_cached(
        path, &digest, &cache_hit, &error));
    assert(!cache_hit);
    assert(flashrt::loader::sha256_file_cached(
        path, &digest, &cache_hit, &error));
    assert(cache_hit);

    struct stat original {};
    assert(::stat(path, &original) == 0);
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file << "abd";
        assert(file.good());
    }
    const timespec restore_times[2] = {original.st_atim, original.st_mtim};
    assert(::utimensat(AT_FDCWD, path, restore_times, 0) == 0);
    assert(flashrt::loader::sha256_file_cached(
        path, &digest, &cache_hit, &error));
    assert(!cache_hit);
    assert(digest ==
           "a52d159f262b2c6ddb724a61840befc36eb30c88877a4030b65cbe86298449c9");

    std::remove((std::string(path) + ".flashrt.sha256").c_str());
    ::unlink(path);
    assert(!flashrt::loader::sha256_file(path, &digest, &error));
    assert(!error.empty());
    std::printf("PASS - SHA-256 file hashing\n");
    return 0;
}

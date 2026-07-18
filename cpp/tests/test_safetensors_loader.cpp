#include "flashrt/cpp/loader/safetensors.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <utility>

namespace {

std::string temp_path() {
    char path[] = "/tmp/frt_safetensors_XXXXXX";
    const int fd = ::mkstemp(path);
    assert(fd >= 0);
    ::close(fd);
    return path;
}

void write_file(const std::string& path, const std::string& header,
                const std::string& payload) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) {
        const char byte = static_cast<char>((n >> (8 * i)) & 0xffu);
        f.write(&byte, 1);
    }
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    assert(f.good());
}

}  // namespace

int main() {
    using flashrt::loader::SafetensorsFile;

    const std::string path = temp_path();
    std::string payload(12, '\0');
    const float expected[] = {1.25f, -2.5f};
    std::memcpy(&payload[4], expected, sizeof(expected));
    write_file(
        path,
        "{\"__metadata__\":{\"format\":\"pt\"},"
        "\"u8\":{\"dtype\":\"U8\",\"shape\":[4],"
        "\"data_offsets\":[0,4]},"
        "\"values\":{\"shape\":[2],\"data_offsets\":[4,12],"
        "\"dtype\":\"F32\"}}",
        payload);

    SafetensorsFile file;
    assert(file.open(path));
    assert(file.is_open());
    assert(file.tensors().size() == 2);
    assert(file.metadata().size() == 1);
    assert(file.metadata().at("format") == "pt");
    const auto* values = file.find("values");
    assert(values);
    assert(values->dtype == "F32");
    assert(values->shape.size() == 1 && values->shape[0] == 2);
    assert(values->bytes == sizeof(expected));
    assert(std::memcmp(file.data(*values), expected, sizeof(expected)) == 0);

    SafetensorsFile moved(std::move(file));
    assert(!file.is_open());
    assert(moved.find("u8"));
    assert(moved.metadata().at("format") == "pt");
    assert(::unlink(path.c_str()) == 0);
    assert(std::memcmp(moved.data(*moved.find("values")), expected,
                       sizeof(expected)) == 0);
    moved.close();
    assert(!moved.is_open());

    const std::string invalid = temp_path();
    write_file(invalid,
               "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],"
               "\"data_offsets\":[0,4]}}",
               std::string(4, '\0'));
    assert(!file.open(invalid));
    assert(file.error().find("does not match") != std::string::npos);

    write_file(invalid,
               "{\"x\":{\"dtype\":\"F4\",\"shape\":[2],"
               "\"data_offsets\":[0,1]}}",
               std::string(1, '\0'));
    assert(!file.open(invalid));
    assert(file.error().find("unsupported") != std::string::npos);

    write_file(invalid,
               "{\"x\":{\"dtype\":\"U8\",\"shape\":[2],"
               "\"data_offsets\":[0,2]},"
               "\"y\":{\"dtype\":\"U8\",\"shape\":[2],"
               "\"data_offsets\":[1,3]}}",
               std::string(3, '\0'));
    assert(!file.open(invalid));
    assert(file.error().find("overlapping") != std::string::npos);

    write_file(invalid,
               "{\"x\":{\"dtype\":\"U8\",\"shape\":[1],"
               "\"data_offsets\":[0,1]},"
               "\"x\":{\"dtype\":\"U8\",\"shape\":[1],"
               "\"data_offsets\":[1,2]}}",
               std::string(2, '\0'));
    assert(!file.open(invalid));
    assert(file.error().find("duplicate") != std::string::npos);

    write_file(invalid,
               "{\"__metadata__\":{\"format\":7},"
               "\"x\":{\"dtype\":\"U8\",\"shape\":[1],"
               "\"data_offsets\":[0,1]}}",
               std::string(1, '\0'));
    assert(!file.open(invalid));
    assert(file.error().find("must be strings") != std::string::npos);

    write_file(invalid,
               "{\"__metadata__\":{\"format\":\"pt\"},"
               "\"__metadata__\":{\"source\":\"test\"},"
               "\"x\":{\"dtype\":\"U8\",\"shape\":[1],"
               "\"data_offsets\":[0,1]}}",
               std::string(1, '\0'));
    assert(!file.open(invalid));
    assert(file.error().find("duplicate") != std::string::npos);

    write_file(invalid,
               "{\"x\":{\"dtype\":\"U8\",\"shape\":[4],"
               "\"data_offsets\":[0,4]}}",
               std::string(2, '\0'));
    assert(!file.open(invalid));
    assert(file.error().find("exceeds") != std::string::npos);

    write_file(invalid,
               "{\"x\":{\"dtype\":\"F64\","
               "\"shape\":[18446744073709551615,2],"
               "\"data_offsets\":[0,0]}}",
               {});
    assert(!file.open(invalid));
    assert(file.error().find("overflowing shape") != std::string::npos);

    write_file(invalid, "{\"x\":", {});
    assert(!file.open(invalid));
    assert(file.error().find("header byte") != std::string::npos);

    assert(::unlink(invalid.c_str()) == 0);
    std::printf("PASS - safetensors mmap loader\n");
    return 0;
}

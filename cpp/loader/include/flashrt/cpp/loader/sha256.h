#ifndef FLASHRT_CPP_LOADER_SHA256_H
#define FLASHRT_CPP_LOADER_SHA256_H

#include <string>

namespace flashrt {
namespace loader {

bool sha256_file(const std::string& path, std::string* hex,
                 std::string* error = nullptr);

}  // namespace loader
}  // namespace flashrt

#endif  // FLASHRT_CPP_LOADER_SHA256_H

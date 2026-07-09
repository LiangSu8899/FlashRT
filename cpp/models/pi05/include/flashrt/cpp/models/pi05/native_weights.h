#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHTS_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHTS_H

#include <cstdint>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

struct NativeTensorRequirement {
    std::string key;
    std::vector<std::uint64_t> shape;
};

const std::vector<NativeTensorRequirement>& native_tensor_requirements();

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_WEIGHTS_H

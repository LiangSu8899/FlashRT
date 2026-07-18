#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_BF16_LINEAR_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_BF16_LINEAR_H

#include "flashrt/cpp/models/pi05/model/resolved_resources.h"

#include <memory>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {

class Sm120Bf16Linear final {
public:
    Sm120Bf16Linear() noexcept;
    ~Sm120Bf16Linear();

    Sm120Bf16Linear(const Sm120Bf16Linear&) = delete;
    Sm120Bf16Linear& operator=(const Sm120Bf16Linear&) = delete;

    modalities::Status status() const;
    modalities::Status run(const Pi05ResolvedWeight& weight,
                           const void* input,
                           void* output,
                           int rows,
                           int input_width,
                           int output_width,
                           Pi05Stream stream) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string error_;
};

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_BF16_LINEAR_H

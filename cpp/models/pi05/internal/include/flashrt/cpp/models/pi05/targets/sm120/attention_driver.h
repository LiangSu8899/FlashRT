#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_ATTENTION_DRIVER_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_ATTENTION_DRIVER_H

#include "flashrt/cpp/models/pi05/targets/sm120/attention.h"

#include <cstdint>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {

class Sm120AttentionDriver final {
public:
    explicit Sm120AttentionDriver(
        const Sm120AttentionBacking* backing) noexcept;

    modalities::Status status() const;
    modalities::Status vision(std::uintptr_t stream) const;
    modalities::Status encoder(int layer, std::uintptr_t stream) const;
    modalities::Status decoder(int layer, std::uintptr_t stream) const;

    void* vision_output() const;
    void* encoder_output() const;
    void* decoder_output() const;
    int multiprocessor_count() const { return multiprocessor_count_; }

private:
    const Sm120AttentionBacking* backing_ = nullptr;
    int multiprocessor_count_ = 0;
    std::string error_;
};

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_ATTENTION_DRIVER_H

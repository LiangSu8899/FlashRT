#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_ATTENTION_DRIVER_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_ATTENTION_DRIVER_H

#include "flashrt/cpp/models/pi05/native_rtx_attention.h"

#include <cstdint>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {

class NativeRtxAttentionDriver {
public:
    explicit NativeRtxAttentionDriver(
        const NativeRtxAttentionWorkspace* workspace) noexcept;

    modalities::Status status() const;
    modalities::Status vision(std::uintptr_t stream) const;
    modalities::Status encoder(int layer, std::uintptr_t stream) const;
    modalities::Status decoder(int layer, std::uintptr_t stream) const;

    void* vision_output() const;
    void* encoder_output() const;
    void* decoder_output() const;
    int num_sms() const { return num_sms_; }

private:
    const NativeAttentionBuffer* find(const char* name) const;

    const NativeRtxAttentionWorkspace* workspace_ = nullptr;
    int num_views_ = 0;
    int encoder_sequence_ = 0;
    int chunk_size_ = 0;
    int total_kv_ = 0;
    int num_sms_ = 0;
    std::string error_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_ATTENTION_DRIVER_H

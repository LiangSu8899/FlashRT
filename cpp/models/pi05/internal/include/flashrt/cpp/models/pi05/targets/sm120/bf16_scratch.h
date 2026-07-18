#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_BF16_SCRATCH_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_BF16_SCRATCH_H

#include "flashrt/cpp/models/pi05/model/dims.h"
#include "flashrt/cpp/models/pi05/targets/sm120/device_buffer.h"

#include <cstddef>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {

struct Sm120VisionBf16Scratch final {
    Sm120DeviceBuffer normalized;
    Sm120DeviceBuffer qkv;
    Sm120DeviceBuffer hidden;
};

struct Sm120EncoderBf16Scratch final {
    Sm120DeviceBuffer normalized;
    Sm120DeviceBuffer qkv;
    Sm120DeviceBuffer gate;
    Sm120DeviceBuffer hidden;
};

struct Sm120DecoderBf16Scratch final {
    Sm120DeviceBuffer normalized;
    Sm120DeviceBuffer gate;
    Sm120DeviceBuffer qkv;
    Sm120DeviceBuffer gate_projection;
    Sm120DeviceBuffer hidden;
};

class Sm120Bf16ScratchBacking final {
public:
    explicit Sm120Bf16ScratchBacking(frt_ctx context) : context_(context) {}

    Sm120Bf16ScratchBacking(const Sm120Bf16ScratchBacking&) = delete;
    Sm120Bf16ScratchBacking& operator=(const Sm120Bf16ScratchBacking&) = delete;

    modalities::Status allocate(const Pi05ResolvedShape& shape,
                                bool fused_gate_up = false);

    const Sm120VisionBf16Scratch& vision() const { return vision_; }
    const Sm120EncoderBf16Scratch& encoder() const { return encoder_; }
    const Sm120DecoderBf16Scratch& decoder() const { return decoder_; }
    const Pi05ResolvedShape& shape() const { return shape_; }

    std::size_t allocation_count() const { return allocation_count_; }
    std::size_t allocated_bytes() const { return allocated_bytes_; }
    bool allocated() const { return allocated_; }
    bool fused_gate_up() const { return fused_gate_up_; }

private:
    modalities::Status add(const char* name,
                           const modalities::Shape& shape,
                           Sm120DeviceBuffer* out);

    frt_ctx context_ = nullptr;
    Pi05ResolvedShape shape_;
    Sm120VisionBf16Scratch vision_;
    Sm120EncoderBf16Scratch encoder_;
    Sm120DecoderBf16Scratch decoder_;
    std::size_t allocation_count_ = 0;
    std::size_t allocated_bytes_ = 0;
    bool fused_gate_up_ = false;
    bool allocation_started_ = false;
    bool allocated_ = false;
};

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_BF16_SCRATCH_H

#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_PHYSICAL_RESOURCES_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_PHYSICAL_RESOURCES_H

#include "flashrt/cpp/models/pi05/model/dims.h"
#include "flashrt/cpp/models/pi05/support/native_calibration.h"
#include "flashrt/cpp/models/pi05/support/native_resource_resolver.h"

#include <cstddef>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm110 {

struct Sm110Buffer final {
    frt_buffer buffer = nullptr;
    modalities::DType dtype = modalities::DType::kUInt8;
    modalities::Shape shape;
    bool alias = false;

    void* device_data() const {
        return buffer ? frt_buffer_dptr(buffer) : nullptr;
    }
    std::size_t bytes() const {
        return buffer ? frt_buffer_bytes(buffer) : 0;
    }
};

struct Sm110VisionResources final {
    Sm110Buffer state_fp8;
    Sm110Buffer qkv;
    Sm110Buffer attention;
    Sm110Buffer post_norm;
    Sm110Buffer hidden;
    Sm110Buffer hidden_fp8;
    Sm110Buffer unit_scale;
};

struct Sm110EncoderResources final {
    Sm110Buffer state_fp8;
    Sm110Buffer qkv;
    Sm110Buffer logits;
    Sm110Buffer attention;
    Sm110Buffer output_fp8;
    Sm110Buffer gate_up;
    Sm110Buffer hidden_fp8;
    Sm110Buffer residual_output;
    Sm110Buffer activation_scales;
    Sm110Buffer key_cache;
    Sm110Buffer value_cache;
};

struct Sm110DecoderResources final {
    Sm110Buffer normalized;
    Sm110Buffer gate;
    Sm110Buffer qkv;
    Sm110Buffer logits;
    Sm110Buffer attention;
    Sm110Buffer projection;
    Sm110Buffer action_delta;
    Sm110Buffer state_fp8;
    Sm110Buffer hidden_fp8;
    Sm110Buffer context_fp8;
    Sm110Buffer activation_scales;
};

struct Sm110ControlResources final {
    Sm110Buffer encoder_valid_tokens;
    Sm110Buffer decoder_valid_tokens;
    Sm110Buffer decoder_position;
};

struct Sm110ObserverResources final {
    Sm110Buffer encoder_normalized;
    Sm110Buffer encoder_residual;
    Sm110Buffer encoder_hidden;
    Sm110Buffer decoder_hidden;
};

class Sm110PhysicalResources final {
public:
    explicit Sm110PhysicalResources(frt_ctx context) : context_(context) {}

    Sm110PhysicalResources(const Sm110PhysicalResources&) = delete;
    Sm110PhysicalResources& operator=(const Sm110PhysicalResources&) = delete;

    modalities::Status allocate(const Pi05ResolvedShape& shape,
                                bool observing);
    modalities::Status initialize_static(
        const NativeCalibrationArtifact& calibration);
    modalities::Status initialize_observer();
    modalities::Status reset_observer(Pi05Stream stream) const;
    modalities::Status download_observer(
        std::vector<float>* encoder,
        std::vector<float>* decoder) const;
    modalities::Status set_prompt_length(int prompt_tokens);
    modalities::Status make_target_bindings(
        Pi05TargetBufferBindings* out) const;

    const Sm110VisionResources& vision() const { return vision_; }
    const Sm110EncoderResources& encoder() const { return encoder_; }
    const Sm110DecoderResources& decoder() const { return decoder_; }
    const Sm110ControlResources& controls() const { return controls_; }
    const Sm110ObserverResources& observer() const { return observer_; }
    const Pi05ResolvedShape& shape() const { return shape_; }

    std::size_t allocation_count() const { return allocation_count_; }
    std::size_t allocated_bytes() const { return allocated_bytes_; }
    int padded_key_stride() const { return padded_key_stride_; }
    bool allocated() const { return allocated_; }
    bool initialized() const { return initialized_; }

private:
    modalities::Status add(const char* name,
                           const modalities::Shape& shape,
                           modalities::DType dtype,
                           Sm110Buffer* out);

    frt_ctx context_ = nullptr;
    Pi05ResolvedShape shape_;
    Sm110VisionResources vision_;
    Sm110EncoderResources encoder_;
    Sm110DecoderResources decoder_;
    Sm110ControlResources controls_;
    Sm110ObserverResources observer_;
    std::size_t allocation_count_ = 0;
    std::size_t allocated_bytes_ = 0;
    int padded_key_stride_ = 0;
    bool allocation_started_ = false;
    bool allocated_ = false;
    bool initialized_ = false;
    bool observing_ = false;
    std::vector<float> encoder_reset_;
    std::vector<float> decoder_reset_;
};

}  // namespace sm110
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_PHYSICAL_RESOURCES_H

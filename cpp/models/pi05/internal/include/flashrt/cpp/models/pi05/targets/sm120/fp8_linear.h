#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_FP8_LINEAR_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_FP8_LINEAR_H

#include "flashrt/cpp/models/pi05/model/linear_weight_groups.h"
#include "flashrt/cpp/models/pi05/support/native_calibration.h"
#include "flashrt/cpp/models/pi05/targets/sm120/device_buffer.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {

class Sm120Fp8ActivationBacking final {
public:
    explicit Sm120Fp8ActivationBacking(frt_ctx context) : context_(context) {}

    Sm120Fp8ActivationBacking(const Sm120Fp8ActivationBacking&) = delete;
    Sm120Fp8ActivationBacking& operator=(
        const Sm120Fp8ActivationBacking&) = delete;

    modalities::Status initialize_static(
        const Pi05ResolvedShape& shape,
        const NativeCalibrationArtifact& artifact);
    modalities::Status download_scales(
        std::vector<float>* vision,
        std::vector<float>* encoder,
        std::vector<float>* decoder) const;

    void* scratch_data() const { return scratch_.device_data(); }
    std::size_t scratch_bytes() const { return scratch_.bytes(); }
    float* scale_data(const Pi05LinearActivationSite& site) const;
    const Pi05LinearScaleLayout& scale_layout() const { return layout_; }
    const Pi05ResolvedShape& shape() const { return shape_; }
    std::size_t allocation_count() const { return allocation_count_; }
    std::size_t allocated_bytes() const { return allocated_bytes_; }
    bool initialized() const { return initialized_; }

private:
    modalities::Status initialize(
        const Pi05ResolvedShape& shape,
        const NativeCalibrationArtifact& artifact);
    modalities::Status add(
        const char* name,
        std::size_t elements,
        modalities::DType dtype,
        Sm120DeviceBuffer* out);
    modalities::Status upload(
        const Sm120DeviceBuffer& destination,
        const std::vector<float>& values,
        Pi05Stream stream) const;
    const Sm120DeviceBuffer* scale_buffer(Pi05LinearDomain domain) const;

    frt_ctx context_ = nullptr;
    Pi05ResolvedShape shape_;
    Pi05LinearScaleLayout layout_;
    Sm120DeviceBuffer scratch_;
    Sm120DeviceBuffer vision_scales_;
    Sm120DeviceBuffer encoder_scales_;
    Sm120DeviceBuffer decoder_scales_;
    std::size_t allocation_count_ = 0;
    std::size_t allocated_bytes_ = 0;
    bool initialization_started_ = false;
    bool initialized_ = false;
};

class Sm120Fp8Linear final {
public:
    explicit Sm120Fp8Linear(
        Sm120Fp8ActivationBacking* activation) noexcept;
    ~Sm120Fp8Linear();

    Sm120Fp8Linear(const Sm120Fp8Linear&) = delete;
    Sm120Fp8Linear& operator=(const Sm120Fp8Linear&) = delete;

    modalities::Status status() const;
    modalities::Status autotune(
        const Pi05ResolvedWeight& weight,
        const Pi05LinearActivationSite& site,
        const void* input,
        void* output,
        int rows,
        int input_width,
        int output_width);
    modalities::Status run(
        const Pi05ResolvedWeight& weight,
        const Pi05LinearActivationSite& site,
        const void* input,
        void* output,
        int rows,
        int input_width,
        int output_width,
        Pi05Stream stream);
    modalities::Status run_prequantized(
        const Pi05ResolvedWeight& weight,
        const Pi05LinearActivationSite& site,
        const void* input,
        void* output,
        int rows,
        int input_width,
        int output_width,
        Pi05Stream stream);
    modalities::Status freeze_autotune();

    float* scale_data(const Pi05LinearActivationSite& site) const;
    void* scratch_data() const;
    std::size_t scratch_bytes() const;
    std::size_t autotuned_shape_count() const;
    bool autotune_frozen() const;

private:
    struct Impl;

    modalities::Status launch(
        const Pi05ResolvedWeight& weight,
        const Pi05LinearActivationSite& site,
        const void* input,
        void* output,
        int rows,
        int input_width,
        int output_width,
        Pi05Stream stream,
        bool prequantized);

    Sm120Fp8ActivationBacking* activation_ = nullptr;
    std::unique_ptr<Impl> impl_;
    modalities::StatusCode error_code_ = modalities::StatusCode::kBackend;
    std::string error_;
};

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_FP8_LINEAR_H

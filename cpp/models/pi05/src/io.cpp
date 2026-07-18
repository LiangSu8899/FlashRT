#include "flashrt/cpp/models/pi05/io.h"

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status validate_pi05_frame_contract(
    const modalities::VisionFrame& frame, bool strict_rgb8) {
    if (strict_rgb8 && frame.format != modalities::PixelFormat::kRGB8) {
        return modalities::Status::error(
            modalities::StatusCode::kShapeMismatch,
            "Pi05 image input must be RGB8");
    }
    if (frame.image.dtype != modalities::DType::kUInt8 ||
        frame.image.layout != modalities::Layout::kHWC) {
        return modalities::Status::error(
            modalities::StatusCode::kShapeMismatch,
            "Pi05 image input must be u8 HWC");
    }
    std::uint64_t channels = 3;
    if (frame.format == modalities::PixelFormat::kRGBA8 ||
        frame.format == modalities::PixelFormat::kBGRA8) {
        channels = 4;
    } else if (frame.format == modalities::PixelFormat::kGRAY8) {
        channels = 1;
    }
    if (frame.width <= 0 || frame.height <= 0 ||
        frame.image.shape.rank != 3 ||
        frame.image.shape.dims[0] != static_cast<std::uint64_t>(frame.height) ||
        frame.image.shape.dims[1] != static_cast<std::uint64_t>(frame.width) ||
        frame.image.shape.dims[2] != channels) {
        return modalities::Status::error(
            modalities::StatusCode::kShapeMismatch,
            "Pi05 image shape must match HWC dimensions");
    }
    if (frame.image.place != modalities::MemoryPlace::kHost &&
        frame.image.place != modalities::MemoryPlace::kHostPinned) {
        return modalities::Status::error(
            modalities::StatusCode::kUnsupported,
            "Pi05 image input must be host memory");
    }
    return modalities::Status::ok();
}

}  // namespace

RuntimeIo::RuntimeIo(int num_views,
                     modalities::TensorView image_input,
                     modalities::TensorView action_output,
                     std::vector<float> action_mean,
                     std::vector<float> action_stddev,
                     void* stream,
                     int chunk,
                     int model_action_dim,
                     int robot_action_dim,
                     modalities::DType image_dtype,
                     modalities::VisionStaging* staging,
                     modalities::ActionStaging* action_staging,
                     bool strict_rgb8)
    : image_input_(image_input),
      action_output_(action_output),
      stream_(stream),
      staging_(staging),
      action_staging_(action_staging),
      strict_rgb8_(strict_rgb8),
      vision_spec_(vision_preprocess_spec(num_views)),
      action_spec_(action_postprocess_spec(action_mean, action_stddev, chunk,
                                           model_action_dim, robot_action_dim)) {
    vision_spec_.output_dtype = image_dtype;
}

modalities::Status RuntimeIo::prepare_vision(
    const std::vector<modalities::VisionFrame>& frames) const {
    for (const auto& frame : frames) {
        auto st = validate_pi05_frame_contract(frame, strict_rgb8_);
        if (!st.ok_status()) return st;
    }
    return modalities::preprocess_vision(vision_spec_, frames, image_input_,
                                         stream_, staging_);
}

modalities::Status RuntimeIo::read_actions(
    std::vector<float>* robot_actions) const {
    return modalities::postprocess_action(action_spec_, action_output_,
                                          robot_actions, stream_,
                                          action_staging_);
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

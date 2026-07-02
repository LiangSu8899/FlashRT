#include "flashrt/cpp/models/pi05/spec.h"

#include <algorithm>

namespace flashrt {
namespace models {
namespace pi05 {

modalities::VisionPreprocessSpec vision_preprocess_spec(int num_views) {
    modalities::VisionPreprocessSpec spec;
    static const char* kViews[] = {"image", "wrist_image", "wrist_image_right"};
    num_views = std::max(1, std::min(3, num_views));
    spec.view_order.reserve(static_cast<std::size_t>(num_views));
    for (int i = 0; i < num_views; ++i) spec.view_order.emplace_back(kViews[i]);
    spec.target_width = kImageSize;
    spec.target_height = kImageSize;
    spec.output_dtype = modalities::DType::kBFloat16;
    spec.output_layout = modalities::Layout::kNHWC;
    spec.normalize.mode = modalities::NormalizeMode::kScaleShift;
    spec.normalize.scale = 1.0f / 127.5f;
    spec.normalize.shift = -1.0f;
    spec.require_exact_views = true;
    return spec;
}

modalities::ActionPostprocessSpec action_postprocess_spec(
    const std::vector<float>& mean,
    const std::vector<float>& stddev,
    int chunk,
    int model_dim,
    int robot_dim) {
    modalities::ActionPostprocessSpec spec;
    spec.chunk = chunk;
    spec.model_dim = model_dim;
    spec.robot_dim = robot_dim;
    spec.schema = "eef_delta_xyz_rpy_gripper";
    spec.mean = mean;
    spec.stddev = stddev;
    spec.clip_model_input = true;
    spec.model_input_min = -1.0f;
    spec.model_input_max = 1.0f;
    return spec;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

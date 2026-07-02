#include "flashrt/models/pi05/cpp_runtime_io.h"

namespace flashrt {
namespace models {
namespace pi05 {

RuntimeIo::RuntimeIo(int num_views,
                     modalities::TensorView image_input,
                     modalities::TensorView action_output,
                     std::vector<float> action_mean,
                     std::vector<float> action_stddev,
                     int chunk,
                     int model_action_dim,
                     int robot_action_dim)
    : image_input_(image_input),
      action_output_(action_output),
      vision_spec_(vision_preprocess_spec(num_views)),
      action_spec_(action_postprocess_spec(action_mean, action_stddev, chunk,
                                           model_action_dim, robot_action_dim)) {}

modalities::Status RuntimeIo::prepare_vision(
    const std::vector<modalities::VisionFrame>& frames) const {
    return modalities::preprocess_vision_cpu(vision_spec_, frames, image_input_);
}

modalities::Status RuntimeIo::read_actions(
    std::vector<float>* robot_actions) const {
    return modalities::postprocess_action_cpu(action_spec_, action_output_,
                                              robot_actions);
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

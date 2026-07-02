#include "flashrt/cpp/models/pi05/io.h"

namespace flashrt {
namespace models {
namespace pi05 {

RuntimeIo::RuntimeIo(int num_views,
                     modalities::TensorView image_input,
                     modalities::TensorView action_output,
                     std::vector<float> action_mean,
                     std::vector<float> action_stddev,
                     void* stream,
                     int chunk,
                     int model_action_dim,
                     int robot_action_dim)
    : image_input_(image_input),
      action_output_(action_output),
      stream_(stream),
      vision_spec_(vision_preprocess_spec(num_views)),
      action_spec_(action_postprocess_spec(action_mean, action_stddev, chunk,
                                           model_action_dim, robot_action_dim)) {}

modalities::Status RuntimeIo::prepare_vision(
    const std::vector<modalities::VisionFrame>& frames) const {
    return modalities::preprocess_vision(vision_spec_, frames, image_input_,
                                         stream_);
}

modalities::Status RuntimeIo::read_actions(
    std::vector<float>* robot_actions) const {
    return modalities::postprocess_action(action_spec_, action_output_,
                                          robot_actions, stream_);
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

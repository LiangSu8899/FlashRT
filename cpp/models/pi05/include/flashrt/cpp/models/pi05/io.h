#ifndef FLASHRT_MODELS_PI05_CPP_RUNTIME_IO_H
#define FLASHRT_MODELS_PI05_CPP_RUNTIME_IO_H

#include "flashrt/cpp/models/pi05/spec.h"

namespace flashrt {
namespace models {
namespace pi05 {

/* Thin Pi0.5 IO adapter. It owns no model weights and does not replay graphs;
 * it binds Pi0.5 modality semantics to concrete tensor buffers supplied by a
 * model runtime. The surrounding runtime remains responsible for capture,
 * frt_runtime_export_v1, and graph replay. */
class RuntimeIo {
public:
    RuntimeIo(int num_views,
              modalities::TensorView image_input,
              modalities::TensorView action_output,
              std::vector<float> action_mean,
              std::vector<float> action_stddev,
              void* stream = nullptr,
              int chunk = kDefaultChunk,
              int model_action_dim = kModelActionDim,
              int robot_action_dim = kLiberoActionDim);

    modalities::Status prepare_vision(
        const std::vector<modalities::VisionFrame>& frames) const;

    modalities::Status read_actions(std::vector<float>* robot_actions) const;

    const modalities::VisionPreprocessSpec& vision_spec() const {
        return vision_spec_;
    }
    const modalities::ActionPostprocessSpec& action_spec() const {
        return action_spec_;
    }

private:
    modalities::TensorView image_input_;
    modalities::TensorView action_output_;
    void* stream_ = nullptr;
    modalities::VisionPreprocessSpec vision_spec_;
    modalities::ActionPostprocessSpec action_spec_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_MODELS_PI05_CPP_RUNTIME_IO_H

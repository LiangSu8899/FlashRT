#ifndef FLASHRT_MODALITIES_ACTION_H
#define FLASHRT_MODALITIES_ACTION_H

#include "flashrt/cpp/modalities/types.h"

#include <string>
#include <vector>

namespace flashrt {
namespace modalities {

struct ActionPostprocessSpec {
    int chunk = 1;
    int model_dim = 0;
    int robot_dim = 0;
    std::string schema;
    std::vector<float> mean;
    std::vector<float> stddev;
    bool clip_model_input = false;
    float model_input_min = -1.0f;
    float model_input_max = 1.0f;
    std::vector<float> min_value;
    std::vector<float> max_value;
    bool clamp = false;
};

Status postprocess_action_cpu(const ActionPostprocessSpec& spec,
                              TensorView model_output,
                              std::vector<float>* robot_actions);

/* Dispatch entry used by model runtimes. Host outputs use the CPU reference.
 * Device outputs use the conservative D2H staging path when CUDA staging is
 * enabled. */
Status postprocess_action(const ActionPostprocessSpec& spec,
                          TensorView model_output,
                          std::vector<float>* robot_actions,
                          void* stream = nullptr);

std::uint64_t required_action_output_bytes(const ActionPostprocessSpec& spec,
                                           DType dtype);

}  // namespace modalities
}  // namespace flashrt

#endif  // FLASHRT_MODALITIES_ACTION_H

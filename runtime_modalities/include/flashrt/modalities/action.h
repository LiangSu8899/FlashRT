#ifndef FLASHRT_MODALITIES_ACTION_H
#define FLASHRT_MODALITIES_ACTION_H

#include "flashrt/modalities/types.h"

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
    std::vector<float> min_value;
    std::vector<float> max_value;
    bool clamp = false;
};

Status postprocess_action_cpu(const ActionPostprocessSpec& spec,
                              TensorView model_output,
                              std::vector<float>* robot_actions);

}  // namespace modalities
}  // namespace flashrt

#endif  // FLASHRT_MODALITIES_ACTION_H

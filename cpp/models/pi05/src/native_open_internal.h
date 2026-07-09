#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_OPEN_INTERNAL_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_OPEN_INTERNAL_H

#include "flashrt/model_runtime.h"

#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

struct NativeOpenConfig {
    std::string checkpoint_path;
    std::string tokenizer_model_path;
    int max_prompt_tokens = 200;
    int state_dim = 0;
    int num_views = 2;
    int chunk = 10;
    int num_steps = 10;
    int vision_pool_factor = 1;
    std::vector<float> state_q01;
    std::vector<float> state_q99;
    std::vector<float> action_q01;
    std::vector<float> action_q99;
};

int build_native_model_runtime(const NativeOpenConfig& config,
                               frt_model_runtime_v1** out,
                               std::string* error);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_OPEN_INTERNAL_H

#ifndef FLASHRT_RUNTIME_CPP_H
#define FLASHRT_RUNTIME_CPP_H

#include "flashrt/runtime.h"
#include "flashrt/modalities/action.h"
#include "flashrt/modalities/vision.h"

#include <string>
#include <vector>

namespace flashrt {
namespace runtime {

/* C++ model runtimes implement this interface. It is intentionally NOT the
 * stable ABI; the stable hand-off to Nexus and other consumers remains
 * frt_runtime_export_v1. This layer owns model semantics: modality preprocess,
 * prompt/state binding, graph inputs, and action postprocess. */
class ModelRuntime {
public:
    virtual ~ModelRuntime() = default;

    virtual const frt_runtime_export_v1* export_runtime() const = 0;
    virtual int set_prompt(const char* text) = 0;

    virtual modalities::Status prepare_vision(
        const std::vector<modalities::VisionFrame>& frames) = 0;

    virtual int replay_tick() = 0;

    virtual modalities::Status read_actions(
        std::vector<float>* robot_actions) = 0;
};

}  // namespace runtime
}  // namespace flashrt

#endif  // FLASHRT_RUNTIME_CPP_H

#ifndef FLASHRT_CPP_MODELS_PI05_MODEL_EXECUTION_PLAN_H
#define FLASHRT_CPP_MODELS_PI05_MODEL_EXECUTION_PLAN_H

#include "flashrt/cpp/modalities/types.h"
#include "flashrt/exec.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace flashrt {
namespace models {
namespace pi05 {

enum class Pi05GraphId : std::uint8_t {
    kInfer = 0,
    kDecodeOnly,
    kContext,
    kCount,
};

enum class Pi05RecordBody : std::uint8_t {
    kFull = 0,
    kDecode,
    kContext,
};

// Graph bindings are stable runtime resource names, not semantic Pipeline
// values. A binding can expose a control buffer that is not a model tensor.
enum class Pi05GraphBindingId : std::uint8_t {
    kImages = 0,
    kPromptEmbedding,
    kEncoderState,
    kNoise,
    kPreviousActions,
    kPrefixWeights,
    kGuidanceWeight,
    kCount,
};

class Pi05ResolvedGraphBindings final {
public:
    modalities::Status bind(Pi05GraphBindingId id, frt_buffer buffer);
    frt_buffer get(Pi05GraphBindingId id) const;

private:
    std::array<frt_buffer,
               static_cast<std::size_t>(Pi05GraphBindingId::kCount)>
        buffers_{};
};

struct Pi05GraphDescriptor final {
    Pi05GraphId id = Pi05GraphId::kCount;
    const char* name = nullptr;
    Pi05RecordBody body = Pi05RecordBody::kFull;
    const Pi05GraphBindingId* bindings = nullptr;
    std::size_t binding_count = 0;
};

struct Pi05StageDescriptor final {
    const char* name = nullptr;
    Pi05GraphId graph = Pi05GraphId::kCount;
    const std::uint32_t* after = nullptr;
    std::size_t after_count = 0;
};

struct Pi05ExecutionPlanDescriptor final {
    const char* name = nullptr;
    const Pi05StageDescriptor* stages = nullptr;
    std::size_t stage_count = 0;
};

const char* pi05_graph_binding_name(Pi05GraphBindingId id);
const Pi05GraphDescriptor* pi05_graph_descriptor(Pi05GraphId id);
const Pi05GraphDescriptor* pi05_graph_catalog(std::size_t* count);
const Pi05ExecutionPlanDescriptor* pi05_execution_plan(const char* name);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_MODEL_EXECUTION_PLAN_H

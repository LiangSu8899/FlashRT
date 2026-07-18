#include "flashrt/cpp/models/pi05/model/execution_plan.h"

#include <cstring>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

constexpr Pi05GraphBindingId kInferBindings[] = {
    Pi05GraphBindingId::kImages,
    Pi05GraphBindingId::kPromptEmbedding,
    Pi05GraphBindingId::kEncoderState,
    Pi05GraphBindingId::kNoise,
    Pi05GraphBindingId::kPreviousActions,
    Pi05GraphBindingId::kPrefixWeights,
    Pi05GraphBindingId::kGuidanceWeight,
};

constexpr Pi05GraphBindingId kDecodeBindings[] = {
    Pi05GraphBindingId::kEncoderState,
    Pi05GraphBindingId::kNoise,
    Pi05GraphBindingId::kPreviousActions,
    Pi05GraphBindingId::kPrefixWeights,
    Pi05GraphBindingId::kGuidanceWeight,
};

constexpr Pi05GraphBindingId kContextBindings[] = {
    Pi05GraphBindingId::kImages,
    Pi05GraphBindingId::kPromptEmbedding,
    Pi05GraphBindingId::kEncoderState,
};

constexpr Pi05GraphDescriptor kGraphCatalog[] = {
    {Pi05GraphId::kInfer, "infer", Pi05RecordBody::kFull,
     kInferBindings, sizeof(kInferBindings) / sizeof(kInferBindings[0])},
    {Pi05GraphId::kDecodeOnly, "decode_only", Pi05RecordBody::kDecode,
     kDecodeBindings, sizeof(kDecodeBindings) / sizeof(kDecodeBindings[0])},
    {Pi05GraphId::kContext, "context", Pi05RecordBody::kContext,
     kContextBindings, sizeof(kContextBindings) / sizeof(kContextBindings[0])},
};

constexpr std::uint32_t kActionAfter[] = {0};
constexpr Pi05StageDescriptor kFullStages[] = {
    {"infer", Pi05GraphId::kInfer, nullptr, 0},
};
constexpr Pi05StageDescriptor kContextActionStages[] = {
    {"context", Pi05GraphId::kContext, nullptr, 0},
    {"action", Pi05GraphId::kDecodeOnly, kActionAfter,
     sizeof(kActionAfter) / sizeof(kActionAfter[0])},
};
constexpr Pi05ExecutionPlanDescriptor kExecutionPlans[] = {
    {"full", kFullStages, sizeof(kFullStages) / sizeof(kFullStages[0])},
    {"context_action", kContextActionStages,
     sizeof(kContextActionStages) / sizeof(kContextActionStages[0])},
};

static_assert(sizeof(kGraphCatalog) / sizeof(kGraphCatalog[0]) ==
                  static_cast<std::size_t>(Pi05GraphId::kCount),
              "PI0.5 graph catalog must be complete");

}  // namespace

modalities::Status Pi05ResolvedGraphBindings::bind(
    Pi05GraphBindingId id,
    frt_buffer buffer) {
    const std::size_t index = static_cast<std::size_t>(id);
    if (index >= buffers_.size() || !buffer || buffers_[index]) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "PI0.5 graph binding request is invalid");
    }
    buffers_[index] = buffer;
    return modalities::Status::ok();
}

frt_buffer Pi05ResolvedGraphBindings::get(Pi05GraphBindingId id) const {
    const std::size_t index = static_cast<std::size_t>(id);
    return index < buffers_.size() ? buffers_[index] : nullptr;
}

const char* pi05_graph_binding_name(Pi05GraphBindingId id) {
    static constexpr const char* kNames[] = {
        "observation_images_normalized",
        "prompt_embedding",
        "encoder_x",
        "diffusion_noise",
        "rtc_prev_action_chunk",
        "rtc_prefix_weights",
        "rtc_guidance_weight",
    };
    static_assert(sizeof(kNames) / sizeof(kNames[0]) ==
                      static_cast<std::size_t>(Pi05GraphBindingId::kCount),
                  "PI0.5 graph binding name catalog must be complete");
    const std::size_t index = static_cast<std::size_t>(id);
    return index < static_cast<std::size_t>(Pi05GraphBindingId::kCount)
               ? kNames[index]
               : nullptr;
}

const Pi05GraphDescriptor* pi05_graph_descriptor(Pi05GraphId id) {
    const std::size_t index = static_cast<std::size_t>(id);
    if (index >= static_cast<std::size_t>(Pi05GraphId::kCount) ||
        kGraphCatalog[index].id != id) {
        return nullptr;
    }
    return &kGraphCatalog[index];
}

const Pi05GraphDescriptor* pi05_graph_catalog(std::size_t* count) {
    if (count) *count = sizeof(kGraphCatalog) / sizeof(kGraphCatalog[0]);
    return kGraphCatalog;
}

const Pi05ExecutionPlanDescriptor* pi05_execution_plan(const char* name) {
    if (!name) return nullptr;
    for (const Pi05ExecutionPlanDescriptor& plan : kExecutionPlans) {
        if (std::strcmp(name, plan.name) == 0) return &plan;
    }
    return nullptr;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

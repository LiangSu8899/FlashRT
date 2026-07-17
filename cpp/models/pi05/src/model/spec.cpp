#include "flashrt/cpp/models/pi05/model/spec.h"

#include <algorithm>
#include <cstring>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

constexpr Pi05GraphSpec kGraphCatalog[] = {
    {GraphKind::kInfer, "infer"},
    {GraphKind::kDecodeOnly, "decode_only"},
    {GraphKind::kContext, "context"},
};

constexpr std::uint32_t kActionAfter[] = {0};
constexpr Pi05StageSpec kFullStages[] = {
    {"infer", GraphKind::kInfer, nullptr, 0},
};
constexpr Pi05StageSpec kContextActionStages[] = {
    {"context", GraphKind::kContext, nullptr, 0},
    {"action", GraphKind::kDecodeOnly, kActionAfter, 1},
};
constexpr Pi05ExecutionPlan kExecutionPlans[] = {
    {"full", kFullStages, 1},
    {"context_action", kContextActionStages, 2},
};

static_assert(sizeof(kGraphCatalog) / sizeof(kGraphCatalog[0]) ==
                  static_cast<std::size_t>(GraphKind::kCount),
              "PI0.5 graph catalog must cover every graph kind");

}  // namespace

const Pi05GraphSpec* pi05_graph_catalog(std::size_t* count) {
    if (count) *count = sizeof(kGraphCatalog) / sizeof(kGraphCatalog[0]);
    return kGraphCatalog;
}

const char* pi05_graph_name(GraphKind kind) {
    const std::size_t index = static_cast<std::size_t>(kind);
    if (index >= static_cast<std::size_t>(GraphKind::kCount) ||
        kGraphCatalog[index].kind != kind) {
        return nullptr;
    }
    return kGraphCatalog[index].name;
}

const Pi05ExecutionPlan* pi05_execution_plan(const char* name) {
    if (!name) return nullptr;
    for (const auto& plan : kExecutionPlans) {
        if (std::strcmp(name, plan.name) == 0) return &plan;
    }
    return nullptr;
}

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
    spec.normalize.mode = modalities::NormalizeMode::kDivideShift;
    spec.normalize.divisor = 127.5f;
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

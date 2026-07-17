#ifndef FLASHRT_MODELS_PI05_CPP_RUNTIME_SPEC_H
#define FLASHRT_MODELS_PI05_CPP_RUNTIME_SPEC_H

#include "flashrt/cpp/modalities/action.h"
#include "flashrt/cpp/modalities/vision.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

static constexpr int kImageSize = 224;
static constexpr int kDefaultChunk = 10;
static constexpr int kModelActionDim = 32;
static constexpr int kLiberoActionDim = 7;
static constexpr int kEncoderWidth = 2048;
static constexpr int kVisionLayers = 27;
static constexpr int kEncoderLayers = 18;
static constexpr int kDecoderLayers = 18;

enum class GraphKind : std::size_t {
    kInfer = 0,
    kDecodeOnly = 1,
    kContext = 2,
    kCount = 3,
};

struct Pi05GraphSpec {
    GraphKind kind;
    const char* name;
};

struct Pi05StageSpec {
    const char* name;
    GraphKind graph;
    const std::uint32_t* after;
    std::uint32_t n_after;
};

struct Pi05ExecutionPlan {
    const char* name;
    const Pi05StageSpec* stages;
    std::size_t n_stages;
};

const Pi05GraphSpec* pi05_graph_catalog(std::size_t* count);
const char* pi05_graph_name(GraphKind kind);
const Pi05ExecutionPlan* pi05_execution_plan(const char* name);

modalities::VisionPreprocessSpec vision_preprocess_spec(int num_views);

modalities::ActionPostprocessSpec action_postprocess_spec(
    const std::vector<float>& mean,
    const std::vector<float>& stddev,
    int chunk = kDefaultChunk,
    int model_dim = kModelActionDim,
    int robot_dim = kLiberoActionDim);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_MODELS_PI05_CPP_RUNTIME_SPEC_H

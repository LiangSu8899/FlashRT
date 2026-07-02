#ifndef FLASHRT_MODALITIES_VISION_H
#define FLASHRT_MODALITIES_VISION_H

#include "flashrt/modalities/types.h"

#include <string>
#include <vector>

namespace flashrt {
namespace modalities {

enum class NormalizeMode {
    kScaleShift,
    kMeanStd,
};

struct NormalizeSpec {
    NormalizeMode mode = NormalizeMode::kScaleShift;
    float scale = 1.0f / 127.5f;
    float shift = -1.0f;
    float mean[3] = {0.0f, 0.0f, 0.0f};
    float inv_std[3] = {1.0f, 1.0f, 1.0f};
};

struct VisionFrame {
    std::string name;
    TensorView image;
    PixelFormat format = PixelFormat::kRGB8;
    int width = 0;
    int height = 0;
    int stride_bytes = 0;
    std::uint64_t timestamp_ns = 0;
};

struct VisionPreprocessSpec {
    std::vector<std::string> view_order;
    int target_width = 224;
    int target_height = 224;
    DType output_dtype = DType::kBFloat16;
    Layout output_layout = Layout::kNHWC;
    NormalizeSpec normalize;
    bool require_exact_views = true;
};

Status preprocess_vision_cpu(const VisionPreprocessSpec& spec,
                             const std::vector<VisionFrame>& frames,
                             TensorView output);

std::uint64_t required_vision_output_bytes(const VisionPreprocessSpec& spec);

}  // namespace modalities
}  // namespace flashrt

#endif  // FLASHRT_MODALITIES_VISION_H

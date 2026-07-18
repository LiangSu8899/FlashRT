#ifndef FLASHRT_CPP_MODELS_PI05_MODEL_DIMS_H
#define FLASHRT_CPP_MODELS_PI05_MODEL_DIMS_H

#include "flashrt/cpp/modalities/types.h"

#include <cstdint>

namespace flashrt {
namespace models {
namespace pi05 {

struct Pi05ModelDims final {
    int image_height;
    int image_width;
    int image_channels;
    int vision_patch;
    int vision_tokens_per_view;

    int vision_width;
    int vision_hidden;
    int vision_layers;
    int vision_heads;
    int vision_head_dim;

    int encoder_width;
    int encoder_hidden;
    int encoder_layers;
    int encoder_heads;
    int encoder_kv_heads;
    int encoder_head_dim;

    int decoder_width;
    int decoder_hidden;
    int decoder_layers;
    int decoder_heads;
    int decoder_kv_heads;
    int decoder_head_dim;

    int action_width;
    int embedding_vocab;
    int embedding_width;
};

inline constexpr Pi05ModelDims kPi05ModelDims{
    224, 224, 3, 14, 256,
    1152, 4304, 27, 16, 72,
    2048, 16384, 18, 8, 1, 256,
    1024, 4096, 18, 8, 1, 256,
    32, 257152, 2048,
};

static_assert(kPi05ModelDims.image_height % kPi05ModelDims.vision_patch == 0 &&
                  kPi05ModelDims.image_width %
                          kPi05ModelDims.vision_patch ==
                      0,
              "PI0.5 image geometry must align to the vision patch");
static_assert(kPi05ModelDims.vision_heads *
                      kPi05ModelDims.vision_head_dim ==
                  kPi05ModelDims.vision_width,
              "PI0.5 vision attention width is inconsistent");
static_assert(kPi05ModelDims.encoder_heads *
                      kPi05ModelDims.encoder_head_dim ==
                  kPi05ModelDims.encoder_width,
              "PI0.5 encoder attention width is inconsistent");
static_assert(kPi05ModelDims.decoder_heads *
                      kPi05ModelDims.decoder_head_dim ==
                  kPi05ModelDims.encoder_width,
              "PI0.5 decoder attention width is inconsistent");

struct Pi05ShapeConfig final {
    std::int64_t num_views = 0;
    std::int64_t max_prompt_tokens = 0;
    std::int64_t chunk = 0;
    std::int64_t num_steps = 0;
    std::int64_t vision_pool_factor = 0;
    std::int64_t state_dim = 0;
    std::int64_t robot_action_dim = 0;
};

struct Pi05ResolvedShape final {
    int num_views = 0;
    int max_prompt_tokens = 0;
    int chunk = 0;
    int num_steps = 0;
    int vision_pool_factor = 0;
    int state_dim = 0;
    int robot_action_dim = 0;

    int pool_area = 0;
    int vision_sequence = 0;
    int encoder_vision_sequence = 0;
    int encoder_sequence = 0;
    int total_attention_keys = 0;
};

modalities::Status resolve_pi05_shape(const Pi05ShapeConfig& config,
                                      Pi05ResolvedShape* out);

modalities::Status validate_pi05_resolved_shape(
    const Pi05ResolvedShape& shape);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_MODEL_DIMS_H

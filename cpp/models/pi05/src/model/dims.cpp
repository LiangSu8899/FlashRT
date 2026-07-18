#include "flashrt/cpp/models/pi05/model/dims.h"

#include <limits>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

}  // namespace

modalities::Status resolve_pi05_shape(const Pi05ShapeConfig& config,
                                      Pi05ResolvedShape* out) {
    if (!out) return invalid("Pi0.5 resolved shape destination is null");

    const std::int64_t int_max = std::numeric_limits<int>::max();
    if (config.num_views < 1 || config.num_views > 3 ||
        config.max_prompt_tokens < 1 ||
        config.max_prompt_tokens > int_max || config.chunk < 1 ||
        config.chunk > int_max || config.num_steps < 1 ||
        config.num_steps > int_max || config.state_dim < 1 ||
        config.state_dim > int_max || config.robot_action_dim < 1 ||
        config.robot_action_dim > kPi05ModelDims.action_width) {
        return invalid("Pi0.5 runtime shape values are out of range");
    }
    if (config.vision_pool_factor != 1 &&
        config.vision_pool_factor != 2 &&
        config.vision_pool_factor != 4) {
        return invalid("Pi0.5 vision pool factor is unsupported");
    }

    const std::uint64_t views =
        static_cast<std::uint64_t>(config.num_views);
    const std::uint64_t prompt =
        static_cast<std::uint64_t>(config.max_prompt_tokens);
    const std::uint64_t chunk = static_cast<std::uint64_t>(config.chunk);
    const std::uint64_t pool_factor =
        static_cast<std::uint64_t>(config.vision_pool_factor);
    const std::uint64_t pool_area = pool_factor * pool_factor;
    const std::uint64_t vision_sequence =
        views * static_cast<std::uint64_t>(
                    kPi05ModelDims.vision_tokens_per_view);
    if (!pool_area || vision_sequence % pool_area != 0) {
        return invalid("Pi0.5 pooled vision sequence is invalid");
    }
    const std::uint64_t encoder_vision_sequence =
        vision_sequence / pool_area;
    const std::uint64_t encoder_sequence =
        encoder_vision_sequence + prompt;
    const std::uint64_t total_attention_keys = encoder_sequence + chunk;
    if (vision_sequence > static_cast<std::uint64_t>(int_max) ||
        encoder_vision_sequence > static_cast<std::uint64_t>(int_max) ||
        encoder_sequence > static_cast<std::uint64_t>(int_max) ||
        total_attention_keys > static_cast<std::uint64_t>(int_max)) {
        return invalid("Pi0.5 runtime sequence exceeds the kernel contract");
    }

    Pi05ResolvedShape resolved;
    resolved.num_views = static_cast<int>(config.num_views);
    resolved.max_prompt_tokens =
        static_cast<int>(config.max_prompt_tokens);
    resolved.chunk = static_cast<int>(config.chunk);
    resolved.num_steps = static_cast<int>(config.num_steps);
    resolved.vision_pool_factor =
        static_cast<int>(config.vision_pool_factor);
    resolved.state_dim = static_cast<int>(config.state_dim);
    resolved.robot_action_dim = static_cast<int>(config.robot_action_dim);
    resolved.pool_area = static_cast<int>(pool_area);
    resolved.vision_sequence = static_cast<int>(vision_sequence);
    resolved.encoder_vision_sequence =
        static_cast<int>(encoder_vision_sequence);
    resolved.encoder_sequence = static_cast<int>(encoder_sequence);
    resolved.total_attention_keys =
        static_cast<int>(total_attention_keys);
    *out = resolved;
    return modalities::Status::ok();
}

modalities::Status validate_pi05_resolved_shape(
    const Pi05ResolvedShape& shape) {
    Pi05ShapeConfig config;
    config.num_views = shape.num_views;
    config.max_prompt_tokens = shape.max_prompt_tokens;
    config.chunk = shape.chunk;
    config.num_steps = shape.num_steps;
    config.vision_pool_factor = shape.vision_pool_factor;
    config.state_dim = shape.state_dim;
    config.robot_action_dim = shape.robot_action_dim;
    Pi05ResolvedShape expected;
    modalities::Status status = resolve_pi05_shape(config, &expected);
    if (!status.ok_status()) return status;
    return pi05_resolved_shape_equal(shape, expected)
               ? modalities::Status::ok()
               : invalid("Pi0.5 resolved shape is internally inconsistent");
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#include "flashrt/modalities/action.h"

#include <algorithm>

namespace flashrt {
namespace modalities {
namespace {

float load_value(const void* base, std::uint64_t index, DType dtype) {
    switch (dtype) {
        case DType::kFloat32:
            return static_cast<const float*>(base)[index];
        case DType::kBFloat16:
            return bfloat16_to_float(static_cast<const std::uint16_t*>(base)[index]);
        case DType::kFloat16:
            return float16_to_float(static_cast<const std::uint16_t*>(base)[index]);
        case DType::kUInt8:
            return static_cast<float>(static_cast<const std::uint8_t*>(base)[index]);
    }
    return 0.0f;
}

bool has_dim(const std::vector<float>& v, int dim) {
    return v.empty() || static_cast<int>(v.size()) >= dim;
}

}  // namespace

Status postprocess_action_cpu(const ActionPostprocessSpec& spec,
                              TensorView model_output,
                              std::vector<float>* robot_actions) {
    if (!robot_actions) {
        return Status::error(StatusCode::kInvalidArgument,
                             "robot_actions is null");
    }
    if (spec.chunk <= 0 || spec.model_dim <= 0 || spec.robot_dim <= 0 ||
        spec.robot_dim > spec.model_dim) {
        return Status::error(StatusCode::kInvalidArgument,
                             "invalid action dimensions");
    }
    if (!has_dim(spec.mean, spec.robot_dim) ||
        !has_dim(spec.stddev, spec.robot_dim) ||
        !has_dim(spec.min_value, spec.robot_dim) ||
        !has_dim(spec.max_value, spec.robot_dim)) {
        return Status::error(StatusCode::kInvalidArgument,
                             "action stats do not cover robot_dim");
    }
    Status st = validate_host_tensor(model_output, "model_output");
    if (!st.ok_status()) return st;

    const std::uint64_t need_elem =
        static_cast<std::uint64_t>(spec.chunk) *
        static_cast<std::uint64_t>(spec.model_dim);
    const std::uint64_t need_bytes = need_elem * dtype_size(model_output.dtype);
    if (model_output.bytes < need_bytes) {
        return Status::error(StatusCode::kInsufficientStorage,
                             "model_output is too small for action spec");
    }

    robot_actions->assign(static_cast<std::size_t>(spec.chunk * spec.robot_dim),
                          0.0f);
    for (int t = 0; t < spec.chunk; ++t) {
        for (int d = 0; d < spec.robot_dim; ++d) {
            const std::uint64_t src_idx =
                static_cast<std::uint64_t>(t) * spec.model_dim +
                static_cast<std::uint64_t>(d);
            float value = load_value(model_output.data, src_idx, model_output.dtype);
            if (!spec.stddev.empty()) value *= spec.stddev[d];
            if (!spec.mean.empty()) value += spec.mean[d];
            if (spec.clamp) {
                if (!spec.min_value.empty()) value = std::max(value, spec.min_value[d]);
                if (!spec.max_value.empty()) value = std::min(value, spec.max_value[d]);
            }
            (*robot_actions)[static_cast<std::size_t>(t * spec.robot_dim + d)] = value;
        }
    }
    return Status::ok();
}

}  // namespace modalities
}  // namespace flashrt

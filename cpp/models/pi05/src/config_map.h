/* config_map.h — shared internal helpers for the Pi0.5 C faces (c_api and
 * model_runtime): C config -> RuntimeConfig mapping and status/enum
 * translation. Not installed. */
#ifndef FLASHRT_CPP_MODELS_PI05_CONFIG_MAP_H
#define FLASHRT_CPP_MODELS_PI05_CONFIG_MAP_H

#include "flashrt/cpp/models/pi05/c_api.h"
#include "flashrt/cpp/models/pi05/runtime.h"

#include <cstddef>

namespace flashrt {
namespace models {
namespace pi05 {
namespace cface {

inline int status_code(const modalities::Status& st) {
    using Code = modalities::StatusCode;
    switch (st.code) {
        case Code::kOk: return 0;
        case Code::kInvalidArgument: return -1;
        case Code::kNotFound: return -2;
        case Code::kUnsupported: return -3;
        case Code::kShapeMismatch: return -4;
        case Code::kInsufficientStorage: return -5;
        case Code::kBackend: return -6;
    }
    return -127;
}

inline modalities::PixelFormat pixel_format(int value) {
    using modalities::PixelFormat;
    switch (value) {
        case FRT_PI05_PIXEL_BGR8: return PixelFormat::kBGR8;
        case FRT_PI05_PIXEL_RGBA8: return PixelFormat::kRGBA8;
        case FRT_PI05_PIXEL_BGRA8: return PixelFormat::kBGRA8;
        case FRT_PI05_PIXEL_GRAY8: return PixelFormat::kGRAY8;
        case FRT_PI05_PIXEL_RGB8:
        default: return PixelFormat::kRGB8;
    }
}

inline modalities::DType dtype(int value) {
    using modalities::DType;
    switch (value) {
        case FRT_PI05_DTYPE_FLOAT16: return DType::kFloat16;
        case FRT_PI05_DTYPE_FLOAT32: return DType::kFloat32;
        case FRT_PI05_DTYPE_BFLOAT16:
        case FRT_PI05_DTYPE_DEFAULT:
        default: return DType::kBFloat16;
    }
}

inline bool has_field(const frt_pi05_runtime_config* in, std::size_t offset,
                      std::size_t bytes) {
    return in && in->struct_size >= offset + bytes;
}

inline RuntimeConfig make_config(const frt_pi05_runtime_config* in) {
    RuntimeConfig cfg;
    if (!in) return cfg;
    if (in->num_views > 0) cfg.num_views = in->num_views;
    if (in->chunk > 0) cfg.chunk = in->chunk;
    if (in->model_action_dim > 0) cfg.model_action_dim = in->model_action_dim;
    if (in->robot_action_dim > 0) cfg.robot_action_dim = in->robot_action_dim;
    if (in->action_mean && in->n_action_mean) {
        cfg.action_mean.assign(in->action_mean,
                               in->action_mean + in->n_action_mean);
    }
    if (in->action_stddev && in->n_action_stddev) {
        cfg.action_stddev.assign(in->action_stddev,
                                 in->action_stddev + in->n_action_stddev);
    }
    if (in->graph_name) cfg.graph_name = in->graph_name;
    if (in->image_buffer_name) cfg.image_buffer_name = in->image_buffer_name;
    if (in->action_buffer_name) cfg.action_buffer_name = in->action_buffer_name;
    if (has_field(in, offsetof(frt_pi05_runtime_config, image_dtype),
                  sizeof(in->image_dtype))) {
        cfg.image_dtype = dtype(in->image_dtype);
    }
    if (has_field(in, offsetof(frt_pi05_runtime_config, action_dtype),
                  sizeof(in->action_dtype))) {
        cfg.action_dtype = dtype(in->action_dtype);
    }
    if (has_field(in, offsetof(frt_pi05_runtime_config, max_frame_width),
                  sizeof(in->max_frame_width)) && in->max_frame_width > 0) {
        cfg.max_frame_width = in->max_frame_width;
    }
    if (has_field(in, offsetof(frt_pi05_runtime_config, max_frame_height),
                  sizeof(in->max_frame_height)) && in->max_frame_height > 0) {
        cfg.max_frame_height = in->max_frame_height;
    }
    return cfg;
}

}  // namespace cface
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_CONFIG_MAP_H

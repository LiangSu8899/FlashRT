#include "flashrt/modalities/vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace flashrt {
namespace modalities {
namespace {

int channels(PixelFormat f) {
    switch (f) {
        case PixelFormat::kRGB8:
        case PixelFormat::kBGR8: return 3;
        case PixelFormat::kRGBA8:
        case PixelFormat::kBGRA8: return 4;
        case PixelFormat::kGRAY8: return 1;
    }
    return 0;
}

const VisionFrame* find_frame(const std::vector<VisionFrame>& frames,
                              const std::string& name) {
    for (const auto& f : frames) {
        if (f.name == name) return &f;
    }
    return nullptr;
}

void read_rgb(const std::uint8_t* p, PixelFormat fmt, float rgb[3]) {
    switch (fmt) {
        case PixelFormat::kRGB8:
        case PixelFormat::kRGBA8:
            rgb[0] = static_cast<float>(p[0]);
            rgb[1] = static_cast<float>(p[1]);
            rgb[2] = static_cast<float>(p[2]);
            return;
        case PixelFormat::kBGR8:
        case PixelFormat::kBGRA8:
            rgb[0] = static_cast<float>(p[2]);
            rgb[1] = static_cast<float>(p[1]);
            rgb[2] = static_cast<float>(p[0]);
            return;
        case PixelFormat::kGRAY8:
            rgb[0] = rgb[1] = rgb[2] = static_cast<float>(p[0]);
            return;
    }
}

float normalize_value(float raw, int c, const NormalizeSpec& spec) {
    if (spec.mode == NormalizeMode::kScaleShift) {
        return raw * spec.scale + spec.shift;
    }
    return (raw / 255.0f - spec.mean[c]) * spec.inv_std[c];
}

void store_value(void* base, std::uint64_t index, DType dtype, float value) {
    switch (dtype) {
        case DType::kFloat32:
            static_cast<float*>(base)[index] = value;
            return;
        case DType::kBFloat16:
            static_cast<std::uint16_t*>(base)[index] = float_to_bfloat16(value);
            return;
        case DType::kFloat16:
            static_cast<std::uint16_t*>(base)[index] = float_to_float16(value);
            return;
        case DType::kUInt8:
            static_cast<std::uint8_t*>(base)[index] =
                static_cast<std::uint8_t>(std::max(0.0f, std::min(255.0f, value)));
            return;
    }
}

Status validate_frame(const VisionFrame& frame) {
    if (frame.width <= 0 || frame.height <= 0) {
        return Status::error(StatusCode::kInvalidArgument,
                             "vision frame has non-positive size");
    }
    const int ch = channels(frame.format);
    if (ch <= 0) {
        return Status::error(StatusCode::kUnsupported,
                             "unsupported pixel format");
    }
    const int stride = frame.stride_bytes > 0 ? frame.stride_bytes : frame.width * ch;
    const std::uint64_t need =
        static_cast<std::uint64_t>(stride) * static_cast<std::uint64_t>(frame.height);
    if (!frame.image.data || frame.image.bytes < need) {
        return Status::error(StatusCode::kInsufficientStorage,
                             "vision frame storage is too small");
    }
    if (frame.image.place != MemoryPlace::kHost &&
        frame.image.place != MemoryPlace::kHostPinned) {
        return Status::error(StatusCode::kUnsupported,
                             "CPU vision preprocess expects host frames");
    }
    return Status::ok();
}

}  // namespace

std::uint64_t required_vision_output_bytes(const VisionPreprocessSpec& spec) {
    return static_cast<std::uint64_t>(spec.view_order.size()) *
           static_cast<std::uint64_t>(spec.target_height) *
           static_cast<std::uint64_t>(spec.target_width) * 3ull *
           dtype_size(spec.output_dtype);
}

Status preprocess_vision_cpu(const VisionPreprocessSpec& spec,
                             const std::vector<VisionFrame>& frames,
                             TensorView output) {
    if (spec.view_order.empty()) {
        return Status::error(StatusCode::kInvalidArgument,
                             "vision view_order is empty");
    }
    if (spec.target_width <= 0 || spec.target_height <= 0) {
        return Status::error(StatusCode::kInvalidArgument,
                             "vision target size is invalid");
    }
    if (spec.output_layout != Layout::kNHWC) {
        return Status::error(StatusCode::kUnsupported,
                             "CPU vision preprocess currently writes NHWC");
    }
    if (spec.require_exact_views && frames.size() != spec.view_order.size()) {
        return Status::error(StatusCode::kShapeMismatch,
                             "vision frame count does not match view_order");
    }
    if (!output.data || output.bytes < required_vision_output_bytes(spec)) {
        return Status::error(StatusCode::kInsufficientStorage,
                             "vision output storage is too small");
    }
    if (output.place != MemoryPlace::kHost &&
        output.place != MemoryPlace::kHostPinned) {
        return Status::error(StatusCode::kUnsupported,
                             "CPU vision preprocess expects host output");
    }

    const int tw = spec.target_width;
    const int th = spec.target_height;
    for (std::size_t v = 0; v < spec.view_order.size(); ++v) {
        const VisionFrame* frame = find_frame(frames, spec.view_order[v]);
        if (!frame) {
            return Status::error(StatusCode::kNotFound,
                                 "missing required vision view: " + spec.view_order[v]);
        }
        Status st = validate_frame(*frame);
        if (!st.ok_status()) return st;

        const int ch = channels(frame->format);
        const int stride = frame->stride_bytes > 0
                               ? frame->stride_bytes
                               : frame->width * ch;
        const auto* src = static_cast<const std::uint8_t*>(frame->image.data);
        for (int y = 0; y < th; ++y) {
            const float fy = (static_cast<float>(y) + 0.5f) *
                             static_cast<float>(frame->height) /
                             static_cast<float>(th) - 0.5f;
            const int y0 = std::max(0, std::min(frame->height - 1,
                                                static_cast<int>(std::floor(fy))));
            const int y1 = std::max(0, std::min(frame->height - 1, y0 + 1));
            const float wy = std::max(0.0f, std::min(1.0f, fy - y0));
            for (int x = 0; x < tw; ++x) {
                const float fx = (static_cast<float>(x) + 0.5f) *
                                 static_cast<float>(frame->width) /
                                 static_cast<float>(tw) - 0.5f;
                const int x0 = std::max(0, std::min(frame->width - 1,
                                                    static_cast<int>(std::floor(fx))));
                const int x1 = std::max(0, std::min(frame->width - 1, x0 + 1));
                const float wx = std::max(0.0f, std::min(1.0f, fx - x0));

                float p00[3], p01[3], p10[3], p11[3];
                read_rgb(src + y0 * stride + x0 * ch, frame->format, p00);
                read_rgb(src + y0 * stride + x1 * ch, frame->format, p01);
                read_rgb(src + y1 * stride + x0 * ch, frame->format, p10);
                read_rgb(src + y1 * stride + x1 * ch, frame->format, p11);

                for (int c = 0; c < 3; ++c) {
                    const float top = p00[c] * (1.0f - wx) + p01[c] * wx;
                    const float bot = p10[c] * (1.0f - wx) + p11[c] * wx;
                    const float raw = top * (1.0f - wy) + bot * wy;
                    const float norm = normalize_value(raw, c, spec.normalize);
                    const std::uint64_t out_idx =
                        (((static_cast<std::uint64_t>(v) * th + y) * tw + x) * 3ull) +
                        static_cast<std::uint64_t>(c);
                    store_value(output.data, out_idx, spec.output_dtype, norm);
                }
            }
        }
    }
    return Status::ok();
}

}  // namespace modalities
}  // namespace flashrt

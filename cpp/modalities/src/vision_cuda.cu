#include "flashrt/cpp/modalities/vision.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace flashrt {
namespace modalities {
namespace {

enum FormatCode {
    kFmtRgb8 = 0,
    kFmtBgr8 = 1,
    kFmtRgba8 = 2,
    kFmtBgra8 = 3,
    kFmtGray8 = 4,
};

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

int format_code(PixelFormat f) {
    switch (f) {
        case PixelFormat::kRGB8: return kFmtRgb8;
        case PixelFormat::kBGR8: return kFmtBgr8;
        case PixelFormat::kRGBA8: return kFmtRgba8;
        case PixelFormat::kBGRA8: return kFmtBgra8;
        case PixelFormat::kGRAY8: return kFmtGray8;
    }
    return -1;
}

const VisionFrame* find_frame(const std::vector<VisionFrame>& frames,
                              const std::string& name) {
    for (const auto& f : frames) {
        if (f.name == name) return &f;
    }
    return nullptr;
}

Status validate_frame_for_cuda(const VisionFrame& frame) {
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
        static_cast<std::uint64_t>(stride) *
        static_cast<std::uint64_t>(frame.height);
    if (!frame.image.data || frame.image.bytes < need) {
        return Status::error(StatusCode::kInsufficientStorage,
                             "vision frame storage is too small");
    }
    if (frame.image.place != MemoryPlace::kHost &&
        frame.image.place != MemoryPlace::kHostPinned) {
        return Status::error(StatusCode::kUnsupported,
                             "CUDA vision preprocess currently expects host frames");
    }
    return Status::ok();
}

__device__ __forceinline__ std::uint16_t f32_to_bf16(float value) {
    std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(bits >> 16);
}

__device__ __forceinline__ float clamp01(float v) {
    return fminf(1.0f, fmaxf(0.0f, v));
}

__device__ __forceinline__ void read_rgb(const std::uint8_t* p,
                                         int fmt,
                                         float rgb[3]) {
    if (fmt == kFmtRgb8 || fmt == kFmtRgba8) {
        rgb[0] = static_cast<float>(p[0]);
        rgb[1] = static_cast<float>(p[1]);
        rgb[2] = static_cast<float>(p[2]);
    } else if (fmt == kFmtBgr8 || fmt == kFmtBgra8) {
        rgb[0] = static_cast<float>(p[2]);
        rgb[1] = static_cast<float>(p[1]);
        rgb[2] = static_cast<float>(p[0]);
    } else {
        rgb[0] = rgb[1] = rgb[2] = static_cast<float>(p[0]);
    }
}

__device__ __forceinline__ float normalize_value(float raw,
                                                 int c,
                                                 int norm_mode,
                                                 float scale,
                                                 float shift,
                                                 const float* mean,
                                                 const float* inv_std) {
    if (norm_mode == 0) return raw * scale + shift;
    return (raw / 255.0f - mean[c]) * inv_std[c];
}

__device__ __forceinline__ void store_value(void* out,
                                           std::uint64_t index,
                                           int dtype,
                                           float value) {
    if (dtype == 1) {
        static_cast<float*>(out)[index] = value;
    } else if (dtype == 2) {
        static_cast<__half*>(out)[index] = __float2half_rn(value);
    } else {
        static_cast<std::uint16_t*>(out)[index] = f32_to_bf16(value);
    }
}

__global__ void resize_normalize_kernel(const std::uint8_t* src,
                                        int sw,
                                        int sh,
                                        int stride,
                                        int fmt,
                                        int src_channels,
                                        void* out,
                                        int out_dtype,
                                        int view,
                                        int tw,
                                        int th,
                                        int norm_mode,
                                        float scale,
                                        float shift,
                                        float mean0,
                                        float mean1,
                                        float mean2,
                                        float inv_std0,
                                        float inv_std1,
                                        float inv_std2) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= tw || y >= th) return;

    const float fx = (static_cast<float>(x) + 0.5f) *
                     static_cast<float>(sw) / static_cast<float>(tw) - 0.5f;
    const float fy = (static_cast<float>(y) + 0.5f) *
                     static_cast<float>(sh) / static_cast<float>(th) - 0.5f;
    const int x0 = max(0, min(sw - 1, static_cast<int>(floorf(fx))));
    const int y0 = max(0, min(sh - 1, static_cast<int>(floorf(fy))));
    const int x1 = max(0, min(sw - 1, x0 + 1));
    const int y1 = max(0, min(sh - 1, y0 + 1));
    const float wx = clamp01(fx - static_cast<float>(x0));
    const float wy = clamp01(fy - static_cast<float>(y0));

    float p00[3], p01[3], p10[3], p11[3];
    read_rgb(src + y0 * stride + x0 * src_channels, fmt, p00);
    read_rgb(src + y0 * stride + x1 * src_channels, fmt, p01);
    read_rgb(src + y1 * stride + x0 * src_channels, fmt, p10);
    read_rgb(src + y1 * stride + x1 * src_channels, fmt, p11);

    float mean[3] = {mean0, mean1, mean2};
    float inv_std[3] = {inv_std0, inv_std1, inv_std2};
    for (int c = 0; c < 3; ++c) {
        const float top = p00[c] * (1.0f - wx) + p01[c] * wx;
        const float bot = p10[c] * (1.0f - wx) + p11[c] * wx;
        const float raw = top * (1.0f - wy) + bot * wy;
        const float norm = normalize_value(raw, c, norm_mode, scale, shift,
                                           mean, inv_std);
        const std::uint64_t out_idx =
            (((static_cast<std::uint64_t>(view) * th + y) * tw + x) * 3ull) +
            static_cast<std::uint64_t>(c);
        store_value(out, out_idx, out_dtype, norm);
    }
}

const char* cuda_error(cudaError_t rc) {
    return cudaGetErrorString(rc);
}

}  // namespace

Status preprocess_vision_cuda(const VisionPreprocessSpec& spec,
                              const std::vector<VisionFrame>& frames,
                              TensorView output,
                              void* stream,
                              VisionStaging* staging) {
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
                             "CUDA vision preprocess currently writes NHWC");
    }
    if (spec.output_dtype != DType::kBFloat16 &&
        spec.output_dtype != DType::kFloat16 &&
        spec.output_dtype != DType::kFloat32) {
        return Status::error(StatusCode::kUnsupported,
                             "CUDA vision preprocess supports bf16/fp16/fp32 output");
    }
    if (spec.require_exact_views && frames.size() != spec.view_order.size()) {
        return Status::error(StatusCode::kShapeMismatch,
                             "vision frame count does not match view_order");
    }
    const std::uint64_t out_bytes = required_vision_output_bytes(spec);
    if (!output.data || output.bytes < out_bytes ||
        output.place != MemoryPlace::kDevice) {
        return Status::error(StatusCode::kInsufficientStorage,
                             "vision device output storage is invalid");
    }

    if (staging && staging->slots < spec.view_order.size()) {
        return Status::error(StatusCode::kInsufficientStorage,
                             "vision staging has fewer slots than views");
    }

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    std::vector<void*> device_frames;   /* fallback path only */
    device_frames.reserve(staging ? 0 : spec.view_order.size());

    auto cleanup = [&]() {
        for (void* p : device_frames) cudaFree(p);
    };

    for (std::size_t v = 0; v < spec.view_order.size(); ++v) {
        const VisionFrame* frame = find_frame(frames, spec.view_order[v]);
        if (!frame) {
            cleanup();
            return Status::error(StatusCode::kNotFound,
                                 "missing required vision view: " + spec.view_order[v]);
        }
        Status st = validate_frame_for_cuda(*frame);
        if (!st.ok_status()) {
            cleanup();
            return st;
        }
        const int ch = channels(frame->format);
        const int stride = frame->stride_bytes > 0 ? frame->stride_bytes
                                                   : frame->width * ch;
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(stride) *
            static_cast<std::uint64_t>(frame->height);
        void* d_src = nullptr;
        cudaError_t rc = cudaSuccess;
        if (staging) {
            /* hot path: fixed slots, no allocation. Bounce through the pinned
             * slot so the H2D copy is a true async transfer, then the caller
             * may reuse its frame memory after the end-of-call sync. */
            if (bytes > staging->slot_bytes) {
                cleanup();
                return Status::error(
                    StatusCode::kInsufficientStorage,
                    "vision frame exceeds the staging slot capacity: " +
                        spec.view_order[v]);
            }
            const std::uint64_t off = staging->slot_bytes * v;
            void* h_slot = static_cast<char*>(staging->host_pinned) + off;
            d_src = static_cast<char*>(staging->device) + off;
            std::memcpy(h_slot, frame->image.data, bytes);
            rc = cudaMemcpyAsync(d_src, h_slot, bytes,
                                 cudaMemcpyHostToDevice, cuda_stream);
        } else {
            /* dev/one-shot path: per-frame allocation */
            rc = cudaMalloc(&d_src, bytes);
            if (rc != cudaSuccess) {
                cleanup();
                return Status::error(
                    StatusCode::kBackend,
                    std::string("cudaMalloc vision source failed: ") +
                        cuda_error(rc));
            }
            device_frames.push_back(d_src);
            rc = cudaMemcpyAsync(d_src, frame->image.data, bytes,
                                 cudaMemcpyHostToDevice, cuda_stream);
        }
        if (rc != cudaSuccess) {
            cleanup();
            return Status::error(StatusCode::kBackend,
                                 std::string("cuda H2D vision source failed: ") +
                                     cuda_error(rc));
        }

        const dim3 block(16, 16);
        const dim3 grid((spec.target_width + block.x - 1) / block.x,
                        (spec.target_height + block.y - 1) / block.y);
        resize_normalize_kernel<<<grid, block, 0, cuda_stream>>>(
            static_cast<const std::uint8_t*>(d_src),
            frame->width, frame->height, stride, format_code(frame->format), ch,
            output.data,
            spec.output_dtype == DType::kFloat32 ? 1 :
            (spec.output_dtype == DType::kFloat16 ? 2 : 0),
            static_cast<int>(v), spec.target_width, spec.target_height,
            spec.normalize.mode == NormalizeMode::kScaleShift ? 0 : 1,
            spec.normalize.scale, spec.normalize.shift,
            spec.normalize.mean[0], spec.normalize.mean[1], spec.normalize.mean[2],
            spec.normalize.inv_std[0], spec.normalize.inv_std[1],
            spec.normalize.inv_std[2]);
        rc = cudaGetLastError();
        if (rc != cudaSuccess) {
            cleanup();
            return Status::error(StatusCode::kBackend,
                                 std::string("vision CUDA kernel launch failed: ") +
                                     cuda_error(rc));
        }
    }

    cudaError_t rc = cudaStreamSynchronize(cuda_stream);
    cleanup();
    if (rc != cudaSuccess) {
        return Status::error(StatusCode::kBackend,
                             std::string("vision CUDA kernel sync failed: ") +
                                 cuda_error(rc));
    }
    return Status::ok();
}

}  // namespace modalities
}  // namespace flashrt

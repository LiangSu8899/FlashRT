#include "flashrt/cpp/models/pi05/c_api.h"

#include "config_map.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

struct frt_pi05_runtime_s {
    std::unique_ptr<flashrt::models::pi05::Runtime> runtime;
    std::string last_error;
};

namespace {

using flashrt::models::pi05::cface::make_config;
using flashrt::models::pi05::cface::pixel_format;
using flashrt::models::pi05::cface::status_code;

}  // namespace

extern "C" int frt_pi05_runtime_create(
    const frt_runtime_export_v1* exp,
    const frt_pi05_runtime_config* config,
    frt_pi05_runtime** out) {
    if (!exp || !out) return -1;
    *out = nullptr;
    constexpr std::size_t kConfigRequiredSize =
        offsetof(frt_pi05_runtime_config, image_dtype);
    if (config && config->struct_size < kConfigRequiredSize) {
        return -1;
    }
    auto* h = new (std::nothrow) frt_pi05_runtime_s();
    if (!h) return -5;
    try {
        h->runtime.reset(
            new flashrt::models::pi05::Runtime(exp, make_config(config)));
    } catch (const std::exception& e) {
        h->last_error = e.what();
        delete h;
        return -6;
    } catch (...) {
        delete h;
        return -6;
    }
    if (!h->runtime->ok()) {
        h->last_error = h->runtime->status().message;
        int rc = status_code(h->runtime->status());
        delete h;
        return rc;
    }
    *out = h;
    return 0;
}

extern "C" void frt_pi05_runtime_destroy(frt_pi05_runtime* h) {
    delete h;
}

extern "C" int frt_pi05_runtime_set_prompt(frt_pi05_runtime* h,
                                           const char* text) {
    if (!h || !h->runtime) return -1;
    int rc = h->runtime->set_prompt(text);
    if (rc != 0) h->last_error = "prompt updates are not supported by adopted-export Pi05 runtime";
    return rc;
}

extern "C" int frt_pi05_runtime_prepare_vision(
    frt_pi05_runtime* h,
    const frt_pi05_vision_frame* frames,
    uint64_t n_frames) {
    if (!h || !h->runtime || (!frames && n_frames)) return -1;
    std::vector<flashrt::modalities::VisionFrame> v;
    v.reserve(static_cast<std::size_t>(n_frames));
    for (uint64_t i = 0; i < n_frames; ++i) {
        const frt_pi05_vision_frame& in = frames[i];
        if (in.struct_size < sizeof(frt_pi05_vision_frame) ||
            !in.name || !in.data) {
            h->last_error = "invalid Pi05 vision frame";
            return -1;
        }
        flashrt::modalities::VisionFrame out;
        out.name = in.name;
        out.image.data = const_cast<void*>(in.data);
        out.image.bytes = in.bytes;
        out.image.dtype = flashrt::modalities::DType::kUInt8;
        out.image.place = flashrt::modalities::MemoryPlace::kHost;
        out.image.layout = flashrt::modalities::Layout::kHWC;
        out.image.shape = flashrt::modalities::Shape{
            static_cast<uint64_t>(std::max(0, in.height)),
            static_cast<uint64_t>(std::max(0, in.width)),
            3};
        out.format = pixel_format(in.pixel_format);
        out.width = in.width;
        out.height = in.height;
        out.stride_bytes = in.stride_bytes;
        out.timestamp_ns = in.timestamp_ns;
        v.push_back(std::move(out));
    }
    auto st = h->runtime->prepare_vision(v);
    if (!st.ok_status()) {
        h->last_error = st.message;
        return status_code(st);
    }
    h->last_error.clear();
    return 0;
}

extern "C" int frt_pi05_runtime_replay_tick(frt_pi05_runtime* h) {
    if (!h || !h->runtime) return -1;
    int rc = h->runtime->replay_tick();
    if (rc != 0) h->last_error = "Pi05 graph replay failed";
    return rc;
}

extern "C" int frt_pi05_runtime_read_actions(frt_pi05_runtime* h,
                                             float* out_actions,
                                             uint64_t out_capacity,
                                             uint64_t* n_written) {
    if (!h || !h->runtime || !out_actions) return -1;
    std::vector<float> actions;
    auto st = h->runtime->read_actions(&actions);
    if (!st.ok_status()) {
        h->last_error = st.message;
        return status_code(st);
    }
    if (out_capacity < actions.size()) {
        h->last_error = "action output buffer is too small";
        if (n_written) *n_written = actions.size();
        return -5;
    }
    std::memcpy(out_actions, actions.data(), actions.size() * sizeof(float));
    if (n_written) *n_written = actions.size();
    h->last_error.clear();
    return 0;
}

extern "C" const frt_runtime_export_v1* frt_pi05_runtime_export(
    frt_pi05_runtime* h) {
    if (!h || !h->runtime) return nullptr;
    return h->runtime->export_runtime();
}

extern "C" const char* frt_pi05_runtime_last_error(frt_pi05_runtime* h) {
    if (!h) return "null Pi05 runtime";
    return h->last_error.c_str();
}

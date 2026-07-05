/* model_runtime.cpp — Pi0.5 behind the generic model-runtime face. */
#include "flashrt/cpp/models/pi05/model_runtime.h"

#include "config_map.h"

#include <cstring>
#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace {

using flashrt::modalities::Status;
using flashrt::models::pi05::cface::status_code;

/* Port indices — fixed for the Pi0.5 face (array order below). No prompt
 * port on the adopted-export path: the prompt embedding is baked before
 * capture and cannot be updated, so declaring it would be an
 * advertise-and-refuse STAGED port. The native tokenizer producer adds it
 * back as a real STAGED port (a fingerprint change, as it should be). */
enum { kPortImages = 0, kPortNoise = 1, kPortActions = 2 };
constexpr uint32_t kNoPort = UINT32_MAX;

struct Adapter {
    std::unique_ptr<flashrt::models::pi05::Runtime> runtime;
    std::string last_error;
    std::vector<std::string> view_order;
    const frt_model_runtime_v1* source_model = nullptr;
    uint32_t images_port = kPortImages;
    uint32_t noise_port = kPortNoise;
    uint32_t actions_port = kPortActions;

    int64_t image_shape[4] = {0, 0, 0, 3};
    int64_t noise_shape[2] = {0, 0};
    int64_t action_shape[2] = {0, 0};
};

uint32_t rt_dtype(flashrt::modalities::DType d) {
    using flashrt::modalities::DType;
    switch (d) {
        case DType::kUInt8: return FRT_RT_DTYPE_U8;
        case DType::kFloat32: return FRT_RT_DTYPE_F32;
        case DType::kFloat16: return FRT_RT_DTYPE_F16;
        case DType::kBFloat16: return FRT_RT_DTYPE_BF16;
    }
    return FRT_RT_DTYPE_BF16;
}

flashrt::modalities::DType model_dtype(uint32_t d) {
    using flashrt::modalities::DType;
    switch (d) {
        case FRT_RT_DTYPE_U8: return DType::kUInt8;
        case FRT_RT_DTYPE_F32: return DType::kFloat32;
        case FRT_RT_DTYPE_F16: return DType::kFloat16;
        case FRT_RT_DTYPE_BF16:
        default: return DType::kBFloat16;
    }
}

flashrt::modalities::Layout model_layout(uint32_t l) {
    using flashrt::modalities::Layout;
    switch (l) {
        case FRT_RT_LAYOUT_HWC: return Layout::kHWC;
        case FRT_RT_LAYOUT_NHWC: return Layout::kNHWC;
        case FRT_RT_LAYOUT_CHW: return Layout::kCHW;
        case FRT_RT_LAYOUT_NCHW: return Layout::kNCHW;
        case FRT_RT_LAYOUT_FLAT:
        default: return Layout::kFlat;
    }
}

flashrt::modalities::TensorView tensor_from_port(
        const frt_runtime_port_desc& p) {
    flashrt::modalities::TensorView view;
    if (!p.buffer) return view;
    auto* base = static_cast<unsigned char*>(frt_buffer_dptr(p.buffer));
    if (!base) return view;
    const uint64_t total = frt_buffer_bytes(p.buffer);
    if (p.offset > total) return view;
    if (p.bytes && p.bytes > total - p.offset) return view;
    view.data = base + p.offset;
    view.bytes = p.bytes ? p.bytes : (total - p.offset);
    view.dtype = model_dtype(p.dtype);
    view.place = flashrt::modalities::MemoryPlace::kDevice;
    view.layout = model_layout(p.layout);
    view.shape.rank = p.rank > flashrt::modalities::Shape::kMaxRank
                          ? flashrt::modalities::Shape::kMaxRank
                          : p.rank;
    for (uint32_t i = 0; i < view.shape.rank; ++i) {
        view.shape.dims[i] = p.shape[i] > 0 ? static_cast<uint64_t>(p.shape[i])
                                            : 0;
    }
    return view;
}

flashrt::modalities::PixelFormat view_pixel_format(uint32_t value) {
    using flashrt::modalities::PixelFormat;
    switch (value) {
        case FRT_RT_PIXEL_BGR8: return PixelFormat::kBGR8;
        case FRT_RT_PIXEL_RGBA8: return PixelFormat::kRGBA8;
        case FRT_RT_PIXEL_BGRA8: return PixelFormat::kBGRA8;
        case FRT_RT_PIXEL_GRAY8: return PixelFormat::kGRAY8;
        case FRT_RT_PIXEL_RGB8:
        default: return PixelFormat::kRGB8;
    }
}

int set_input(void* self, uint32_t port, const void* data, uint64_t bytes,
              int stream) {
    (void)stream;  /* the pi05 runtime stages on the export's graph stream */
    auto* a = static_cast<Adapter*>(self);
    if (!a || !a->runtime) return -1;
    if (port == a->images_port) {
            if (!data || !bytes || bytes % sizeof(frt_image_view)) {
                a->last_error = "images payload must be frt_image_view[]";
                return -1;
            }
            const auto* views = static_cast<const frt_image_view*>(data);
            const uint64_t n = bytes / sizeof(frt_image_view);
            std::vector<flashrt::modalities::VisionFrame> frames;
            frames.reserve(n);
            for (uint64_t i = 0; i < n; ++i) {
                const frt_image_view& in = views[i];
                if (in.struct_size < sizeof(frt_image_view) || !in.data) {
                    a->last_error = "invalid image view";
                    return -1;
                }
                flashrt::modalities::VisionFrame f;
                /* generic views carry no names: positional, declared order */
                f.name = i < a->view_order.size() ? a->view_order[i]
                                                  : "view" + std::to_string(i);
                f.image.data = const_cast<void*>(in.data);
                f.image.bytes = in.bytes;
                f.image.dtype = flashrt::modalities::DType::kUInt8;
                f.image.place = flashrt::modalities::MemoryPlace::kHost;
                f.image.layout = flashrt::modalities::Layout::kHWC;
                f.image.shape = flashrt::modalities::Shape{
                    static_cast<uint64_t>(in.height > 0 ? in.height : 0),
                    static_cast<uint64_t>(in.width > 0 ? in.width : 0), 3};
                f.format = view_pixel_format(in.pixel_format);
                f.width = in.width;
                f.height = in.height;
                f.stride_bytes = in.stride_bytes;
                f.timestamp_ns = in.timestamp_ns;
                frames.push_back(std::move(f));
            }
            Status st = a->runtime->prepare_vision(frames);
            if (!st.ok_status()) {
                a->last_error = st.message;
                return status_code(st);
            }
            a->last_error.clear();
            return 0;
    }
    if (port == a->noise_port) {
        a->last_error =
            "noise is a SWAP port: write its buffer window directly";
        return -3;
    }
    a->last_error = "unknown or non-input port";
    return -1;
}

int get_output(void* self, uint32_t port, void* out, uint64_t capacity,
               uint64_t* written, int stream) {
    (void)stream;
    auto* a = static_cast<Adapter*>(self);
    if (!a || !a->runtime || !out) return -1;
    if (port != a->actions_port) {
        a->last_error = "unknown or non-output port";
        return -1;
    }
    std::vector<float> actions;
    Status st = a->runtime->read_actions(&actions);
    if (!st.ok_status()) {
        a->last_error = st.message;
        return status_code(st);
    }
    const uint64_t need = actions.size() * sizeof(float);
    if (written) *written = need;
    if (capacity < need) {
        a->last_error = "action output buffer is too small";
        return -5;
    }
    std::memcpy(out, actions.data(), need);
    a->last_error.clear();
    return 0;
}

int prepare(void* self, uint32_t graph, frt_shape_key key) {
    (void)graph;
    (void)key;
    auto* a = static_cast<Adapter*>(self);
    if (a) a->last_error = "adopted-export Pi05 runtime has fixed variants";
    return -3;
}

int step(void* self) {
    auto* a = static_cast<Adapter*>(self);
    if (!a || !a->runtime) return -1;
    if (a->source_model) {
        const frt_model_runtime_v1* m = a->source_model;
        const frt_runtime_export_v1* exp = m->exp;
        if (!exp || (!exp->graphs && exp->n_graphs)) return -1;
        for (uint64_t i = 0; i < m->n_stages; ++i) {
            const auto& stage = m->stages[i];
            if (stage.graph >= exp->n_graphs) {
                a->last_error = "Pi05 stage references a missing graph";
                return -1;
            }
            const int stream_id = exp->graphs[stage.graph].stream_id;
            for (uint32_t d = 0; d < stage.n_after; ++d) {
                const uint32_t dep = stage.after[d];
                if (dep >= i || m->stages[dep].graph >= exp->n_graphs) {
                    a->last_error = "Pi05 stage dependency is invalid";
                    return -1;
                }
                if (exp->graphs[m->stages[dep].graph].stream_id != stream_id) {
                    a->last_error =
                        "cross-stream stage DAG requires a host scheduler";
                    return -3;
                }
            }
            const auto& graph = exp->graphs[stage.graph];
            if (!graph.handle) {
                a->last_error = "Pi05 stage graph has no handle";
                return -1;
            }
            const int rc = frt_graph_replay(graph.handle, graph.default_key,
                                            stream_id);
            if (rc != 0) {
                a->last_error = "Pi05 stage replay failed";
                return rc;
            }
        }
        a->last_error.clear();
        return 0;
    }
    int rc = a->runtime->replay_tick();
    if (rc != 0) a->last_error = "Pi05 graph replay failed";
    return rc;
}

const char* last_error(void* self) {
    auto* a = static_cast<Adapter*>(self);
    return a ? a->last_error.c_str() : "null Pi05 model runtime";
}

void destroy_adapter(void* p) { delete static_cast<Adapter*>(p); }

const frt_runtime_buffer_desc* find_buffer(const frt_runtime_export_v1* exp,
                                           const std::string& name) {
    for (uint64_t i = 0; i < exp->n_buffers; ++i)
        if (exp->buffers[i].name && name == exp->buffers[i].name)
            return &exp->buffers[i];
    return nullptr;
}

const frt_runtime_graph_desc* graph_for_first_stage(
        const frt_model_runtime_v1* model) {
    if (!model || !model->exp || !model->n_stages) return nullptr;
    const uint32_t graph = model->stages[0].graph;
    if (graph >= model->exp->n_graphs) return nullptr;
    return &model->exp->graphs[graph];
}

uint32_t find_port_index(const frt_model_runtime_v1* model,
                         const char* name) {
    if (!model || !name) return kNoPort;
    for (uint64_t i = 0; i < model->n_ports; ++i) {
        const char* n = model->ports[i].name;
        if (n && std::strcmp(n, name) == 0) return static_cast<uint32_t>(i);
    }
    return kNoPort;
}

bool compatible_port(const frt_model_runtime_v1* model, uint32_t port,
                     uint32_t modality, uint32_t direction,
                     uint32_t update) {
    if (!model || port == kNoPort || port >= model->n_ports) return false;
    const auto& p = model->ports[port];
    return p.modality == modality && p.direction == direction &&
           p.update == update;
}

}  // namespace

extern "C" int frt_pi05_model_runtime_create(
        const frt_runtime_export_v1* exp,
        const frt_pi05_runtime_config* config,
        frt_model_runtime_v1** out) {
    if (!exp || !out) return -1;
    *out = nullptr;
    constexpr std::size_t kConfigRequiredSize =
        offsetof(frt_pi05_runtime_config, image_dtype);
    if (config && config->struct_size < kConfigRequiredSize) return -1;

    auto a = std::unique_ptr<Adapter>(new (std::nothrow) Adapter());
    if (!a) return -5;
    a->runtime.reset(new (std::nothrow) flashrt::models::pi05::Runtime(
        exp, flashrt::models::pi05::cface::make_config(config)));
    if (!a->runtime) return -5;
    if (!a->runtime->ok()) return status_code(a->runtime->status());

    const auto& manifest = a->runtime->manifest();
    a->view_order = manifest.vision.view_order;
    a->image_shape[0] = static_cast<int64_t>(a->view_order.size());
    a->image_shape[1] = manifest.vision.target_height;
    a->image_shape[2] = manifest.vision.target_width;
    a->noise_shape[0] = manifest.action.chunk;
    a->noise_shape[1] = manifest.action.model_dim;
    a->action_shape[0] = manifest.action.chunk;
    a->action_shape[1] = manifest.action.robot_dim;

    const std::string image_name = config && config->image_buffer_name
                                       ? config->image_buffer_name
                                       : "observation_images_normalized";
    const std::string action_name = config && config->action_buffer_name
                                        ? config->action_buffer_name
                                        : "diffusion_noise";
    const frt_runtime_buffer_desc* image_buf = find_buffer(exp, image_name);
    const frt_runtime_buffer_desc* action_buf = find_buffer(exp, action_name);
    const uint32_t io_dtype = rt_dtype(manifest.vision.output_dtype);

    frt_runtime_port_desc ports[3] = {};
    ports[kPortImages] = {"images", FRT_RT_MOD_IMAGE, io_dtype,
                          FRT_RT_LAYOUT_NHWC, FRT_RT_PORT_IN,
                          FRT_RT_PORT_STAGED, 1, a->image_shape, 4, 30,
                          image_buf ? image_buf->handle : nullptr, 0,
                          image_buf ? image_buf->bytes : 0};
    ports[kPortNoise] = {"noise", FRT_RT_MOD_TENSOR, io_dtype,
                         FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_IN,
                         FRT_RT_PORT_SWAP, 0, a->noise_shape, 2, 0,
                         action_buf ? action_buf->handle : nullptr, 0,
                         action_buf ? action_buf->bytes : 0};
    ports[kPortActions] = {"actions", FRT_RT_MOD_ACTION, io_dtype,
                           FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_OUT,
                           FRT_RT_PORT_STAGED, 0, a->action_shape, 2, 0,
                           action_buf ? action_buf->handle : nullptr, 0,
                           action_buf ? action_buf->bytes : 0};

    const std::string graph_name =
        config && config->graph_name ? config->graph_name : "infer";
    uint32_t graph_index = 0;
    bool found = false;
    for (uint64_t i = 0; i < exp->n_graphs; ++i) {
        if (exp->graphs[i].name && graph_name == exp->graphs[i].name) {
            graph_index = static_cast<uint32_t>(i);
            found = true;
            break;
        }
    }
    if (!found) return -2;
    frt_runtime_stage_desc stages[1] = {};
    stages[0].graph = graph_index;

    frt_model_runtime_verbs verbs{};
    verbs.struct_size = sizeof(verbs);
    verbs.set_input = set_input;
    verbs.get_output = get_output;
    verbs.prepare = prepare;
    verbs.step = step;
    verbs.last_error = last_error;

    Adapter* raw = a.release();
    frt_model_runtime_v1* m = frt_model_runtime_wrap(
        exp, ports, 3, stages, 1, &verbs, raw, raw, destroy_adapter);
    if (!m) {
        delete raw;
        return -1;
    }
    *out = m;
    return 0;
}

extern "C" int frt_pi05_model_runtime_create_over(
        const frt_model_runtime_v1* model,
        const frt_pi05_runtime_config* config,
        frt_model_runtime_v1** out) {
    if (!model || !out) return -1;
    *out = nullptr;
    if (model->abi_version != FRT_MODEL_RUNTIME_ABI_VERSION ||
        model->struct_size < sizeof(frt_model_runtime_v1) ||
        !model->exp || !model->retain || !model->release) {
        return -1;
    }
    constexpr std::size_t kConfigRequiredSize =
        offsetof(frt_pi05_runtime_config, image_dtype);
    if (config && config->struct_size < kConfigRequiredSize) return -1;

    const uint32_t images = find_port_index(model, "images");
    const uint32_t noise = find_port_index(model, "noise");
    const uint32_t actions = find_port_index(model, "actions");
    if (!compatible_port(model, images, FRT_RT_MOD_IMAGE, FRT_RT_PORT_IN,
                         FRT_RT_PORT_STAGED) ||
        !compatible_port(model, actions, FRT_RT_MOD_ACTION, FRT_RT_PORT_OUT,
                         FRT_RT_PORT_STAGED)) {
        return -2;
    }
    if (noise != kNoPort &&
        !compatible_port(model, noise, FRT_RT_MOD_TENSOR, FRT_RT_PORT_IN,
                         FRT_RT_PORT_SWAP)) {
        return -2;
    }

    auto cfg = flashrt::models::pi05::cface::make_config(config);
    if (model->n_stages) {
        const auto* graph = graph_for_first_stage(model);
        if (!graph || !graph->name) return -2;
        cfg.graph_name = graph->name;
    }
    cfg.image_input_override = tensor_from_port(model->ports[images]);
    cfg.action_output_override = tensor_from_port(model->ports[actions]);

    auto a = std::unique_ptr<Adapter>(new (std::nothrow) Adapter());
    if (!a) return -5;
    a->runtime.reset(new (std::nothrow)
                         flashrt::models::pi05::Runtime(model->exp, cfg));
    if (!a->runtime) return -5;
    if (!a->runtime->ok()) return status_code(a->runtime->status());
    a->source_model = model;
    a->images_port = images;
    a->noise_port = noise;
    a->actions_port = actions;
    a->view_order = a->runtime->manifest().vision.view_order;

    frt_model_runtime_verbs verbs{};
    verbs.struct_size = sizeof(verbs);
    verbs.set_input = set_input;
    verbs.get_output = get_output;
    verbs.prepare = prepare;
    verbs.step = step;
    verbs.last_error = last_error;

    Adapter* raw = a.release();
    frt_model_runtime_v1* m = frt_model_runtime_override_verbs(
        model, &verbs, raw, raw, nullptr, destroy_adapter);
    if (!m) {
        delete raw;
        return -1;
    }
    *out = m;
    return 0;
}

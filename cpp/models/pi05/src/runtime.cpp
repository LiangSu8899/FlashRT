#include "flashrt/cpp/models/pi05/runtime.h"

#include <cstring>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

const frt_runtime_graph_desc* find_graph(const frt_runtime_export_v1* exp,
                                         const std::string& name) {
    if (!exp || (!exp->graphs && exp->n_graphs)) return nullptr;
    for (std::uint64_t i = 0; i < exp->n_graphs; ++i) {
        const char* n = exp->graphs[i].name;
        if (n && name == n) return &exp->graphs[i];
    }
    return nullptr;
}

const frt_runtime_buffer_desc* find_buffer(const frt_runtime_export_v1* exp,
                                           const std::string& name) {
    if (!exp || (!exp->buffers && exp->n_buffers)) return nullptr;
    for (std::uint64_t i = 0; i < exp->n_buffers; ++i) {
        const char* n = exp->buffers[i].name;
        if (n && name == n) return &exp->buffers[i];
    }
    return nullptr;
}

void* find_native_stream(const frt_runtime_export_v1* exp, int stream_id) {
    if (!exp || (!exp->streams && exp->n_streams)) return nullptr;
    for (std::uint64_t i = 0; i < exp->n_streams; ++i) {
        if (exp->streams[i].stream_id == stream_id) {
            return exp->streams[i].native_handle;
        }
    }
    return nullptr;
}

modalities::TensorView device_tensor_from_buffer(
    const frt_runtime_buffer_desc* desc,
    modalities::DType dtype,
    modalities::Layout layout,
    modalities::Shape shape) {
    modalities::TensorView view;
    if (!desc || !desc->handle) return view;
    view.data = frt_buffer_dptr(desc->handle);
    view.bytes = desc->bytes;
    view.dtype = dtype;
    view.place = modalities::MemoryPlace::kDevice;
    view.layout = layout;
    view.shape = shape;
    return view;
}

bool has_tensor_override(const modalities::TensorView& view) {
    return view.data != nullptr;
}

}  // namespace

Runtime::Runtime(const frt_runtime_export_v1* exp, RuntimeConfig config)
    : exp_(exp),
      config_(std::move(config)),
      status_(modalities::Status::ok()),
      io_(1, modalities::TensorView{}, modalities::TensorView{}, {}, {},
          nullptr) {
    status_ = bind();
    if (status_.ok_status()) retain_export();
}

Runtime::~Runtime() {
    release_export();
}

void Runtime::retain_export() {
    if (exp_ && exp_->retain) exp_->retain(exp_->owner);
}

void Runtime::release_export() {
    if (exp_ && exp_->release) exp_->release(exp_->owner);
    exp_ = nullptr;
}

modalities::Status Runtime::bind() {
    if (!exp_) {
        return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                         "Pi05 Runtime requires an export");
    }
    if (exp_->abi_version != FRT_RUNTIME_ABI_VERSION ||
        exp_->struct_size < sizeof(frt_runtime_export_v1)) {
        return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                         "Pi05 Runtime export ABI mismatch");
    }
    if (!exp_->retain || !exp_->release) {
        return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                         "Pi05 Runtime export has no lifetime hooks");
    }

    const frt_runtime_graph_desc* graph = find_graph(exp_, config_.graph_name);
    if (!graph || !graph->handle) {
        return modalities::Status::error(modalities::StatusCode::kNotFound,
                                         "Pi05 Runtime graph not found: " +
                                             config_.graph_name);
    }
    graph_ = graph->handle;
    graph_key_ = graph->default_key;
    stream_id_ = graph->stream_id;

    modalities::TensorView image = config_.image_input_override;
    if (!has_tensor_override(image)) {
        const auto* b = find_buffer(exp_, config_.image_buffer_name);
        image = device_tensor_from_buffer(
            b, modalities::DType::kBFloat16, modalities::Layout::kNHWC,
            modalities::Shape{static_cast<std::uint64_t>(config_.num_views),
                              kImageSize, kImageSize, 3});
    }
    modalities::TensorView action = config_.action_output_override;
    if (!has_tensor_override(action)) {
        const auto* b = find_buffer(exp_, config_.action_buffer_name);
        action = device_tensor_from_buffer(
            b, modalities::DType::kBFloat16, modalities::Layout::kFlat,
            modalities::Shape{static_cast<std::uint64_t>(config_.chunk),
                              static_cast<std::uint64_t>(config_.model_action_dim)});
    }
    if (!image.data) {
        return modalities::Status::error(modalities::StatusCode::kNotFound,
                                         "Pi05 Runtime image buffer not found: " +
                                             config_.image_buffer_name);
    }
    if (!action.data) {
        return modalities::Status::error(modalities::StatusCode::kNotFound,
                                         "Pi05 Runtime action buffer not found: " +
                                             config_.action_buffer_name);
    }

    manifest_.vision = vision_preprocess_spec(config_.num_views);
    manifest_.action = action_postprocess_spec(
        config_.action_mean, config_.action_stddev, config_.chunk,
        config_.model_action_dim, config_.robot_action_dim);
    manifest_.graphs.infer = config_.graph_name;
    manifest_.graphs.decode_only = "decode_only";
    for (std::uint64_t i = 0; i < exp_->n_capsule_regions; ++i) {
        const auto& r = exp_->capsule_regions[i];
        families::vla::StateRegion region;
        region.name = r.name ? r.name : "";
        region.buffer = r.buffer ? frt_buffer_name(r.buffer) : "";
        region.offset = r.offset;
        region.bytes = r.bytes;
        manifest_.state_regions.push_back(std::move(region));
    }

    io_ = RuntimeIo(config_.num_views, image, action, config_.action_mean,
                    config_.action_stddev, find_native_stream(exp_, stream_id_),
                    config_.chunk,
                    config_.model_action_dim, config_.robot_action_dim);
    return modalities::Status::ok();
}

int Runtime::set_prompt(const char* text) {
    /* The adopted-export path assumes prompt/token embedding was prepared by
     * the producer before capture/export. A native Pi0.5 producer will replace
     * this with tokenizer + prompt-region binding without changing Nexus. */
    return (text == nullptr || text[0] == '\0') ? 0 : -1;
}

modalities::Status Runtime::prepare_vision(
    const std::vector<modalities::VisionFrame>& frames) {
    if (!ok()) return status_;
    return io_.prepare_vision(frames);
}

int Runtime::replay_tick() {
    if (!ok()) return -1;
    ReplayFn fn = config_.replay_fn ? config_.replay_fn : default_replay;
    return fn(graph_, graph_key_, stream_id_, config_.replay_user);
}

modalities::Status Runtime::read_actions(std::vector<float>* robot_actions) {
    if (!ok()) return status_;
    return io_.read_actions(robot_actions);
}

int Runtime::default_replay(frt_graph graph, frt_shape_key key,
                            int stream_id, void* user) {
    (void)user;
    return frt_graph_replay(graph, key, stream_id);
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

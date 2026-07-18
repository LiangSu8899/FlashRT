#include "flashrt/cpp/models/pi05/runtime.h"

#include "flashrt/cpp/models/pi05/model/prompt_embed.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

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

bool checked_tensor_bytes(std::uint64_t rows, std::uint64_t cols,
                          modalities::DType dtype, std::uint64_t* out) {
    const std::uint64_t width = modalities::dtype_size(dtype);
    if (!out || !width ||
        (cols && rows > std::numeric_limits<std::uint64_t>::max() / cols)) {
        return false;
    }
    const std::uint64_t elements = rows * cols;
    if (width && elements > std::numeric_limits<std::uint64_t>::max() / width) {
        return false;
    }
    *out = elements * width;
    return true;
}

modalities::Status validate_prompt_matrix(
    const modalities::TensorView& view, const char* name,
    std::uint64_t rows, std::uint64_t cols) {
    if (!view.data) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            std::string(name) + " has null data");
    }
    if (view.place != modalities::MemoryPlace::kHost &&
        view.place != modalities::MemoryPlace::kHostPinned &&
        view.place != modalities::MemoryPlace::kDevice) {
        return modalities::Status::error(
            modalities::StatusCode::kUnsupported,
            std::string(name) + " memory place is unsupported");
    }
    if (view.dtype == modalities::DType::kUInt8 ||
        view.layout != modalities::Layout::kFlat || view.shape.rank != 2 ||
        view.shape.dims[0] != rows || view.shape.dims[1] != cols) {
        return modalities::Status::error(
            modalities::StatusCode::kShapeMismatch,
            std::string(name) + " shape or dtype is invalid");
    }
    std::uint64_t bytes = 0;
    if (!checked_tensor_bytes(rows, cols, view.dtype, &bytes)) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            std::string(name) + " byte size overflows");
    }
    if (view.bytes < bytes) {
        return modalities::Status::error(
            modalities::StatusCode::kInsufficientStorage,
            std::string(name) + " storage is too small");
    }
    return modalities::Status::ok();
}

}  // namespace

struct Runtime::PromptState {
    modalities::SentencePieceTokenizer tokenizer;
    modalities::TextEmbeddingStaging embedding_staging;
    PromptEmbeddingSpec spec;
    modalities::TensorView embedding_table;
    modalities::TensorView embedding_output;
    modalities::Status status;
    std::vector<std::int32_t> token_ids;
    std::vector<float> normalized_state;
    std::string task_workspace;
    std::string formatted_workspace;
    std::size_t max_task_bytes = 0;
    std::uint64_t current_len = 0;
    bool enabled = false;

    ~PromptState() {
        modalities::text_embedding_staging_destroy(&embedding_staging);
    }
};

Runtime::Runtime(const frt_runtime_export_v1* exp, RuntimeConfig config)
    : exp_(exp),
      config_(std::move(config)),
      status_(modalities::Status::ok()),
      io_(1, modalities::TensorView{}, modalities::TensorView{}, {}, {},
          nullptr),
      prompt_(new PromptState()) {
    status_ = bind();
    if (status_.ok_status()) retain_export();
}

Runtime::~Runtime() {
    prompt_.reset();
    modalities::action_staging_destroy(&action_staging_);
    modalities::vision_staging_destroy(&staging_);
    release_export();
}

void Runtime::retain_export() {
    if (exp_ && exp_->retain) {
        exp_->retain(exp_->owner);
        export_retained_ = true;
    }
}

void Runtime::release_export() {
    if (export_retained_ && exp_ && exp_->release) exp_->release(exp_->owner);
    export_retained_ = false;
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
            b, config_.image_dtype, modalities::Layout::kNHWC,
            modalities::Shape{static_cast<std::uint64_t>(config_.num_views),
                              kImageSize, kImageSize, 3});
    }
    modalities::TensorView action = config_.action_output_override;
    if (!has_tensor_override(action)) {
        const auto* b = find_buffer(exp_, config_.action_buffer_name);
        action = device_tensor_from_buffer(
            b, config_.action_dtype, modalities::Layout::kFlat,
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
    manifest_.vision.output_dtype = config_.image_dtype;
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

    /* Persistent staging: the per-frame hot path never allocates. Only the
     * device path needs it (host-tensor overrides preprocess on the CPU). */
    modalities::VisionStaging* staging = nullptr;
    if (image.place == modalities::MemoryPlace::kDevice) {
        if (config_.max_frame_width <= 0 || config_.max_frame_height <= 0) {
            return modalities::Status::error(
                modalities::StatusCode::kInvalidArgument,
                "Pi05 frame staging dimensions must be positive");
        }
        const std::uint64_t width =
            static_cast<std::uint64_t>(config_.max_frame_width);
        const std::uint64_t height =
            static_cast<std::uint64_t>(config_.max_frame_height);
        if (width > std::numeric_limits<std::uint64_t>::max() / height / 4ull) {
            return modalities::Status::error(
                modalities::StatusCode::kInvalidArgument,
                "Pi05 frame staging capacity overflows");
        }
        const std::uint64_t max_frame_bytes = width * height * 4ull;
        modalities::Status st = modalities::vision_staging_create(
            &staging_, static_cast<std::uint32_t>(config_.num_views),
            max_frame_bytes);
        if (!st.ok_status()) return st;
        staging = &staging_;
    }

    modalities::ActionStaging* action_staging = nullptr;
    if (action.place == modalities::MemoryPlace::kDevice) {
        modalities::ActionPostprocessSpec action_spec = action_postprocess_spec(
            config_.action_mean, config_.action_stddev, config_.chunk,
            config_.model_action_dim, config_.robot_action_dim);
        const std::uint64_t bytes =
            modalities::required_action_output_bytes(action_spec, action.dtype);
        if (!bytes) {
            return modalities::Status::error(
                modalities::StatusCode::kInvalidArgument,
                "Pi05 action staging dimensions are invalid");
        }
        modalities::Status st = modalities::action_staging_create(
            &action_staging_, bytes);
        if (!st.ok_status()) return st;
        action_staging = &action_staging_;
    }

    io_ = RuntimeIo(config_.num_views, image, action, config_.action_mean,
                    config_.action_stddev, find_native_stream(exp_, stream_id_),
                    config_.chunk, config_.model_action_dim,
                    config_.robot_action_dim, config_.image_dtype, staging,
                    action_staging, config_.strict_rgb8);
    return bind_prompt_staging();
}

int Runtime::set_prompt(const char* text) {
    return set_prompt_state(text, nullptr, 0);
}

int Runtime::set_prompt_state(const char* text, const float* state,
                              std::uint64_t n_state) {
    if (!prompt_->enabled) {
        return (text == nullptr || text[0] == '\0') ? 0 : -1;
    }
    if (!text || (!state && n_state) || (state && !n_state)) {
        prompt_->status = modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "prompt/state arguments are invalid");
        return -1;
    }
    const std::size_t text_bytes = std::strlen(text);
    if (text_bytes > prompt_->max_task_bytes) {
        prompt_->status = modalities::Status::error(
            modalities::StatusCode::kShapeMismatch,
            "prompt text exceeds the configured hot-path capacity");
        return -1;
    }
    prompt_->task_workspace.assign(text, text_bytes);

    const float* state_for_prompt = state;
    if (state) {
        if (!state_normalization_enabled()) {
            prompt_->status = modalities::Status::error(
                modalities::StatusCode::kInvalidArgument,
                "state updates require normalization statistics");
            return -1;
        }
        if (n_state != config_.state_q01.size()) {
            prompt_->status = modalities::Status::error(
                modalities::StatusCode::kShapeMismatch,
                "state dimension does not match norm stats");
            return -1;
        }
        for (std::uint64_t i = 0; i < n_state; ++i) {
            if (!std::isfinite(state[i])) {
                prompt_->status = modalities::Status::error(
                    modalities::StatusCode::kInvalidArgument,
                    "state contains non-finite data");
                return -1;
            }
        }
    }
    if (state) {
        for (std::uint64_t i = 0; i < n_state; ++i) {
            const float lo = config_.state_q01[i];
            const float hi = config_.state_q99[i];
            prompt_->normalized_state[i] =
                ((state[i] - lo) / (hi - lo + 1e-6f)) * 2.0f - 1.0f;
        }
        state_for_prompt = prompt_->normalized_state.data();
    }

    prompt_->status = embed_prompt(
        prompt_->tokenizer, prompt_->spec, prompt_->task_workspace,
        state_for_prompt, n_state, prompt_->embedding_table,
        prompt_->embedding_output, &prompt_->token_ids, &prompt_->current_len,
        find_native_stream(exp_, stream_id_),
        prompt_->embedding_output.place == modalities::MemoryPlace::kDevice
            ? &prompt_->embedding_staging
            : nullptr,
        &prompt_->formatted_workspace);
    if (prompt_->status.ok_status() && config_.prompt_length_update_fn) {
        const int rc = config_.prompt_length_update_fn(
            config_.prompt_length_update_user, prompt_->current_len);
        if (rc != 0) {
            prompt_->status = modalities::Status::error(
                modalities::StatusCode::kBackend,
                "prompt length device update failed");
        }
    }
    return prompt_->status.ok_status() ? 0 : -1;
}

const modalities::Status& Runtime::prompt_status() const {
    return prompt_->status;
}

bool Runtime::prompt_staging_enabled() const {
    return prompt_->enabled;
}

bool Runtime::state_normalization_enabled() const {
    return !config_.state_q01.empty() &&
           config_.state_q01.size() == config_.state_q99.size();
}

std::uint64_t Runtime::current_prompt_len() const {
    return prompt_->current_len;
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

modalities::Status Runtime::bind_prompt_staging() {
    const bool any =
        !config_.prompt_tokenizer_model_path.empty() ||
        config_.prompt_embedding_table.data ||
        config_.prompt_embedding_output.data || config_.prompt_vocab_size ||
        config_.prompt_hidden_dim || config_.prompt_max_tokens ||
        !config_.state_q01.empty() || !config_.state_q99.empty() ||
        config_.prompt_length_update_fn;
    if (!any) {
        prompt_->status = modalities::Status::ok();
        return modalities::Status::ok();
    }
    if (config_.prompt_tokenizer_model_path.empty() ||
        !config_.prompt_embedding_table.data ||
        !config_.prompt_embedding_output.data || !config_.prompt_vocab_size ||
        !config_.prompt_hidden_dim || !config_.prompt_max_tokens) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "incomplete Pi05 prompt staging config");
    }
    if (config_.prompt_vocab_size <= 108 ||
        !std::isfinite(config_.prompt_embedding_scale) ||
        config_.prompt_embedding_scale < 0.0f) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "invalid Pi05 prompt embedding config");
    }
    if (config_.state_q01.size() != config_.state_q99.size()) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "Pi05 state quantile dimensions do not match");
    }
    for (std::size_t i = 0; i < config_.state_q01.size(); ++i) {
        if (!std::isfinite(config_.state_q01[i]) ||
            !std::isfinite(config_.state_q99[i]) ||
            config_.state_q99[i] <= config_.state_q01[i]) {
            return modalities::Status::error(
                modalities::StatusCode::kInvalidArgument,
                "Pi05 state quantiles are invalid");
        }
    }

    modalities::Status st = validate_prompt_matrix(
        config_.prompt_embedding_table, "prompt embedding table",
        config_.prompt_vocab_size, config_.prompt_hidden_dim);
    if (!st.ok_status()) return st;
    st = validate_prompt_matrix(
        config_.prompt_embedding_output, "prompt embedding output",
        config_.prompt_max_tokens, config_.prompt_hidden_dim);
    if (!st.ok_status()) return st;

    const auto table_place = config_.prompt_embedding_table.place;
    const auto output_place = config_.prompt_embedding_output.place;
    if (output_place == modalities::MemoryPlace::kDevice) {
        if (table_place != modalities::MemoryPlace::kDevice) {
            return modalities::Status::error(
                modalities::StatusCode::kUnsupported,
                "device prompt output requires a device embedding table");
        }
#if !defined(FLASHRT_CPP_WITH_CUDA_STAGING) || \
    !defined(FLASHRT_CPP_WITH_CUDA_KERNELS)
        return modalities::Status::error(
            modalities::StatusCode::kUnsupported,
            "device prompt embedding requires CUDA staging and kernels");
#endif
    } else if (table_place != modalities::MemoryPlace::kHost &&
               table_place != modalities::MemoryPlace::kHostPinned) {
        return modalities::Status::error(
            modalities::StatusCode::kUnsupported,
            "host prompt output requires a host embedding table");
    }

    prompt_->status =
        prompt_->tokenizer.load_model(config_.prompt_tokenizer_model_path);
    if (!prompt_->status.ok_status()) return prompt_->status;
    if (prompt_->tokenizer.vocab_size() != config_.prompt_vocab_size) {
        return modalities::Status::error(
            modalities::StatusCode::kShapeMismatch,
            "tokenizer vocabulary does not match the embedding table");
    }

    prompt_->embedding_table = config_.prompt_embedding_table;
    prompt_->embedding_output = config_.prompt_embedding_output;
    prompt_->spec.vocab_size = config_.prompt_vocab_size;
    prompt_->spec.hidden_dim = config_.prompt_hidden_dim;
    prompt_->spec.max_tokens = config_.prompt_max_tokens;
    prompt_->spec.scale = config_.prompt_embedding_scale > 0.0f
                              ? config_.prompt_embedding_scale
                              : std::sqrt(static_cast<float>(
                                    config_.prompt_hidden_dim));

    constexpr std::size_t kStateCharsPerValue = 5;
    constexpr std::size_t kPromptCharsPerToken = 8;
    constexpr std::size_t kFormatOverhead = 32;
    if (config_.state_q01.size() >
        (std::numeric_limits<std::size_t>::max() - kFormatOverhead) /
            kStateCharsPerValue) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "state workspace capacity overflows size_t");
    }
    const std::size_t state_bytes =
        config_.state_q01.size() * kStateCharsPerValue + kFormatOverhead;
    if (config_.prompt_max_tokens >
        (std::numeric_limits<std::size_t>::max() - state_bytes) /
            kPromptCharsPerToken) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "prompt workspace capacity overflows size_t");
    }
    const std::size_t max_prompt_bytes =
        static_cast<std::size_t>(config_.prompt_max_tokens) *
        kPromptCharsPerToken;
    prompt_->max_task_bytes = max_prompt_bytes;
    prompt_->task_workspace.reserve(max_prompt_bytes);
    prompt_->formatted_workspace.reserve(max_prompt_bytes + state_bytes);
    prompt_->token_ids.reserve(
        static_cast<std::size_t>(config_.prompt_max_tokens) + 1);
    prompt_->normalized_state.resize(config_.state_q01.size());
    prompt_->tokenizer.reserve(config_.prompt_max_tokens);
    if (output_place == modalities::MemoryPlace::kDevice) {
        prompt_->status = modalities::text_embedding_staging_create(
            &prompt_->embedding_staging, config_.prompt_max_tokens);
        if (!prompt_->status.ok_status()) return prompt_->status;
    }
    prompt_->enabled = true;
    return modalities::Status::ok();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

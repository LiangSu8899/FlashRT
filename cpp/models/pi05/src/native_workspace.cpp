#include "flashrt/cpp/models/pi05/native_workspace.h"

#ifdef FLASHRT_CPP_WITH_CUDA_STAGING
#include <cuda_runtime_api.h>
#endif

#include <limits>
#include <cmath>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

bool element_count(std::initializer_list<std::uint64_t> shape,
                   std::size_t* out) {
    std::size_t count = 1;
    for (std::uint64_t dim : shape) {
        if (!dim || dim > std::numeric_limits<std::size_t>::max() ||
            count > std::numeric_limits<std::size_t>::max() /
                        static_cast<std::size_t>(dim)) {
            return false;
        }
        count *= static_cast<std::size_t>(dim);
    }
    if (out) *out = count;
    return true;
}

}  // namespace

modalities::Status NativeWorkspace::add(
    const std::string& name,
    std::initializer_list<std::uint64_t> shape,
    modalities::DType dtype) {
    if (!ctx_ || name.empty() || buffers_.find(name) != buffers_.end()) {
        return invalid("native workspace buffer definition is invalid");
    }
    std::size_t elements = 0;
    const std::size_t width = modalities::dtype_size(dtype);
    if (!width || !element_count(shape, &elements) ||
        elements > std::numeric_limits<std::size_t>::max() / width) {
        return invalid("native workspace buffer shape is invalid");
    }
    const std::size_t bytes = elements * width;
    frt_buffer buffer = frt_buffer_alloc(ctx_, name.c_str(), bytes);
    if (!buffer) return backend("native workspace allocation failed");
    buffers_.emplace(name, NativeWorkspaceBuffer{
                               buffer, std::vector<std::uint64_t>(shape),
                               dtype, false});
    ++allocation_count_;
    allocated_bytes_ += bytes;
    return modalities::Status::ok();
}

modalities::Status NativeWorkspace::add_alias(
    const std::string& name,
    const std::string& source_name,
    std::initializer_list<std::uint64_t> shape) {
    if (name.empty() || buffers_.find(name) != buffers_.end()) {
        return invalid("native workspace alias definition is invalid");
    }
    const auto source = buffers_.find(source_name);
    if (source == buffers_.end() || !source->second.buffer) {
        return invalid("native workspace alias source was not found");
    }
    std::size_t elements = 0;
    const std::size_t width = modalities::dtype_size(source->second.dtype);
    if (!width || !element_count(shape, &elements) ||
        elements > std::numeric_limits<std::size_t>::max() / width ||
        elements * width !=
            frt_buffer_bytes(source->second.buffer)) {
        return invalid("native workspace alias shape does not match source");
    }
    buffers_.emplace(name, NativeWorkspaceBuffer{
                               source->second.buffer,
                               std::vector<std::uint64_t>(shape),
                               source->second.dtype, true});
    return modalities::Status::ok();
}

modalities::Status NativeWorkspace::initialize_rms_ones() {
#ifndef FLASHRT_CPP_WITH_CUDA_STAGING
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "native workspace initialization requires the CUDA build");
#else
    for (const char* name : {"encoder_rms_ones", "decoder_rms_ones"}) {
        const NativeWorkspaceBuffer* target = find(name);
        if (!target) return invalid("native RMS buffer was not allocated");
        std::vector<std::uint16_t> ones(
            target->shape[0], modalities::float_to_bfloat16(1.0f));
        const cudaError_t rc = cudaMemcpy(
            frt_buffer_dptr(target->buffer), ones.data(),
            ones.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
        if (rc != cudaSuccess) return backend("native RMS upload failed");
    }
    return modalities::Status::ok();
#endif
}

modalities::Status NativeWorkspace::initialize_rope() {
#ifndef FLASHRT_CPP_WITH_CUDA_STAGING
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "native RoPE initialization requires the CUDA build");
#else
    const int max_positions = encoder_sequence_ + chunk_size_;
    rope_table_.resize(static_cast<std::size_t>(max_positions) * 256);
    for (int position = 0; position < max_positions; ++position) {
        const std::size_t row = static_cast<std::size_t>(position) * 256;
        for (int i = 0; i < 128; ++i) {
            const double exponent = static_cast<double>(2 * i) / 256.0;
            const double inverse_frequency =
                1.0 / std::pow(10000.0, exponent);
            const double phase =
                static_cast<double>(position) * inverse_frequency;
            rope_table_[row + 2 * i] =
                modalities::float_to_bfloat16(
                    static_cast<float>(std::cos(phase)));
            rope_table_[row + 2 * i + 1] =
                modalities::float_to_bfloat16(
                    static_cast<float>(std::sin(phase)));
        }
    }
    const NativeWorkspaceBuffer* encoder = find("encoder_rope_weights");
    if (!encoder) return invalid("encoder RoPE buffer was not allocated");
    const std::size_t encoder_bytes =
        static_cast<std::size_t>(encoder_sequence_) * 256 *
        sizeof(std::uint16_t);
    const cudaError_t rc = cudaMemcpy(
        frt_buffer_dptr(encoder->buffer), rope_table_.data(), encoder_bytes,
        cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return backend("encoder RoPE upload failed");
    return update_decoder_rope(0);
#endif
}

modalities::Status NativeWorkspace::update_decoder_rope(int prompt_tokens) {
    if (prompt_tokens < 0 || prompt_tokens > max_prompt_tokens_ ||
        rope_table_.empty()) {
        return invalid("Pi0.5 decoder RoPE prompt length is invalid");
    }
#ifndef FLASHRT_CPP_WITH_CUDA_STAGING
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "decoder RoPE update requires the CUDA build");
#else
    const NativeWorkspaceBuffer* decoder = find("decoder_rope_weights");
    if (!decoder) return invalid("decoder RoPE buffer was not allocated");
    const std::size_t start =
        static_cast<std::size_t>(encoder_vision_sequence_ + prompt_tokens) *
        256;
    const std::size_t elements =
        static_cast<std::size_t>(chunk_size_) * 256;
    if (start > rope_table_.size() ||
        elements > rope_table_.size() - start) {
        return invalid("decoder RoPE slice exceeds the generated table");
    }
    const cudaError_t rc = cudaMemcpy(
        frt_buffer_dptr(decoder->buffer), rope_table_.data() + start,
        elements * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    return rc == cudaSuccess
               ? modalities::Status::ok()
               : backend("decoder RoPE upload failed");
#endif
}

modalities::Status NativeWorkspace::expand_vision_position_embedding(
    const NativeDeviceWeightStore& weights) {
    const NativeDeviceWeight* source =
        weights.find("vision_position_embedding");
    const NativeWorkspaceBuffer* destination =
        find("vision_pos_embed_expanded");
    if (!source || !destination ||
        source->dtype != NativeWeightDType::kBf16 ||
        source->shape != std::vector<std::uint64_t>({256, 1152})) {
        return invalid("vision position embedding source is invalid");
    }
#ifndef FLASHRT_CPP_WITH_CUDA_STAGING
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "position embedding expansion requires the CUDA build");
#else
    const std::size_t view_bytes = 256 * 1152 * sizeof(std::uint16_t);
    if (frt_buffer_bytes(destination->buffer) !=
        static_cast<std::size_t>(num_views_) * view_bytes) {
        return invalid("expanded position embedding buffer size is invalid");
    }
    for (int view = 0; view < num_views_; ++view) {
        auto* target = static_cast<unsigned char*>(
                           frt_buffer_dptr(destination->buffer)) +
                       static_cast<std::size_t>(view) * view_bytes;
        const cudaError_t rc = cudaMemcpy(
            target, frt_buffer_dptr(source->buffer), view_bytes,
            cudaMemcpyDeviceToDevice);
        if (rc != cudaSuccess) {
            return backend("vision position embedding expansion failed");
        }
    }
    return modalities::Status::ok();
#endif
}

modalities::Status NativeWorkspace::allocate(
    const NativeWorkspaceConfig& config) {
    if (!ctx_ || !buffers_.empty() || config.num_views < 1 ||
        config.num_views > 3 || config.max_prompt_tokens <= 0 ||
        config.max_prompt_tokens > std::numeric_limits<int>::max() - 768 ||
        config.chunk_size <= 0 || config.num_steps <= 0 ||
        (config.vision_pool_factor != 1 &&
         config.vision_pool_factor != 2 &&
         config.vision_pool_factor != 4)) {
        return invalid("Pi0.5 native workspace configuration is invalid");
    }
    const int pool_area =
        config.vision_pool_factor * config.vision_pool_factor;
    num_views_ = config.num_views;
    max_prompt_tokens_ = config.max_prompt_tokens;
    chunk_size_ = config.chunk_size;
    vision_sequence_ = config.num_views * 256;
    encoder_vision_sequence_ = vision_sequence_ / pool_area;
    encoder_sequence_ =
        encoder_vision_sequence_ + config.max_prompt_tokens;
    const std::uint64_t nv = static_cast<std::uint64_t>(config.num_views);
    const std::uint64_t vs = static_cast<std::uint64_t>(vision_sequence_);
    const std::uint64_t vs_enc =
        static_cast<std::uint64_t>(encoder_vision_sequence_);
    const std::uint64_t es = static_cast<std::uint64_t>(encoder_sequence_);
    const std::uint64_t ds = static_cast<std::uint64_t>(config.chunk_size);
    const std::uint64_t steps = static_cast<std::uint64_t>(config.num_steps);
    modalities::Status st;
#define FRT_ADD(...)                      \
    do {                                  \
        st = add(__VA_ARGS__);             \
        if (!st.ok_status()) return st;    \
    } while (false)
    FRT_ADD("observation_images_normalized", {nv, 224, 224, 3},
            modalities::DType::kBFloat16);
    FRT_ADD("vision_x", {vs, 1152}, modalities::DType::kBFloat16);
    FRT_ADD("vision_x_norm", {vs, 1152}, modalities::DType::kBFloat16);
    if (config.vision_pool_factor == 1) {
        st = add_alias("vision_x_pooled", "vision_x", {vs_enc, 1152});
        if (!st.ok_status()) return st;
    } else {
        FRT_ADD("vision_x_pooled", {vs_enc, 1152},
                modalities::DType::kBFloat16);
    }
    FRT_ADD("vision_QKV", {vs, 3456}, modalities::DType::kBFloat16);
    FRT_ADD("vision_hidden", {vs, 4304}, modalities::DType::kBFloat16);
    FRT_ADD("vision_pos_embed_expanded", {vs, 1152},
            modalities::DType::kBFloat16);
    FRT_ADD("vision_patches", {vs, 588}, modalities::DType::kBFloat16);

    FRT_ADD("encoder_rope_weights", {es, 256},
            modalities::DType::kBFloat16);
    FRT_ADD("prompt_embedding",
            {static_cast<std::uint64_t>(max_prompt_tokens_), 2048},
            modalities::DType::kBFloat16);
    FRT_ADD("encoder_x", {es, 2048}, modalities::DType::kBFloat16);
    FRT_ADD("encoder_x_norm", {es, 2048}, modalities::DType::kBFloat16);
    FRT_ADD("encoder_QKV", {es, 2560}, modalities::DType::kBFloat16);
    FRT_ADD("encoder_hidden", {es, 16384}, modalities::DType::kBFloat16);
    FRT_ADD("encoder_gate_merged", {es, 32768},
            modalities::DType::kBFloat16);
    FRT_ADD("encoder_gate_buf", {es, 16384},
            modalities::DType::kBFloat16);
    FRT_ADD("encoder_rms_ones", {2048}, modalities::DType::kBFloat16);

    FRT_ADD("decoder_rope_weights", {ds, 256},
            modalities::DType::kBFloat16);
    FRT_ADD("decoder_x", {ds, 1024}, modalities::DType::kBFloat16);
    FRT_ADD("decoder_action_buf", {ds, 32}, modalities::DType::kBFloat16);
    FRT_ADD("decoder_time_emb", {steps, ds, 1024},
            modalities::DType::kBFloat16);
    FRT_ADD("decoder_style_attn", {steps, 18, ds, 3072},
            modalities::DType::kBFloat16);
    FRT_ADD("decoder_style_ffn", {steps, 18, ds, 3072},
            modalities::DType::kBFloat16);
    FRT_ADD("decoder_style_final", {steps, ds, 3072},
            modalities::DType::kBFloat16);
    FRT_ADD("decoder_QKV", {ds, 2560}, modalities::DType::kBFloat16);
    FRT_ADD("decoder_hidden", {ds, 4096}, modalities::DType::kBFloat16);
    FRT_ADD("decoder_gate_merged", {ds, 8192},
            modalities::DType::kBFloat16);
    FRT_ADD("decoder_gate_buf", {ds, 4096},
            modalities::DType::kBFloat16);
    FRT_ADD("diffusion_noise", {ds, 32}, modalities::DType::kBFloat16);
    FRT_ADD("rtc_prev_action_chunk", {ds, 32},
            modalities::DType::kBFloat16);
    FRT_ADD("rtc_prefix_weights", {ds}, modalities::DType::kFloat32);
    FRT_ADD("rtc_guidance_weight", {1}, modalities::DType::kFloat32);
    FRT_ADD("x_normed_buf", {ds, 1024}, modalities::DType::kBFloat16);
    FRT_ADD("gate_buf", {ds, 1024}, modalities::DType::kBFloat16);
    FRT_ADD("decoder_rms_ones", {1024}, modalities::DType::kBFloat16);
#undef FRT_ADD
    st = initialize_rms_ones();
    if (!st.ok_status()) return st;
    return initialize_rope();
}

const NativeWorkspaceBuffer* NativeWorkspace::find(
    const std::string& name) const {
    const auto it = buffers_.find(name);
    return it == buffers_.end() ? nullptr : &it->second;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#include "flashrt/cpp/models/pi05/support/native_workspace.h"

#include "flashrt/cpp/models/pi05/model/dims.h"
#include "flashrt/cpp/models/pi05/support/native_rope.h"

#ifdef FLASHRT_CPP_WITH_CUDA_STAGING
#include <cuda_runtime_api.h>
#endif

#include <cmath>
#include <limits>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

constexpr int kRopeWidth = kPi05ModelDims.encoder_head_dim;
static_assert(kPi05ModelDims.encoder_head_dim ==
                  kPi05ModelDims.decoder_head_dim,
              "PI0.5 encoder and decoder RoPE widths must match");

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

bool element_count(const std::vector<std::uint64_t>& shape,
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
    const std::vector<std::uint64_t>& shape,
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
    buffers_.emplace(
        name, NativeWorkspaceBuffer{buffer, shape, dtype, false});
    ++allocation_count_;
    allocated_bytes_ += bytes;
    return modalities::Status::ok();
}

modalities::Status NativeWorkspace::add_alias(
    const std::string& name,
    const std::string& source_name,
    const std::vector<std::uint64_t>& shape) {
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
    buffers_.emplace(
        name, NativeWorkspaceBuffer{
                  source->second.buffer, shape, source->second.dtype, true});
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
        if (target->shape.size() != 1 ||
            (target->dtype != modalities::DType::kBFloat16 &&
             target->dtype != modalities::DType::kFloat16)) {
            return invalid("native RMS buffer layout is invalid");
        }
        const std::uint16_t one =
            target->dtype == modalities::DType::kFloat16
                ? modalities::float_to_float16(1.0f)
                : modalities::float_to_bfloat16(1.0f);
        std::vector<std::uint16_t> ones(target->shape[0], one);
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
    const NativeWorkspaceBuffer* encoder = find("encoder_rope_weights");
    if (!encoder || encoder->dtype != activation_dtype_) {
        return invalid("encoder RoPE buffer was not allocated");
    }
    if (activation_dtype_ == modalities::DType::kFloat16) {
        rope_table_.clear();
        modalities::Status st = generate_native_rope_f16(
            frt_buffer_dptr(encoder->buffer), 0, encoder_sequence_, 0);
        return st.ok_status() ? update_decoder_rope(0) : st;
    }
    if (activation_dtype_ != modalities::DType::kBFloat16) {
        return invalid("native RoPE activation dtype is unsupported");
    }

    const int max_positions = encoder_sequence_ + chunk_size_;
    rope_table_.resize(
        static_cast<std::size_t>(max_positions) * kRopeWidth);
    for (int position = 0; position < max_positions; ++position) {
        const std::size_t row =
            static_cast<std::size_t>(position) * kRopeWidth;
        for (int i = 0; i < kRopeWidth / 2; ++i) {
            const double exponent =
                static_cast<double>(2 * i) / kRopeWidth;
            const double inverse_frequency =
                1.0 / std::pow(10000.0, exponent);
            const double phase =
                static_cast<double>(position) * inverse_frequency;
            rope_table_[row + 2 * i] = modalities::float_to_bfloat16(
                static_cast<float>(std::cos(phase)));
            rope_table_[row + 2 * i + 1] = modalities::float_to_bfloat16(
                static_cast<float>(std::sin(phase)));
        }
    }
    const std::size_t encoder_bytes =
        static_cast<std::size_t>(encoder_sequence_) * kRopeWidth *
        sizeof(std::uint16_t);
    const cudaError_t rc = cudaMemcpy(
        frt_buffer_dptr(encoder->buffer), rope_table_.data(), encoder_bytes,
        cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return backend("encoder RoPE upload failed");
    return update_decoder_rope(0);
#endif
}

modalities::Status NativeWorkspace::set_fixed_prompt_length(
    int prompt_tokens) {
    if (!fixed_prompt_controls_) {
        return update_decoder_rope(prompt_tokens);
    }
    if (prompt_tokens < 0 || prompt_tokens > max_prompt_tokens_ ||
        !prompt_embedding_buffer_) {
        return invalid("fixed prompt length is invalid");
    }
    const int rounded_prompt = prompt_tokens + (prompt_tokens & 1);
    if (rounded_prompt > max_prompt_tokens_) {
        return invalid("fixed prompt length exceeds its even capacity");
    }
#ifndef FLASHRT_CPP_WITH_CUDA_STAGING
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "fixed prompt update requires the CUDA build");
#else
    if (rounded_prompt != prompt_tokens && prompt_tokens > 0) {
        const std::size_t row_bytes =
            kPi05ModelDims.embedding_width * sizeof(std::uint16_t);
        auto* base = static_cast<unsigned char*>(
            frt_buffer_dptr(prompt_embedding_buffer_));
        const cudaError_t copy_rc = cudaMemcpy(
            base + static_cast<std::size_t>(prompt_tokens) * row_bytes,
            base + static_cast<std::size_t>(prompt_tokens - 1) * row_bytes,
            row_bytes, cudaMemcpyDeviceToDevice);
        if (copy_rc != cudaSuccess) {
            return backend("prompt padding copy failed");
        }
    }
    const std::int32_t valid = encoder_vision_sequence_ + rounded_prompt;
    const std::int32_t values[] = {valid, valid + chunk_size_, valid};
    for (int i = 0; i < 3; ++i) {
        if (!prompt_length_buffers_[i] ||
            cudaMemcpy(frt_buffer_dptr(prompt_length_buffers_[i]), &values[i],
                       sizeof(values[i]), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            return backend("fixed prompt control upload failed");
        }
    }
    return update_decoder_rope(rounded_prompt);
#endif
}

modalities::Status NativeWorkspace::update_decoder_rope(
    int prompt_tokens) {
    if (prompt_tokens < 0 || prompt_tokens > max_prompt_tokens_) {
        return invalid("Pi0.5 decoder RoPE prompt length is invalid");
    }
#ifndef FLASHRT_CPP_WITH_CUDA_STAGING
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "decoder RoPE update requires the CUDA build");
#else
    if (!decoder_rope_buffer_) {
        return invalid("decoder RoPE buffer was not allocated");
    }
    if (activation_dtype_ == modalities::DType::kFloat16) {
        return generate_native_rope_f16(
            frt_buffer_dptr(decoder_rope_buffer_),
            encoder_vision_sequence_ + prompt_tokens, chunk_size_, 0);
    }
    if (activation_dtype_ != modalities::DType::kBFloat16 ||
        rope_table_.empty()) {
        return invalid("decoder RoPE activation dtype is unsupported");
    }
    const std::size_t start =
        static_cast<std::size_t>(encoder_vision_sequence_ + prompt_tokens) *
        kRopeWidth;
    const std::size_t elements =
        static_cast<std::size_t>(chunk_size_) * kRopeWidth;
    if (start > rope_table_.size() ||
        elements > rope_table_.size() - start) {
        return invalid("decoder RoPE slice exceeds the generated table");
    }
    const cudaError_t rc = cudaMemcpy(
        frt_buffer_dptr(decoder_rope_buffer_), rope_table_.data() + start,
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
    if (!destination) {
        return invalid("vision position embedding destination is invalid");
    }

    NativeWeightDType expected_weight;
    if (destination->dtype == modalities::DType::kFloat16) {
        expected_weight = NativeWeightDType::kFloat16;
    } else if (destination->dtype == modalities::DType::kBFloat16) {
        expected_weight = NativeWeightDType::kBf16;
    } else {
        return invalid("vision position embedding dtype is invalid");
    }
    if (!source || source->dtype != expected_weight ||
        source->shape != std::vector<std::uint64_t>({
                             kPi05ModelDims.vision_tokens_per_view,
                             kPi05ModelDims.vision_width})) {
        return invalid("vision position embedding source is invalid");
    }
#ifndef FLASHRT_CPP_WITH_CUDA_STAGING
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "position embedding expansion requires the CUDA build");
#else
    const std::size_t view_bytes =
        static_cast<std::size_t>(
            kPi05ModelDims.vision_tokens_per_view) *
        kPi05ModelDims.vision_width * sizeof(std::uint16_t);
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
    const NativeWorkspaceConfig& config,
    const NativeWorkspaceRequirements& requirements) {
    if (!ctx_ || !buffers_.empty() ||
        (requirements.activation_dtype != modalities::DType::kFloat16 &&
         requirements.activation_dtype != modalities::DType::kBFloat16)) {
        return invalid("Pi0.5 native workspace configuration is invalid");
    }

    Pi05ShapeConfig shape_config;
    shape_config.num_views = config.num_views;
    shape_config.max_prompt_tokens = config.max_prompt_tokens;
    shape_config.chunk = config.chunk_size;
    shape_config.num_steps = config.num_steps;
    shape_config.vision_pool_factor = config.vision_pool_factor;
    shape_config.state_dim = 1;
    shape_config.robot_action_dim = kPi05ModelDims.action_width;
    Pi05ResolvedShape shape;
    modalities::Status st = resolve_pi05_shape(shape_config, &shape);
    if (!st.ok_status()) return st;

    num_views_ = shape.num_views;
    max_prompt_tokens_ = shape.max_prompt_tokens;
    chunk_size_ = shape.chunk;
    num_steps_ = shape.num_steps;
    activation_dtype_ = requirements.activation_dtype;
    fixed_prompt_controls_ = requirements.fixed_prompt_controls;
    vision_sequence_ = shape.vision_sequence;
    encoder_vision_sequence_ = shape.encoder_vision_sequence;
    encoder_sequence_ = shape.encoder_sequence;

    const std::uint64_t nv = static_cast<std::uint64_t>(num_views_);
    const std::uint64_t vs = static_cast<std::uint64_t>(vision_sequence_);
    const std::uint64_t vs_enc =
        static_cast<std::uint64_t>(encoder_vision_sequence_);
    const std::uint64_t es = static_cast<std::uint64_t>(encoder_sequence_);
    const std::uint64_t ds = static_cast<std::uint64_t>(chunk_size_);
    const std::uint64_t steps = static_cast<std::uint64_t>(num_steps_);
#define FRT_ADD(...)                   \
    do {                               \
        st = add(__VA_ARGS__);         \
        if (!st.ok_status()) return st; \
    } while (false)

    FRT_ADD("observation_images_normalized",
            {nv, kPi05ModelDims.image_height, kPi05ModelDims.image_width,
             kPi05ModelDims.image_channels},
            activation_dtype_);
    FRT_ADD("vision_x", {vs, kPi05ModelDims.vision_width},
            activation_dtype_);
    if (config.vision_pool_factor == 1) {
        st = add_alias("vision_x_pooled", "vision_x",
                       {vs_enc, kPi05ModelDims.vision_width});
        if (!st.ok_status()) return st;
    } else {
        FRT_ADD("vision_x_pooled",
                {vs_enc, kPi05ModelDims.vision_width}, activation_dtype_);
    }
    FRT_ADD("vision_pos_embed_expanded",
            {vs, kPi05ModelDims.vision_width}, activation_dtype_);
    FRT_ADD("vision_patches",
            {vs, kPi05ModelDims.vision_patch *
                     kPi05ModelDims.vision_patch *
                     kPi05ModelDims.image_channels},
            activation_dtype_);

    FRT_ADD("encoder_rope_weights", {es, kRopeWidth}, activation_dtype_);
    FRT_ADD("prompt_embedding",
            {static_cast<std::uint64_t>(max_prompt_tokens_),
             kPi05ModelDims.embedding_width},
            activation_dtype_);
    FRT_ADD("encoder_x", {es, kPi05ModelDims.encoder_width},
            activation_dtype_);
    FRT_ADD("encoder_rms_ones", {kPi05ModelDims.encoder_width},
            activation_dtype_);

    FRT_ADD("decoder_rope_weights", {ds, kRopeWidth}, activation_dtype_);
    FRT_ADD("decoder_x", {ds, kPi05ModelDims.decoder_width},
            activation_dtype_);
    FRT_ADD("decoder_time_emb",
            {steps, ds, kPi05ModelDims.decoder_width}, activation_dtype_);
    FRT_ADD("decoder_style_attn",
            {steps, kPi05ModelDims.decoder_layers, ds,
             3 * kPi05ModelDims.decoder_width},
            activation_dtype_);
    FRT_ADD("decoder_style_ffn",
            {steps, kPi05ModelDims.decoder_layers, ds,
             3 * kPi05ModelDims.decoder_width},
            activation_dtype_);
    FRT_ADD("decoder_style_final",
            {steps, ds, 3 * kPi05ModelDims.decoder_width},
            activation_dtype_);
    FRT_ADD("diffusion_noise", {ds, kPi05ModelDims.action_width},
            activation_dtype_);
    FRT_ADD("rtc_prev_action_chunk", {ds, kPi05ModelDims.action_width},
            activation_dtype_);
    FRT_ADD("rtc_prefix_weights", {ds}, modalities::DType::kFloat32);
    FRT_ADD("rtc_guidance_weight", {1}, modalities::DType::kFloat32);
    FRT_ADD("decoder_rms_ones", {kPi05ModelDims.decoder_width},
            activation_dtype_);
    logical_size_ = buffers_.size();

    for (const NativeWorkspaceBufferRequirement& buffer :
         requirements.buffers) {
        st = add(buffer.name, buffer.shape, buffer.dtype);
        if (!st.ok_status()) return st;
    }
    for (const NativeWorkspaceAliasRequirement& alias :
         requirements.aliases) {
        st = add_alias(alias.name, alias.source, alias.shape);
        if (!st.ok_status()) return st;
    }
#undef FRT_ADD

    const NativeWorkspaceBuffer* decoder = find("decoder_rope_weights");
    const NativeWorkspaceBuffer* prompt = find("prompt_embedding");
    if (!decoder || !prompt || decoder->dtype != activation_dtype_ ||
        prompt->dtype != activation_dtype_) {
        return invalid("model workspace controls were not allocated");
    }
    decoder_rope_buffer_ = decoder->buffer;
    prompt_embedding_buffer_ = prompt->buffer;

    if (fixed_prompt_controls_) {
        const char* controls[] = {
            "attn_enc_seqused", "attn_dec_seqused", "attn_dec_devpos"};
        for (int i = 0; i < 3; ++i) {
            const NativeWorkspaceBuffer* control = find(controls[i]);
            if (!control || control->dtype != modalities::DType::kUInt8 ||
                frt_buffer_bytes(control->buffer) != sizeof(std::int32_t)) {
                return invalid("fixed prompt control is missing");
            }
            prompt_length_buffers_[i] = control->buffer;
        }
    }

    st = initialize_rms_ones();
    if (!st.ok_status()) return st;
    st = initialize_rope();
    if (!st.ok_status()) return st;
    return fixed_prompt_controls_
               ? set_fixed_prompt_length(0)
               : modalities::Status::ok();
}

const NativeWorkspaceBuffer* NativeWorkspace::find(
    const std::string& name) const {
    const auto it = buffers_.find(name);
    return it == buffers_.end() ? nullptr : &it->second;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

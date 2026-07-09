#include "flashrt/cpp/models/pi05/native_rtx_attention.h"

#ifdef FLASHRT_CPP_WITH_CUDA_STAGING
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <limits>

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

std::size_t dtype_size(NativeAttentionDType dtype) {
    switch (dtype) {
        case NativeAttentionDType::kBf16: return sizeof(std::uint16_t);
        case NativeAttentionDType::kFloat32: return sizeof(float);
        case NativeAttentionDType::kInt32: return sizeof(std::int32_t);
    }
    return 0;
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

std::uint64_t round_up_128(std::uint64_t value) {
    return ((value + 127) / 128) * 128;
}

}  // namespace

modalities::Status NativeRtxAttentionWorkspace::add(
    const std::string& name,
    std::initializer_list<std::uint64_t> shape,
    NativeAttentionDType dtype) {
    if (!ctx_ || name.empty() || buffers_.find(name) != buffers_.end()) {
        return invalid("native attention buffer definition is invalid");
    }
    std::size_t elements = 0;
    const std::size_t width = dtype_size(dtype);
    if (!width || !element_count(shape, &elements) ||
        elements > std::numeric_limits<std::size_t>::max() / width) {
        return invalid("native attention buffer shape is invalid");
    }
    const std::size_t bytes = elements * width;
    frt_buffer buffer = frt_buffer_alloc(ctx_, name.c_str(), bytes);
    if (!buffer) return backend("native attention allocation failed");
    buffers_.emplace(name, NativeAttentionBuffer{
                               buffer, std::vector<std::uint64_t>(shape),
                               dtype});
    allocated_bytes_ += bytes;
    return modalities::Status::ok();
}

modalities::Status NativeRtxAttentionWorkspace::allocate(
    const NativeRtxAttentionConfig& config) {
    if (!ctx_ || !buffers_.empty() || config.num_views < 1 ||
        config.num_views > 3 || config.encoder_sequence <= 0 ||
        config.encoder_vision_sequence <= 0 ||
        config.encoder_vision_sequence > config.encoder_sequence ||
        config.chunk_size <= 0 || config.encoder_layers != 18) {
        return invalid("Pi0.5 RTX attention configuration is invalid");
    }
    num_views_ = config.num_views;
    encoder_sequence_ = config.encoder_sequence;
    encoder_vision_sequence_ = config.encoder_vision_sequence;
    chunk_size_ = config.chunk_size;
    encoder_layers_ = config.encoder_layers;
    const std::uint64_t nv = static_cast<std::uint64_t>(num_views_);
    const std::uint64_t es = static_cast<std::uint64_t>(encoder_sequence_);
    const std::uint64_t ds = static_cast<std::uint64_t>(chunk_size_);
    const std::uint64_t layers = static_cast<std::uint64_t>(encoder_layers_);
    const std::uint64_t total_kv = es + ds;
    encoder_splits_ = std::min(128, (encoder_sequence_ + 63) / 64);
    decoder_splits_ =
        std::min(128, (encoder_sequence_ + chunk_size_ + 63) / 64);
    kv_layer_stride_bytes_ =
        static_cast<std::size_t>(total_kv) * 256 * sizeof(std::uint16_t);
    modalities::Status st;
#define FRT_ADD(...)                   \
    do {                               \
        st = add(__VA_ARGS__);          \
        if (!st.ok_status()) return st; \
    } while (false)
    FRT_ADD("attn_vis_Q", {nv, 256, 16, 72}, NativeAttentionDType::kBf16);
    FRT_ADD("attn_vis_K", {nv, 256, 16, 72}, NativeAttentionDType::kBf16);
    FRT_ADD("attn_vis_V", {nv, 256, 16, 72}, NativeAttentionDType::kBf16);
    FRT_ADD("attn_enc_Q", {es, 8, 256}, NativeAttentionDType::kBf16);
    FRT_ADD("attn_enc_K", {layers, total_kv, 1, 256},
            NativeAttentionDType::kBf16);
    FRT_ADD("attn_enc_V", {layers, total_kv, 1, 256},
            NativeAttentionDType::kBf16);
    FRT_ADD("attn_dec_Q", {ds, 8, 256}, NativeAttentionDType::kBf16);
    FRT_ADD("attn_enc_seqused", {1}, NativeAttentionDType::kInt32);
    FRT_ADD("attn_dec_seqused", {1}, NativeAttentionDType::kInt32);
    FRT_ADD("attn_dec_devpos", {1}, NativeAttentionDType::kInt32);

    FRT_ADD("attn_vis_O", {nv, 256, 16, 72}, NativeAttentionDType::kBf16);
    FRT_ADD("attn_vis_lse", {nv, 16, 256}, NativeAttentionDType::kFloat32);
    FRT_ADD("attn_vis_lse_accum", {2, nv, 16, 256},
            NativeAttentionDType::kFloat32);
    FRT_ADD("attn_vis_o_accum", {2, nv, 16, 256, 96},
            NativeAttentionDType::kFloat32);

    FRT_ADD("attn_enc_O", {1, es, 8, 256}, NativeAttentionDType::kBf16);
    FRT_ADD("attn_enc_lse", {1, 8, round_up_128(es)},
            NativeAttentionDType::kFloat32);
    FRT_ADD("attn_enc_lse_accum",
            {static_cast<std::uint64_t>(encoder_splits_), 1, 8, es},
            NativeAttentionDType::kFloat32);
    FRT_ADD("attn_enc_o_accum",
            {static_cast<std::uint64_t>(encoder_splits_), 1, 8, es, 256},
            NativeAttentionDType::kFloat32);

    FRT_ADD("attn_dec_O", {1, ds, 8, 256}, NativeAttentionDType::kBf16);
    FRT_ADD("attn_dec_lse", {1, 8, round_up_128(ds)},
            NativeAttentionDType::kFloat32);
    FRT_ADD("attn_dec_lse_accum",
            {static_cast<std::uint64_t>(decoder_splits_), 1, 8, ds},
            NativeAttentionDType::kFloat32);
    FRT_ADD("attn_dec_o_accum",
            {static_cast<std::uint64_t>(decoder_splits_), 1, 8, ds, 256},
            NativeAttentionDType::kFloat32);
#undef FRT_ADD
    return set_fixed_prompt_length(0);
}

modalities::Status NativeRtxAttentionWorkspace::set_fixed_prompt_length(
    int prompt_tokens) {
    const int max_prompt = encoder_sequence_ - encoder_vision_sequence_;
    if (prompt_tokens < 0 || prompt_tokens > max_prompt || buffers_.empty()) {
        return invalid("Pi0.5 fixed attention prompt length is invalid");
    }
#ifndef FLASHRT_CPP_WITH_CUDA_STAGING
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "fixed attention update requires the CUDA build");
#else
    const std::int32_t valid = encoder_vision_sequence_ + prompt_tokens;
    const std::int32_t values[] = {valid, valid + chunk_size_, valid};
    const char* names[] = {"attn_enc_seqused", "attn_dec_seqused",
                           "attn_dec_devpos"};
    for (int i = 0; i < 3; ++i) {
        const NativeAttentionBuffer* target = find(names[i]);
        if (!target ||
            cudaMemcpy(frt_buffer_dptr(target->buffer), &values[i],
                       sizeof(values[i]), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            return backend("fixed attention length upload failed");
        }
    }
    return modalities::Status::ok();
#endif
}

const NativeAttentionBuffer* NativeRtxAttentionWorkspace::find(
    const std::string& name) const {
    const auto it = buffers_.find(name);
    return it == buffers_.end() ? nullptr : &it->second;
}

void* NativeRtxAttentionWorkspace::encoder_k_layer_dptr(int layer) const {
    const NativeAttentionBuffer* cache = find("attn_enc_K");
    if (!cache || layer < 0 || layer >= encoder_layers_) return nullptr;
    return static_cast<unsigned char*>(frt_buffer_dptr(cache->buffer)) +
           static_cast<std::size_t>(layer) * kv_layer_stride_bytes_;
}

void* NativeRtxAttentionWorkspace::encoder_v_layer_dptr(int layer) const {
    const NativeAttentionBuffer* cache = find("attn_enc_V");
    if (!cache || layer < 0 || layer >= encoder_layers_) return nullptr;
    return static_cast<unsigned char*>(frt_buffer_dptr(cache->buffer)) +
           static_cast<std::size_t>(layer) * kv_layer_stride_bytes_;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

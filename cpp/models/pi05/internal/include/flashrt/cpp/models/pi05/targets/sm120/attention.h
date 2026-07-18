#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_ATTENTION_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_ATTENTION_H

#include "flashrt/cpp/models/pi05/support/native_resource_resolver.h"

#include <cstddef>
#include <cstdint>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {

struct Sm120AttentionBuffer final {
    frt_buffer buffer = nullptr;
    modalities::DType dtype = modalities::DType::kUInt8;
    modalities::Shape shape;

    void* device_data() const {
        return buffer ? frt_buffer_dptr(buffer) : nullptr;
    }
    std::size_t bytes() const {
        return buffer ? frt_buffer_bytes(buffer) : 0;
    }
};

struct Sm120VisionAttentionBuffers final {
    Sm120AttentionBuffer query;
    Sm120AttentionBuffer key;
    Sm120AttentionBuffer value;
    Sm120AttentionBuffer output;
    Sm120AttentionBuffer logsumexp;
    Sm120AttentionBuffer logsumexp_accumulator;
    Sm120AttentionBuffer output_accumulator;
};

struct Sm120EncoderAttentionBuffers final {
    Sm120AttentionBuffer query;
    Sm120AttentionBuffer output;
    Sm120AttentionBuffer logsumexp;
};

struct Sm120DecoderAttentionBuffers final {
    Sm120AttentionBuffer query;
    Sm120AttentionBuffer output;
    Sm120AttentionBuffer logsumexp;
    Sm120AttentionBuffer logsumexp_accumulator;
    Sm120AttentionBuffer output_accumulator;
};

struct Sm120AttentionCacheBuffers final {
    Sm120AttentionBuffer key;
    Sm120AttentionBuffer value;
};

struct Sm120AttentionControlBuffers final {
    Sm120AttentionBuffer encoder_valid_tokens;
    Sm120AttentionBuffer decoder_valid_tokens;
    Sm120AttentionBuffer decoder_position;
};

class Sm120AttentionBacking final {
public:
    explicit Sm120AttentionBacking(frt_ctx context) : context_(context) {}

    Sm120AttentionBacking(const Sm120AttentionBacking&) = delete;
    Sm120AttentionBacking& operator=(const Sm120AttentionBacking&) = delete;

    modalities::Status allocate(const Pi05ResolvedShape& shape);
    modalities::Status set_prompt_length(int prompt_tokens);
    modalities::Status make_target_bindings(
        Pi05TargetBufferBindings* out) const;

    const Sm120VisionAttentionBuffers& vision() const { return vision_; }
    const Sm120EncoderAttentionBuffers& encoder() const { return encoder_; }
    const Sm120DecoderAttentionBuffers& decoder() const { return decoder_; }
    const Sm120AttentionCacheBuffers& cache() const { return cache_; }
    const Sm120AttentionControlBuffers& controls() const { return controls_; }
    const Pi05ResolvedShape& shape() const { return shape_; }

    void* key_layer_data(int layer) const;
    void* value_layer_data(int layer) const;

    std::size_t allocation_count() const { return allocation_count_; }
    std::size_t allocated_bytes() const { return allocated_bytes_; }
    std::size_t cache_layer_stride_bytes() const {
        return cache_layer_stride_bytes_;
    }
    int decoder_splits() const { return decoder_splits_; }
    bool allocated() const { return allocated_; }

private:
    modalities::Status add(const char* name,
                           const modalities::Shape& shape,
                           modalities::DType dtype,
                           Sm120AttentionBuffer* out);
    modalities::Status upload_controls(int prompt_tokens);

    frt_ctx context_ = nullptr;
    Pi05ResolvedShape shape_;
    Sm120VisionAttentionBuffers vision_;
    Sm120EncoderAttentionBuffers encoder_;
    Sm120DecoderAttentionBuffers decoder_;
    Sm120AttentionCacheBuffers cache_;
    Sm120AttentionControlBuffers controls_;
    std::size_t allocation_count_ = 0;
    std::size_t allocated_bytes_ = 0;
    std::size_t cache_layer_stride_bytes_ = 0;
    int decoder_splits_ = 0;
    bool allocation_started_ = false;
    bool allocated_ = false;
};

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_ATTENTION_H

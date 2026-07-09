#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_ATTENTION_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_ATTENTION_H

#include "flashrt/cpp/modalities/types.h"
#include "flashrt/exec.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

enum class NativeAttentionDType {
    kBf16,
    kFloat32,
    kInt32,
};

struct NativeRtxAttentionConfig {
    int num_views = 2;
    int encoder_sequence = 712;
    int encoder_vision_sequence = 512;
    int chunk_size = 10;
    int encoder_layers = 18;
};

struct NativeAttentionBuffer {
    frt_buffer buffer = nullptr;
    std::vector<std::uint64_t> shape;
    NativeAttentionDType dtype = NativeAttentionDType::kBf16;
};

class NativeRtxAttentionWorkspace {
public:
    explicit NativeRtxAttentionWorkspace(frt_ctx ctx) : ctx_(ctx) {}

    modalities::Status allocate(const NativeRtxAttentionConfig& config);
    modalities::Status set_fixed_prompt_length(int prompt_tokens);
    const NativeAttentionBuffer* find(const std::string& name) const;
    void* encoder_k_layer_dptr(int layer) const;
    void* encoder_v_layer_dptr(int layer) const;

    std::size_t size() const { return buffers_.size(); }
    std::size_t allocated_bytes() const { return allocated_bytes_; }
    std::size_t kv_layer_stride_bytes() const { return kv_layer_stride_bytes_; }
    int encoder_splits() const { return encoder_splits_; }
    int decoder_splits() const { return decoder_splits_; }

private:
    modalities::Status add(const std::string& name,
                           std::initializer_list<std::uint64_t> shape,
                           NativeAttentionDType dtype);

    frt_ctx ctx_ = nullptr;
    std::map<std::string, NativeAttentionBuffer> buffers_;
    std::size_t allocated_bytes_ = 0;
    std::size_t kv_layer_stride_bytes_ = 0;
    int num_views_ = 0;
    int encoder_sequence_ = 0;
    int encoder_vision_sequence_ = 0;
    int chunk_size_ = 0;
    int encoder_layers_ = 0;
    int encoder_splits_ = 0;
    int decoder_splits_ = 0;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_RTX_ATTENTION_H

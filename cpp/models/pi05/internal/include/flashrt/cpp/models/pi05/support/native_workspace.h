#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_WORKSPACE_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_WORKSPACE_H

#include "flashrt/cpp/modalities/types.h"
#include "flashrt/cpp/models/pi05/support/native_device_weights.h"
#include "flashrt/exec.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {

struct NativeWorkspaceConfig {
    int num_views = 2;
    int max_prompt_tokens = 200;
    int chunk_size = 10;
    int num_steps = 10;
    int vision_pool_factor = 1;
};

struct NativeWorkspaceBufferRequirement {
    std::string name;
    std::vector<std::uint64_t> shape;
    modalities::DType dtype = modalities::DType::kBFloat16;
};

struct NativeWorkspaceAliasRequirement {
    std::string name;
    std::string source;
    std::vector<std::uint64_t> shape;
};

struct NativeWorkspaceRequirements {
    modalities::DType activation_dtype = modalities::DType::kBFloat16;
    bool fixed_prompt_controls = false;
    std::vector<NativeWorkspaceBufferRequirement> buffers;
    std::vector<NativeWorkspaceAliasRequirement> aliases;

    void add_buffer(const char* name,
                    std::initializer_list<std::uint64_t> shape,
                    modalities::DType dtype) {
        buffers.push_back(
            {name, std::vector<std::uint64_t>(shape), dtype});
    }

    void add_alias(const char* name,
                   const char* source,
                   std::initializer_list<std::uint64_t> shape) {
        aliases.push_back(
            {name, source, std::vector<std::uint64_t>(shape)});
    }
};

struct NativeWorkspaceBuffer {
    frt_buffer buffer = nullptr;
    std::vector<std::uint64_t> shape;
    modalities::DType dtype = modalities::DType::kBFloat16;
    bool alias = false;
};

class NativeWorkspace {
public:
    explicit NativeWorkspace(frt_ctx ctx) : ctx_(ctx) {}

    NativeWorkspace(const NativeWorkspace&) = delete;
    NativeWorkspace& operator=(const NativeWorkspace&) = delete;

    modalities::Status allocate(
        const NativeWorkspaceConfig& config,
        const NativeWorkspaceRequirements& requirements);
    modalities::Status update_decoder_rope(int prompt_tokens);
    modalities::Status set_fixed_prompt_length(int prompt_tokens);
    modalities::Status expand_vision_position_embedding(
        const NativeDeviceWeightStore& weights);
    const NativeWorkspaceBuffer* find(const std::string& name) const;

    std::size_t logical_size() const { return logical_size_; }
    std::size_t allocation_count() const { return allocation_count_; }
    std::size_t allocated_bytes() const { return allocated_bytes_; }
    int vision_sequence() const { return vision_sequence_; }
    int encoder_vision_sequence() const { return encoder_vision_sequence_; }
    int encoder_sequence() const { return encoder_sequence_; }
    int total_keys() const { return encoder_sequence_ + chunk_size_; }
    int num_views() const { return num_views_; }
    int chunk_size() const { return chunk_size_; }
    int num_steps() const { return num_steps_; }
    modalities::DType activation_dtype() const { return activation_dtype_; }

private:
    modalities::Status add(const std::string& name,
                           const std::vector<std::uint64_t>& shape,
                           modalities::DType dtype);
    modalities::Status add_alias(const std::string& name,
                                 const std::string& source_name,
                                 const std::vector<std::uint64_t>& shape);
    modalities::Status initialize_rms_ones();
    modalities::Status initialize_rope();

    frt_ctx ctx_ = nullptr;
    std::map<std::string, NativeWorkspaceBuffer> buffers_;
    std::size_t logical_size_ = 0;
    std::size_t allocation_count_ = 0;
    std::size_t allocated_bytes_ = 0;
    int vision_sequence_ = 0;
    int encoder_vision_sequence_ = 0;
    int encoder_sequence_ = 0;
    int num_views_ = 0;
    int max_prompt_tokens_ = 0;
    int chunk_size_ = 0;
    int num_steps_ = 0;
    modalities::DType activation_dtype_ = modalities::DType::kBFloat16;
    bool fixed_prompt_controls_ = false;
    frt_buffer decoder_rope_buffer_ = nullptr;
    frt_buffer prompt_embedding_buffer_ = nullptr;
    frt_buffer prompt_length_buffers_[3] = {};
    std::vector<std::uint16_t> rope_table_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_WORKSPACE_H

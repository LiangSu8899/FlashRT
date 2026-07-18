#include "flashrt/cpp/models/pi05/targets/sm120/bf16_operations.h"

#include "activation.cuh"
#include "dit_bf16.cuh"
#include "elementwise.cuh"
#include "norm.cuh"
#include "patch_embed.cuh"
#include "rope.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <new>
#include <utility>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

modalities::Status launch_status() {
    const cudaError_t result = cudaGetLastError();
    return result == cudaSuccess ? modalities::Status::ok()
                                 : backend(cudaGetErrorString(result));
}

void set_status(modalities::Status* destination,
                modalities::Status status) {
    if (destination) *destination = std::move(status);
}

void* data(const Pi05ResolvedBuffer& buffer) {
    return buffer.buffer ? frt_buffer_dptr(buffer.buffer) : nullptr;
}

const void* weight_data(const Pi05ResolvedWeight& weight) {
    return weight.device_data;
}

cudaStream_t cuda_stream(Pi05Stream stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

modalities::Status add_bias(void* values,
                            const Pi05ResolvedWeight& bias,
                            int rows,
                            int columns,
                            Pi05Stream stream) {
    add_bias_bf16(static_cast<__nv_bfloat16*>(values),
                  static_cast<const __nv_bfloat16*>(weight_data(bias)), rows,
                  columns, cuda_stream(stream));
    return launch_status();
}

modalities::Status bias_residual(void* residual,
                                 const void* values,
                                 const Pi05ResolvedWeight& bias,
                                 int rows,
                                 int columns,
                                 Pi05Stream stream) {
    ::bias_residual(
        static_cast<__nv_bfloat16*>(residual),
        static_cast<const __nv_bfloat16*>(values),
        static_cast<const __nv_bfloat16*>(weight_data(bias)), rows, columns,
        cuda_stream(stream));
    return launch_status();
}

modalities::Status layer_norm(const void* values,
                              const Pi05ResolvedWeight& weight,
                              const Pi05ResolvedWeight& bias,
                              void* output,
                              int rows,
                              int columns,
                              float epsilon,
                              Pi05Stream stream) {
    ::layer_norm(
        static_cast<const __nv_bfloat16*>(values),
        static_cast<const __nv_bfloat16*>(weight_data(weight)),
        static_cast<const __nv_bfloat16*>(weight_data(bias)),
        static_cast<__nv_bfloat16*>(output), rows, columns, epsilon,
        cuda_stream(stream));
    return launch_status();
}

}  // namespace

std::unique_ptr<Sm120Bf16Operations> Sm120Bf16Operations::create(
    const Pi05ResolvedShape& shape,
    const Pi05ResolvedResources& resources,
    const Pi05NativeSupportBuffers& support,
    const Sm120Bf16ScratchBacking& scratch,
    const Sm120AttentionBacking& attention,
    const Sm120AttentionDriver& attention_driver,
    const Sm120Bf16Linear& linear,
    modalities::Status* status) {
    modalities::Status validation =
        validate_pi05_resolved_resources(resources, shape);
    if (!validation.ok_status() || !scratch.allocated() ||
        !attention.allocated() ||
        !pi05_resolved_shape_equal(shape, scratch.shape()) ||
        !pi05_resolved_shape_equal(shape, attention.shape())) {
        set_status(status, validation.ok_status()
                               ? invalid("SM120 BF16 operation resources are invalid")
                               : validation);
        return nullptr;
    }
    validation = attention_driver.status();
    if (validation.ok_status()) validation = linear.status();
    if (!validation.ok_status()) {
        set_status(status, std::move(validation));
        return nullptr;
    }
    if (!data(support.vision_patches) ||
        !data(support.pooled_vision_state) ||
        !data(support.expanded_vision_position) ||
        !data(support.encoder_rms_weight) ||
        !data(support.decoder_rms_weight)) {
        set_status(status, invalid("SM120 BF16 support handles are invalid"));
        return nullptr;
    }
    std::unique_ptr<Sm120Bf16Operations> result(new (std::nothrow)
        Sm120Bf16Operations(shape, resources, support, scratch, attention,
                            attention_driver, linear));
    set_status(status, result ? modalities::Status::ok()
                              : backend("SM120 BF16 operation allocation failed"));
    return result;
}

modalities::Status Sm120Bf16Operations::compose_prompt(
    Pi05Stream stream) const {
    const Pi05ResolvedBuffer& prompt = resources_.buffers.prompt_embedding;
    const Pi05ResolvedBuffer& encoder = resources_.buffers.encoder_state;
    const std::size_t offset =
        static_cast<std::size_t>(shape_.encoder_vision_sequence) *
        static_cast<std::size_t>(kPi05ModelDims.encoder_width) *
        sizeof(std::uint16_t);
    if (!data(prompt) || !data(encoder) ||
        offset > encoder.physical_bytes ||
        prompt.physical_bytes > encoder.physical_bytes - offset) {
        return invalid("SM120 prompt composition buffers are invalid");
    }
    const cudaError_t result = cudaMemcpyAsync(
        static_cast<unsigned char*>(data(encoder)) + offset, data(prompt),
        static_cast<std::size_t>(prompt.physical_bytes),
        cudaMemcpyDeviceToDevice, cuda_stream(stream));
    return result == cudaSuccess
               ? modalities::Status::ok()
               : backend("SM120 prompt composition failed");
}

modalities::Status Sm120Bf16Operations::vision_embed(
    Pi05Stream stream) const {
    const int rows = shape_.vision_sequence;
    const int width = kPi05ModelDims.vision_width;
    const int patch_width = kPi05ModelDims.vision_patch *
                            kPi05ModelDims.vision_patch *
                            kPi05ModelDims.image_channels;
    const Pi05VisionGlobalWeights& weights = resources_.weights.vision;
    void* patches = data(support_.vision_patches);
    void* state = data(resources_.buffers.vision_state);
    void* normalized = scratch_.vision().normalized.device_data();
    ::patch_im2col(
        static_cast<const __half*>(data(resources_.buffers.images)),
        static_cast<__half*>(patches), shape_.num_views, cuda_stream(stream));
    modalities::Status status = launch_status();
    if (!status.ok_status()) return status;
    status = linear_.run(weights.patch_weight, patches, state, rows,
                         patch_width, width, stream);
    if (!status.ok_status()) return status;
    status = bias_residual(state, data(support_.expanded_vision_position),
                           weights.patch_bias, rows, width, stream);
    if (!status.ok_status()) return status;
    const Pi05VisionLayerWeights& first = resources_.weights.vision_layers[0];
    return layer_norm(state, first.pre_attention_norm_weight,
                      first.pre_attention_norm_bias, normalized, rows, width,
                      kPi05ModelNumerics.vision_layer_norm_epsilon, stream);
}

modalities::Status Sm120Bf16Operations::vision_attention(
    int layer,
    Pi05Stream stream) const {
    if (layer < 0 || layer >= kPi05ModelDims.vision_layers) {
        return invalid("SM120 vision attention layer is invalid");
    }
    const int rows = shape_.vision_sequence;
    const int width = kPi05ModelDims.vision_width;
    const Pi05VisionLayerWeights& weights =
        resources_.weights.vision_layers[static_cast<std::size_t>(layer)];
    const Sm120VisionAttentionBuffers& attention = attention_.vision();
    void* state = data(resources_.buffers.vision_state);
    void* normalized = scratch_.vision().normalized.device_data();
    void* qkv = scratch_.vision().qkv.device_data();

    modalities::Status status = linear_.run(
        weights.attention_qkv_weight, normalized, qkv, rows, width,
        3 * width, stream);
    if (!status.ok_status()) return status;
    status = add_bias(qkv, weights.attention_qkv_bias, rows, 3 * width,
                      stream);
    if (!status.ok_status()) return status;
    ::qkv_split(
        static_cast<const __nv_bfloat16*>(qkv),
        static_cast<__nv_bfloat16*>(attention.query.device_data()),
        static_cast<__nv_bfloat16*>(attention.key.device_data()),
        static_cast<__nv_bfloat16*>(attention.value.device_data()), rows,
        width, width, width, cuda_stream(stream));
    status = launch_status();
    if (!status.ok_status()) return status;
    status = attention_driver_.vision(stream);
    if (!status.ok_status()) return status;
    status = linear_.run(
        weights.attention_output_weight, attention_driver_.vision_output(),
        normalized, rows, width, width, stream);
    if (!status.ok_status()) return status;
    status = bias_residual(state, normalized, weights.attention_output_bias,
                           rows, width, stream);
    return status.ok_status()
               ? layer_norm(state, weights.pre_mlp_norm_weight,
                            weights.pre_mlp_norm_bias, normalized, rows, width,
                            kPi05ModelNumerics.vision_layer_norm_epsilon,
                            stream)
               : status;
}

modalities::Status Sm120Bf16Operations::vision_mlp(
    int layer,
    Pi05Stream stream) const {
    if (layer < 0 || layer >= kPi05ModelDims.vision_layers) {
        return invalid("SM120 vision MLP layer is invalid");
    }
    const int rows = shape_.vision_sequence;
    const int width = kPi05ModelDims.vision_width;
    const int hidden_width = kPi05ModelDims.vision_hidden;
    const Pi05VisionLayerWeights& weights =
        resources_.weights.vision_layers[static_cast<std::size_t>(layer)];
    void* state = data(resources_.buffers.vision_state);
    void* normalized = scratch_.vision().normalized.device_data();
    void* hidden = scratch_.vision().hidden.device_data();

    modalities::Status status = linear_.run(
        weights.mlp_up_weight, normalized, hidden, rows, width, hidden_width,
        stream);
    if (!status.ok_status()) return status;
    status = add_bias(hidden, weights.mlp_up_bias, rows, hidden_width, stream);
    if (!status.ok_status()) return status;
    ::gelu_inplace(static_cast<__nv_bfloat16*>(hidden), rows * hidden_width,
                   cuda_stream(stream));
    status = launch_status();
    if (!status.ok_status()) return status;
    status = linear_.run(weights.mlp_down_weight, hidden, normalized, rows,
                         hidden_width, width, stream);
    if (!status.ok_status()) return status;
    status = bias_residual(state, normalized, weights.mlp_down_bias, rows,
                           width, stream);
    if (!status.ok_status() || layer + 1 == kPi05ModelDims.vision_layers) {
        return status;
    }
    const Pi05VisionLayerWeights& next =
        resources_.weights.vision_layers[static_cast<std::size_t>(layer + 1)];
    return layer_norm(state, next.pre_attention_norm_weight,
                      next.pre_attention_norm_bias, normalized, rows, width,
                      kPi05ModelNumerics.vision_layer_norm_epsilon, stream);
}

modalities::Status Sm120Bf16Operations::vision_project(
    Pi05Stream stream) const {
    const int rows = shape_.encoder_vision_sequence;
    const int width = kPi05ModelDims.vision_width;
    const Pi05VisionGlobalWeights& weights = resources_.weights.vision;
    void* state = data(resources_.buffers.vision_state);
    void* pooled = data(support_.pooled_vision_state);
    void* normalized = scratch_.vision().normalized.device_data();
    void* encoder = data(resources_.buffers.encoder_state);
    modalities::Status status = modalities::Status::ok();
    if (shape_.vision_pool_factor > 1) {
        ::avg_pool_vision_tokens(
            static_cast<const __nv_bfloat16*>(state),
            static_cast<__nv_bfloat16*>(pooled), shape_.num_views,
            kPi05ModelDims.image_height / kPi05ModelDims.vision_patch,
            kPi05ModelDims.image_width / kPi05ModelDims.vision_patch, width,
            shape_.vision_pool_factor, cuda_stream(stream));
        status = launch_status();
        if (!status.ok_status()) return status;
    }
    status = layer_norm(pooled, weights.final_norm_weight,
                        weights.final_norm_bias, normalized, rows, width,
                        kPi05ModelNumerics.vision_layer_norm_epsilon, stream);
    if (!status.ok_status()) return status;
    status = linear_.run(weights.projector_weight, normalized, encoder, rows,
                         width, kPi05ModelDims.encoder_width, stream);
    return status.ok_status()
               ? add_bias(encoder, weights.projector_bias, rows,
                          kPi05ModelDims.encoder_width, stream)
               : status;
}

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

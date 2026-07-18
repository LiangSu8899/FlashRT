#include "flashrt/cpp/models/pi05/targets/sm120/operations.h"

#include "activation.cuh"
#include "dit_bf16.cuh"
#include "elementwise.cuh"
#include "fusion.cuh"
#include "norm.cuh"
#include "patch_embed.cuh"
#include "rope.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <limits>
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

modalities::Status rms_norm(const void* values,
                            const Pi05ResolvedBuffer& weight,
                            void* output,
                            int rows,
                            int columns,
                            float epsilon,
                            Pi05Stream stream) {
    ::rms_norm(
        static_cast<const __nv_bfloat16*>(values),
        static_cast<const __nv_bfloat16*>(data(weight)),
        static_cast<__nv_bfloat16*>(output), rows, columns, epsilon,
        cuda_stream(stream));
    return launch_status();
}

modalities::Status add_residual(void* residual,
                                const void* values,
                                int elements,
                                Pi05Stream stream) {
    ::residual_add(static_cast<__nv_bfloat16*>(residual),
                   static_cast<const __nv_bfloat16*>(values), elements,
                   cuda_stream(stream));
    return launch_status();
}

modalities::Status split_qkv_rope(const void* qkv,
                                  const Pi05ResolvedBuffer& rope,
                                  void* query,
                                  void* key,
                                  void* value,
                                  int rows,
                                  int query_width,
                                  int key_width,
                                  int value_width,
                                  int head_width,
                                  Pi05Stream stream) {
    ::qkv_split_rope(
        static_cast<const __nv_bfloat16*>(qkv),
        static_cast<const __nv_bfloat16*>(data(rope)),
        static_cast<__nv_bfloat16*>(query),
        static_cast<__nv_bfloat16*>(key),
        static_cast<__nv_bfloat16*>(value), rows, query_width, key_width,
        value_width, head_width, cuda_stream(stream));
    return launch_status();
}

modalities::Status gated_silu(const void* gate,
                              const void* up,
                              void* output,
                              int elements,
                              Pi05Stream stream) {
    ::gate_silu_mul(static_cast<const __nv_bfloat16*>(gate),
                    static_cast<const __nv_bfloat16*>(up),
                    static_cast<__nv_bfloat16*>(output), elements,
                    cuda_stream(stream));
    return launch_status();
}

modalities::Status gated_silu_merged(const void* merged,
                                     void* output,
                                     int rows,
                                     int hidden,
                                     Pi05Stream stream) {
    ::gate_silu_mul_merged(
        static_cast<const __nv_bfloat16*>(merged),
        static_cast<__nv_bfloat16*>(output), rows, hidden,
        cuda_stream(stream));
    return launch_status();
}

modalities::Status adaptive_rms_norm(const void* values,
                                     const Pi05ResolvedBuffer& weight,
                                     const void* style,
                                     void* output,
                                     void* gate,
                                     int rows,
                                     int columns,
                                     float epsilon,
                                     Pi05Stream stream) {
    ::ada_rms_norm_style(
        static_cast<const __nv_bfloat16*>(values),
        static_cast<const __nv_bfloat16*>(data(weight)),
        static_cast<const __nv_bfloat16*>(style),
        static_cast<__nv_bfloat16*>(output),
        static_cast<__nv_bfloat16*>(gate), rows, columns, epsilon,
        cuda_stream(stream));
    return launch_status();
}

modalities::Status gated_residual(void* residual,
                                  const void* values,
                                  const void* gate,
                                  int elements,
                                  Pi05Stream stream) {
    ::gate_mul_residual(
        static_cast<__nv_bfloat16*>(residual),
        static_cast<const __nv_bfloat16*>(values),
        static_cast<const __nv_bfloat16*>(gate), elements,
        cuda_stream(stream));
    return launch_status();
}

modalities::Status split_qkv_rope_at_position(
    const void* qkv,
    const Pi05ResolvedBuffer& rope,
    void* query,
    void* key,
    void* value,
    const Pi05ResolvedBuffer& position,
    int rows,
    int query_width,
    int key_width,
    int value_width,
    int head_width,
    Pi05Stream stream) {
    ::qkv_split_rope_devpos(
        static_cast<const __nv_bfloat16*>(qkv),
        static_cast<const __nv_bfloat16*>(data(rope)),
        static_cast<__nv_bfloat16*>(query),
        static_cast<__nv_bfloat16*>(key),
        static_cast<__nv_bfloat16*>(value),
        static_cast<const int*>(data(position)), rows, query_width, key_width,
        value_width, head_width, cuda_stream(stream));
    return launch_status();
}

const void* style_slice(const Pi05ResolvedBuffer& styles,
                        const Pi05ResolvedShape& shape,
                        int step,
                        int layer) {
    std::size_t index = static_cast<std::size_t>(step);
    if (layer >= 0) {
        index = index * kPi05ModelDims.decoder_layers +
                static_cast<std::size_t>(layer);
    }
    const std::size_t elements_per_style =
        static_cast<std::size_t>(shape.chunk) *
        3 * kPi05ModelDims.decoder_width;
    return static_cast<const std::uint16_t*>(data(styles)) +
           index * elements_per_style;
}

bool bf16_span_fits(const void* pointer,
                    std::size_t bytes,
                    int rows,
                    int columns) {
    return pointer && rows > 0 && columns > 0 &&
           static_cast<std::size_t>(rows) <=
               std::numeric_limits<std::size_t>::max() /
                   static_cast<std::size_t>(columns) &&
           static_cast<std::size_t>(rows) *
                   static_cast<std::size_t>(columns) <=
               bytes / sizeof(std::uint16_t);
}

class Sm120Fp8AutotuneSink final : public Pi05LinearWeightGroupSink {
public:
    Sm120Fp8AutotuneSink(const Pi05ResolvedShape& shape,
                         Pi05ResolvedResources* resources,
                         const Sm120Bf16ScratchBacking& scratch,
                         Sm120Fp8Linear* linear)
        : shape_(shape),
          resources_(resources),
          scratch_(scratch),
          linear_(linear) {}

    modalities::Status record(
        const Pi05LinearWeightGroup& group) override {
        const Pi05ResolvedWeight* weight =
            group.fused && group.fused->device_data ? group.fused
                                                    : group.first;
        if (!resources_ || !linear_ || !weight || !weight->device_data ||
            weight->shape.rank != 2 ||
            weight->shape.dims[0] >
                static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
            weight->shape.dims[1] >
                static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            return invalid("SM120 FP8 autotune weight is invalid");
        }

        const void* input = nullptr;
        std::size_t input_bytes = 0;
        void* output = nullptr;
        std::size_t output_bytes = 0;
        int rows = 0;
        switch (group.key.domain) {
            case Pi05LinearDomain::kVision: {
                rows = shape_.vision_sequence;
                input = scratch_.vision().normalized.device_data();
                input_bytes = scratch_.vision().normalized.bytes();
                output = scratch_.vision().normalized.device_data();
                output_bytes = scratch_.vision().normalized.bytes();
                switch (group.key.role) {
                    case Pi05LinearRole::kAttentionQkv:
                        output = scratch_.vision().qkv.device_data();
                        output_bytes = scratch_.vision().qkv.bytes();
                        break;
                    case Pi05LinearRole::kMlpUp:
                        output = scratch_.vision().hidden.device_data();
                        output_bytes = scratch_.vision().hidden.bytes();
                        break;
                    case Pi05LinearRole::kMlpDown:
                        input = scratch_.vision().hidden.device_data();
                        input_bytes = scratch_.vision().hidden.bytes();
                        break;
                    case Pi05LinearRole::kProjector:
                        rows = shape_.encoder_vision_sequence;
                        output = data(resources_->buffers.encoder_state);
                        output_bytes = static_cast<std::size_t>(
                            resources_->buffers.encoder_state.physical_bytes);
                        break;
                    case Pi05LinearRole::kAttentionOutput:
                        break;
                    case Pi05LinearRole::kMlpGateUpGroup:
                        return invalid("SM120 vision autotune role is invalid");
                }
                break;
            }
            case Pi05LinearDomain::kEncoder: {
                rows = shape_.encoder_sequence;
                input = scratch_.encoder().normalized.device_data();
                input_bytes = scratch_.encoder().normalized.bytes();
                output = scratch_.encoder().normalized.device_data();
                output_bytes = scratch_.encoder().normalized.bytes();
                switch (group.key.role) {
                    case Pi05LinearRole::kAttentionQkv:
                        output = scratch_.encoder().qkv.device_data();
                        output_bytes = scratch_.encoder().qkv.bytes();
                        break;
                    case Pi05LinearRole::kMlpGateUpGroup:
                        output = scratch_.encoder().gate.device_data();
                        output_bytes = scratch_.encoder().gate.bytes();
                        break;
                    case Pi05LinearRole::kMlpDown:
                        input = scratch_.encoder().hidden.device_data();
                        input_bytes = scratch_.encoder().hidden.bytes();
                        break;
                    case Pi05LinearRole::kAttentionOutput:
                        break;
                    case Pi05LinearRole::kMlpUp:
                    case Pi05LinearRole::kProjector:
                        return invalid("SM120 encoder autotune role is invalid");
                }
                break;
            }
            case Pi05LinearDomain::kDecoder: {
                rows = shape_.chunk;
                input = scratch_.decoder().normalized.device_data();
                input_bytes = scratch_.decoder().normalized.bytes();
                output = scratch_.decoder().normalized.device_data();
                output_bytes = scratch_.decoder().normalized.bytes();
                switch (group.key.role) {
                    case Pi05LinearRole::kAttentionQkv:
                        output = scratch_.decoder().qkv.device_data();
                        output_bytes = scratch_.decoder().qkv.bytes();
                        break;
                    case Pi05LinearRole::kAttentionOutput:
                        input = scratch_.decoder().qkv.device_data();
                        input_bytes = scratch_.decoder().qkv.bytes();
                        break;
                    case Pi05LinearRole::kMlpGateUpGroup:
                        output = scratch_.decoder().gate_projection.device_data();
                        output_bytes = scratch_.decoder().gate_projection.bytes();
                        break;
                    case Pi05LinearRole::kMlpDown:
                        input = scratch_.decoder().hidden.device_data();
                        input_bytes = scratch_.decoder().hidden.bytes();
                        break;
                    case Pi05LinearRole::kMlpUp:
                    case Pi05LinearRole::kProjector:
                        return invalid("SM120 decoder autotune role is invalid");
                }
                break;
            }
        }

        const int input_width = static_cast<int>(weight->shape.dims[0]);
        const int output_width = static_cast<int>(weight->shape.dims[1]);
        if (!bf16_span_fits(input, input_bytes, rows, input_width) ||
            !bf16_span_fits(output, output_bytes, rows, output_width)) {
            return invalid("SM120 FP8 autotune scratch is too small");
        }
        Pi05LinearActivationSite site;
        const int step = group.key.domain == Pi05LinearDomain::kDecoder
                             ? 0
                             : -1;
        modalities::Status status = resolve_pi05_linear_activation_site(
            group.key, step, shape_, &site);
        return status.ok_status()
                   ? linear_->autotune(*weight, site, input, output, rows,
                                       input_width, output_width)
                   : status;
    }

private:
    const Pi05ResolvedShape& shape_;
    Pi05ResolvedResources* resources_ = nullptr;
    const Sm120Bf16ScratchBacking& scratch_;
    Sm120Fp8Linear* linear_ = nullptr;
};

}  // namespace

modalities::Status Sm120Operations::autotune_fp8(
    const Pi05ResolvedShape& shape,
    Pi05ResolvedResources* resources,
    const Sm120Bf16ScratchBacking& scratch,
    Sm120Fp8Linear* linear) {
    if (!resources || !linear || !scratch.allocated() ||
        !scratch.fused_gate_up() ||
        !pi05_resolved_shape_equal(shape, scratch.shape()) ||
        linear->autotune_frozen()) {
        return invalid("SM120 FP8 autotune state is invalid");
    }
    modalities::Status status =
        validate_pi05_resolved_resources(*resources, shape);
    if (!status.ok_status()) return status;
    Sm120Fp8AutotuneSink sink(shape, resources, scratch, linear);
    status = visit_pi05_linear_weight_groups(&resources->weights, &sink);
    return status.ok_status() ? linear->freeze_autotune() : status;
}

std::unique_ptr<Sm120Operations> Sm120Operations::create(
    const Pi05ResolvedShape& shape,
    const Pi05ResolvedResources& resources,
    const Pi05NativeSupportBuffers& support,
    const Sm120Bf16ScratchBacking& scratch,
    const Sm120AttentionBacking& attention,
    const Sm120AttentionDriver& attention_driver,
    const Sm120Bf16Linear& bf16_linear,
    Sm120Fp8Linear* fp8_linear,
    modalities::Status* status) {
    modalities::Status validation =
        validate_pi05_resolved_resources(resources, shape);
    if (!validation.ok_status() || !scratch.allocated() ||
        !attention.allocated() ||
        !pi05_resolved_shape_equal(shape, scratch.shape()) ||
        !pi05_resolved_shape_equal(shape, attention.shape()) ||
        scratch.fused_gate_up() != (fp8_linear != nullptr)) {
        set_status(status, validation.ok_status()
                               ? invalid("SM120 operation resources are invalid")
                               : validation);
        return nullptr;
    }
    validation = attention_driver.status();
    if (validation.ok_status()) validation = bf16_linear.status();
    if (validation.ok_status() && fp8_linear) {
        validation = fp8_linear->status();
        if (validation.ok_status() && !fp8_linear->autotune_frozen()) {
            validation = invalid("SM120 FP8 operation autotune is incomplete");
        }
    }
    if (!validation.ok_status()) {
        set_status(status, std::move(validation));
        return nullptr;
    }
    if (!data(support.vision_patches) ||
        !data(support.pooled_vision_state) ||
        !data(support.expanded_vision_position) ||
        !data(support.encoder_rms_weight) ||
        !data(support.decoder_rms_weight)) {
        set_status(status, invalid("SM120 support handles are invalid"));
        return nullptr;
    }
    std::unique_ptr<Sm120Operations> result(new (std::nothrow)
        Sm120Operations(shape, resources, support, scratch, attention,
                        attention_driver, bf16_linear, fp8_linear));
    set_status(status, result ? modalities::Status::ok()
                              : backend("SM120 operation allocation failed"));
    return result;
}

modalities::Status Sm120Operations::linear(
    const Pi05ResolvedWeight& weight,
    Pi05LinearWeightKey key,
    int step,
    const void* input,
    void* output,
    int rows,
    int input_width,
    int output_width,
    Pi05Stream stream,
    bool prequantized) const {
    if (!fp8_linear_) {
        return prequantized
                   ? invalid("SM120 BF16 input cannot be prequantized")
                   : bf16_linear_.run(weight, input, output, rows,
                                      input_width, output_width, stream);
    }
    Pi05LinearActivationSite site;
    modalities::Status status = resolve_pi05_linear_activation_site(
        key, step, shape_, &site);
    if (!status.ok_status()) return status;
    return prequantized
               ? fp8_linear_->run_prequantized(
                     weight, site, input, output, rows, input_width,
                     output_width, stream)
               : fp8_linear_->run(weight, site, input, output, rows,
                                  input_width, output_width, stream);
}

float* Sm120Operations::scale(Pi05LinearWeightKey key, int step) const {
    Pi05LinearActivationSite site;
    return fp8_linear_ &&
                   resolve_pi05_linear_activation_site(
                       key, step, shape_, &site).ok_status()
               ? fp8_linear_->scale_data(site)
               : nullptr;
}

modalities::Status Sm120Operations::compose_prompt(
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

modalities::Status Sm120Operations::vision_embed(
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
    status = bf16_linear_.run(weights.patch_weight, patches, state, rows,
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

modalities::Status Sm120Operations::vision_attention(
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

    modalities::Status status = linear(
        weights.attention_qkv_weight,
        {Pi05LinearDomain::kVision, Pi05LinearRole::kAttentionQkv, layer},
        -1, normalized, qkv, rows, width, 3 * width, stream);
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
    status = linear(
        weights.attention_output_weight,
        {Pi05LinearDomain::kVision, Pi05LinearRole::kAttentionOutput, layer},
        -1, attention_driver_.vision_output(), normalized, rows, width, width,
        stream);
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

modalities::Status Sm120Operations::vision_mlp(
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

    modalities::Status status = linear(
        weights.mlp_up_weight,
        {Pi05LinearDomain::kVision, Pi05LinearRole::kMlpUp, layer}, -1,
        normalized, hidden, rows, width, hidden_width, stream);
    if (!status.ok_status()) return status;
    status = add_bias(hidden, weights.mlp_up_bias, rows, hidden_width, stream);
    if (!status.ok_status()) return status;
    ::gelu_inplace(static_cast<__nv_bfloat16*>(hidden), rows * hidden_width,
                   cuda_stream(stream));
    status = launch_status();
    if (!status.ok_status()) return status;
    status = linear(
        weights.mlp_down_weight,
        {Pi05LinearDomain::kVision, Pi05LinearRole::kMlpDown, layer}, -1,
        hidden, normalized, rows, hidden_width, width, stream);
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

modalities::Status Sm120Operations::vision_project(
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
    status = linear(
        weights.projector_weight,
        {Pi05LinearDomain::kVision, Pi05LinearRole::kProjector, -1}, -1,
        normalized, encoder, rows, width, kPi05ModelDims.encoder_width,
        stream);
    return status.ok_status()
               ? add_bias(encoder, weights.projector_bias, rows,
                          kPi05ModelDims.encoder_width, stream)
               : status;
}

modalities::Status Sm120Operations::encoder_attention(
    int layer,
    Pi05Stream stream) const {
    if (layer < 0 || layer >= kPi05ModelDims.encoder_layers - 1) {
        return invalid("SM120 encoder attention layer is invalid");
    }
    const int rows = shape_.encoder_sequence;
    const int width = kPi05ModelDims.encoder_width;
    const int key_width =
        kPi05ModelDims.encoder_kv_heads * kPi05ModelDims.encoder_head_dim;
    const int qkv_width = width + 2 * key_width;
    const Pi05EncoderLayerWeights& weights =
        resources_.weights.encoder_layers[static_cast<std::size_t>(layer)];
    const Sm120EncoderAttentionBuffers& attention = attention_.encoder();
    void* state = data(resources_.buffers.encoder_state);
    void* normalized = scratch_.encoder().normalized.device_data();
    void* qkv = scratch_.encoder().qkv.device_data();
    void* key = attention_.key_layer_data(layer);
    void* value = attention_.value_layer_data(layer);

    const Pi05LinearWeightKey qkv_key{
        Pi05LinearDomain::kEncoder, Pi05LinearRole::kAttentionQkv, layer};
    const Pi05LinearWeightKey output_key{
        Pi05LinearDomain::kEncoder, Pi05LinearRole::kAttentionOutput, layer};
    const Pi05LinearWeightKey gate_up_key{
        Pi05LinearDomain::kEncoder, Pi05LinearRole::kMlpGateUpGroup, layer};
    modalities::Status status;
    const void* qkv_input = normalized;
    if (static_fp8()) {
        float* qkv_scale = scale(qkv_key, -1);
        if (!qkv_scale) return invalid("SM120 encoder QKV scale is invalid");
        if (layer == 0) {
            ::rms_norm_fp8(
                static_cast<const __nv_bfloat16*>(state),
                static_cast<const __nv_bfloat16*>(
                    data(support_.encoder_rms_weight)),
                static_cast<__nv_fp8_e4m3*>(fp8_linear_->scratch_data()),
                rows, width, kPi05ModelNumerics.encoder_rms_norm_epsilon,
                qkv_scale, cuda_stream(stream));
        } else {
            ::residual_add_rms_norm_fp8(
                static_cast<__nv_bfloat16*>(state),
                static_cast<const __nv_bfloat16*>(normalized),
                static_cast<const __nv_bfloat16*>(
                    data(support_.encoder_rms_weight)),
                static_cast<__nv_fp8_e4m3*>(fp8_linear_->scratch_data()),
                rows, width, kPi05ModelNumerics.encoder_rms_norm_epsilon,
                qkv_scale, cuda_stream(stream));
        }
        status = launch_status();
        qkv_input = fp8_linear_->scratch_data();
    } else {
        status = rms_norm(
            state, support_.encoder_rms_weight, normalized, rows, width,
            kPi05ModelNumerics.encoder_rms_norm_epsilon, stream);
    }
    if (!status.ok_status()) return status;
    status = linear(weights.attention_qkv_weight, qkv_key, -1, qkv_input,
                    qkv, rows, width, qkv_width, stream, static_fp8());
    if (!status.ok_status()) return status;
    status = split_qkv_rope(
        qkv, resources_.buffers.encoder_rope, attention.query.device_data(),
        key, value, rows, width, key_width, key_width,
        kPi05ModelDims.encoder_head_dim, stream);
    if (!status.ok_status()) return status;
    status = attention_driver_.encoder(layer, stream);
    if (!status.ok_status()) return status;
    status = linear(weights.attention_output_weight, output_key, -1,
                    attention_driver_.encoder_output(), normalized, rows,
                    width, width, stream);
    if (!status.ok_status()) return status;
    if (static_fp8()) {
        float* gate_up_scale = scale(gate_up_key, -1);
        if (!gate_up_scale) {
            return invalid("SM120 encoder gate-up scale is invalid");
        }
        ::residual_add_rms_norm_fp8(
            static_cast<__nv_bfloat16*>(state),
            static_cast<const __nv_bfloat16*>(normalized),
            static_cast<const __nv_bfloat16*>(
                data(support_.encoder_rms_weight)),
            static_cast<__nv_fp8_e4m3*>(fp8_linear_->scratch_data()), rows,
            width, kPi05ModelNumerics.encoder_rms_norm_epsilon,
            gate_up_scale, cuda_stream(stream));
        return launch_status();
    }
    status = add_residual(state, normalized, rows * width, stream);
    return status.ok_status()
               ? rms_norm(state, support_.encoder_rms_weight, normalized,
                          rows, width,
                          kPi05ModelNumerics.encoder_rms_norm_epsilon, stream)
               : status;
}

modalities::Status Sm120Operations::encoder_mlp(
    int layer,
    Pi05Stream stream) const {
    if (layer < 0 || layer >= kPi05ModelDims.encoder_layers - 1) {
        return invalid("SM120 encoder MLP layer is invalid");
    }
    const int rows = shape_.encoder_sequence;
    const int width = kPi05ModelDims.encoder_width;
    const int hidden_width = kPi05ModelDims.encoder_hidden;
    const Pi05FeedForwardWeights& weights =
        resources_.weights.encoder_layers[static_cast<std::size_t>(layer)].mlp;
    void* state = data(resources_.buffers.encoder_state);
    void* normalized = scratch_.encoder().normalized.device_data();
    void* gate = scratch_.encoder().gate.device_data();
    void* hidden = scratch_.encoder().hidden.device_data();

    const Pi05LinearWeightKey gate_up_key{
        Pi05LinearDomain::kEncoder, Pi05LinearRole::kMlpGateUpGroup, layer};
    const Pi05LinearWeightKey down_key{
        Pi05LinearDomain::kEncoder, Pi05LinearRole::kMlpDown, layer};
    if (static_fp8()) {
        float* down_scale = scale(down_key, -1);
        if (!down_scale || !weights.gate_up_weight.device_data) {
            return invalid("SM120 encoder fused FP8 storage is invalid");
        }
        modalities::Status status = linear(
            weights.gate_up_weight, gate_up_key, -1,
            fp8_linear_->scratch_data(), gate, rows, width,
            2 * hidden_width, stream, true);
        if (!status.ok_status()) return status;
        ::gate_silu_mul_merged_fp8(
            static_cast<const __nv_bfloat16*>(gate),
            static_cast<__nv_fp8_e4m3*>(fp8_linear_->scratch_data()), rows,
            hidden_width, down_scale, cuda_stream(stream));
        status = launch_status();
        return status.ok_status()
                   ? linear(weights.down_weight, down_key, -1,
                            fp8_linear_->scratch_data(), normalized, rows,
                            hidden_width, width, stream, true)
                   : status;
    }
    if (observed_fp8()) {
        modalities::Status status = linear(
            weights.gate_up_weight, gate_up_key, -1, normalized, gate, rows,
            width, 2 * hidden_width, stream);
        if (!status.ok_status()) return status;
        status = gated_silu_merged(
            gate, hidden, rows, hidden_width, stream);
        if (!status.ok_status()) return status;
        status = linear(
            weights.down_weight, down_key, -1, hidden, normalized, rows,
            hidden_width, width, stream);
        return status.ok_status()
                   ? add_residual(state, normalized, rows * width, stream)
                   : status;
    }
    modalities::Status status = bf16_linear_.run(
        weights.gate_weight, normalized, gate, rows, width, hidden_width,
        stream);
    if (!status.ok_status()) return status;
    status = bf16_linear_.run(weights.up_weight, normalized, hidden, rows,
                              width, hidden_width, stream);
    if (!status.ok_status()) return status;
    status = gated_silu(gate, hidden, hidden, rows * hidden_width, stream);
    if (!status.ok_status()) return status;
    status = bf16_linear_.run(weights.down_weight, hidden, normalized, rows,
                              hidden_width, width, stream);
    return status.ok_status()
               ? add_residual(state, normalized, rows * width, stream)
               : status;
}

modalities::Status Sm120Operations::encoder_cache_finalize(
    int layer,
    Pi05Stream stream) const {
    if (layer != kPi05ModelDims.encoder_layers - 1) {
        return invalid("SM120 encoder cache-finalize layer is invalid");
    }
    const int rows = shape_.encoder_sequence;
    const int width = kPi05ModelDims.encoder_width;
    const int key_width =
        kPi05ModelDims.encoder_kv_heads * kPi05ModelDims.encoder_head_dim;
    const int qkv_width = width + 2 * key_width;
    const Pi05ResolvedWeight& weight =
        resources_.weights.encoder_layers[static_cast<std::size_t>(layer)]
            .attention_qkv_weight;
    void* normalized = scratch_.encoder().normalized.device_data();
    void* qkv = scratch_.encoder().qkv.device_data();
    void* key = attention_.key_layer_data(layer);
    void* value = attention_.value_layer_data(layer);

    const Pi05LinearWeightKey qkv_key{
        Pi05LinearDomain::kEncoder, Pi05LinearRole::kAttentionQkv, layer};
    modalities::Status status;
    const void* qkv_input = normalized;
    if (static_fp8()) {
        float* qkv_scale = scale(qkv_key, -1);
        if (!qkv_scale) return invalid("SM120 encoder QKV scale is invalid");
        ::residual_add_rms_norm_fp8(
            static_cast<__nv_bfloat16*>(
                data(resources_.buffers.encoder_state)),
            static_cast<const __nv_bfloat16*>(normalized),
            static_cast<const __nv_bfloat16*>(
                data(support_.encoder_rms_weight)),
            static_cast<__nv_fp8_e4m3*>(fp8_linear_->scratch_data()), rows,
            width, kPi05ModelNumerics.encoder_rms_norm_epsilon, qkv_scale,
            cuda_stream(stream));
        status = launch_status();
        qkv_input = fp8_linear_->scratch_data();
    } else {
        status = rms_norm(
            data(resources_.buffers.encoder_state),
            support_.encoder_rms_weight, normalized, rows, width,
            kPi05ModelNumerics.encoder_rms_norm_epsilon, stream);
    }
    if (!status.ok_status()) return status;
    status = linear(weight, qkv_key, -1, qkv_input, qkv, rows, width,
                    qkv_width, stream, static_fp8());
    return status.ok_status()
               ? split_qkv_rope(
                     qkv, resources_.buffers.encoder_rope,
                     attention_.encoder().query.device_data(), key, value,
                     rows, width, key_width, key_width,
                     kPi05ModelDims.encoder_head_dim, stream)
               : status;
}

modalities::Status Sm120Operations::diffusion_input_project(
    int step,
    Pi05Stream stream) const {
    if (step < 0 || step >= shape_.num_steps) {
        return invalid("SM120 diffusion input step is invalid");
    }
    const Pi05DecoderGlobalWeights& weights = resources_.weights.decoder;
    modalities::Status status = bf16_linear_.run(
        weights.action_in_weight, data(resources_.buffers.noise),
        data(resources_.buffers.decoder_state), shape_.chunk,
        kPi05ModelDims.action_width, kPi05ModelDims.decoder_width, stream);
    return status.ok_status()
               ? add_bias(data(resources_.buffers.decoder_state),
                          weights.action_in_bias, shape_.chunk,
                          kPi05ModelDims.decoder_width, stream)
               : status;
}

modalities::Status Sm120Operations::decoder_attention(
    int layer,
    int step,
    Pi05Stream stream) const {
    if (layer < 0 || layer >= kPi05ModelDims.decoder_layers || step < 0 ||
        step >= shape_.num_steps) {
        return invalid("SM120 decoder attention index is invalid");
    }
    const int rows = shape_.chunk;
    const int width = kPi05ModelDims.decoder_width;
    const int query_width =
        kPi05ModelDims.decoder_heads * kPi05ModelDims.decoder_head_dim;
    const int key_width =
        kPi05ModelDims.decoder_kv_heads * kPi05ModelDims.decoder_head_dim;
    const int qkv_width = query_width + 2 * key_width;
    const Pi05DecoderLayerWeights& weights =
        resources_.weights.decoder_layers[static_cast<std::size_t>(layer)];
    const Sm120DecoderAttentionBuffers& attention = attention_.decoder();
    void* state = data(resources_.buffers.decoder_state);
    void* normalized = scratch_.decoder().normalized.device_data();
    void* gate = scratch_.decoder().gate.device_data();
    void* qkv = scratch_.decoder().qkv.device_data();

    const Pi05LinearWeightKey qkv_key{
        Pi05LinearDomain::kDecoder, Pi05LinearRole::kAttentionQkv, layer};
    const Pi05LinearWeightKey output_key{
        Pi05LinearDomain::kDecoder, Pi05LinearRole::kAttentionOutput, layer};
    modalities::Status status = modalities::Status::ok();
    const void* qkv_input = normalized;
    if (static_fp8()) {
        if (layer == 0) {
            float* qkv_scale = scale(qkv_key, step);
            if (!qkv_scale) {
                return invalid("SM120 decoder QKV scale is invalid");
            }
            ::ada_rms_norm_style_fp8(
                static_cast<const __nv_bfloat16*>(state),
                static_cast<const __nv_bfloat16*>(
                    data(support_.decoder_rms_weight)),
                static_cast<const __nv_bfloat16*>(style_slice(
                    resources_.buffers.attention_style, shape_, step, layer)),
                static_cast<__nv_fp8_e4m3*>(fp8_linear_->scratch_data()),
                static_cast<__nv_bfloat16*>(gate), rows, width,
                kPi05ModelNumerics.decoder_rms_norm_epsilon, qkv_scale,
                cuda_stream(stream));
            status = launch_status();
        }
        qkv_input = fp8_linear_->scratch_data();
    } else {
        status = adaptive_rms_norm(
            state, support_.decoder_rms_weight,
            style_slice(resources_.buffers.attention_style, shape_, step,
                        layer),
            normalized, gate, rows, width,
            kPi05ModelNumerics.decoder_rms_norm_epsilon, stream);
    }
    if (!status.ok_status()) return status;
    status = linear(weights.attention_qkv_weight, qkv_key, step, qkv_input,
                    qkv, rows, width, qkv_width, stream, static_fp8());
    if (!status.ok_status()) return status;
    status = split_qkv_rope_at_position(
        qkv, resources_.buffers.decoder_rope, attention.query.device_data(),
        attention_.key_layer_data(layer), attention_.value_layer_data(layer),
        resources_.buffers.decoder_position, rows, query_width, key_width,
        key_width, kPi05ModelDims.decoder_head_dim, stream);
    if (!status.ok_status()) return status;
    status = attention_driver_.decoder(layer, stream);
    if (!status.ok_status()) return status;
    status = linear(weights.attention_output_weight, output_key, step,
                    attention_driver_.decoder_output(), normalized, rows,
                    query_width, width, stream);
    if (!status.ok_status() || static_fp8()) return status;
    return gated_residual(state, normalized, gate, rows * width, stream);
}

modalities::Status Sm120Operations::decoder_mlp(
    int layer,
    int step,
    Pi05Stream stream) const {
    if (layer < 0 || layer >= kPi05ModelDims.decoder_layers || step < 0 ||
        step >= shape_.num_steps) {
        return invalid("SM120 decoder MLP index is invalid");
    }
    const int rows = shape_.chunk;
    const int width = kPi05ModelDims.decoder_width;
    const int hidden_width = kPi05ModelDims.decoder_hidden;
    const Pi05FeedForwardWeights& weights =
        resources_.weights.decoder_layers[static_cast<std::size_t>(layer)].mlp;
    void* state = data(resources_.buffers.decoder_state);
    void* normalized = scratch_.decoder().normalized.device_data();
    void* gate = scratch_.decoder().gate.device_data();
    void* gate_projection =
        scratch_.decoder().gate_projection.device_data();
    void* hidden = scratch_.decoder().hidden.device_data();

    const Pi05LinearWeightKey gate_up_key{
        Pi05LinearDomain::kDecoder, Pi05LinearRole::kMlpGateUpGroup, layer};
    const Pi05LinearWeightKey down_key{
        Pi05LinearDomain::kDecoder, Pi05LinearRole::kMlpDown, layer};
    if (static_fp8()) {
        float* gate_up_scale = scale(gate_up_key, step);
        float* down_scale = scale(down_key, step);
        if (!gate_up_scale || !down_scale ||
            !weights.gate_up_weight.device_data) {
            return invalid("SM120 decoder fused FP8 storage is invalid");
        }
        ::gate_residual_ada_norm_fp8(
            static_cast<__nv_bfloat16*>(state),
            static_cast<const __nv_bfloat16*>(normalized),
            static_cast<const __nv_bfloat16*>(gate),
            static_cast<const __nv_bfloat16*>(
                data(support_.decoder_rms_weight)),
            static_cast<const __nv_bfloat16*>(
                style_slice(resources_.buffers.mlp_style, shape_, step,
                            layer)),
            static_cast<__nv_fp8_e4m3*>(fp8_linear_->scratch_data()),
            static_cast<__nv_bfloat16*>(gate), rows, width,
            kPi05ModelNumerics.decoder_rms_norm_epsilon, gate_up_scale,
            cuda_stream(stream));
        modalities::Status status = launch_status();
        if (!status.ok_status()) return status;
        status = linear(weights.gate_up_weight, gate_up_key, step,
                        fp8_linear_->scratch_data(), gate_projection, rows,
                        width, 2 * hidden_width, stream, true);
        if (!status.ok_status()) return status;
        ::gate_silu_mul_merged_fp8(
            static_cast<const __nv_bfloat16*>(gate_projection),
            static_cast<__nv_fp8_e4m3*>(fp8_linear_->scratch_data()), rows,
            hidden_width, down_scale, cuda_stream(stream));
        status = launch_status();
        if (!status.ok_status()) return status;
        status = linear(weights.down_weight, down_key, step,
                        fp8_linear_->scratch_data(), normalized, rows,
                        hidden_width, width, stream, true);
        if (!status.ok_status()) return status;
        if (layer + 1 == kPi05ModelDims.decoder_layers) {
            return gated_residual(state, normalized, gate, rows * width,
                                  stream);
        }
        const Pi05LinearWeightKey next_qkv_key{
            Pi05LinearDomain::kDecoder, Pi05LinearRole::kAttentionQkv,
            layer + 1};
        float* next_qkv_scale = scale(next_qkv_key, step);
        if (!next_qkv_scale) {
            return invalid("SM120 decoder next QKV scale is invalid");
        }
        ::gate_residual_ada_norm_fp8(
            static_cast<__nv_bfloat16*>(state),
            static_cast<const __nv_bfloat16*>(normalized),
            static_cast<const __nv_bfloat16*>(gate),
            static_cast<const __nv_bfloat16*>(
                data(support_.decoder_rms_weight)),
            static_cast<const __nv_bfloat16*>(
                style_slice(resources_.buffers.attention_style, shape_, step,
                            layer + 1)),
            static_cast<__nv_fp8_e4m3*>(fp8_linear_->scratch_data()),
            static_cast<__nv_bfloat16*>(gate), rows, width,
            kPi05ModelNumerics.decoder_rms_norm_epsilon, next_qkv_scale,
            cuda_stream(stream));
        return launch_status();
    }
    modalities::Status status = adaptive_rms_norm(
        state, support_.decoder_rms_weight,
        style_slice(resources_.buffers.mlp_style, shape_, step, layer),
        normalized, gate, rows, width,
        kPi05ModelNumerics.decoder_rms_norm_epsilon, stream);
    if (!status.ok_status()) return status;
    if (observed_fp8()) {
        status = linear(
            weights.gate_up_weight, gate_up_key, step, normalized,
            gate_projection, rows, width, 2 * hidden_width, stream);
        if (!status.ok_status()) return status;
        status = gated_silu_merged(
            gate_projection, hidden, rows, hidden_width, stream);
        if (!status.ok_status()) return status;
        status = linear(
            weights.down_weight, down_key, step, hidden, normalized, rows,
            hidden_width, width, stream);
        return status.ok_status()
                   ? gated_residual(
                         state, normalized, gate, rows * width, stream)
                   : status;
    }
    status = bf16_linear_.run(weights.gate_weight, normalized,
                              gate_projection, rows, width, hidden_width,
                              stream);
    if (!status.ok_status()) return status;
    status = bf16_linear_.run(weights.up_weight, normalized, hidden, rows,
                              width, hidden_width, stream);
    if (!status.ok_status()) return status;
    status = gated_silu(gate_projection, hidden, hidden,
                        rows * hidden_width, stream);
    if (!status.ok_status()) return status;
    status = bf16_linear_.run(weights.down_weight, hidden, normalized, rows,
                              hidden_width, width, stream);
    return status.ok_status()
               ? gated_residual(state, normalized, gate, rows * width,
                                stream)
               : status;
}

modalities::Status Sm120Operations::action_project(
    int step,
    Pi05Stream stream) const {
    if (step < 0 || step >= shape_.num_steps) {
        return invalid("SM120 action projection step is invalid");
    }
    const int rows = shape_.chunk;
    const int width = kPi05ModelDims.decoder_width;
    const Pi05DecoderGlobalWeights& weights = resources_.weights.decoder;
    void* normalized = scratch_.decoder().normalized.device_data();
    void* gate = scratch_.decoder().gate.device_data();
    void* action = data(resources_.buffers.action_delta);
    modalities::Status status = adaptive_rms_norm(
        data(resources_.buffers.decoder_state),
        support_.decoder_rms_weight,
        style_slice(resources_.buffers.final_style, shape_, step, -1),
        normalized, gate, rows, width,
        kPi05ModelNumerics.decoder_rms_norm_epsilon, stream);
    if (!status.ok_status()) return status;
    status = bf16_linear_.run(weights.action_out_weight, normalized, action,
                              rows, width, kPi05ModelDims.action_width,
                              stream);
    return status.ok_status()
               ? add_bias(action, weights.action_out_bias, rows,
                          kPi05ModelDims.action_width, stream)
               : status;
}

modalities::Status Sm120Operations::diffusion_update(
    int step,
    Pi05Stream stream) const {
    if (step < 0 || step >= shape_.num_steps) {
        return invalid("SM120 diffusion update step is invalid");
    }
    return add_residual(
        data(resources_.buffers.noise),
        data(resources_.buffers.action_delta),
        shape_.chunk * kPi05ModelDims.action_width, stream);
}

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

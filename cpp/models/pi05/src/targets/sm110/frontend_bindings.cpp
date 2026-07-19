#include "flashrt/cpp/models/pi05/targets/sm110/frontend_bindings.h"

#include "flashrt/cpp/models/pi05/targets/sm110/fp8_weight_packer.h"
#include "flashrt/cpp/models/pi05/targets/sm110/operation_driver.h"
#include "flashrt/cpp/models/pi05/targets/sm110/physical_resources.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm110 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

Sm110FrontendBindings* bindings(void* state) {
    return static_cast<Sm110FrontendBindings*>(state);
}

bool ready(const Sm110FrontendBindings* value) {
    return value && value->shape && value->calibration && value->physical &&
           value->weights && value->weights->finished() && value->driver &&
           value->driver->status().ok_status();
}

void* data(const Pi05ResolvedBuffer& buffer) {
    return buffer.buffer ? frt_buffer_dptr(buffer.buffer) : nullptr;
}

const float* scale_data(const Sm110Buffer& buffer, std::size_t index) {
    return static_cast<const float*>(buffer.device_data()) + index;
}

modalities::Status activation_site(
    const Sm110FrontendBindings& state,
    Pi05LinearWeightKey key,
    int step,
    Pi05LinearActivationSite* site,
    const float** device_scale) {
    modalities::Status status = resolve_pi05_linear_activation_site(
        key, step, *state.shape, site);
    if (!status.ok_status()) return status;
    switch (site->domain) {
        case Pi05LinearDomain::kVision:
            *device_scale = static_cast<const float*>(
                state.physical->vision().unit_scale.device_data());
            break;
        case Pi05LinearDomain::kEncoder:
            if (site->index >= state.calibration->encoder_scales.size()) {
                return invalid("SM110 encoder activation site is invalid");
            }
            *device_scale = scale_data(
                state.physical->encoder().activation_scales, site->index);
            break;
        case Pi05LinearDomain::kDecoder:
            if (site->index >= state.calibration->decoder_scales.size()) {
                return invalid("SM110 decoder activation site is invalid");
            }
            *device_scale = scale_data(
                state.physical->decoder().activation_scales, site->index);
            break;
    }
    return *device_scale ? modalities::Status::ok()
                         : invalid("SM110 activation scale is unavailable");
}

bool empty_epilogue(const Pi05LinearEpilogue& epilogue) {
    return epilogue.kind == Pi05LinearEpilogueKind::kNone &&
           !epilogue.bias && !epilogue.residual;
}

modalities::Status frontend_linear(
    void* state,
    const Pi05ResolvedWeight& weight,
    const void* input,
    void* output,
    int rows,
    int input_width,
    int output_width,
    Pi05Stream stream) {
    Sm110FrontendBindings* binding = bindings(state);
    if (!ready(binding) || !weight.device_data || !input || !output) {
        return invalid("SM110 F16 linear binding is invalid");
    }
    if (input_width == kPi05ModelDims.decoder_width &&
        output_width == kPi05ModelDims.action_width) {
        return binding->driver->gmm_fp16_out_fp32(
            input, weight.device_data, static_cast<float*>(output), rows,
            output_width, input_width, stream);
    }
    if (input_width == kPi05ModelDims.action_width &&
        output_width == kPi05ModelDims.decoder_width) {
        return binding->driver->gmm_fp16(
            input, weight.device_data, output, rows, output_width,
            input_width, 0.0f, stream);
    }
    return binding->driver->fp16_nn(
        const_cast<void*>(input), const_cast<void*>(weight.device_data),
        output, rows, output_width, input_width, stream);
}

modalities::Status frontend_projected_linear(
    void* state,
    const Pi05ResolvedWeight& weight,
    Pi05LinearWeightKey key,
    int step,
    const void* input,
    void* output,
    int rows,
    int input_width,
    int output_width,
    bool prequantized,
    const Pi05LinearEpilogue& epilogue,
    Pi05Stream stream) {
    Sm110FrontendBindings* binding = bindings(state);
    if (!ready(binding) || !input || !output) {
        return invalid("SM110 projected-linear binding is invalid");
    }
    if (key.domain == Pi05LinearDomain::kVision &&
        key.role == Pi05LinearRole::kProjector) {
        if (prequantized || !weight.device_data ||
            epilogue.kind != Pi05LinearEpilogueKind::kBias ||
            !epilogue.bias || epilogue.residual) {
            return invalid("SM110 vision projector binding is invalid");
        }
        modalities::Status status = frontend_linear(
            state, weight, input, output, rows, input_width, output_width,
            stream);
        return status.ok_status()
                   ? binding->driver->add_bias_fp16(
                         output, epilogue.bias->device_data, rows,
                         output_width, stream)
                   : status;
    }

    const Sm110Fp8PackedWeight* packed =
        binding->weights->packed_weight(key);
    Pi05LinearActivationSite site;
    const float* activation_scale = nullptr;
    modalities::Status status = activation_site(
        *binding, key, step, &site, &activation_scale);
    if (!status.ok_status() || !packed ||
        packed->view.device_data != weight.device_data ||
        packed->view.scale_data != weight.scale_data) {
        return status.ok_status()
                   ? invalid("SM110 packed linear weight is invalid")
                   : status;
    }

    if (key.domain == Pi05LinearDomain::kVision) {
        const void* linear_input = input;
        if (!prequantized) {
            Sm110Buffer scratch;
            if (key.role == Pi05LinearRole::kAttentionOutput) {
                scratch = binding->physical->vision().state_fp8;
            } else if (key.role == Pi05LinearRole::kMlpDown) {
                scratch = binding->physical->vision().hidden_fp8;
            } else {
                return invalid("SM110 vision FP8 input is invalid");
            }
            status = binding->driver->quantize_fp8_static(
                input, scratch.device_data(), activation_scale,
                static_cast<std::size_t>(rows) * input_width, stream);
            if (!status.ok_status()) return status;
            linear_input = scratch.device_data();
        }
        switch (epilogue.kind) {
            case Pi05LinearEpilogueKind::kBias:
                if (!epilogue.bias || epilogue.residual) break;
                return binding->driver->fp8_nn_bias(
                    const_cast<void*>(linear_input),
                    const_cast<void*>(weight.device_data), output,
                    const_cast<void*>(epilogue.bias->device_data), rows,
                    output_width, input_width, packed->host_scale, stream);
            case Pi05LinearEpilogueKind::kBiasGelu:
                if (!epilogue.bias || epilogue.residual) break;
                return binding->driver->fp8_nn_gelu_bias(
                    const_cast<void*>(linear_input),
                    const_cast<void*>(weight.device_data), output,
                    const_cast<void*>(epilogue.bias->device_data), rows,
                    output_width, input_width, packed->host_scale, stream);
            case Pi05LinearEpilogueKind::kBiasResidual:
                if (!epilogue.bias || !epilogue.residual) break;
                return binding->driver->fp8_nn_bias_residual(
                    const_cast<void*>(linear_input),
                    const_cast<void*>(weight.device_data),
                    epilogue.residual,
                    const_cast<void*>(epilogue.bias->device_data), rows,
                    output_width, input_width, packed->host_scale, stream);
            case Pi05LinearEpilogueKind::kNone: break;
        }
        return invalid("SM110 vision epilogue is invalid");
    }

    if (!prequantized || !empty_epilogue(epilogue)) {
        return invalid("SM110 transformer linear input is not FP8");
    }
    if (key.domain == Pi05LinearDomain::kEncoder) {
        float alpha = packed->host_scale *
                      binding->calibration->encoder_scales[site.index];
        if (!std::isfinite(alpha) || !(alpha > 0.0f)) {
            return invalid("SM110 encoder GEMM scale is invalid");
        }
        Sm110Fp8Tactic tactic = Sm110Fp8Tactic::kSquare;
        if (key.role == Pi05LinearRole::kMlpGateUpGroup) {
            tactic = Sm110Fp8Tactic::kT1;
        } else if (key.role == Pi05LinearRole::kMlpDown) {
            tactic = Sm110Fp8Tactic::kWide;
        }
        return binding->driver->fp8_cutlass(
            const_cast<void*>(input), const_cast<void*>(weight.device_data),
            output, rows, output_width, input_width, alpha, 0.0f, tactic,
            stream);
    }
    return binding->driver->fp8_descale(
        const_cast<void*>(input), const_cast<void*>(weight.device_data),
        output, rows, output_width, input_width, activation_scale,
        weight.scale_data, stream);
}

modalities::Status frontend_add_bias(
    void* state,
    void* values,
    const Pi05ResolvedWeight& bias,
    int rows,
    int columns,
    Pi05Stream stream) {
    Sm110FrontendBindings* binding = bindings(state);
    return ready(binding)
               ? binding->driver->add_bias_fp16(
                     values, bias.device_data, rows, columns, stream)
               : invalid("SM110 bias binding is invalid");
}

modalities::Status frontend_silu(
    void* state, void* values, std::size_t elements, Pi05Stream stream) {
    Sm110FrontendBindings* binding = bindings(state);
    return ready(binding)
               ? binding->driver->precise_silu_fp16(
                     values, elements, stream)
               : invalid("SM110 SiLU binding is invalid");
}

modalities::Status frontend_copy(
    void*, void* destination, const void* source, std::size_t bytes,
    Pi05Stream stream) {
    const cudaError_t result = cudaMemcpyAsync(
        destination, source, bytes, cudaMemcpyDeviceToDevice,
        reinterpret_cast<cudaStream_t>(stream));
    return result == cudaSuccess ? modalities::Status::ok()
                                 : backend(cudaGetErrorString(result));
}

modalities::Status frontend_patchify(
    void* state, const void* images, void* patches, int views,
    Pi05Stream stream) {
    Sm110FrontendBindings* binding = bindings(state);
    return ready(binding)
               ? binding->driver->patch_im2col_fp16(
                     images, patches, views, stream)
               : invalid("SM110 patch binding is invalid");
}

modalities::Status frontend_bias_residual(
    void* state,
    void* residual,
    const void* values,
    const Pi05ResolvedWeight& bias,
    int rows,
    int columns,
    Pi05Stream stream) {
    Sm110FrontendBindings* binding = bindings(state);
    if (!ready(binding) || rows != binding->shape->vision_sequence ||
        columns != kPi05ModelDims.vision_width) {
        return invalid("SM110 patch position binding is invalid");
    }
    return binding->driver->patch_bias_position_fp16(
        residual, bias.device_data, values, rows, columns,
        binding->shape->vision_sequence / binding->shape->num_views,
        stream);
}

modalities::Status frontend_layer_norm(
    void* state,
    const void* values,
    const Pi05ResolvedWeight& weight,
    const Pi05ResolvedWeight& bias,
    void* output,
    int rows,
    int columns,
    float epsilon,
    bool quantize,
    Pi05LinearInput* linear_input,
    Pi05Stream stream) {
    Sm110FrontendBindings* binding = bindings(state);
    if (!ready(binding) || !linear_input) {
        return invalid("SM110 LayerNorm binding is invalid");
    }
    if (quantize) {
        void* fp8 = binding->physical->vision().state_fp8.device_data();
        modalities::Status status = binding->driver->layer_norm_fp8(
            values, fp8, weight.device_data, bias.device_data, rows,
            columns, epsilon, stream);
        if (status.ok_status()) *linear_input = {fp8, true};
        return status;
    }
    modalities::Status status = binding->driver->layer_norm_fp16(
        values, weight.device_data, bias.device_data, output, rows, columns,
        epsilon, stream);
    if (status.ok_status()) *linear_input = {output, false};
    return status;
}

modalities::Status frontend_qkv_split(
    void*, const void* qkv, void* query, void* key, void* value, int,
    int query_width, int key_width, int value_width, Pi05Stream) {
    const auto* base = static_cast<const unsigned char*>(qkv);
    if (!base || query != qkv ||
        key != base + static_cast<std::size_t>(query_width) * 2 ||
        value != base + static_cast<std::size_t>(query_width + key_width) * 2 ||
        key_width != value_width) {
        return invalid("SM110 vision QKV view is invalid");
    }
    return modalities::Status::ok();
}

modalities::Status frontend_normalize_for_linear(
    void* state,
    void* residual,
    const void* update,
    const Pi05ResolvedBuffer&,
    void*,
    Pi05LinearWeightKey key,
    int step,
    int rows,
    int columns,
    float,
    const void** linear_input,
    bool* prequantized,
    Pi05Stream stream) {
    Sm110FrontendBindings* binding = bindings(state);
    if (!ready(binding) || !linear_input || !prequantized ||
        key.domain != Pi05LinearDomain::kEncoder) {
        return invalid("SM110 encoder normalization binding is invalid");
    }
    Pi05LinearActivationSite site;
    const float* scale = nullptr;
    modalities::Status status = activation_site(
        *binding, key, step, &site, &scale);
    if (!status.ok_status()) return status;
    void* fp8 = binding->physical->encoder().state_fp8.device_data();
    if (!update) {
        status = binding->driver->rms_norm_fp8_noweight(
            residual, fp8, rows, columns, scale, stream);
    } else if (key.role == Pi05LinearRole::kAttentionQkv) {
        status = binding->driver->residual_add_fp16(
            residual, update, static_cast<std::size_t>(rows) * columns,
            stream);
        if (status.ok_status()) {
            status = binding->driver->rms_norm_fp8_noweight(
                residual, fp8, rows, columns, scale, stream);
        }
    } else if (key.role == Pi05LinearRole::kMlpGateUpGroup) {
        status = binding->driver->residual_rms_norm_fp8_noweight(
            residual, update, fp8, rows, columns, scale, stream);
    } else {
        return invalid("SM110 encoder normalization role is invalid");
    }
    if (status.ok_status()) {
        *linear_input = fp8;
        *prequantized = true;
    }
    return status;
}

modalities::Status frontend_qkv_rope(
    void* state,
    const void* qkv,
    const Pi05ResolvedBuffer& rope,
    const Pi05ResolvedBuffer* position,
    void* query,
    void* key,
    void* value,
    int rows,
    int query_width,
    int key_width,
    int value_width,
    int head_width,
    Pi05Stream stream) {
    Sm110FrontendBindings* binding = bindings(state);
    if (!ready(binding) || key_width != value_width) {
        return invalid("SM110 QKV/RoPE binding is invalid");
    }
    const int stride = query_width + key_width + value_width;
    if (!position) {
        return binding->driver->qkv_rope_cache_fp16(
            qkv, data(rope), query, key, value, rows, query_width,
            key_width, head_width, stride, 0, head_width, stream);
    }
    return binding->driver->qkv_rope_cache_devpos_fp16(
        qkv, data(rope), query, key, value,
        static_cast<const int*>(data(*position)), rows, query_width,
        key_width, head_width, stride, 0, head_width, stream);
}

modalities::Status frontend_attention(
    void* state,
    Pi05LinearDomain domain,
    int,
    const void* query,
    const void* key,
    const void* value,
    void* output,
    int batches,
    int query_rows,
    int key_rows,
    int query_heads,
    int key_heads,
    int head_width,
    Pi05Stream stream) {
    Sm110FrontendBindings* binding = bindings(state);
    if (!ready(binding)) return invalid("SM110 attention binding is invalid");
    if (domain == Pi05LinearDomain::kVision) {
        return binding->driver->vision_fmha_fp16(
            query, key, value, output, batches, query_rows, key_rows,
            query_heads, key_heads, head_width,
            3 * kPi05ModelDims.vision_width,
            3 * kPi05ModelDims.vision_width, stream);
    }
    const Sm110Buffer& logits =
        domain == Pi05LinearDomain::kEncoder
            ? binding->physical->encoder().logits
            : binding->physical->decoder().logits;
    const Sm110Buffer& valid =
        domain == Pi05LinearDomain::kEncoder
            ? binding->physical->controls().encoder_valid_tokens
            : binding->physical->controls().decoder_valid_tokens;
    const float attention_scale =
        1.0f / std::sqrt(static_cast<float>(head_width));
    return binding->driver->attention_seqused_fp16(
        query, key, value, logits.device_data(), output, query_rows,
        key_rows, query_heads, head_width,
        static_cast<const int*>(valid.device_data()), attention_scale,
        stream);
}

modalities::Status frontend_adaptive_normalize(
    void* state,
    void* residual,
    const void* update,
    const void* update_gate,
    const Pi05ResolvedBuffer&,
    const void* style,
    void* normalized,
    void* gate,
    Pi05LinearWeightKey key,
    int step,
    int rows,
    int columns,
    float,
    bool quantize,
    const void** linear_input,
    bool* prequantized,
    Pi05Stream stream) {
    Sm110FrontendBindings* binding = bindings(state);
    if (!ready(binding) || !linear_input || !prequantized) {
        return invalid("SM110 adaptive normalization binding is invalid");
    }
    modalities::Status status;
    if (!quantize) {
        if (update) {
            status = binding->driver->gate_res_fp16(
                update, update_gate, residual,
                static_cast<std::size_t>(rows) * columns, stream);
            if (!status.ok_status()) return status;
        }
        status = binding->driver->adarms_fp16(
            residual, style, normalized, gate, rows, columns, stream);
        if (status.ok_status()) {
            *linear_input = normalized;
            *prequantized = false;
        }
        return status;
    }

    Pi05LinearActivationSite site;
    const float* scale = nullptr;
    status = activation_site(*binding, key, step, &site, &scale);
    if (!status.ok_status()) return status;
    void* fp8 = binding->physical->decoder().state_fp8.device_data();
    status = update
        ? binding->driver->gate_res_adarms_fp8(
              update, update_gate, residual, style, fp8, gate, rows,
              columns, scale, stream)
        : binding->driver->fused_adarms_fp8(
              residual, style, fp8, gate, rows, columns, scale, stream);
    if (status.ok_status()) {
        *linear_input = fp8;
        *prequantized = true;
    }
    return status;
}

modalities::Status frontend_diffusion_update(
    void* state,
    void* residual,
    const void* update,
    const Pi05ResolvedWeight& bias,
    int rows,
    int columns,
    int num_steps,
    Pi05Stream stream) {
    Sm110FrontendBindings* binding = bindings(state);
    if (!ready(binding) || num_steps <= 0) {
        return invalid("SM110 diffusion update binding is invalid");
    }
    return binding->driver->action_update_fp16(
        static_cast<const float*>(update), bias.device_data, residual, rows,
        columns, -1.0f / static_cast<float>(num_steps), stream);
}

modalities::Status frontend_gate_up(
    void* state,
    const Pi05FeedForwardWeights& weights,
    Pi05LinearWeightKey key,
    int step,
    const void* input,
    bool prequantized,
    void* gate,
    void*,
    int rows,
    int width,
    int hidden_width,
    bool* merged,
    Pi05Stream stream) {
    if (!merged || !weights.gate_up_weight.device_data) {
        return invalid("SM110 gate/up binding is invalid");
    }
    modalities::Status status = frontend_projected_linear(
        state, weights.gate_up_weight, key, step, input, gate, rows, width,
        2 * hidden_width, prequantized, {}, stream);
    if (status.ok_status()) *merged = true;
    return status;
}

modalities::Status frontend_gated_activation(
    void* state,
    const void* gate,
    const void*,
    bool merged,
    void*,
    int rows,
    int hidden_width,
    Pi05LinearWeightKey output_key,
    int step,
    const void** linear_input,
    bool* prequantized,
    Pi05Stream stream) {
    Sm110FrontendBindings* binding = bindings(state);
    if (!ready(binding) || !merged || !linear_input || !prequantized) {
        return invalid("SM110 gated activation binding is invalid");
    }
    Pi05LinearActivationSite site;
    const float* scale = nullptr;
    modalities::Status status = activation_site(
        *binding, output_key, step, &site, &scale);
    if (!status.ok_status()) return status;
    void* output = output_key.domain == Pi05LinearDomain::kEncoder
                       ? binding->physical->encoder().hidden_fp8.device_data()
                       : binding->physical->decoder().hidden_fp8.device_data();
    status = binding->driver->gate_gelu_fp8(
        gate, output, rows, hidden_width, scale, stream);
    if (status.ok_status()) {
        *linear_input = output;
        *prequantized = true;
    }
    return status;
}

modalities::Status frontend_gelu(
    void* state, void* values, std::size_t elements, Pi05Stream stream) {
    Sm110FrontendBindings* binding = bindings(state);
    return ready(binding)
               ? binding->driver->gelu_fp16(values, elements, stream)
               : invalid("SM110 GELU binding is invalid");
}

modalities::Status frontend_vision_pool(
    void*, const void* input, void* output, int views, int grid_height,
    int grid_width, int columns, int factor, Pi05Stream stream) {
    if (!input || !output || views <= 0 || grid_height <= 0 ||
        grid_width <= 0 || columns <= 0 || factor != 1) {
        return invalid("SM110 vision pool binding is unsupported");
    }
    if (input == output) return modalities::Status::ok();
    const std::size_t bytes = static_cast<std::size_t>(views) * grid_height *
                              grid_width * columns * sizeof(std::uint16_t);
    return frontend_copy(nullptr, output, input, bytes, stream);
}

}  // namespace

modalities::Status initialize_sm110_frontend_ops(
    Sm110FrontendBindings* state,
    Pi05FrontendOps* out) {
    if (!ready(state) || !out) {
        return invalid("SM110 frontend bindings are incomplete");
    }
    Pi05FrontendOps result;
    result.profile.activation_dtype = modalities::DType::kFloat16;
    Pi05PrimitiveSet& ops = result.f16;
    ops.state = state;
    ops.linear = frontend_linear;
    ops.projected_linear = frontend_projected_linear;
    ops.add_bias = frontend_add_bias;
    ops.silu = frontend_silu;
    ops.copy = frontend_copy;
    ops.patchify = frontend_patchify;
    ops.bias_residual = frontend_bias_residual;
    ops.layer_norm = frontend_layer_norm;
    ops.qkv_split = frontend_qkv_split;
    ops.normalize_for_linear = frontend_normalize_for_linear;
    ops.qkv_rope = frontend_qkv_rope;
    ops.adaptive_normalize = frontend_adaptive_normalize;
    ops.diffusion_update = frontend_diffusion_update;
    ops.attention = frontend_attention;
    ops.gate_up = frontend_gate_up;
    ops.gated_activation = frontend_gated_activation;
    ops.gelu = frontend_gelu;
    ops.vision_pool = frontend_vision_pool;
    *out = result;
    return modalities::Status::ok();
}

}  // namespace sm110
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

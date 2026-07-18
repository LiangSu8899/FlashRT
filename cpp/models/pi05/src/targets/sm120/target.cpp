#include "flashrt/cpp/models/pi05/targets/sm120/target.h"

#include "activation.cuh"
#include "dit_bf16.cuh"
#include "flashrt/cpp/loader/safetensors.h"
#include "flashrt/cpp/models/pi05/support/native_resource_resolver.h"
#include "flashrt/cpp/models/pi05/support/native_weight_materializer.h"
#include "flashrt/cpp/models/pi05/targets/sm120/attention.h"
#include "flashrt/cpp/models/pi05/targets/sm120/attention_driver.h"
#include "flashrt/cpp/models/pi05/targets/sm120/bf16_linear.h"
#include "flashrt/cpp/models/pi05/targets/sm120/bf16_scratch.h"
#include "flashrt/cpp/models/pi05/targets/sm120/fp8_linear.h"
#include "flashrt/cpp/models/pi05/targets/sm120/fp8_weight_packer.h"
#include "flashrt/cpp/models/pi05/targets/sm120/operations.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <exception>
#include <filesystem>
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

modalities::Status unsupported(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kUnsupported,
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

modalities::Status validate_device() {
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t result = cudaGetDevice(&device);
    if (result == cudaSuccess) {
        result = cudaGetDeviceProperties(&properties, device);
    }
    if (result != cudaSuccess) return backend(cudaGetErrorString(result));
    return properties.major == 12 && properties.minor == 0
               ? modalities::Status::ok()
               : unsupported("SM120 target requires compute capability 12.0");
}

bool vector_weight(const Pi05ResolvedWeight& weight, int width) {
    return width > 0 && weight.device_data &&
           weight.storage == Pi05WeightStorage::kBFloat16 &&
           weight.shape.rank == 1 &&
           weight.shape.dims[0] == static_cast<std::uint64_t>(width) &&
           weight.bytes ==
               static_cast<std::uint64_t>(width) * sizeof(std::uint16_t);
}

void* offset(void* base, std::size_t elements) {
    return static_cast<unsigned char*>(base) +
           elements * sizeof(std::uint16_t);
}

const void* offset(const void* base, std::size_t elements) {
    return static_cast<const unsigned char*>(base) +
           elements * sizeof(std::uint16_t);
}

modalities::Status add_bias(void* values,
                            const Pi05ResolvedWeight& bias,
                            int rows,
                            int columns,
                            Pi05Stream stream) {
    if (!values || rows <= 0 || !vector_weight(bias, columns)) {
        return invalid("SM120 BF16 bias arguments are invalid");
    }
    add_bias_bf16(
        static_cast<__nv_bfloat16*>(values),
        static_cast<const __nv_bfloat16*>(bias.device_data), rows, columns,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status silu(void* values,
                       int elements,
                       Pi05Stream stream) {
    if (!values || elements <= 0) {
        return invalid("SM120 BF16 SiLU arguments are invalid");
    }
    silu_inplace_bf16(static_cast<__nv_bfloat16*>(values), elements,
                      reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status record_time_mlp(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    const Pi05ResolvedResources& resources,
    const Sm120Bf16ScratchBacking& scratch,
    const Sm120Bf16Linear& linear,
    Pi05Stream stream) {
    const int width = kPi05ModelDims.decoder_width;
    const Pi05DecoderGlobalWeights& weights = resources.weights.decoder;
    void* scratch_a = scratch.decoder().normalized.device_data();
    void* scratch_b = scratch.decoder().gate.device_data();
    void* output = frt_buffer_dptr(resources.buffers.time_state.buffer);
    if (!scratch_a || !scratch_b || !output) {
        return invalid("SM120 time MLP buffers are invalid");
    }

    const void* source = offset(
        weights.time_embeddings.device_data,
        static_cast<std::size_t>(call.step) * width);
    modalities::Status status = linear.run(
        weights.time_mlp_in_weight, source, scratch_a, 1, width, width,
        stream);
    if (!status.ok_status()) return status;
    status = add_bias(scratch_a, weights.time_mlp_in_bias, 1, width, stream);
    if (!status.ok_status()) return status;
    status = silu(scratch_a, width, stream);
    if (!status.ok_status()) return status;
    status = linear.run(weights.time_mlp_out_weight, scratch_a, scratch_b, 1,
                        width, width, stream);
    if (!status.ok_status()) return status;
    status = add_bias(scratch_b, weights.time_mlp_out_bias, 1, width, stream);
    if (!status.ok_status()) return status;
    status = silu(scratch_b, width, stream);
    if (!status.ok_status()) return status;

    void* step_output = offset(
        output, static_cast<std::size_t>(call.step) * shape.chunk * width);
    const std::size_t row_bytes =
        static_cast<std::size_t>(width) * sizeof(std::uint16_t);
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    for (int row = 0; row < shape.chunk; ++row) {
        const cudaError_t result = cudaMemcpyAsync(
            offset(step_output, static_cast<std::size_t>(row) * width),
            scratch_b, row_bytes, cudaMemcpyDeviceToDevice, cuda_stream);
        if (result != cudaSuccess) {
            return backend("SM120 time state expansion failed");
        }
    }
    return modalities::Status::ok();
}

modalities::Status record_style(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    const Pi05ResolvedResources& resources,
    const Pi05ResolvedWeight& weight,
    const Pi05ResolvedWeight& bias,
    const Pi05ResolvedBuffer& destination,
    bool layered,
    const Sm120Bf16Linear& linear,
    Pi05Stream stream) {
    const int input_width = kPi05ModelDims.decoder_width;
    const int output_width = 3 * input_width;
    const void* time_state = frt_buffer_dptr(resources.buffers.time_state.buffer);
    void* output = frt_buffer_dptr(destination.buffer);
    if (!time_state || !output) {
        return invalid("SM120 style buffers are invalid");
    }
    const std::size_t input_offset =
        static_cast<std::size_t>(call.step) * shape.chunk * input_width;
    std::size_t output_index = static_cast<std::size_t>(call.step);
    if (layered) {
        output_index = output_index * kPi05ModelDims.decoder_layers +
                       static_cast<std::size_t>(call.layer);
    }
    const std::size_t output_offset =
        output_index * shape.chunk * output_width;
    void* target = offset(output, output_offset);
    modalities::Status status = linear.run(
        weight, offset(time_state, input_offset), target, shape.chunk,
        input_width, output_width, stream);
    return status.ok_status()
               ? add_bias(target, bias, shape.chunk, output_width, stream)
               : status;
}

bool prepare_operation(Pi05OperationId id) {
    return id == Pi05OperationId::kTimeMlp ||
           id == Pi05OperationId::kAttentionStyle ||
           id == Pi05OperationId::kMlpStyle ||
           id == Pi05OperationId::kFinalStyle;
}

bool fp8(Sm120ExecutionMode mode) {
    return mode == Sm120ExecutionMode::kStaticFp8E4M3 ||
           mode == Sm120ExecutionMode::kObservedFp8E4M3;
}

bool static_fp8(Sm120ExecutionMode mode) {
    return mode == Sm120ExecutionMode::kStaticFp8E4M3;
}

bool observed_fp8(Sm120ExecutionMode mode) {
    return mode == Sm120ExecutionMode::kObservedFp8E4M3;
}

bool valid_config(const Sm120TargetConfig& config) {
    if (config.checkpoint_path.empty()) return false;
    switch (config.execution_mode) {
        case Sm120ExecutionMode::kBf16:
            return !config.calibration.has_value();
        case Sm120ExecutionMode::kStaticFp8E4M3:
            return config.calibration.has_value();
        case Sm120ExecutionMode::kObservedFp8E4M3:
            return !config.calibration.has_value();
    }
    return false;
}

}  // namespace

struct Sm120TargetBundle::Impl final {
    enum class State {
        kConstructed = 0,
        kResourcesInitialized,
        kResourcesResolved,
        kPrepareComplete,
        kSetupFinalized,
        kCaptureInputsInitialized,
        kFailed,
    };

    Impl(frt_ctx context,
         Pi05ResolvedShape resolved_shape,
         Sm120TargetConfig target_config)
        : shape(resolved_shape),
          config(std::move(target_config)),
          weights(context),
          workspace(context),
          attention(context),
          scratch(context) {}

    modalities::Status fail(modalities::Status status) {
        state = State::kFailed;
        return status;
    }

    Pi05ResolvedShape shape;
    Sm120TargetConfig config;
    NativeDeviceWeightStore weights;
    NativeWorkspace workspace;
    Sm120AttentionBacking attention;
    Sm120Bf16ScratchBacking scratch;
    std::unique_ptr<Sm120Bf16Linear> bf16_linear;
    std::unique_ptr<Sm120Fp8WeightPacker> fp8_packer;
    std::unique_ptr<Sm120Fp8ActivationBacking> fp8_activation;
    std::unique_ptr<Sm120Fp8Linear> fp8_linear;
    std::unique_ptr<Sm120AttentionDriver> attention_driver;
    std::unique_ptr<Sm120Operations> operations;
    Pi05ResolvedResources resources;
    Pi05NativeSupportBuffers support;
    std::size_t prepare_cursor = 0;
    int prompt_length = 0;
    State state = State::kConstructed;
};

Sm120TargetBundle::Sm120TargetBundle(frt_ctx context,
                                     std::unique_ptr<Impl> impl)
    : Pi05TargetBundle(context, false), impl_(std::move(impl)) {}

Sm120TargetBundle::~Sm120TargetBundle() = default;

std::unique_ptr<Sm120TargetBundle> Sm120TargetBundle::create(
    frt_ctx context,
    const Pi05ResolvedShape& shape,
    Sm120TargetConfig config,
    modalities::Status* status) {
    modalities::Status validation = validate_pi05_resolved_shape(shape);
    if (!context || !valid_config(config) || !validation.ok_status()) {
        set_status(status,
                   context && !validation.ok_status()
                       ? validation
                       : invalid("SM120 target configuration is invalid"));
        return nullptr;
    }
    try {
        std::unique_ptr<Impl> impl(
            new Impl(context, shape, std::move(config)));
        std::unique_ptr<Sm120TargetBundle> target(
            new Sm120TargetBundle(context, std::move(impl)));
        set_status(status, modalities::Status::ok());
        return target;
    } catch (const std::exception& error) {
        set_status(status, backend(error.what()));
    } catch (...) {
        set_status(status, backend("SM120 target allocation failed"));
    }
    return nullptr;
}

modalities::Status Sm120TargetBundle::initialize_resources() {
    if (!impl_ || impl_->state != Impl::State::kConstructed) {
        return invalid("SM120 target resource state is invalid");
    }
    try {
        modalities::Status status = validate_device();
        if (!status.ok_status()) return impl_->fail(std::move(status));
        if (fp8(impl_->config.execution_mode)) {
            status = Sm120Fp8Linear::runtime_status();
            if (!status.ok_status()) return impl_->fail(std::move(status));
        }

        const std::filesystem::path checkpoint =
            std::filesystem::path(impl_->config.checkpoint_path) /
            "model.safetensors";
        loader::SafetensorsFile source;
        if (!source.open(checkpoint.string())) {
            return impl_->fail(modalities::Status::error(
                modalities::StatusCode::kNotFound, source.error()));
        }
        NativeWeightMaterializer materializer(source, &impl_->weights);
        NativeMaterializationOptions options;
        options.num_steps = impl_->shape.num_steps;
        options.merge_decoder_gate_up = false;
        options.include_embedding = true;
        status = materializer.materialize_all(options);
        if (!status.ok_status()) return impl_->fail(std::move(status));

        NativeWorkspaceConfig workspace_config;
        workspace_config.num_views = impl_->shape.num_views;
        workspace_config.max_prompt_tokens =
            impl_->shape.max_prompt_tokens;
        workspace_config.chunk_size = impl_->shape.chunk;
        workspace_config.num_steps = impl_->shape.num_steps;
        workspace_config.vision_pool_factor =
            impl_->shape.vision_pool_factor;
        NativeWorkspaceRequirements requirements;
        requirements.activation_dtype = modalities::DType::kBFloat16;
        status = impl_->workspace.allocate(workspace_config, requirements);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = impl_->workspace.expand_vision_position_embedding(
            impl_->weights);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = impl_->attention.allocate(impl_->shape);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = impl_->scratch.allocate(
            impl_->shape, fp8(impl_->config.execution_mode));
        if (!status.ok_status()) return impl_->fail(std::move(status));

        impl_->bf16_linear.reset(new (std::nothrow) Sm120Bf16Linear());
        if (!impl_->bf16_linear) {
            return impl_->fail(backend("SM120 BF16 linear allocation failed"));
        }
        status = impl_->bf16_linear->status();
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = impl_->attention.set_prompt_length(0);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = impl_->workspace.update_decoder_rope(0);
        if (!status.ok_status()) return impl_->fail(std::move(status));

        impl_->state = Impl::State::kResourcesInitialized;
        return modalities::Status::ok();
    } catch (const std::exception& error) {
        return impl_->fail(backend(error.what()));
    } catch (...) {
        return impl_->fail(backend("SM120 resource initialization failed"));
    }
}

modalities::Status Sm120TargetBundle::resolve_resources(
    Pi05ResolvedResources* out) {
    if (!impl_ || !out ||
        impl_->state != Impl::State::kResourcesInitialized) {
        return invalid("SM120 target resolve state is invalid");
    }
    try {
        Pi05TargetBufferBindings target_bindings;
        modalities::Status status =
            impl_->attention.make_target_bindings(&target_bindings);
        if (!status.ok_status()) return impl_->fail(std::move(status));

        Pi05ResolvedResources resolved;
        status = resolve_pi05_native_buffers(
            impl_->workspace, target_bindings, impl_->shape,
            &resolved.buffers);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        Pi05NativeWeightLayout layout;
        layout.encoder = NativeFeedForwardLayout::kSeparateGateUp;
        layout.decoder = NativeFeedForwardLayout::kSeparateGateUp;
        status = resolve_pi05_materialized_weights(
            impl_->weights, impl_->shape, modalities::DType::kBFloat16,
            layout, &resolved.weights);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        Pi05NativeSupportBuffers support;
        status = resolve_pi05_native_support_buffers(
            impl_->workspace, impl_->shape, &support);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        if (fp8(impl_->config.execution_mode)) {
            impl_->fp8_packer.reset(new (std::nothrow)
                                        Sm120Fp8WeightPacker(context()));
            if (!impl_->fp8_packer) {
                return impl_->fail(
                    backend("SM120 FP8 weight packer allocation failed"));
            }
            status = visit_pi05_linear_weight_groups(
                &resolved.weights, impl_->fp8_packer.get());
            if (status.ok_status()) status = impl_->fp8_packer->finish();
            if (!status.ok_status()) return impl_->fail(std::move(status));
        }
        status = validate_pi05_resolved_resources(resolved, impl_->shape);
        if (!status.ok_status()) return impl_->fail(std::move(status));

        if (fp8(impl_->config.execution_mode)) {
            impl_->fp8_activation.reset(new (std::nothrow)
                                            Sm120Fp8ActivationBacking(
                                                context()));
            if (!impl_->fp8_activation) {
                return impl_->fail(
                    backend("SM120 FP8 activation allocation failed"));
            }
            if (static_fp8(impl_->config.execution_mode)) {
                status = impl_->fp8_activation->initialize_static(
                    impl_->shape, *impl_->config.calibration);
            } else {
                status = impl_->fp8_activation->initialize_observer(
                    impl_->shape);
            }
            if (!status.ok_status()) return impl_->fail(std::move(status));
            const Sm120Fp8ExecutionMode linear_mode =
                observed_fp8(impl_->config.execution_mode)
                    ? Sm120Fp8ExecutionMode::kObserve
                    : Sm120Fp8ExecutionMode::kStatic;
            impl_->fp8_linear.reset(new (std::nothrow) Sm120Fp8Linear(
                impl_->fp8_activation.get(), linear_mode));
            if (!impl_->fp8_linear) {
                return impl_->fail(
                    backend("SM120 FP8 linear allocation failed"));
            }
            status = impl_->fp8_linear->status();
            if (!status.ok_status()) return impl_->fail(std::move(status));
        }

        impl_->resources = resolved;
        impl_->support = support;
        impl_->state = Impl::State::kResourcesResolved;
        *out = resolved;
        return modalities::Status::ok();
    } catch (const std::exception& error) {
        return impl_->fail(backend(error.what()));
    } catch (...) {
        return impl_->fail(backend("SM120 resource resolution failed"));
    }
}

modalities::Status Sm120TargetBundle::record(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    Pi05Stream stream) {
    if (!impl_ || !pi05_resolved_shape_equal(shape, impl_->shape)) {
        return invalid("SM120 target operation shape is invalid");
    }
    modalities::Status status = validate_pi05_operation_call(call, shape);
    if (!status.ok_status()) return status;
    if (prepare_operation(call.id)) {
        if (impl_->state != Impl::State::kResourcesResolved) {
            return invalid("SM120 prepare operation state is invalid");
        }
        switch (call.id) {
            case Pi05OperationId::kTimeMlp:
                status = record_time_mlp(
                    call, shape, impl_->resources, impl_->scratch,
                    *impl_->bf16_linear, stream);
                break;
            case Pi05OperationId::kAttentionStyle: {
                const Pi05DecoderLayerWeights& layer =
                    impl_->resources.weights.decoder_layers[
                        static_cast<std::size_t>(call.layer)];
                status = record_style(
                    call, shape, impl_->resources,
                    layer.attention_mod_weight, layer.attention_mod_bias,
                    impl_->resources.buffers.attention_style, true,
                    *impl_->bf16_linear, stream);
                break;
            }
            case Pi05OperationId::kMlpStyle: {
                const Pi05DecoderLayerWeights& layer =
                    impl_->resources.weights.decoder_layers[
                        static_cast<std::size_t>(call.layer)];
                status = record_style(
                    call, shape, impl_->resources, layer.mlp_mod_weight,
                    layer.mlp_mod_bias, impl_->resources.buffers.mlp_style,
                    true, *impl_->bf16_linear, stream);
                break;
            }
            case Pi05OperationId::kFinalStyle:
                status = record_style(
                    call, shape, impl_->resources,
                    impl_->resources.weights.decoder.final_norm_mod_weight,
                    impl_->resources.weights.decoder.final_norm_mod_bias,
                    impl_->resources.buffers.final_style, false,
                    *impl_->bf16_linear, stream);
                break;
            default:
                return unsupported("SM120 prepare operation is unsupported");
        }
        if (!status.ok_status()) return impl_->fail(std::move(status));
        ++impl_->prepare_cursor;
        if (call.id == Pi05OperationId::kFinalStyle &&
            call.step == shape.num_steps - 1) {
            impl_->state = Impl::State::kPrepareComplete;
        }
        return modalities::Status::ok();
    }

    if (impl_->state != Impl::State::kCaptureInputsInitialized ||
        !impl_->operations) {
        return invalid("SM120 forward operation state is invalid");
    }
    switch (call.id) {
        case Pi05OperationId::kComposePrompt:
            status = impl_->operations->compose_prompt(stream);
            break;
        case Pi05OperationId::kVisionEmbed:
            status = impl_->operations->vision_embed(stream);
            break;
        case Pi05OperationId::kVisionAttention:
            status = impl_->operations->vision_attention(call.layer, stream);
            break;
        case Pi05OperationId::kVisionMlp:
            status = impl_->operations->vision_mlp(call.layer, stream);
            break;
        case Pi05OperationId::kVisionProject:
            status = impl_->operations->vision_project(stream);
            break;
        case Pi05OperationId::kEncoderAttention:
            status = impl_->operations->encoder_attention(call.layer, stream);
            break;
        case Pi05OperationId::kEncoderMlp:
            status = impl_->operations->encoder_mlp(call.layer, stream);
            break;
        case Pi05OperationId::kEncoderCacheFinalize:
            status = impl_->operations->encoder_cache_finalize(call.layer,
                                                                stream);
            break;
        case Pi05OperationId::kDiffusionInputProject:
            status = impl_->operations->diffusion_input_project(call.step,
                                                                 stream);
            break;
        case Pi05OperationId::kDecoderAttention:
            status = impl_->operations->decoder_attention(
                call.layer, call.step, stream);
            break;
        case Pi05OperationId::kDecoderMlp:
            status = impl_->operations->decoder_mlp(call.layer, call.step,
                                                     stream);
            break;
        case Pi05OperationId::kActionProject:
            status = impl_->operations->action_project(call.step, stream);
            break;
        case Pi05OperationId::kDiffusionUpdate:
            status = impl_->operations->diffusion_update(call.step, stream);
            break;
        default:
            return unsupported("SM120 forward operation is not installed");
    }
    if (!status.ok_status()) return impl_->fail(std::move(status));
    return modalities::Status::ok();
}

modalities::Status Sm120TargetBundle::finalize_setup() {
    if (!impl_ || impl_->state != Impl::State::kPrepareComplete) {
        return invalid("SM120 target prepare is incomplete");
    }
    const cudaError_t synchronized = cudaDeviceSynchronize();
    if (synchronized != cudaSuccess) {
        return impl_->fail(backend(cudaGetErrorString(synchronized)));
    }
    impl_->attention_driver.reset(
        new (std::nothrow) Sm120AttentionDriver(&impl_->attention));
    if (!impl_->attention_driver) {
        return impl_->fail(backend("SM120 attention driver allocation failed"));
    }
    modalities::Status status = impl_->attention_driver->status();
    if (!status.ok_status()) return impl_->fail(std::move(status));
    if (impl_->fp8_linear) {
        status = Sm120Operations::autotune_fp8(
            impl_->shape, &impl_->resources, impl_->scratch,
            impl_->fp8_linear.get());
        if (!status.ok_status()) return impl_->fail(std::move(status));
    }
    impl_->operations = Sm120Operations::create(
        impl_->shape, impl_->resources, impl_->support, impl_->scratch,
        impl_->attention, *impl_->attention_driver, *impl_->bf16_linear,
        impl_->fp8_linear.get(), &status);
    if (!impl_->operations || !status.ok_status()) {
        return impl_->fail(std::move(status));
    }
    impl_->state = Impl::State::kSetupFinalized;
    return modalities::Status::ok();
}

modalities::Status Sm120TargetBundle::initialize_capture_inputs() {
    if (!impl_ || impl_->state != Impl::State::kSetupFinalized) {
        return invalid("SM120 capture input state is invalid");
    }
    const Pi05ResolvedBuffer* buffers[] = {
        &impl_->resources.buffers.images,
        &impl_->resources.buffers.prompt_embedding,
        &impl_->resources.buffers.encoder_state,
        &impl_->resources.buffers.noise,
    };
    for (const Pi05ResolvedBuffer* buffer : buffers) {
        if (!buffer->buffer ||
            cudaMemset(frt_buffer_dptr(buffer->buffer), 0,
                       frt_buffer_bytes(buffer->buffer)) != cudaSuccess) {
            return impl_->fail(backend("SM120 capture input reset failed"));
        }
    }
    const cudaError_t synchronized = cudaDeviceSynchronize();
    if (synchronized != cudaSuccess) {
        return impl_->fail(backend(cudaGetErrorString(synchronized)));
    }
    impl_->state = Impl::State::kCaptureInputsInitialized;
    return modalities::Status::ok();
}

modalities::Status Sm120TargetBundle::reset_after_warmup() {
    if (!impl_ ||
        impl_->state != Impl::State::kCaptureInputsInitialized ||
        !impl_->resources.buffers.noise.buffer) {
        return invalid("SM120 warmup reset state is invalid");
    }
    const cudaError_t result = cudaMemset(
        frt_buffer_dptr(impl_->resources.buffers.noise.buffer), 0,
        frt_buffer_bytes(impl_->resources.buffers.noise.buffer));
    return result == cudaSuccess
               ? modalities::Status::ok()
               : impl_->fail(backend(cudaGetErrorString(result)));
}

modalities::Status Sm120TargetBundle::set_prompt_length(
    int prompt_tokens) {
    if (!impl_ || impl_->state == Impl::State::kConstructed ||
        impl_->state == Impl::State::kFailed || prompt_tokens < 0 ||
        prompt_tokens > impl_->shape.max_prompt_tokens) {
        return invalid("SM120 prompt length is out of range");
    }
    const int previous = impl_->prompt_length;
    modalities::Status status =
        impl_->attention.set_prompt_length(prompt_tokens);
    if (!status.ok_status()) return status;
    status = impl_->workspace.update_decoder_rope(prompt_tokens);
    if (!status.ok_status()) {
        impl_->attention.set_prompt_length(previous);
        impl_->workspace.update_decoder_rope(previous);
        return status;
    }
    impl_->prompt_length = prompt_tokens;
    return modalities::Status::ok();
}

const Pi05ResolvedResources* Sm120TargetBundle::resolved_resources() const {
    if (!impl_ || impl_->state == Impl::State::kConstructed ||
        impl_->state == Impl::State::kResourcesInitialized ||
        impl_->state == Impl::State::kFailed) {
        return nullptr;
    }
    return &impl_->resources;
}

std::size_t Sm120TargetBundle::materialized_weight_count() const {
    return impl_ ? impl_->weights.size() : 0;
}

std::size_t Sm120TargetBundle::packed_weight_count() const {
    return impl_ && impl_->fp8_packer ? impl_->fp8_packer->size() : 0;
}

std::size_t Sm120TargetBundle::autotuned_shape_count() const {
    return impl_ && impl_->fp8_linear
               ? impl_->fp8_linear->autotuned_shape_count()
               : 0;
}

std::size_t Sm120TargetBundle::prepare_call_count() const {
    return impl_ ? impl_->prepare_cursor : 0;
}

Sm120ExecutionMode Sm120TargetBundle::execution_mode() const {
    return impl_ ? impl_->config.execution_mode
                 : Sm120ExecutionMode::kBf16;
}

modalities::Status Sm120TargetBundle::reset_observer_scales(
    Pi05Stream stream) {
    if (!impl_ || impl_->state != Impl::State::kCaptureInputsInitialized ||
        !observed_fp8(impl_->config.execution_mode) ||
        !impl_->fp8_activation) {
        return invalid("SM120 observer reset state is invalid");
    }
    return impl_->fp8_activation->reset_observer_scales(stream);
}

modalities::Status Sm120TargetBundle::download_observer_scales(
    std::vector<float>* vision,
    std::vector<float>* encoder,
    std::vector<float>* decoder) const {
    if (!impl_ || impl_->state != Impl::State::kCaptureInputsInitialized ||
        !observed_fp8(impl_->config.execution_mode) ||
        !impl_->fp8_activation) {
        return invalid("SM120 observer download state is invalid");
    }
    return impl_->fp8_activation->download_observer_scales(
        vision, encoder, decoder);
}

bool Sm120TargetBundle::ready_for_capture() const {
    return impl_ &&
           impl_->state == Impl::State::kCaptureInputsInitialized;
}

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

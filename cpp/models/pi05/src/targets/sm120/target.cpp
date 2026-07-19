#include "flashrt/cpp/models/pi05/targets/sm120/target.h"

#include "activation.cuh"
#include "dit_bf16.cuh"
#include "elementwise.cuh"
#include "flashrt/cpp/loader/safetensors.h"
#include "flashrt/cpp/models/pi05/model/frontend_ops.h"
#include "flashrt/cpp/models/pi05/support/native_resource_resolver.h"
#include "flashrt/cpp/models/pi05/support/native_weight_materializer.h"
#include "flashrt/cpp/models/pi05/targets/sm120/attention.h"
#include "flashrt/cpp/models/pi05/targets/sm120/attention_driver.h"
#include "flashrt/cpp/models/pi05/targets/sm120/bf16_linear.h"
#include "flashrt/cpp/models/pi05/targets/sm120/bf16_scratch.h"
#include "flashrt/cpp/models/pi05/targets/sm120/fp8_linear.h"
#include "flashrt/cpp/models/pi05/targets/sm120/fp8_weight_packer.h"
#include "flashrt/cpp/models/pi05/targets/sm120/operations.h"
#include "norm.cuh"
#include "patch_embed.cuh"
#include "quantize.cuh"
#include "rope.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <exception>
#include <filesystem>
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

bool matrix_weight(const Pi05ResolvedWeight& weight,
                   int rows,
                   int columns) {
    return rows > 0 && columns > 0 && weight.device_data &&
           weight.storage == Pi05WeightStorage::kBFloat16 &&
           weight.shape.rank == 2 &&
           weight.shape.dims[0] == static_cast<std::uint64_t>(rows) &&
           weight.shape.dims[1] == static_cast<std::uint64_t>(columns) &&
           static_cast<std::uint64_t>(rows) <=
               std::numeric_limits<std::uint64_t>::max() /
                   static_cast<std::uint64_t>(columns) &&
           weight.bytes == static_cast<std::uint64_t>(rows) *
                               static_cast<std::uint64_t>(columns) *
                               sizeof(std::uint16_t);
}

modalities::Status derive_action_step_weights(
    frt_ctx context,
    const Pi05ResolvedShape& shape,
    Pi05ResolvedWeights* weights,
    frt_buffer* weight_backing,
    frt_buffer* bias_backing) {
    if (!context || !weights || !weight_backing || !bias_backing ||
        shape.num_steps <= 0 ||
        !matrix_weight(weights->decoder.action_out_weight,
                       kPi05ModelDims.decoder_width,
                       kPi05ModelDims.action_width) ||
        !vector_weight(weights->decoder.action_out_bias,
                       kPi05ModelDims.action_width)) {
        return invalid("SM120 action weight derivation is invalid");
    }
    const std::uint64_t weight_bytes =
        weights->decoder.action_out_weight.bytes;
    const std::uint64_t bias_bytes = weights->decoder.action_out_bias.bytes;
    frt_buffer derived_weight = frt_buffer_alloc(
        context, "pi05_sm120_action_out_step_weight", weight_bytes);
    frt_buffer derived_bias = frt_buffer_alloc(
        context, "pi05_sm120_action_out_step_bias", bias_bytes);
    if (!derived_weight || !derived_bias ||
        !frt_buffer_dptr(derived_weight) || !frt_buffer_dptr(derived_bias)) {
        return backend("SM120 action weight derivation allocation failed");
    }
    const float scale = -1.0f / static_cast<float>(shape.num_steps);
    scale_bf16_weight_device(
        static_cast<const __nv_bfloat16*>(
            weights->decoder.action_out_weight.device_data),
        static_cast<__nv_bfloat16*>(frt_buffer_dptr(derived_weight)), scale,
        kPi05ModelDims.decoder_width * kPi05ModelDims.action_width);
    scale_bf16_weight_device(
        static_cast<const __nv_bfloat16*>(
            weights->decoder.action_out_bias.device_data),
        static_cast<__nv_bfloat16*>(frt_buffer_dptr(derived_bias)), scale,
        kPi05ModelDims.action_width);
    const cudaError_t launched = cudaGetLastError();
    if (launched != cudaSuccess) return backend(cudaGetErrorString(launched));
    const cudaError_t synchronized = cudaStreamSynchronize(nullptr);
    if (synchronized != cudaSuccess) {
        return backend(cudaGetErrorString(synchronized));
    }

    Pi05ResolvedWeight private_weight = weights->decoder.action_out_weight;
    Pi05ResolvedWeight private_bias = weights->decoder.action_out_bias;
    private_weight.device_data = frt_buffer_dptr(derived_weight);
    private_bias.device_data = frt_buffer_dptr(derived_bias);
    weights->decoder.action_out_weight = private_weight;
    weights->decoder.action_out_bias = private_bias;
    *weight_backing = derived_weight;
    *bias_backing = derived_bias;
    return modalities::Status::ok();
}

struct Sm120FrontendBindings final {
    const Pi05ResolvedShape* shape = nullptr;
    Sm120Bf16Linear* bf16_linear = nullptr;
    Sm120Fp8Linear* fp8_linear = nullptr;
    Sm120AttentionBacking* attention_buffers = nullptr;
    Sm120AttentionDriver* attention = nullptr;
};

Sm120FrontendBindings* frontend(void* state) {
    return static_cast<Sm120FrontendBindings*>(state);
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
    Sm120FrontendBindings* binding = frontend(state);
    if (!binding || !binding->bf16_linear) {
        return invalid("SM120 BF16 linear binding is invalid");
    }
    return binding->bf16_linear->run(
        weight, input, output, rows, input_width, output_width, stream);
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
    Pi05Stream stream) {
    Sm120FrontendBindings* binding = frontend(state);
    if (!binding || !binding->shape || !binding->bf16_linear) {
        return invalid("SM120 projected-linear binding is invalid");
    }
    if (!binding->fp8_linear) {
        return prequantized
                   ? invalid("SM120 BF16 input cannot be prequantized")
                   : binding->bf16_linear->run(
                         weight, input, output, rows, input_width,
                         output_width, stream);
    }
    Pi05LinearActivationSite site;
    modalities::Status status = resolve_pi05_linear_activation_site(
        key, step, *binding->shape, &site);
    if (!status.ok_status()) return status;
    return prequantized
               ? binding->fp8_linear->run_prequantized(
                     weight, site, input, output, rows, input_width,
                     output_width, stream)
               : binding->fp8_linear->run(
                     weight, site, input, output, rows, input_width,
                     output_width, stream);
}

modalities::Status frontend_add_bias(void*,
                                     void* values,
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

modalities::Status frontend_silu(void*,
                                void* values,
                                std::size_t elements,
                                Pi05Stream stream) {
    if (!values || elements <= 0) {
        return invalid("SM120 BF16 SiLU arguments are invalid");
    }
    silu_inplace_bf16(static_cast<__nv_bfloat16*>(values),
                      static_cast<int>(elements),
                      reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status frontend_copy(void*,
                                 void* destination,
                                 const void* source,
                                 std::size_t bytes,
                                 Pi05Stream stream) {
    const cudaError_t result = cudaMemcpyAsync(
        destination, source, bytes, cudaMemcpyDeviceToDevice,
        reinterpret_cast<cudaStream_t>(stream));
    return result == cudaSuccess
               ? modalities::Status::ok()
               : backend(cudaGetErrorString(result));
}

modalities::Status frontend_patchify(void*,
                                     const void* images,
                                     void* patches,
                                     int views,
                                     Pi05Stream stream) {
    ::patch_im2col(
        static_cast<const __half*>(images), static_cast<__half*>(patches),
        views, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status frontend_bias_residual(
    void*,
    void* residual,
    const void* values,
    const Pi05ResolvedWeight& bias,
    int rows,
    int columns,
    Pi05Stream stream) {
    if (!residual || !values || !vector_weight(bias, columns)) {
        return invalid("SM120 bias-residual binding is invalid");
    }
    ::bias_residual(
        static_cast<__nv_bfloat16*>(residual),
        static_cast<const __nv_bfloat16*>(values),
        static_cast<const __nv_bfloat16*>(bias.device_data), rows, columns,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status frontend_layer_norm(
    void*,
    const void* values,
    const Pi05ResolvedWeight& weight,
    const Pi05ResolvedWeight& bias,
    void* output,
    int rows,
    int columns,
    float epsilon,
    Pi05Stream stream) {
    if (!values || !output || !vector_weight(weight, columns) ||
        !vector_weight(bias, columns)) {
        return invalid("SM120 layer-norm binding is invalid");
    }
    ::layer_norm(
        static_cast<const __nv_bfloat16*>(values),
        static_cast<const __nv_bfloat16*>(weight.device_data),
        static_cast<const __nv_bfloat16*>(bias.device_data),
        static_cast<__nv_bfloat16*>(output), rows, columns, epsilon,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status frontend_qkv_split(
    void*,
    const void* qkv,
    void* query,
    void* key,
    void* value,
    int rows,
    int query_width,
    int key_width,
    int value_width,
    Pi05Stream stream) {
    ::qkv_split(
        static_cast<const __nv_bfloat16*>(qkv),
        static_cast<__nv_bfloat16*>(query),
        static_cast<__nv_bfloat16*>(key),
        static_cast<__nv_bfloat16*>(value), rows, query_width, key_width,
        value_width, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status frontend_vision_attention(
    void* state,
    const void* query,
    const void* key,
    const void* value,
    void* output,
    int views,
    int rows,
    int heads,
    int head_width,
    Pi05Stream stream) {
    Sm120FrontendBindings* binding = frontend(state);
    if (!binding || !binding->shape || !binding->attention_buffers ||
        !binding->attention) {
        return invalid("SM120 vision-attention binding is invalid");
    }
    const Sm120VisionAttentionBuffers& buffers =
        binding->attention_buffers->vision();
    if (query != buffers.query.device_data() ||
        key != buffers.key.device_data() ||
        value != buffers.value.device_data() ||
        output != binding->attention->vision_output() ||
        views != binding->shape->num_views ||
        rows != binding->shape->vision_sequence ||
        heads != kPi05ModelDims.vision_heads ||
        head_width != kPi05ModelDims.vision_head_dim) {
        return invalid("SM120 vision-attention contract is invalid");
    }
    return binding->attention->vision(stream);
}

modalities::Status frontend_gelu(void*,
                                 void* values,
                                 std::size_t elements,
                                 Pi05Stream stream) {
    if (!values || !elements ||
        elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return invalid("SM120 GELU binding is invalid");
    }
    ::gelu_inplace(static_cast<__nv_bfloat16*>(values),
                   static_cast<int>(elements),
                   reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status frontend_vision_pool(void*,
                                        const void* input,
                                        void* output,
                                        int views,
                                        int grid_height,
                                        int grid_width,
                                        int columns,
                                        int factor,
                                        Pi05Stream stream) {
    ::avg_pool_vision_tokens(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(output), views, grid_height, grid_width,
        columns, factor, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
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
    frt_buffer action_out_step_weight = nullptr;
    frt_buffer action_out_step_bias = nullptr;
    NativeWorkspace workspace;
    Sm120AttentionBacking attention;
    Sm120Bf16ScratchBacking scratch;
    std::unique_ptr<Sm120Bf16Linear> bf16_linear;
    Sm120FrontendBindings frontend;
    Pi05FrontendOps frontend_ops;
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
        status = derive_action_step_weights(
            context(), impl_->shape, &resolved.weights,
            &impl_->action_out_step_weight, &impl_->action_out_step_bias);
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
        impl_->frontend = {&impl_->shape, impl_->bf16_linear.get(),
                           impl_->fp8_linear.get(), &impl_->attention,
                           nullptr};
        impl_->frontend_ops.profile.activation_dtype =
            modalities::DType::kBFloat16;
        Pi05PrimitiveSet& primitives = impl_->frontend_ops.bf16;
        primitives.state = &impl_->frontend;
        primitives.linear = frontend_linear;
        primitives.projected_linear = frontend_projected_linear;
        primitives.add_bias = frontend_add_bias;
        primitives.silu = frontend_silu;
        primitives.copy = frontend_copy;
        primitives.patchify = frontend_patchify;
        primitives.bias_residual = frontend_bias_residual;
        primitives.layer_norm = frontend_layer_norm;
        primitives.qkv_split = frontend_qkv_split;
        primitives.vision_attention = frontend_vision_attention;
        primitives.gelu = frontend_gelu;
        primitives.vision_pool = frontend_vision_pool;
        impl_->state = Impl::State::kResourcesResolved;
        *out = resolved;
        return modalities::Status::ok();
    } catch (const std::exception& error) {
        return impl_->fail(backend(error.what()));
    } catch (...) {
        return impl_->fail(backend("SM120 resource resolution failed"));
    }
}

modalities::Status Sm120TargetBundle::make_prepare_execution(
    Pi05PrepareExecution* out) {
    if (!impl_ || !out || impl_->state != Impl::State::kResourcesResolved ||
        !impl_->frontend_ops.bf16.linear) {
        return invalid("SM120 prepare execution state is invalid");
    }
    const Sm120DecoderBf16Scratch& decoder = impl_->scratch.decoder();
    *out = {&impl_->resources,
            &impl_->frontend_ops,
            {decoder.normalized.device_data(), decoder.gate.device_data(),
             decoder.normalized.bytes()}};
    return modalities::Status::ok();
}

modalities::Status Sm120TargetBundle::complete_prepare() {
    if (!impl_ || impl_->state != Impl::State::kResourcesResolved) {
        return invalid("SM120 prepare completion state is invalid");
    }
    impl_->prepare_cursor =
        static_cast<std::size_t>(impl_->shape.num_steps) *
        (2 + 2 * kPi05ModelDims.decoder_layers);
    impl_->state = Impl::State::kPrepareComplete;
    return modalities::Status::ok();
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
    if (impl_->state != Impl::State::kCaptureInputsInitialized ||
        !impl_->operations) {
        return invalid("SM120 forward operation state is invalid");
    }
    switch (call.id) {
        case Pi05OperationId::kComposePrompt:
            status = impl_->operations->compose_prompt(stream);
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
    impl_->frontend.attention = impl_->attention_driver.get();
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

modalities::Status Sm120TargetBundle::make_forward_execution(
    Pi05ForwardExecution* out) {
    if (!impl_ || !out || impl_->state != Impl::State::kSetupFinalized ||
        !impl_->attention_driver || !impl_->operations) {
        return invalid("SM120 forward execution state is invalid");
    }
    const Sm120VisionBf16Scratch& scratch = impl_->scratch.vision();
    const Sm120VisionAttentionBuffers& attention = impl_->attention.vision();
    auto data = [](const Pi05ResolvedBuffer& buffer) {
        return buffer.buffer ? frt_buffer_dptr(buffer.buffer) : nullptr;
    };
    out->resources = &impl_->resources;
    out->ops = &impl_->frontend_ops;
    out->vision = {
        data(impl_->support.vision_patches),
        data(impl_->support.expanded_vision_position),
        data(impl_->support.pooled_vision_state),
        scratch.normalized.device_data(),
        scratch.qkv.device_data(),
        scratch.hidden.device_data(),
        attention.query.device_data(),
        attention.key.device_data(),
        attention.value.device_data(),
        impl_->attention_driver->vision_output(),
    };
    out->fallback = this;
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

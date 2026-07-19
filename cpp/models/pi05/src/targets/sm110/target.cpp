#include "flashrt/cpp/models/pi05/targets/sm110/target.h"

#include "flashrt/cpp/loader/safetensors.h"
#include "flashrt/cpp/models/pi05/model/frontend_ops.h"
#include "flashrt/cpp/models/pi05/model/linear_weight_groups.h"
#include "flashrt/cpp/models/pi05/support/native_device_weights.h"
#include "flashrt/cpp/models/pi05/support/native_resource_resolver.h"
#include "flashrt/cpp/models/pi05/support/native_weight_materializer.h"
#include "flashrt/cpp/models/pi05/support/native_workspace.h"
#include "flashrt/cpp/models/pi05/targets/sm110/fp8_weight_packer.h"
#include "flashrt/cpp/models/pi05/targets/sm110/operation_driver.h"
#include "flashrt/cpp/models/pi05/targets/sm110/physical_resources.h"

#include <cuda_runtime_api.h>

#include <exception>
#include <filesystem>
#include <new>
#include <utility>

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

modalities::Status unsupported(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kUnsupported,
                                     message);
}

modalities::Status backend(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
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
    return properties.major == 11 && properties.minor == 0
               ? modalities::Status::ok()
               : unsupported("SM110 target requires compute capability 11.0");
}

bool calibration_matches(const NativeCalibrationArtifact& calibration,
                         const Pi05ResolvedShape& shape) {
    return calibration.activation_dtype == "float16" &&
           calibration.hardware == "sm110" &&
           calibration.num_views == shape.num_views &&
           calibration.max_prompt_tokens == shape.max_prompt_tokens &&
           calibration.chunk_size == shape.chunk &&
           calibration.num_steps == shape.num_steps &&
           calibration.vision_pool_factor == shape.vision_pool_factor &&
           calibration.state_dim == shape.state_dim;
}

bool valid_shape_for_target(const Pi05ResolvedShape& shape) {
    return shape.vision_pool_factor == 1 &&
           (shape.max_prompt_tokens & 1) == 0;
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
    if (!state) return invalid("SM110 F16 linear binding is invalid");
    return static_cast<Sm110OperationDriver*>(state)->fp16_nn(
        const_cast<void*>(input), const_cast<void*>(weight.device_data),
        output, rows, output_width, input_width, stream);
}

modalities::Status frontend_add_bias(void* state,
                                     void* values,
                                     const Pi05ResolvedWeight& bias,
                                     int rows,
                                     int columns,
                                     Pi05Stream stream) {
    if (!state) return invalid("SM110 F16 bias binding is invalid");
    return static_cast<Sm110OperationDriver*>(state)->add_bias_fp16(
        values, bias.device_data, rows, columns, stream);
}

modalities::Status frontend_silu(void* state,
                                void* values,
                                std::size_t elements,
                                Pi05Stream stream) {
    if (!state) return invalid("SM110 F16 SiLU binding is invalid");
    return static_cast<Sm110OperationDriver*>(state)->precise_silu_fp16(
        values, elements, stream);
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

modalities::Status zero_prepare_storage(
    const Pi05ResolvedResources& resources,
    const Sm110PhysicalResources& physical) {
    const Pi05ResolvedBuffer* outputs[] = {
        &resources.buffers.time_state,
        &resources.buffers.attention_style,
        &resources.buffers.mlp_style,
        &resources.buffers.final_style,
    };
    for (const Pi05ResolvedBuffer* output : outputs) {
        if (!output->buffer || !output->physical_bytes ||
            cudaMemset(frt_buffer_dptr(output->buffer), 0,
                       output->physical_bytes) != cudaSuccess) {
            return backend("SM110 prepare output initialization failed");
        }
    }
    const Sm110Buffer* scratch[] = {
        &physical.decoder().normalized,
        &physical.decoder().gate,
    };
    for (const Sm110Buffer* buffer : scratch) {
        if (!buffer->device_data() || !buffer->bytes() ||
            cudaMemset(buffer->device_data(), 0, buffer->bytes()) !=
                cudaSuccess) {
            return backend("SM110 prepare scratch initialization failed");
        }
    }
    const cudaError_t synchronized = cudaDeviceSynchronize();
    return synchronized == cudaSuccess
               ? modalities::Status::ok()
               : backend(cudaGetErrorString(synchronized));
}

}  // namespace

struct Sm110TargetBundle::Impl final {
    enum class State {
        kConstructed = 0,
        kResourcesInitialized,
        kResourcesResolved,
        kPrepareComplete,
        kFailed,
    };

    Impl(frt_ctx context,
         Pi05ResolvedShape resolved_shape,
         Sm110TargetConfig target_config)
        : shape(resolved_shape),
          config(std::move(target_config)),
          weights(context),
          workspace(context),
          physical(context) {}

    modalities::Status fail(modalities::Status status) {
        state = State::kFailed;
        return status;
    }

    bool resources_available() const {
        return state == State::kResourcesResolved ||
               state == State::kPrepareComplete;
    }

    Pi05ResolvedShape shape;
    Sm110TargetConfig config;
    NativeDeviceWeightStore weights;
    NativeWorkspace workspace;
    Sm110PhysicalResources physical;
    std::unique_ptr<Sm110Fp8WeightPacker> packer;
    std::unique_ptr<Sm110OperationDriver> driver;
    Pi05FrontendOps frontend_ops;
    Pi05ResolvedResources resources;
    Pi05NativeSupportBuffers support;
    int committed_prompt_length = 0;
    State state = State::kConstructed;
};

Sm110TargetBundle::Sm110TargetBundle(frt_ctx context,
                                     std::unique_ptr<Impl> impl)
    : Pi05TargetBundle(context, true), impl_(std::move(impl)) {}

Sm110TargetBundle::~Sm110TargetBundle() = default;

std::unique_ptr<Sm110TargetBundle> Sm110TargetBundle::create(
    frt_ctx context,
    const Pi05ResolvedShape& shape,
    Sm110TargetConfig config,
    modalities::Status* status) {
    modalities::Status validation = validate_pi05_resolved_shape(shape);
    if (validation.ok_status()) {
        validation = validate_native_calibration_artifact(config.calibration);
    }
    if (!context || config.checkpoint_path.empty() ||
        !validation.ok_status() || !valid_shape_for_target(shape) ||
        !calibration_matches(config.calibration, shape)) {
        set_status(status,
                   context && !validation.ok_status()
                       ? validation
                       : invalid("SM110 target configuration is invalid"));
        return nullptr;
    }
    try {
        std::unique_ptr<Impl> impl(
            new Impl(context, shape, std::move(config)));
        std::unique_ptr<Sm110TargetBundle> target(
            new Sm110TargetBundle(context, std::move(impl)));
        set_status(status, modalities::Status::ok());
        return target;
    } catch (const std::exception& error) {
        set_status(status, backend(error.what()));
    } catch (...) {
        set_status(status, backend("SM110 target allocation failed"));
    }
    return nullptr;
}

modalities::Status Sm110TargetBundle::initialize_resources() {
    if (!impl_ || impl_->state != Impl::State::kConstructed) {
        return invalid("SM110 target resource state is invalid");
    }
    try {
        modalities::Status status = validate_device();
        if (!status.ok_status()) return impl_->fail(std::move(status));

        impl_->driver.reset(new (std::nothrow) Sm110OperationDriver());
        if (!impl_->driver) {
            return impl_->fail(
                backend("SM110 operation driver allocation failed"));
        }
        status = impl_->driver->status();
        if (!status.ok_status()) return impl_->fail(std::move(status));

        const std::filesystem::path checkpoint =
            std::filesystem::path(impl_->config.checkpoint_path) /
            "model.safetensors";
        loader::SafetensorsFile source;
        if (!source.open(checkpoint.string())) {
            return impl_->fail(modalities::Status::error(
                modalities::StatusCode::kNotFound, source.error()));
        }
        NativeWeightMaterializer materializer(
            source, &impl_->weights, modalities::DType::kFloat16);
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
        requirements.activation_dtype = modalities::DType::kFloat16;
        status = impl_->workspace.allocate(workspace_config, requirements);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = impl_->workspace.expand_vision_position_embedding(
            impl_->weights);
        if (!status.ok_status()) return impl_->fail(std::move(status));

        status = impl_->physical.allocate(impl_->shape);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = impl_->physical.initialize_static(
            impl_->config.calibration);
        if (!status.ok_status()) return impl_->fail(std::move(status));

        impl_->state = Impl::State::kResourcesInitialized;
        return modalities::Status::ok();
    } catch (const std::exception& error) {
        return impl_->fail(backend(error.what()));
    } catch (...) {
        return impl_->fail(backend("SM110 resource initialization failed"));
    }
}

modalities::Status Sm110TargetBundle::resolve_resources(
    Pi05ResolvedResources* out) {
    if (!impl_ || !out ||
        impl_->state != Impl::State::kResourcesInitialized) {
        return invalid("SM110 target resolve state is invalid");
    }
    try {
        Pi05TargetBufferBindings target_bindings;
        modalities::Status status =
            impl_->physical.make_target_bindings(&target_bindings);
        if (!status.ok_status()) return impl_->fail(std::move(status));

        Pi05ResolvedResources resolved;
        status = resolve_pi05_native_buffers(
            impl_->workspace, target_bindings, impl_->shape,
            &resolved.buffers);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        Pi05NativeWeightLayout layout;
        status = resolve_pi05_materialized_weights(
            impl_->weights, impl_->shape, modalities::DType::kFloat16,
            layout, &resolved.weights);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = resolve_pi05_native_support_buffers(
            impl_->workspace, impl_->shape, &impl_->support);
        if (!status.ok_status()) return impl_->fail(std::move(status));

        impl_->packer.reset(
            new (std::nothrow) Sm110Fp8WeightPacker(context()));
        if (!impl_->packer) {
            return impl_->fail(
                backend("SM110 FP8 weight packer allocation failed"));
        }
        status = visit_pi05_linear_weight_groups(
            &resolved.weights, impl_->packer.get());
        if (status.ok_status()) status = impl_->packer->finish();
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = validate_pi05_resolved_resources(resolved, impl_->shape);
        if (!status.ok_status()) return impl_->fail(std::move(status));

        impl_->resources = resolved;
        impl_->frontend_ops.profile.activation_dtype =
            modalities::DType::kFloat16;
        impl_->frontend_ops.f16 = {
            impl_->driver.get(), frontend_linear, frontend_add_bias,
            frontend_silu, frontend_copy};
        status = zero_prepare_storage(impl_->resources, impl_->physical);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        impl_->state = Impl::State::kResourcesResolved;
        *out = impl_->resources;
        return modalities::Status::ok();
    } catch (const std::exception& error) {
        return impl_->fail(backend(error.what()));
    } catch (...) {
        return impl_->fail(backend("SM110 resource resolution failed"));
    }
}

modalities::Status Sm110TargetBundle::make_prepare_execution(
    Pi05PrepareExecution* out) {
    if (!impl_ || !out || impl_->state != Impl::State::kResourcesResolved ||
        !impl_->frontend_ops.f16.linear) {
        return invalid("SM110 prepare execution state is invalid");
    }
    const Sm110DecoderResources& decoder = impl_->physical.decoder();
    *out = {&impl_->resources,
            &impl_->frontend_ops,
            {decoder.normalized.device_data(), decoder.gate.device_data(),
             decoder.normalized.bytes()}};
    return modalities::Status::ok();
}

modalities::Status Sm110TargetBundle::complete_prepare() {
    if (!impl_ || impl_->state != Impl::State::kResourcesResolved) {
        return invalid("SM110 prepare completion state is invalid");
    }
    impl_->state = Impl::State::kPrepareComplete;
    return modalities::Status::ok();
}

modalities::Status Sm110TargetBundle::finalize_setup() {
    return unsupported("SM110 operation composition is not connected");
}

modalities::Status Sm110TargetBundle::initialize_capture_inputs() {
    return unsupported("SM110 operation composition is not connected");
}

modalities::Status Sm110TargetBundle::reset_after_warmup() {
    return unsupported("SM110 operation composition is not connected");
}

modalities::Status Sm110TargetBundle::record(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape,
    Pi05Stream stream) {
    if (!impl_ || !pi05_resolved_shape_equal(shape, impl_->shape)) {
        return invalid("SM110 target operation shape is invalid");
    }
    modalities::Status status = validate_pi05_operation_call(call, shape);
    if (!status.ok_status()) return status;
    (void)stream;
    return unsupported("SM110 forward operation is not connected");
}

modalities::Status Sm110TargetBundle::set_prompt_length(
    int prompt_tokens) {
    if (!impl_ || !impl_->resources_available() ||
        prompt_tokens < 0 ||
        prompt_tokens > impl_->shape.max_prompt_tokens) {
        return invalid("SM110 prompt length is invalid");
    }
    const int rounded_prompt = prompt_tokens + (prompt_tokens & 1);
    if (rounded_prompt > impl_->shape.max_prompt_tokens) {
        return invalid("SM110 prompt length exceeds its even capacity");
    }

    const Pi05ResolvedBuffer& prompt =
        impl_->resources.buffers.prompt_embedding;
    if (!prompt.buffer ||
        prompt.physical_dtype != modalities::DType::kFloat16 ||
        prompt.physical_shape.rank != 2 ||
        prompt.physical_shape.dims[0] !=
            static_cast<std::uint64_t>(impl_->shape.max_prompt_tokens) ||
        prompt.physical_shape.dims[1] !=
            static_cast<std::uint64_t>(kPi05ModelDims.embedding_width)) {
        return impl_->fail(invalid("SM110 prompt backing is invalid"));
    }
    if (rounded_prompt != prompt_tokens && prompt_tokens > 0) {
        const std::size_t row_bytes =
            static_cast<std::size_t>(kPi05ModelDims.embedding_width) *
            sizeof(std::uint16_t);
        auto* base = static_cast<unsigned char*>(
            frt_buffer_dptr(prompt.buffer));
        const cudaError_t copied = cudaMemcpy(
            base + static_cast<std::size_t>(prompt_tokens) * row_bytes,
            base + static_cast<std::size_t>(prompt_tokens - 1) * row_bytes,
            row_bytes, cudaMemcpyDeviceToDevice);
        if (copied != cudaSuccess) {
            return impl_->fail(backend(cudaGetErrorString(copied)));
        }
    }

    modalities::Status status =
        impl_->workspace.update_decoder_rope(rounded_prompt);
    if (!status.ok_status()) return impl_->fail(std::move(status));
    status = impl_->physical.set_prompt_length(rounded_prompt);
    if (!status.ok_status()) {
        modalities::Status rollback = impl_->workspace.update_decoder_rope(
            impl_->committed_prompt_length);
        if (!rollback.ok_status()) {
            return impl_->fail(
                backend("SM110 prompt update rollback failed"));
        }
        return impl_->fail(std::move(status));
    }
    impl_->committed_prompt_length = rounded_prompt;
    return modalities::Status::ok();
}

const Pi05ResolvedResources* Sm110TargetBundle::resolved_resources() const {
    return impl_ && impl_->resources_available()
               ? &impl_->resources
               : nullptr;
}

const Pi05NativeSupportBuffers* Sm110TargetBundle::support_buffers() const {
    return impl_ && impl_->resources_available()
               ? &impl_->support
               : nullptr;
}

const Sm110PhysicalResources* Sm110TargetBundle::physical_resources() const {
    return impl_ ? &impl_->physical : nullptr;
}

const Sm110Fp8WeightPacker* Sm110TargetBundle::weight_packer() const {
    return impl_ ? impl_->packer.get() : nullptr;
}

const Sm110OperationDriver* Sm110TargetBundle::operation_driver() const {
    return impl_ ? impl_->driver.get() : nullptr;
}

std::size_t Sm110TargetBundle::materialized_weight_count() const {
    return impl_ ? impl_->weights.size() : 0;
}

std::size_t Sm110TargetBundle::logical_workspace_count() const {
    return impl_ ? impl_->workspace.logical_size() : 0;
}

std::size_t Sm110TargetBundle::logical_workspace_allocation_count() const {
    return impl_ ? impl_->workspace.allocation_count() : 0;
}

std::size_t Sm110TargetBundle::logical_workspace_bytes() const {
    return impl_ ? impl_->workspace.allocated_bytes() : 0;
}

bool Sm110TargetBundle::resources_ready() const {
    return impl_ && impl_->resources_available();
}

}  // namespace sm110
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

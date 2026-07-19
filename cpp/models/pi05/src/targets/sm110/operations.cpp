#include "flashrt/cpp/models/pi05/targets/sm110/operations.h"

#include "flashrt/cpp/models/pi05/targets/sm110/fp8_weight_packer.h"
#include "flashrt/cpp/models/pi05/targets/sm110/operation_driver.h"
#include "flashrt/cpp/models/pi05/targets/sm110/physical_resources.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
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

modalities::Status backend(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

void set_status(modalities::Status* destination,
                modalities::Status status) {
    if (destination) *destination = std::move(status);
}

void* data(const Pi05ResolvedBuffer& buffer) {
    return buffer.buffer ? frt_buffer_dptr(buffer.buffer) : nullptr;
}

void* offset(void* base, std::size_t elements) {
    return static_cast<std::uint16_t*>(base) + elements;
}

const void* offset(const void* base, std::size_t elements) {
    return static_cast<const std::uint16_t*>(base) + elements;
}

bool f16_weight(const Pi05ResolvedWeight& weight) {
    return weight.device_data && !weight.scale_data && weight.bytes &&
           weight.storage == Pi05WeightStorage::kFloat16;
}

bool f16_private_buffer(const Sm110Buffer& buffer,
                        int rows,
                        int columns) {
    return buffer.device_data() &&
           buffer.dtype == modalities::DType::kFloat16 &&
           buffer.shape.rank == 2 &&
           buffer.shape.dims[0] == static_cast<std::uint64_t>(rows) &&
           buffer.shape.dims[1] == static_cast<std::uint64_t>(columns);
}

}  // namespace

std::unique_ptr<Sm110Operations> Sm110Operations::create(
    const Pi05ResolvedShape& shape,
    const Pi05ResolvedResources& resources,
    const Pi05NativeSupportBuffers& support,
    const Sm110PhysicalResources& physical,
    const Sm110Fp8WeightPacker& packer,
    const Sm110OperationDriver& driver,
    modalities::Status* status) {
    modalities::Status validation =
        validate_pi05_resolved_resources(resources, shape);
    if (!validation.ok_status() || !physical.allocated() ||
        !physical.initialized() ||
        !pi05_resolved_shape_equal(shape, physical.shape()) ||
        !packer.finished()) {
        set_status(status,
                   validation.ok_status()
                       ? invalid("SM110 operation resources are invalid")
                       : validation);
        return nullptr;
    }
    validation = driver.status();
    if (!validation.ok_status()) {
        set_status(status, std::move(validation));
        return nullptr;
    }
    const Sm110DecoderResources& decoder = physical.decoder();
    if (!f16_private_buffer(decoder.normalized, shape.chunk,
                            kPi05ModelDims.decoder_width) ||
        !f16_private_buffer(decoder.gate, shape.chunk,
                            kPi05ModelDims.decoder_width) ||
        !data(resources.buffers.time_state) ||
        !data(resources.buffers.attention_style) ||
        !data(resources.buffers.mlp_style) ||
        !data(resources.buffers.final_style) ||
        !f16_weight(resources.weights.decoder.time_embeddings) ||
        !f16_weight(resources.weights.decoder.time_mlp_in_weight) ||
        !f16_weight(resources.weights.decoder.time_mlp_in_bias) ||
        !f16_weight(resources.weights.decoder.time_mlp_out_weight) ||
        !f16_weight(resources.weights.decoder.time_mlp_out_bias) ||
        !f16_weight(resources.weights.decoder.final_norm_mod_weight) ||
        !f16_weight(resources.weights.decoder.final_norm_mod_bias)) {
        set_status(status, invalid("SM110 prepare resources are invalid"));
        return nullptr;
    }

    std::unique_ptr<Sm110Operations> operations(new (std::nothrow)
        Sm110Operations(shape, resources, support, physical, packer, driver));
    if (!operations) {
        set_status(status, backend("SM110 operation allocation failed"));
        return nullptr;
    }
    set_status(status, modalities::Status::ok());
    return operations;
}

modalities::Status Sm110Operations::time_mlp(
    int step,
    Pi05Stream stream) const {
    if (step < 0 || step >= shape_.num_steps) {
        return invalid("SM110 time MLP step is invalid");
    }
    const int width = kPi05ModelDims.decoder_width;
    const Pi05DecoderGlobalWeights& weights = resources_.weights.decoder;
    void* scratch_a = physical_.decoder().normalized.device_data();
    void* scratch_b = physical_.decoder().gate.device_data();
    const void* source = offset(
        weights.time_embeddings.device_data,
        static_cast<std::size_t>(step) * width);

    modalities::Status status = driver_.fp16_nn(
        const_cast<void*>(source),
        const_cast<void*>(weights.time_mlp_in_weight.device_data), scratch_a,
        1, width, width, stream);
    if (!status.ok_status()) return status;
    status = driver_.add_bias_fp16(
        scratch_a, weights.time_mlp_in_bias.device_data, 1, width, stream);
    if (!status.ok_status()) return status;
    status = driver_.precise_silu_fp16(
        scratch_a, static_cast<std::size_t>(width), stream);
    if (!status.ok_status()) return status;
    status = driver_.fp16_nn(
        scratch_a,
        const_cast<void*>(weights.time_mlp_out_weight.device_data), scratch_b,
        1, width, width, stream);
    if (!status.ok_status()) return status;
    status = driver_.add_bias_fp16(
        scratch_b, weights.time_mlp_out_bias.device_data, 1, width, stream);
    if (!status.ok_status()) return status;
    status = driver_.precise_silu_fp16(
        scratch_b, static_cast<std::size_t>(width), stream);
    if (!status.ok_status()) return status;

    void* destination = offset(
        data(resources_.buffers.time_state),
        static_cast<std::size_t>(step) * shape_.chunk * width);
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    for (int row = 0; row < shape_.chunk; ++row) {
        const cudaError_t copied = cudaMemcpyAsync(
            offset(destination, static_cast<std::size_t>(row) * width),
            scratch_b, static_cast<std::size_t>(width) * sizeof(std::uint16_t),
            cudaMemcpyDeviceToDevice, cuda_stream);
        if (copied != cudaSuccess) {
            return backend("SM110 time-state expansion failed");
        }
    }
    return modalities::Status::ok();
}

modalities::Status Sm110Operations::style(
    const Pi05ResolvedWeight& weight,
    const Pi05ResolvedWeight& bias,
    const Pi05ResolvedBuffer& destination,
    int layer,
    int step,
    Pi05Stream stream) const {
    if (step < 0 || step >= shape_.num_steps ||
        layer < -1 || layer >= kPi05ModelDims.decoder_layers ||
        !f16_weight(weight) || !f16_weight(bias) || !data(destination)) {
        return invalid("SM110 style operation is invalid");
    }
    const int input_width = kPi05ModelDims.decoder_width;
    const int output_width = 3 * input_width;
    const std::size_t input_offset =
        static_cast<std::size_t>(step) * shape_.chunk * input_width;
    std::size_t output_index = static_cast<std::size_t>(step);
    if (layer >= 0) {
        output_index = output_index * kPi05ModelDims.decoder_layers +
                       static_cast<std::size_t>(layer);
    }
    void* target = offset(data(destination),
                          output_index * shape_.chunk * output_width);
    modalities::Status status = driver_.fp16_nn(
        const_cast<void*>(offset(data(resources_.buffers.time_state),
                                 input_offset)),
        const_cast<void*>(weight.device_data), target, shape_.chunk,
        output_width, input_width, stream);
    return status.ok_status()
               ? driver_.add_bias_fp16(target, bias.device_data, shape_.chunk,
                                       output_width, stream)
               : status;
}

modalities::Status Sm110Operations::attention_style(
    int layer,
    int step,
    Pi05Stream stream) const {
    if (layer < 0 || layer >= kPi05ModelDims.decoder_layers) {
        return invalid("SM110 attention-style layer is invalid");
    }
    const Pi05DecoderLayerWeights& weights =
        resources_.weights.decoder_layers[static_cast<std::size_t>(layer)];
    return style(weights.attention_mod_weight, weights.attention_mod_bias,
                 resources_.buffers.attention_style, layer, step, stream);
}

modalities::Status Sm110Operations::mlp_style(
    int layer,
    int step,
    Pi05Stream stream) const {
    if (layer < 0 || layer >= kPi05ModelDims.decoder_layers) {
        return invalid("SM110 MLP-style layer is invalid");
    }
    const Pi05DecoderLayerWeights& weights =
        resources_.weights.decoder_layers[static_cast<std::size_t>(layer)];
    return style(weights.mlp_mod_weight, weights.mlp_mod_bias,
                 resources_.buffers.mlp_style, layer, step, stream);
}

modalities::Status Sm110Operations::final_style(
    int step,
    Pi05Stream stream) const {
    return style(resources_.weights.decoder.final_norm_mod_weight,
                 resources_.weights.decoder.final_norm_mod_bias,
                 resources_.buffers.final_style, -1, step, stream);
}

}  // namespace sm110
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#include "flashrt/cpp/models/pi05/targets/sm110/fp8_weight_packer.h"

#include "quantize.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <limits>
#include <string>
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

modalities::Status backend(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

cudaStream_t cuda_stream(Pi05Stream stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

bool matrix(const Pi05ResolvedWeight& weight,
            int* rows,
            int* columns) {
    if (!weight.device_data || weight.scale_data ||
        weight.storage != Pi05WeightStorage::kFloat16 ||
        weight.shape.rank != 2 || !weight.shape.dims[0] ||
        !weight.shape.dims[1] ||
        weight.shape.dims[0] >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        weight.shape.dims[1] >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    const int matrix_rows = static_cast<int>(weight.shape.dims[0]);
    const int matrix_columns = static_cast<int>(weight.shape.dims[1]);
    if (matrix_rows > std::numeric_limits<int>::max() / matrix_columns ||
        weight.bytes != static_cast<std::uint64_t>(matrix_rows) *
                            static_cast<std::uint64_t>(matrix_columns) *
                            sizeof(__half)) {
        return false;
    }
    if (rows) *rows = matrix_rows;
    if (columns) *columns = matrix_columns;
    return true;
}

std::string buffer_name(const char* kind, std::size_t index) {
    return std::string("pi05_sm110_fp8_") + kind + "_" +
           std::to_string(index);
}

}  // namespace

Sm110Fp8WeightPacker::Sm110Fp8WeightPacker(
    frt_ctx context,
    Pi05Stream stream)
    : context_(context), stream_(stream) {}

modalities::Status Sm110Fp8WeightPacker::fail(
    modalities::Status status) {
    failed_ = true;
    return status;
}

modalities::Status Sm110Fp8WeightPacker::record(
    const Pi05LinearWeightGroup& group) {
    if (!context_ || finished_ || failed_ || !group.first) {
        return fail(invalid("SM110 FP8 weight packer state is invalid"));
    }
    if (group.key.domain == Pi05LinearDomain::kVision &&
        group.key.role == Pi05LinearRole::kProjector) {
        if (group.key.layer != -1 || group.second || group.fused) {
            return fail(invalid("SM110 projector weight group is invalid"));
        }
        return modalities::Status::ok();
    }
    const bool pair = group.key.role ==
                      Pi05LinearRole::kMlpGateUpGroup;
    if ((pair && (!group.second || !group.fused)) ||
        (!pair && (group.second || group.fused))) {
        return fail(invalid("SM110 FP8 weight group is incomplete"));
    }
    const bool transpose = group.key.domain == Pi05LinearDomain::kEncoder;
    modalities::Status status = pack(group, pair, transpose);
    return status.ok_status() ? status : fail(std::move(status));
}

modalities::Status Sm110Fp8WeightPacker::pack(
    const Pi05LinearWeightGroup& group,
    bool pair,
    bool transpose) {
    int rows = 0;
    int columns = 0;
    int second_rows = 0;
    int second_columns = 0;
    if (!matrix(*group.first, &rows, &columns) ||
        (pair && (!matrix(*group.second, &second_rows, &second_columns) ||
                  second_rows != rows || second_columns != columns))) {
        return invalid("SM110 FP8 source weight is invalid");
    }
    const int factor = pair ? 2 : 1;
    if (rows > std::numeric_limits<int>::max() / columns / factor) {
        return invalid("SM110 FP8 packed weight shape overflows");
    }
    const int elements = rows * columns * factor;
    const std::size_t index = packed_.size();
    frt_buffer values = frt_buffer_alloc(
        context_, buffer_name("values", index).c_str(),
        static_cast<std::size_t>(elements));
    frt_buffer scale = frt_buffer_alloc(
        context_, buffer_name("scale", index).c_str(), sizeof(float));
    if (!values || !scale || !frt_buffer_dptr(values) ||
        !frt_buffer_dptr(scale)) {
        return backend("SM110 FP8 packed weight allocation failed");
    }

    if (pair) {
        quantize_fp8_weight_f16_pair_device(
            static_cast<const __half*>(group.first->device_data),
            static_cast<const __half*>(group.second->device_data),
            static_cast<__nv_fp8_e4m3*>(frt_buffer_dptr(values)),
            static_cast<float*>(frt_buffer_dptr(scale)), rows, columns,
            transpose, cuda_stream(stream_));
    } else {
        quantize_fp8_weight_f16_device(
            static_cast<const __half*>(group.first->device_data),
            static_cast<__nv_fp8_e4m3*>(frt_buffer_dptr(values)),
            static_cast<float*>(frt_buffer_dptr(scale)), rows, columns,
            transpose, cuda_stream(stream_));
    }
    const cudaError_t launched = cudaGetLastError();
    if (launched != cudaSuccess) return backend(cudaGetErrorString(launched));

    Pi05ResolvedWeight view;
    view.device_data = frt_buffer_dptr(values);
    view.scale_data = static_cast<const float*>(frt_buffer_dptr(scale));
    view.bytes = static_cast<std::uint64_t>(elements);
    view.storage = Pi05WeightStorage::kFp8E4M3;
    view.shape = pair
        ? modalities::Shape({static_cast<std::uint64_t>(rows),
                             static_cast<std::uint64_t>(2 * columns)})
        : group.first->shape;
    const modalities::Shape physical_shape = transpose
        ? modalities::Shape({static_cast<std::uint64_t>(factor * columns),
                             static_cast<std::uint64_t>(rows)})
        : view.shape;
    packed_.push_back(
        {group.key, values, scale, view, physical_shape, 0.0f});
    if (pair) {
        *group.first = Pi05ResolvedWeight{};
        *group.second = Pi05ResolvedWeight{};
        *group.fused = view;
    } else {
        *group.first = view;
    }
    return modalities::Status::ok();
}

modalities::Status Sm110Fp8WeightPacker::finish() {
    const std::size_t expected =
        static_cast<std::size_t>(
            kPi05ModelDims.vision_layers + kPi05ModelDims.encoder_layers +
            kPi05ModelDims.decoder_layers) *
        kPi05LinearScalesPerLayer;
    if (!context_ || packed_.size() != expected || finished_ || failed_) {
        return fail(invalid("SM110 FP8 weight packer cannot finish"));
    }
    const cudaError_t synchronized =
        cudaStreamSynchronize(cuda_stream(stream_));
    if (synchronized != cudaSuccess) {
        return fail(backend(cudaGetErrorString(synchronized)));
    }
    for (Sm110Fp8PackedWeight& weight : packed_) {
        const cudaError_t copied = cudaMemcpy(
            &weight.host_scale, weight.view.scale_data, sizeof(float),
            cudaMemcpyDeviceToHost);
        if (copied != cudaSuccess) {
            return fail(backend(cudaGetErrorString(copied)));
        }
    }
    finished_ = true;
    return modalities::Status::ok();
}

const Sm110Fp8PackedWeight* Sm110Fp8WeightPacker::packed_weight(
    std::size_t index) const {
    return index < packed_.size() ? &packed_[index] : nullptr;
}

}  // namespace sm110
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

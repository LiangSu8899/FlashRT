#include "flashrt/cpp/models/pi05/targets/sm120/fp8_weight_packer.h"

#include "flashrt/native_cpp/operations.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <limits>
#include <string>
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

cudaStream_t cuda_stream(Pi05Stream stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

bool matrix_elements(const Pi05ResolvedWeight& weight,
                     std::size_t* elements) {
    if (!weight.device_data || weight.scale_data ||
        weight.storage != Pi05WeightStorage::kBFloat16 ||
        weight.shape.rank != 2 || !weight.shape.dims[0] ||
        !weight.shape.dims[1] ||
        weight.shape.dims[0] >
            std::numeric_limits<std::size_t>::max() /
                weight.shape.dims[1]) {
        return false;
    }
    const std::size_t count =
        static_cast<std::size_t>(weight.shape.dims[0]) *
        static_cast<std::size_t>(weight.shape.dims[1]);
    if (count > std::numeric_limits<std::uint64_t>::max() /
                    sizeof(__nv_bfloat16) ||
        weight.bytes != count * sizeof(__nv_bfloat16)) {
        return false;
    }
    if (elements) *elements = count;
    return true;
}

std::string buffer_name(const char* kind, std::size_t index) {
    return std::string("pi05_sm120_fp8_") + kind + "_" +
           std::to_string(index);
}

}  // namespace

Sm120Fp8WeightPacker::Sm120Fp8WeightPacker(
    frt_ctx context,
    Pi05Stream stream)
    : context_(context), stream_(stream) {}

modalities::Status Sm120Fp8WeightPacker::fail(
    modalities::Status status) {
    failed_ = true;
    return status;
}

modalities::Status Sm120Fp8WeightPacker::record(
    const Pi05LinearWeightGroup& group) {
    if (!context_ || finished_ || failed_ || !group.first) {
        return fail(invalid("SM120 FP8 weight packer state is invalid"));
    }
    const bool pair = group.second && group.fused;
    if ((group.second != nullptr) != (group.fused != nullptr)) {
        return fail(invalid("SM120 FP8 weight group is incomplete"));
    }
    modalities::Status status = pair ? pack_pair(group)
                                      : pack_single(group);
    return status.ok_status() ? status : fail(std::move(status));
}

modalities::Status Sm120Fp8WeightPacker::pack_view(
    const Pi05LinearWeightKey& key,
    const Pi05ResolvedWeight& source,
    Pi05ResolvedWeight* destination) {
    std::size_t elements = 0;
    if (!destination || !matrix_elements(source, &elements) ||
        (elements & 1u) != 0 ||
        elements > static_cast<std::size_t>(
                       std::numeric_limits<int>::max())) {
        return invalid("SM120 FP8 source weight is invalid");
    }
    const std::size_t index = packed_.size();
    frt_buffer values = frt_buffer_alloc(
        context_, buffer_name("values", index).c_str(), elements);
    frt_buffer scale = frt_buffer_alloc(
        context_, buffer_name("scale", index).c_str(), sizeof(float));
    if (!values || !scale || !frt_buffer_dptr(values) ||
        !frt_buffer_dptr(scale)) {
        return backend("SM120 FP8 packed weight allocation failed");
    }
    flashrt_native_quantize_fp8_weight_bf16(
        static_cast<const __nv_bfloat16*>(source.device_data),
        static_cast<__nv_fp8_e4m3*>(frt_buffer_dptr(values)),
        static_cast<float*>(frt_buffer_dptr(scale)),
        static_cast<int>(elements), cuda_stream(stream_));
    const cudaError_t launched = cudaGetLastError();
    if (launched != cudaSuccess) {
        return backend(cudaGetErrorString(launched));
    }

    Pi05ResolvedWeight view;
    view.device_data = frt_buffer_dptr(values);
    view.scale_data = static_cast<const float*>(frt_buffer_dptr(scale));
    view.bytes = elements;
    view.storage = Pi05WeightStorage::kFp8E4M3;
    view.shape = source.shape;
    packed_.push_back({key, values, scale, view});
    *destination = view;
    return modalities::Status::ok();
}

modalities::Status Sm120Fp8WeightPacker::pack_single(
    const Pi05LinearWeightGroup& group) {
    const Pi05ResolvedWeight source = *group.first;
    return pack_view(group.key, source, group.first);
}

modalities::Status Sm120Fp8WeightPacker::ensure_merge_scratch(
    std::size_t bytes) {
    if (!bytes) return invalid("SM120 FP8 merge scratch size is invalid");
    if (merge_scratch_ && merge_scratch_bytes_ >= bytes) {
        return modalities::Status::ok();
    }
    frt_buffer scratch = frt_buffer_alloc(
        context_, buffer_name("merge", scratch_generation_++).c_str(),
        bytes);
    if (!scratch || !frt_buffer_dptr(scratch)) {
        return backend("SM120 FP8 merge scratch allocation failed");
    }
    merge_scratch_ = scratch;
    merge_scratch_bytes_ = bytes;
    return modalities::Status::ok();
}

modalities::Status Sm120Fp8WeightPacker::pack_pair(
    const Pi05LinearWeightGroup& group) {
    const Pi05ResolvedWeight left = *group.first;
    const Pi05ResolvedWeight right = *group.second;
    std::size_t left_elements = 0;
    std::size_t right_elements = 0;
    if (!matrix_elements(left, &left_elements) ||
        !matrix_elements(right, &right_elements) ||
        left_elements != right_elements ||
        left.shape.dims[0] != right.shape.dims[0] ||
        left.shape.dims[1] != right.shape.dims[1] ||
        left.shape.dims[1] >
            std::numeric_limits<std::uint64_t>::max() / 2 ||
        left.bytes > std::numeric_limits<std::size_t>::max() / 2) {
        return invalid("SM120 FP8 gate/up source weights are invalid");
    }
    const std::size_t merged_bytes =
        static_cast<std::size_t>(left.bytes) * 2;
    modalities::Status status = ensure_merge_scratch(merged_bytes);
    if (!status.ok_status()) return status;

    const std::size_t rows = static_cast<std::size_t>(left.shape.dims[0]);
    const std::size_t columns =
        static_cast<std::size_t>(left.shape.dims[1]);
    const std::size_t source_pitch = columns * sizeof(__nv_bfloat16);
    const std::size_t destination_pitch = source_pitch * 2;
    auto* destination = static_cast<unsigned char*>(
        frt_buffer_dptr(merge_scratch_));
    cudaError_t copied = cudaMemcpy2DAsync(
        destination, destination_pitch, left.device_data, source_pitch,
        source_pitch, rows, cudaMemcpyDeviceToDevice,
        cuda_stream(stream_));
    if (copied == cudaSuccess) {
        copied = cudaMemcpy2DAsync(
            destination + source_pitch, destination_pitch,
            right.device_data, source_pitch, source_pitch, rows,
            cudaMemcpyDeviceToDevice, cuda_stream(stream_));
    }
    if (copied != cudaSuccess) return backend(cudaGetErrorString(copied));

    Pi05ResolvedWeight merged;
    merged.device_data = destination;
    merged.bytes = merged_bytes;
    merged.storage = Pi05WeightStorage::kBFloat16;
    merged.shape = modalities::Shape({
        left.shape.dims[0], left.shape.dims[1] * 2});
    status = pack_view(group.key, merged, group.fused);
    if (!status.ok_status()) return status;
    *group.first = Pi05ResolvedWeight{};
    *group.second = Pi05ResolvedWeight{};
    return modalities::Status::ok();
}

modalities::Status Sm120Fp8WeightPacker::finish() {
    if (!context_ || packed_.empty() || finished_ || failed_) {
        return fail(invalid("SM120 FP8 weight packer cannot finish"));
    }
    const cudaError_t synchronized =
        cudaStreamSynchronize(cuda_stream(stream_));
    if (synchronized != cudaSuccess) {
        return fail(backend(cudaGetErrorString(synchronized)));
    }
    finished_ = true;
    return modalities::Status::ok();
}

const Sm120Fp8PackedWeight* Sm120Fp8WeightPacker::packed_weight(
    std::size_t index) const {
    return index < packed_.size() ? &packed_[index] : nullptr;
}

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

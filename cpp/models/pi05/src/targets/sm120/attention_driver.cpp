#include "flashrt/cpp/models/pi05/targets/sm120/attention_driver.h"

#include "attention/fa2_wrapper.h"
#include "flashrt/cpp/models/pi05/model/dims.h"
#include "flashrt/native_cpp/operations.h"

#include <cuda_runtime_api.h>

#include <cmath>

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
    const cudaError_t status = cudaGetLastError();
    return status == cudaSuccess ? modalities::Status::ok()
                                 : backend(cudaGetErrorString(status));
}

float inverse_sqrt(int dimension) {
    return static_cast<float>(1.0 / std::sqrt(static_cast<double>(dimension)));
}

}  // namespace

Sm120AttentionDriver::Sm120AttentionDriver(
    const Sm120AttentionBacking* backing) noexcept
    : backing_(backing) {
    if (!backing_ || !backing_->allocated()) {
        error_ = "SM120 attention backing is not allocated";
        return;
    }
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDevice(&device);
    if (status == cudaSuccess) {
        status = cudaGetDeviceProperties(&properties, device);
    }
    if (status != cudaSuccess) {
        error_ = cudaGetErrorString(status);
        return;
    }
    if (properties.major != 12 || properties.minor != 0) {
        error_ = "SM120 attention target requires compute capability 12.0";
        return;
    }
    multiprocessor_count_ = properties.multiProcessorCount;
}

modalities::Status Sm120AttentionDriver::status() const {
    return error_.empty() ? modalities::Status::ok() : backend(error_);
}

modalities::Status Sm120AttentionDriver::vision(
    std::uintptr_t stream) const {
    if (!error_.empty()) return backend(error_);
    const Sm120VisionAttentionBuffers& buffers = backing_->vision();
    const int row_stride =
        kPi05ModelDims.vision_heads * kPi05ModelDims.vision_head_dim;
    const int batch_stride =
        kPi05ModelDims.vision_tokens_per_view * row_stride;
    fvk_attention_fa2_fwd_bf16(
        buffers.query.device_data(), buffers.key.device_data(),
        buffers.value.device_data(), buffers.output.device_data(),
        buffers.logsumexp.device_data(),
        buffers.logsumexp_accumulator.device_data(),
        buffers.output_accumulator.device_data(), backing_->shape().num_views,
        kPi05ModelDims.vision_tokens_per_view,
        kPi05ModelDims.vision_tokens_per_view, kPi05ModelDims.vision_heads,
        kPi05ModelDims.vision_heads, kPi05ModelDims.vision_head_dim,
        batch_stride, row_stride, kPi05ModelDims.vision_head_dim,
        batch_stride, row_stride, kPi05ModelDims.vision_head_dim,
        batch_stride, row_stride, kPi05ModelDims.vision_head_dim,
        batch_stride, row_stride, kPi05ModelDims.vision_head_dim,
        inverse_sqrt(kPi05ModelDims.vision_head_dim),
        multiprocessor_count_, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm120AttentionDriver::encoder(
    int layer,
    std::uintptr_t stream) const {
    if (!error_.empty()) return backend(error_);
    void* key = backing_->key_layer_data(layer);
    void* value = backing_->value_layer_data(layer);
    if (!key || !value) {
        return invalid("SM120 encoder attention layer is invalid");
    }
    const Sm120EncoderAttentionBuffers& buffers = backing_->encoder();
    const Sm120AttentionControlBuffers& controls = backing_->controls();
    const int query_row_stride =
        kPi05ModelDims.encoder_heads * kPi05ModelDims.encoder_head_dim;
    const int query_batch_stride =
        backing_->shape().encoder_sequence * query_row_stride;
    const int cache_batch_stride =
        backing_->shape().total_attention_keys *
        kPi05ModelDims.encoder_head_dim;
    fvk_attention_fa2_fwd_bf16_seqused(
        buffers.query.device_data(), key, value, buffers.output.device_data(),
        buffers.logsumexp.device_data(),
        controls.encoder_valid_tokens.device_data(), 1,
        backing_->shape().encoder_sequence,
        backing_->shape().encoder_sequence, kPi05ModelDims.encoder_heads,
        kPi05ModelDims.encoder_kv_heads, kPi05ModelDims.encoder_head_dim,
        query_batch_stride, query_row_stride,
        kPi05ModelDims.encoder_head_dim, cache_batch_stride,
        kPi05ModelDims.encoder_head_dim, kPi05ModelDims.encoder_head_dim,
        cache_batch_stride, kPi05ModelDims.encoder_head_dim,
        kPi05ModelDims.encoder_head_dim, query_batch_stride,
        query_row_stride, kPi05ModelDims.encoder_head_dim,
        inverse_sqrt(kPi05ModelDims.encoder_head_dim),
        multiprocessor_count_, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm120AttentionDriver::decoder(
    int layer,
    std::uintptr_t stream) const {
    if (!error_.empty()) return backend(error_);
    void* key = backing_->key_layer_data(layer);
    void* value = backing_->value_layer_data(layer);
    if (!key || !value) {
        return invalid("SM120 decoder attention layer is invalid");
    }
    const Sm120DecoderAttentionBuffers& buffers = backing_->decoder();
    const Sm120AttentionControlBuffers& controls = backing_->controls();
    const std::size_t accumulator_count =
        buffers.logsumexp_accumulator.bytes() / sizeof(float);
    flashrt_native_fill_negative_infinity_f32(
        static_cast<float*>(buffers.logsumexp_accumulator.device_data()),
        accumulator_count, reinterpret_cast<cudaStream_t>(stream));
    modalities::Status status = launch_status();
    if (!status.ok_status()) return status;

    const int query_row_stride =
        kPi05ModelDims.decoder_heads * kPi05ModelDims.decoder_head_dim;
    const int query_batch_stride =
        backing_->shape().chunk * query_row_stride;
    const int cache_batch_stride =
        backing_->shape().total_attention_keys *
        kPi05ModelDims.decoder_head_dim;
    fvk_attention_fa2_fwd_bf16_seqused_splitkv(
        buffers.query.device_data(), key, value, buffers.output.device_data(),
        buffers.logsumexp.device_data(),
        controls.decoder_valid_tokens.device_data(),
        buffers.logsumexp_accumulator.device_data(),
        buffers.output_accumulator.device_data(), 1,
        backing_->shape().chunk, backing_->shape().total_attention_keys,
        kPi05ModelDims.decoder_heads, kPi05ModelDims.decoder_kv_heads,
        kPi05ModelDims.decoder_head_dim, query_batch_stride,
        query_row_stride, kPi05ModelDims.decoder_head_dim,
        cache_batch_stride, kPi05ModelDims.decoder_head_dim,
        kPi05ModelDims.decoder_head_dim, cache_batch_stride,
        kPi05ModelDims.decoder_head_dim, kPi05ModelDims.decoder_head_dim,
        query_batch_stride, query_row_stride,
        kPi05ModelDims.decoder_head_dim,
        inverse_sqrt(kPi05ModelDims.decoder_head_dim),
        multiprocessor_count_, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

void* Sm120AttentionDriver::vision_output() const {
    return backing_ ? backing_->vision().output.device_data() : nullptr;
}

void* Sm120AttentionDriver::encoder_output() const {
    return backing_ ? backing_->encoder().output.device_data() : nullptr;
}

void* Sm120AttentionDriver::decoder_output() const {
    return backing_ ? backing_->decoder().output.device_data() : nullptr;
}

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

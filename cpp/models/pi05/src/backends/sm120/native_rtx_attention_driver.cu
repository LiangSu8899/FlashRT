#include "flashrt/cpp/models/pi05/backends/sm120/native_rtx_attention_driver.h"

#include "attention/fa2_wrapper.h"

#include <cuda_runtime_api.h>

#include <cmath>

namespace flashrt {
namespace models {
namespace pi05 {
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
    const cudaError_t rc = cudaGetLastError();
    return rc == cudaSuccess ? modalities::Status::ok()
                             : backend(cudaGetErrorString(rc));
}

__global__ void fill_negative_infinity(float* values, std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = __int_as_float(0xff800000);
}

bool exact_shape(const NativeAttentionBuffer* buffer,
                 std::initializer_list<std::uint64_t> expected) {
    return buffer && buffer->shape == std::vector<std::uint64_t>(expected);
}

float inverse_sqrt(int dimension) {
    // Match the Python producer: evaluate in binary64, then narrow once.
    return static_cast<float>(1.0 / std::sqrt(static_cast<double>(dimension)));
}

}  // namespace

NativeRtxAttentionDriver::NativeRtxAttentionDriver(
    const NativeRtxAttentionWorkspace* workspace) noexcept
    : workspace_(workspace) {
    const NativeAttentionBuffer* vis = find("attn_vis_Q");
    const NativeAttentionBuffer* enc = find("attn_enc_Q");
    const NativeAttentionBuffer* dec = find("attn_dec_Q");
    const NativeAttentionBuffer* kv = find("attn_enc_K");
    if (!vis || !enc || !dec || !kv || vis->shape.size() != 4 ||
        enc->shape.size() != 3 || dec->shape.size() != 3 ||
        kv->shape.size() != 4 || vis->shape[1] != 256 ||
        vis->shape[2] != 16 || vis->shape[3] != 72 ||
        enc->shape[1] != 8 || enc->shape[2] != 256 ||
        dec->shape[1] != 8 || dec->shape[2] != 256 ||
        kv->shape[0] != 18 || kv->shape[2] != 1 || kv->shape[3] != 256) {
        error_ = "Pi0.5 RTX attention workspace is not allocated";
        return;
    }
    num_views_ = static_cast<int>(vis->shape[0]);
    encoder_sequence_ = static_cast<int>(enc->shape[0]);
    chunk_size_ = static_cast<int>(dec->shape[0]);
    total_kv_ = static_cast<int>(kv->shape[1]);
    if (total_kv_ != encoder_sequence_ + chunk_size_ ||
        !exact_shape(find("attn_vis_O"),
                     {static_cast<std::uint64_t>(num_views_), 256, 16, 72}) ||
        !exact_shape(find("attn_enc_O"),
                     {1, static_cast<std::uint64_t>(encoder_sequence_), 8,
                      256}) ||
        !exact_shape(find("attn_dec_O"),
                     {1, static_cast<std::uint64_t>(chunk_size_), 8, 256})) {
        error_ = "Pi0.5 RTX attention workspace shapes are inconsistent";
        return;
    }
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t rc = cudaGetDevice(&device);
    if (rc == cudaSuccess) rc = cudaGetDeviceProperties(&properties, device);
    if (rc != cudaSuccess) {
        error_ = cudaGetErrorString(rc);
        return;
    }
    if (properties.major < 8) {
        error_ = "native BF16 FA2 requires compute capability 8.0 or newer";
        return;
    }
    num_sms_ = properties.multiProcessorCount;
}

const NativeAttentionBuffer* NativeRtxAttentionDriver::find(
    const char* name) const {
    return workspace_ ? workspace_->find(name) : nullptr;
}

modalities::Status NativeRtxAttentionDriver::status() const {
    return error_.empty() ? modalities::Status::ok() : backend(error_);
}

modalities::Status NativeRtxAttentionDriver::vision(
    std::uintptr_t stream) const {
    if (!error_.empty()) return backend(error_);
    const NativeAttentionBuffer* q = find("attn_vis_Q");
    const NativeAttentionBuffer* k = find("attn_vis_K");
    const NativeAttentionBuffer* v = find("attn_vis_V");
    const NativeAttentionBuffer* o = find("attn_vis_O");
    const NativeAttentionBuffer* lse = find("attn_vis_lse");
    const NativeAttentionBuffer* lse_accum = find("attn_vis_lse_accum");
    const NativeAttentionBuffer* o_accum = find("attn_vis_o_accum");
    if (!q || !k || !v || !o || !lse || !lse_accum || !o_accum) {
        return invalid("native vision attention buffers are incomplete");
    }
    constexpr int row_stride = 16 * 72;
    constexpr int batch_stride = 256 * row_stride;
    fvk_attention_fa2_fwd_bf16(
        frt_buffer_dptr(q->buffer), frt_buffer_dptr(k->buffer),
        frt_buffer_dptr(v->buffer), frt_buffer_dptr(o->buffer),
        frt_buffer_dptr(lse->buffer), frt_buffer_dptr(lse_accum->buffer),
        frt_buffer_dptr(o_accum->buffer), num_views_, 256, 256, 16, 16, 72,
        batch_stride, row_stride, 72, batch_stride, row_stride, 72,
        batch_stride, row_stride, 72, batch_stride, row_stride, 72,
        inverse_sqrt(72), num_sms_,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeRtxAttentionDriver::encoder(
    int layer,
    std::uintptr_t stream) const {
    if (!error_.empty()) return backend(error_);
    void* k = workspace_->encoder_k_layer_dptr(layer);
    void* v = workspace_->encoder_v_layer_dptr(layer);
    const NativeAttentionBuffer* q = find("attn_enc_Q");
    const NativeAttentionBuffer* o = find("attn_enc_O");
    const NativeAttentionBuffer* lse = find("attn_enc_lse");
    const NativeAttentionBuffer* seqused = find("attn_enc_seqused");
    if (!q || !k || !v || !o || !lse || !seqused) {
        return invalid("native encoder attention arguments are invalid");
    }
    const int q_row_stride = 8 * 256;
    const int q_batch_stride = encoder_sequence_ * q_row_stride;
    const int kv_batch_stride = total_kv_ * 256;
    fvk_attention_fa2_fwd_bf16_seqused(
        frt_buffer_dptr(q->buffer), k, v, frt_buffer_dptr(o->buffer),
        frt_buffer_dptr(lse->buffer), frt_buffer_dptr(seqused->buffer), 1,
        encoder_sequence_, encoder_sequence_, 8, 1, 256, q_batch_stride,
        q_row_stride, 256, kv_batch_stride, 256, 256, kv_batch_stride, 256,
        256, q_batch_stride, q_row_stride, 256,
        inverse_sqrt(256), num_sms_,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeRtxAttentionDriver::decoder(
    int layer,
    std::uintptr_t stream) const {
    if (!error_.empty()) return backend(error_);
    void* k = workspace_->encoder_k_layer_dptr(layer);
    void* v = workspace_->encoder_v_layer_dptr(layer);
    const NativeAttentionBuffer* q = find("attn_dec_Q");
    const NativeAttentionBuffer* o = find("attn_dec_O");
    const NativeAttentionBuffer* lse = find("attn_dec_lse");
    const NativeAttentionBuffer* seqused = find("attn_dec_seqused");
    const NativeAttentionBuffer* lse_accum = find("attn_dec_lse_accum");
    const NativeAttentionBuffer* o_accum = find("attn_dec_o_accum");
    if (!q || !k || !v || !o || !lse || !seqused || !lse_accum ||
        !o_accum) {
        return invalid("native decoder attention arguments are invalid");
    }
    const std::size_t accum_count =
        frt_buffer_bytes(lse_accum->buffer) / sizeof(float);
    fill_negative_infinity<<<(accum_count + 255) / 256, 256, 0,
                              reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<float*>(frt_buffer_dptr(lse_accum->buffer)), accum_count);
    cudaError_t rc = cudaGetLastError();
    if (rc != cudaSuccess) return backend(cudaGetErrorString(rc));

    const int q_row_stride = 8 * 256;
    const int q_batch_stride = chunk_size_ * q_row_stride;
    const int kv_batch_stride = total_kv_ * 256;
    fvk_attention_fa2_fwd_bf16_seqused_splitkv(
        frt_buffer_dptr(q->buffer), k, v, frt_buffer_dptr(o->buffer),
        frt_buffer_dptr(lse->buffer), frt_buffer_dptr(seqused->buffer),
        frt_buffer_dptr(lse_accum->buffer), frt_buffer_dptr(o_accum->buffer),
        1, chunk_size_, total_kv_, 8, 1, 256, q_batch_stride, q_row_stride,
        256, kv_batch_stride, 256, 256, kv_batch_stride, 256, 256,
        q_batch_stride, q_row_stride, 256, inverse_sqrt(256),
        num_sms_, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

void* NativeRtxAttentionDriver::vision_output() const {
    const NativeAttentionBuffer* output = find("attn_vis_O");
    return output ? frt_buffer_dptr(output->buffer) : nullptr;
}

void* NativeRtxAttentionDriver::encoder_output() const {
    const NativeAttentionBuffer* output = find("attn_enc_O");
    return output ? frt_buffer_dptr(output->buffer) : nullptr;
}

void* NativeRtxAttentionDriver::decoder_output() const {
    const NativeAttentionBuffer* output = find("attn_dec_O");
    return output ? frt_buffer_dptr(output->buffer) : nullptr;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

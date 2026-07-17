#include "flashrt/cpp/models/pi05/backends/sm110/native_thor_fp8_forward.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

constexpr int kVisionWidth = 1152;
constexpr int kVisionHidden = 4304;
constexpr int kVisionHeads = 16;
constexpr int kVisionHeadDimension = 72;
constexpr int kEncoderWidth = 2048;
constexpr int kEncoderHidden = 16384;
constexpr int kDecoderWidth = 1024;
constexpr int kDecoderHidden = 4096;
constexpr int kHeads = 8;
constexpr int kHeadDimension = 256;
constexpr int kLayers = 18;

static_assert(kVisionHeads * kVisionHeadDimension == kVisionWidth,
              "Pi0.5 vision attention shape must cover the hidden width");

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

const NativeWorkspaceBuffer* buffer(
    const NativeWorkspace& workspace, const char* name,
    modalities::DType dtype,
    std::initializer_list<std::uint64_t> shape) {
    const NativeWorkspaceBuffer* value = workspace.find(name);
    return value && value->dtype == dtype &&
                   value->shape == std::vector<std::uint64_t>(shape)
               ? value
               : nullptr;
}

const NativeDeviceWeight* weight(
    const NativeDeviceWeightStore& weights, const std::string& name,
    NativeWeightDType dtype,
    std::initializer_list<std::uint64_t> shape) {
    const NativeDeviceWeight* value = weights.find(name);
    return value && value->dtype == dtype &&
                   value->shape == std::vector<std::uint64_t>(shape)
               ? value
               : nullptr;
}

void* dptr(const NativeWorkspaceBuffer* value) {
    return value ? frt_buffer_dptr(value->buffer) : nullptr;
}

void* dptr(const NativeDeviceWeight* value) {
    return value ? frt_buffer_dptr(value->buffer) : nullptr;
}

void* offset_bytes(void* base, std::size_t bytes) {
    return static_cast<unsigned char*>(base) + bytes;
}

const void* offset_bytes(const void* base, std::size_t bytes) {
    return static_cast<const unsigned char*>(base) + bytes;
}

float* scale_ptr(const NativeWorkspaceBuffer* scales, std::size_t index) {
    return static_cast<float*>(dptr(scales)) + index;
}

float* scale_ptr(const NativeDeviceWeight* scales, std::size_t index) {
    return static_cast<float*>(dptr(scales)) + index;
}

modalities::Status copy_device(void* destination, const void* source,
                               std::size_t bytes, std::uintptr_t stream) {
    const cudaError_t rc = cudaMemcpyAsync(
        destination, source, bytes, cudaMemcpyDeviceToDevice,
        reinterpret_cast<cudaStream_t>(stream));
    return rc == cudaSuccess ? modalities::Status::ok()
                             : backend(cudaGetErrorString(rc));
}

modalities::Status synchronize(std::uintptr_t stream) {
    const cudaError_t rc = cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(stream));
    return rc == cudaSuccess ? modalities::Status::ok()
                             : backend(cudaGetErrorString(rc));
}

modalities::Status measure_scale(
    const NativeThorKernelDriver& driver, const void* values, void* fp8_scratch,
    float* dynamic_scale, float* destination_scale, std::size_t elements,
    std::uintptr_t stream, float* host_value = nullptr) {
    modalities::Status st = driver.quantize_fp8_dynamic(
        values, fp8_scratch, dynamic_scale, elements, stream);
    if (!st.ok_status()) return st;
    st = synchronize(stream);
    if (!st.ok_status()) return st;
    if (host_value) {
        const cudaError_t rc = cudaMemcpy(
            host_value, dynamic_scale, sizeof(*host_value),
            cudaMemcpyDeviceToHost);
        if (rc != cudaSuccess) return backend(cudaGetErrorString(rc));
    }
    return copy_device(destination_scale, dynamic_scale, sizeof(float), stream);
}

}  // namespace

modalities::Status NativeThorFp8Forward::vision(
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    const NativeThorWeightScales& weight_scales,
    std::uintptr_t stream) const {
    if (!driver_ || !driver_->status().ok_status() || !workspace ||
        workspace->flavor() != NativeWorkspaceFlavor::kThorFp8 ||
        weight_scales.vision.size() != 27 * 4) {
        return invalid("Thor vision forward owner is invalid");
    }
    const std::uint64_t sequence = workspace->vision_sequence();
    const std::uint64_t views = workspace->num_views();
    const NativeWorkspaceBuffer* images = buffer(
        *workspace, "observation_images_normalized",
        modalities::DType::kFloat16, {views, 224, 224, 3});
    const NativeWorkspaceBuffer* patches = buffer(
        *workspace, "vision_patches", modalities::DType::kFloat16,
        {sequence, 588});
    const NativeWorkspaceBuffer* position = buffer(
        *workspace, "vision_pos_embed_expanded",
        modalities::DType::kFloat16, {sequence, kVisionWidth});
    const NativeWorkspaceBuffer* x = buffer(
        *workspace, "vision_x", modalities::DType::kFloat16,
        {sequence, kVisionWidth});
    const NativeWorkspaceBuffer* x_fp8 = buffer(
        *workspace, "vision_x_fp8", modalities::DType::kUInt8,
        {sequence, kVisionWidth});
    const NativeWorkspaceBuffer* qkv = buffer(
        *workspace, "vision_QKV", modalities::DType::kFloat16,
        {sequence, 3 * kVisionWidth});
    const NativeWorkspaceBuffer* attention = buffer(
        *workspace, "vision_attn", modalities::DType::kFloat16,
        {sequence, kVisionWidth});
    const NativeWorkspaceBuffer* hidden = buffer(
        *workspace, "vision_hidden", modalities::DType::kFloat16,
        {sequence, kVisionHidden});
    const NativeWorkspaceBuffer* hidden_fp8 = buffer(
        *workspace, "vision_hidden_fp8", modalities::DType::kUInt8,
        {sequence, kVisionHidden});
    const NativeWorkspaceBuffer* unit_scale = buffer(
        *workspace, "vision_unit_scale", modalities::DType::kFloat32, {1});
    const NativeWorkspaceBuffer* encoder_x = buffer(
        *workspace, "encoder_x", modalities::DType::kFloat16,
        {static_cast<std::uint64_t>(workspace->encoder_sequence()),
         kEncoderWidth});
    const NativeDeviceWeight* patch_w = weight(
        weights, "vision_patch_embedding_w", NativeWeightDType::kFloat16,
        {14, 14, 3, kVisionWidth});
    const NativeDeviceWeight* patch_b = weight(
        weights, "vision_patch_embedding_b", NativeWeightDType::kFloat16,
        {kVisionWidth});
    if (!images || !patches || !position || !x || !x_fp8 || !qkv ||
        !attention || !hidden || !hidden_fp8 || !unit_scale || !encoder_x ||
        !patch_w || !patch_b) {
        return invalid("Thor vision buffers or weights are incomplete");
    }

    modalities::Status st = driver_->patch_im2col_fp16(
        dptr(images), dptr(patches), static_cast<int>(views), stream);
    if (!st.ok_status()) return st;
    st = driver_->fp16_nn(dptr(patches), dptr(patch_w), dptr(x),
                          static_cast<int>(sequence), kVisionWidth, 588,
                          stream);
    if (!st.ok_status()) return st;
    st = driver_->patch_bias_position_fp16(
        dptr(x), dptr(patch_b), dptr(position), static_cast<int>(sequence),
        kVisionWidth, 256, stream);
    if (!st.ok_status()) return st;

    for (int layer = 0; layer < 27; ++layer) {
        const std::string suffix = std::to_string(layer);
        const NativeDeviceWeight* ln_attn_w = weight(
            weights, "vision_pre_attn_norm_w_" + suffix,
            NativeWeightDType::kFloat16, {kVisionWidth});
        const NativeDeviceWeight* ln_attn_b = weight(
            weights, "vision_pre_attn_norm_b_" + suffix,
            NativeWeightDType::kFloat16, {kVisionWidth});
        const NativeDeviceWeight* qkv_w = weight(
            weights, "vision_attn_qkv_w_" + suffix,
            NativeWeightDType::kFp8E4M3,
            {kVisionWidth, 3 * kVisionWidth});
        const NativeDeviceWeight* qkv_b = weight(
            weights, "vision_attn_qkv_b_" + suffix,
            NativeWeightDType::kFloat16, {3 * kVisionWidth});
        const NativeDeviceWeight* o_w = weight(
            weights, "vision_attn_o_w_" + suffix,
            NativeWeightDType::kFp8E4M3, {kVisionWidth, kVisionWidth});
        const NativeDeviceWeight* o_b = weight(
            weights, "vision_attn_o_b_" + suffix,
            NativeWeightDType::kFloat16, {kVisionWidth});
        const NativeDeviceWeight* ln_ffn_w = weight(
            weights, "vision_pre_ffn_norm_w_" + suffix,
            NativeWeightDType::kFloat16, {kVisionWidth});
        const NativeDeviceWeight* ln_ffn_b = weight(
            weights, "vision_pre_ffn_norm_b_" + suffix,
            NativeWeightDType::kFloat16, {kVisionWidth});
        const NativeDeviceWeight* up_w = weight(
            weights, "vision_ffn_up_w_" + suffix,
            NativeWeightDType::kFp8E4M3, {kVisionWidth, kVisionHidden});
        const NativeDeviceWeight* up_b = weight(
            weights, "vision_ffn_up_b_" + suffix,
            NativeWeightDType::kFloat16, {kVisionHidden});
        const NativeDeviceWeight* down_w = weight(
            weights, "vision_ffn_down_w_" + suffix,
            NativeWeightDType::kFp8E4M3, {kVisionHidden, kVisionWidth});
        const NativeDeviceWeight* down_b = weight(
            weights, "vision_ffn_down_b_" + suffix,
            NativeWeightDType::kFloat16, {kVisionWidth});
        if (!ln_attn_w || !ln_attn_b || !qkv_w || !qkv_b || !o_w || !o_b ||
            !ln_ffn_w || !ln_ffn_b || !up_w || !up_b || !down_w || !down_b) {
            return invalid("Thor vision layer weights are incomplete");
        }
        const std::size_t scale = static_cast<std::size_t>(layer) * 4;
        st = driver_->layer_norm_fp8(
            dptr(x), dptr(x_fp8), dptr(ln_attn_w), dptr(ln_attn_b),
            static_cast<int>(sequence), kVisionWidth, 1e-5f, stream);
        if (!st.ok_status()) return st;
        st = driver_->fp8_nn_bias(
            dptr(x_fp8), dptr(qkv_w), dptr(qkv), dptr(qkv_b),
            static_cast<int>(sequence), 3 * kVisionWidth, kVisionWidth,
            weight_scales.vision[scale], stream);
        if (!st.ok_status()) return st;
        st = driver_->vision_fmha_fp16(
            dptr(qkv), offset_bytes(dptr(qkv), kVisionWidth * 2),
            offset_bytes(dptr(qkv), 2 * kVisionWidth * 2), dptr(attention),
            static_cast<int>(views), 256, 256, kVisionHeads, kVisionHeads,
            kVisionHeadDimension,
            3 * kVisionWidth, 3 * kVisionWidth, stream);
        if (!st.ok_status()) return st;
        st = driver_->quantize_fp8_static(
            dptr(attention), dptr(x_fp8),
            static_cast<const float*>(dptr(unit_scale)),
            sequence * kVisionWidth, stream);
        if (!st.ok_status()) return st;
        st = driver_->fp8_nn_bias_residual(
            dptr(x_fp8), dptr(o_w), dptr(x), dptr(o_b),
            static_cast<int>(sequence), kVisionWidth, kVisionWidth,
            weight_scales.vision[scale + 1], stream);
        if (!st.ok_status()) return st;
        st = driver_->layer_norm_fp8(
            dptr(x), dptr(x_fp8), dptr(ln_ffn_w), dptr(ln_ffn_b),
            static_cast<int>(sequence), kVisionWidth, 1e-5f, stream);
        if (!st.ok_status()) return st;
        st = driver_->fp8_nn_gelu_bias(
            dptr(x_fp8), dptr(up_w), dptr(hidden), dptr(up_b),
            static_cast<int>(sequence), kVisionHidden, kVisionWidth,
            weight_scales.vision[scale + 2], stream);
        if (!st.ok_status()) return st;
        st = driver_->quantize_fp8_static(
            dptr(hidden), dptr(hidden_fp8),
            static_cast<const float*>(dptr(unit_scale)),
            sequence * kVisionHidden, stream);
        if (!st.ok_status()) return st;
        st = driver_->fp8_nn_bias_residual(
            dptr(hidden_fp8), dptr(down_w), dptr(x), dptr(down_b),
            static_cast<int>(sequence), kVisionWidth, kVisionHidden,
            weight_scales.vision[scale + 3], stream);
        if (!st.ok_status()) return st;
    }

    const NativeDeviceWeight* final_w = weight(
        weights, "vision_final_norm_w", NativeWeightDType::kFloat16,
        {kVisionWidth});
    const NativeDeviceWeight* final_b = weight(
        weights, "vision_final_norm_b", NativeWeightDType::kFloat16,
        {kVisionWidth});
    const NativeDeviceWeight* projector_w = weight(
        weights, "encoder_multi_modal_projector_w",
        NativeWeightDType::kFloat16, {kVisionWidth, kEncoderWidth});
    const NativeDeviceWeight* projector_b = weight(
        weights, "encoder_multi_modal_projector_b",
        NativeWeightDType::kFloat16, {kEncoderWidth});
    if (!final_w || !final_b || !projector_w || !projector_b) {
        return invalid("Thor vision projection weights are incomplete");
    }
    st = driver_->layer_norm_fp16(
        dptr(x), dptr(final_w), dptr(final_b), dptr(attention),
        static_cast<int>(sequence), kVisionWidth, 1e-6f, stream);
    if (!st.ok_status()) return st;
    st = driver_->fp16_nn(
        dptr(attention), dptr(projector_w), dptr(encoder_x),
        static_cast<int>(sequence), kEncoderWidth, kVisionWidth, stream);
    if (!st.ok_status()) return st;
    return driver_->add_bias_fp16(
        dptr(encoder_x), dptr(projector_b), static_cast<int>(sequence),
        kEncoderWidth, stream);
}

modalities::Status NativeThorFp8Forward::encoder(
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    const std::vector<float>& activation_weight_alphas,
    std::uintptr_t stream) const {
    if (!driver_ || !driver_->status().ok_status() || !workspace ||
        workspace->flavor() != NativeWorkspaceFlavor::kThorFp8 ||
        activation_weight_alphas.size() != kLayers * 4) {
        return invalid("Thor encoder forward owner is invalid");
    }
    const std::uint64_t sequence = workspace->encoder_sequence();
    const std::uint64_t keys = workspace->total_keys();
    const NativeWorkspaceBuffer* x = buffer(
        *workspace, "encoder_x", modalities::DType::kFloat16,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* x_fp8 = buffer(
        *workspace, "encoder_x_fp8", modalities::DType::kUInt8,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* qkv = buffer(
        *workspace, "encoder_QKV", modalities::DType::kFloat16,
        {sequence, 2560});
    const NativeWorkspaceBuffer* logits = buffer(
        *workspace, "encoder_logits", modalities::DType::kFloat16,
        {sequence * kHeads, keys});
    const NativeWorkspaceBuffer* attention = buffer(
        *workspace, "encoder_attn", modalities::DType::kFloat16,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* o_fp8 = buffer(
        *workspace, "encoder_o_fp8", modalities::DType::kUInt8,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* gate = buffer(
        *workspace, "encoder_gate_merged", modalities::DType::kFloat16,
        {sequence, 2 * kEncoderHidden});
    const NativeWorkspaceBuffer* hidden_fp8 = buffer(
        *workspace, "encoder_hidden_fp8", modalities::DType::kUInt8,
        {sequence, kEncoderHidden});
    const NativeWorkspaceBuffer* fg = buffer(
        *workspace, "encoder_fg", modalities::DType::kFloat16,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* rope = buffer(
        *workspace, "encoder_rope_weights", modalities::DType::kFloat16,
        {sequence, kHeadDimension});
    const NativeWorkspaceBuffer* activation_scales = buffer(
        *workspace, "encoder_activation_scales",
        modalities::DType::kFloat32, {kLayers, 4});
    const NativeWorkspaceBuffer* key_cache = buffer(
        *workspace, "encoder_k_cache", modalities::DType::kFloat16,
        {kLayers, keys, kHeadDimension});
    const NativeWorkspaceBuffer* value_cache = buffer(
        *workspace, "encoder_v_cache", modalities::DType::kFloat16,
        {kLayers, keys, kHeadDimension});
    const NativeWorkspaceBuffer* valid_keys = buffer(
        *workspace, "attn_enc_seqused", modalities::DType::kUInt8,
        {sizeof(std::int32_t)});
    if (!x || !x_fp8 || !qkv || !logits || !attention || !o_fp8 || !gate ||
        !hidden_fp8 || !fg || !rope || !activation_scales || !key_cache ||
        !value_cache || !valid_keys) {
        return invalid("Thor encoder workspace is incomplete");
    }

    const float attention_scale =
        1.0f / std::sqrt(static_cast<float>(kHeadDimension));
    const std::size_t cache_layer_elements = keys * kHeadDimension;
    for (int layer = 0; layer < kLayers; ++layer) {
        const std::string suffix = std::to_string(layer);
        const NativeDeviceWeight* qkv_w = weight(
            weights, "encoder_attn_qkv_w_" + suffix,
            NativeWeightDType::kFp8E4M3, {2560, kEncoderWidth});
        if (!qkv_w) return invalid("Thor encoder QKV weight is invalid");
        const std::size_t scale = static_cast<std::size_t>(layer) * 4;
        modalities::Status st = driver_->rms_norm_fp8_noweight(
            dptr(x), dptr(x_fp8), static_cast<int>(sequence), kEncoderWidth,
            scale_ptr(activation_scales, scale), stream);
        if (!st.ok_status()) return st;
        st = driver_->fp8_cutlass(
            dptr(x_fp8), dptr(qkv_w), dptr(qkv), static_cast<int>(sequence),
            2560, kEncoderWidth, activation_weight_alphas[scale], 0.0f,
            NativeThorFp8Tactic::kSquare, stream);
        if (!st.ok_status()) return st;
        st = driver_->qkv_rope_cache_fp16(
            dptr(qkv), dptr(rope), dptr(attention), dptr(key_cache),
            dptr(value_cache), static_cast<int>(sequence), kEncoderWidth,
            kHeadDimension, kHeadDimension, 2560,
            static_cast<int>(scale / 4 * cache_layer_elements),
            kHeadDimension, stream);
        if (!st.ok_status() || layer == kLayers - 1) return st;

        void* layer_key = offset_bytes(
            dptr(key_cache), static_cast<std::size_t>(layer) *
                                 cache_layer_elements * sizeof(std::uint16_t));
        void* layer_value = offset_bytes(
            dptr(value_cache), static_cast<std::size_t>(layer) *
                                   cache_layer_elements * sizeof(std::uint16_t));
        st = driver_->attention_seqused_fp16(
            dptr(attention), layer_key, layer_value, dptr(logits),
            dptr(attention), static_cast<int>(sequence),
            static_cast<int>(sequence), kHeads, kHeadDimension,
            static_cast<const int*>(dptr(valid_keys)), attention_scale, stream);
        if (!st.ok_status()) return st;

        const NativeDeviceWeight* o_w = weight(
            weights, "encoder_attn_o_w_" + suffix,
            NativeWeightDType::kFp8E4M3, {kEncoderWidth, kEncoderWidth});
        const NativeDeviceWeight* gate_w = weight(
            weights, "encoder_ffn_gate_up_w_" + suffix,
            NativeWeightDType::kFp8E4M3,
            {2 * kEncoderHidden, kEncoderWidth});
        const NativeDeviceWeight* down_w = weight(
            weights, "encoder_ffn_down_w_" + suffix,
            NativeWeightDType::kFp8E4M3,
            {kEncoderWidth, kEncoderHidden});
        if (!o_w || !gate_w || !down_w) {
            return invalid("Thor encoder layer weights are incomplete");
        }
        st = driver_->quantize_fp8_static(
            dptr(attention), dptr(o_fp8), scale_ptr(activation_scales, scale + 1),
            sequence * kEncoderWidth, stream);
        if (!st.ok_status()) return st;
        st = driver_->fp8_cutlass(
            dptr(o_fp8), dptr(o_w), dptr(fg), static_cast<int>(sequence),
            kEncoderWidth, kEncoderWidth,
            activation_weight_alphas[scale + 1], 0.0f,
            NativeThorFp8Tactic::kSquare, stream);
        if (!st.ok_status()) return st;
        st = driver_->residual_rms_norm_fp8_noweight(
            dptr(x), dptr(fg), dptr(x_fp8), static_cast<int>(sequence),
            kEncoderWidth, scale_ptr(activation_scales, scale + 2), stream);
        if (!st.ok_status()) return st;
        st = driver_->fp8_cutlass(
            dptr(x_fp8), dptr(gate_w), dptr(gate), static_cast<int>(sequence),
            2 * kEncoderHidden, kEncoderWidth,
            activation_weight_alphas[scale + 2], 0.0f,
            NativeThorFp8Tactic::kT1, stream);
        if (!st.ok_status()) return st;
        st = driver_->gate_gelu_fp8(
            dptr(gate), dptr(hidden_fp8), static_cast<int>(sequence),
            kEncoderHidden, scale_ptr(activation_scales, scale + 3), stream);
        if (!st.ok_status()) return st;
        st = driver_->fp8_cutlass(
            dptr(hidden_fp8), dptr(down_w), dptr(fg),
            static_cast<int>(sequence), kEncoderWidth, kEncoderHidden,
            activation_weight_alphas[scale + 3], 0.0f,
            NativeThorFp8Tactic::kWide, stream);
        if (!st.ok_status()) return st;
        st = driver_->residual_add_fp16(
            dptr(x), dptr(fg), sequence * kEncoderWidth, stream);
        if (!st.ok_status()) return st;
    }
    return modalities::Status::ok();
}

modalities::Status NativeThorFp8Forward::calibrate_encoder(
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    const NativeThorWeightScales& weight_scales,
    std::vector<float>* sample_scales,
    std::uintptr_t stream) const {
    if (!driver_ || !driver_->status().ok_status() || !workspace ||
        !sample_scales ||
        workspace->flavor() != NativeWorkspaceFlavor::kThorFp8 ||
        weight_scales.encoder.size() != kLayers * 4) {
        return invalid("Thor encoder calibration owner is invalid");
    }
    const std::uint64_t sequence = workspace->encoder_sequence();
    const std::uint64_t keys = workspace->total_keys();
    const NativeWorkspaceBuffer* x = buffer(
        *workspace, "encoder_x", modalities::DType::kFloat16,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* x_fp8 = buffer(
        *workspace, "encoder_x_fp8", modalities::DType::kUInt8,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* qkv = buffer(
        *workspace, "encoder_QKV", modalities::DType::kFloat16,
        {sequence, 2560});
    const NativeWorkspaceBuffer* logits = buffer(
        *workspace, "encoder_logits", modalities::DType::kFloat16,
        {sequence * kHeads, keys});
    const NativeWorkspaceBuffer* attention = buffer(
        *workspace, "encoder_attn", modalities::DType::kFloat16,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* o_fp8 = buffer(
        *workspace, "encoder_o_fp8", modalities::DType::kUInt8,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* gate = buffer(
        *workspace, "encoder_gate_merged", modalities::DType::kFloat16,
        {sequence, 2 * kEncoderHidden});
    const NativeWorkspaceBuffer* hidden = buffer(
        *workspace, "encoder_hidden", modalities::DType::kFloat16,
        {sequence, kEncoderHidden});
    const NativeWorkspaceBuffer* hidden_fp8 = buffer(
        *workspace, "encoder_hidden_fp8", modalities::DType::kUInt8,
        {sequence, kEncoderHidden});
    const NativeWorkspaceBuffer* fg = buffer(
        *workspace, "encoder_fg", modalities::DType::kFloat16,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* rope = buffer(
        *workspace, "encoder_rope_weights", modalities::DType::kFloat16,
        {sequence, kHeadDimension});
    const NativeWorkspaceBuffer* key_cache = buffer(
        *workspace, "encoder_k_cache", modalities::DType::kFloat16,
        {kLayers, keys, kHeadDimension});
    const NativeWorkspaceBuffer* value_cache = buffer(
        *workspace, "encoder_v_cache", modalities::DType::kFloat16,
        {kLayers, keys, kHeadDimension});
    const NativeWorkspaceBuffer* valid_keys = buffer(
        *workspace, "attn_enc_seqused", modalities::DType::kUInt8,
        {sizeof(std::int32_t)});
    const NativeWorkspaceBuffer* norm_scratch = buffer(
        *workspace, "encoder_norm_scratch", modalities::DType::kFloat16,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* x_scratch = buffer(
        *workspace, "encoder_x_scratch", modalities::DType::kFloat16,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* fp8_scratch = buffer(
        *workspace, "encoder_fp8_scratch", modalities::DType::kUInt8,
        {sequence, kEncoderHidden});
    const NativeWorkspaceBuffer* scales = buffer(
        *workspace, "encoder_sample_scales", modalities::DType::kFloat32,
        {kLayers, 4});
    const NativeWorkspaceBuffer* dynamic_scale = buffer(
        *workspace, "calibration_scale", modalities::DType::kFloat32, {1});
    const NativeWorkspaceBuffer* ones = buffer(
        *workspace, "encoder_rms_ones", modalities::DType::kFloat16,
        {kEncoderWidth});
    if (!x || !x_fp8 || !qkv || !logits || !attention || !o_fp8 || !gate ||
        !hidden || !hidden_fp8 || !fg || !rope || !key_cache || !value_cache ||
        !valid_keys || !norm_scratch || !x_scratch || !fp8_scratch ||
        !scales || !dynamic_scale || !ones) {
        return invalid("Thor encoder calibration workspace is incomplete");
    }
    cudaError_t rc = cudaMemsetAsync(
        dptr(scales), 0, kLayers * 4 * sizeof(float),
        reinterpret_cast<cudaStream_t>(stream));
    if (rc != cudaSuccess) return backend(cudaGetErrorString(rc));

    const float attention_scale =
        1.0f / std::sqrt(static_cast<float>(kHeadDimension));
    const std::size_t cache_layer_elements = keys * kHeadDimension;
    for (int layer = 0; layer < kLayers; ++layer) {
        const std::string suffix = std::to_string(layer);
        const NativeDeviceWeight* qkv_w = weight(
            weights, "encoder_attn_qkv_w_" + suffix,
            NativeWeightDType::kFp8E4M3, {2560, kEncoderWidth});
        if (!qkv_w) return invalid("Thor encoder QKV weight is invalid");
        const std::size_t site = static_cast<std::size_t>(layer) * 4;
        modalities::Status st = driver_->rms_norm_fp16(
            dptr(x), dptr(ones), dptr(norm_scratch),
            static_cast<int>(sequence), kEncoderWidth, 1e-6f, stream);
        if (!st.ok_status()) return st;
        float qkv_scale = 0.0f;
        st = measure_scale(
            *driver_, dptr(norm_scratch), dptr(fp8_scratch),
            static_cast<float*>(dptr(dynamic_scale)), scale_ptr(scales, site),
            sequence * kEncoderWidth, stream, &qkv_scale);
        if (!st.ok_status()) return st;
        st = driver_->rms_norm_fp8_noweight(
            dptr(x), dptr(x_fp8), static_cast<int>(sequence), kEncoderWidth,
            scale_ptr(scales, site), stream);
        if (!st.ok_status()) return st;
        const float qkv_alpha = qkv_scale * weight_scales.encoder[site];
        st = driver_->fp8_cutlass(
            dptr(x_fp8), dptr(qkv_w), dptr(qkv), static_cast<int>(sequence),
            2560, kEncoderWidth, qkv_alpha, 0.0f,
            NativeThorFp8Tactic::kSquare, stream);
        if (!st.ok_status()) return st;
        st = driver_->qkv_rope_cache_fp16(
            dptr(qkv), dptr(rope), dptr(attention), dptr(key_cache),
            dptr(value_cache), static_cast<int>(sequence), kEncoderWidth,
            kHeadDimension, kHeadDimension, 2560,
            static_cast<int>(static_cast<std::size_t>(layer) *
                             cache_layer_elements),
            kHeadDimension, stream);
        if (!st.ok_status()) return st;
        if (layer == kLayers - 1) break;

        void* layer_key = offset_bytes(
            dptr(key_cache), static_cast<std::size_t>(layer) *
                                 cache_layer_elements * 2);
        void* layer_value = offset_bytes(
            dptr(value_cache), static_cast<std::size_t>(layer) *
                                   cache_layer_elements * 2);
        st = driver_->attention_seqused_fp16(
            dptr(attention), layer_key, layer_value, dptr(logits),
            dptr(attention), static_cast<int>(sequence),
            static_cast<int>(sequence), kHeads, kHeadDimension,
            static_cast<const int*>(dptr(valid_keys)), attention_scale, stream);
        if (!st.ok_status()) return st;

        const NativeDeviceWeight* o_w = weight(
            weights, "encoder_attn_o_w_" + suffix,
            NativeWeightDType::kFp8E4M3, {kEncoderWidth, kEncoderWidth});
        const NativeDeviceWeight* gate_w = weight(
            weights, "encoder_ffn_gate_up_w_" + suffix,
            NativeWeightDType::kFp8E4M3,
            {2 * kEncoderHidden, kEncoderWidth});
        const NativeDeviceWeight* down_w = weight(
            weights, "encoder_ffn_down_w_" + suffix,
            NativeWeightDType::kFp8E4M3,
            {kEncoderWidth, kEncoderHidden});
        if (!o_w || !gate_w || !down_w) {
            return invalid("Thor encoder calibration weights are incomplete");
        }
        float o_scale = 0.0f;
        st = measure_scale(
            *driver_, dptr(attention), dptr(fp8_scratch),
            static_cast<float*>(dptr(dynamic_scale)),
            scale_ptr(scales, site + 1), sequence * kEncoderWidth, stream,
            &o_scale);
        if (!st.ok_status()) return st;
        st = driver_->quantize_fp8_static(
            dptr(attention), dptr(o_fp8), scale_ptr(scales, site + 1),
            sequence * kEncoderWidth, stream);
        if (!st.ok_status()) return st;
        st = driver_->fp8_cutlass(
            dptr(o_fp8), dptr(o_w), dptr(fg), static_cast<int>(sequence),
            kEncoderWidth, kEncoderWidth,
            o_scale * weight_scales.encoder[site + 1], 0.0f,
            NativeThorFp8Tactic::kSquare, stream);
        if (!st.ok_status()) return st;

        st = copy_device(dptr(x_scratch), dptr(x),
                         sequence * kEncoderWidth * 2, stream);
        if (!st.ok_status()) return st;
        st = driver_->residual_add_fp16(
            dptr(x_scratch), dptr(fg), sequence * kEncoderWidth, stream);
        if (!st.ok_status()) return st;
        st = driver_->rms_norm_fp16(
            dptr(x_scratch), dptr(ones), dptr(norm_scratch),
            static_cast<int>(sequence), kEncoderWidth, 1e-6f, stream);
        if (!st.ok_status()) return st;
        float gate_scale = 0.0f;
        st = measure_scale(
            *driver_, dptr(norm_scratch), dptr(fp8_scratch),
            static_cast<float*>(dptr(dynamic_scale)),
            scale_ptr(scales, site + 2), sequence * kEncoderWidth, stream,
            &gate_scale);
        if (!st.ok_status()) return st;
        st = driver_->residual_rms_norm_fp8_noweight(
            dptr(x), dptr(fg), dptr(x_fp8), static_cast<int>(sequence),
            kEncoderWidth, scale_ptr(scales, site + 2), stream);
        if (!st.ok_status()) return st;
        st = driver_->fp8_cutlass(
            dptr(x_fp8), dptr(gate_w), dptr(gate), static_cast<int>(sequence),
            2 * kEncoderHidden, kEncoderWidth,
            gate_scale * weight_scales.encoder[site + 2], 0.0f,
            NativeThorFp8Tactic::kT1, stream);
        if (!st.ok_status()) return st;
        st = driver_->gate_gelu_fp16(
            dptr(gate), dptr(hidden), static_cast<int>(sequence),
            kEncoderHidden, stream);
        if (!st.ok_status()) return st;
        float down_scale = 0.0f;
        st = measure_scale(
            *driver_, dptr(hidden), dptr(fp8_scratch),
            static_cast<float*>(dptr(dynamic_scale)),
            scale_ptr(scales, site + 3), sequence * kEncoderHidden, stream,
            &down_scale);
        if (!st.ok_status()) return st;
        st = driver_->gate_gelu_fp8(
            dptr(gate), dptr(hidden_fp8), static_cast<int>(sequence),
            kEncoderHidden, scale_ptr(scales, site + 3), stream);
        if (!st.ok_status()) return st;
        st = driver_->fp8_cutlass(
            dptr(hidden_fp8), dptr(down_w), dptr(fg),
            static_cast<int>(sequence), kEncoderWidth, kEncoderHidden,
            down_scale * weight_scales.encoder[site + 3], 0.0f,
            NativeThorFp8Tactic::kWide, stream);
        if (!st.ok_status()) return st;
        st = driver_->residual_add_fp16(
            dptr(x), dptr(fg), sequence * kEncoderWidth, stream);
        if (!st.ok_status()) return st;
    }

    // The last encoder layer only writes Q/K/V. Canonical non-zero values keep
    // the artifact valid without advertising measurements for skipped sites.
    const float unused_scales[] = {1.0f, 1.0f, 1.0f};
    rc = cudaMemcpyAsync(
        scale_ptr(scales, (kLayers - 1) * 4 + 1), unused_scales,
        sizeof(unused_scales), cudaMemcpyHostToDevice,
        reinterpret_cast<cudaStream_t>(stream));
    if (rc != cudaSuccess) return backend(cudaGetErrorString(rc));
    modalities::Status st = synchronize(stream);
    if (!st.ok_status()) return st;
    sample_scales->resize(kLayers * 4);
    rc = cudaMemcpy(sample_scales->data(), dptr(scales),
                    sample_scales->size() * sizeof(float),
                    cudaMemcpyDeviceToHost);
    return rc == cudaSuccess ? modalities::Status::ok()
                             : backend(cudaGetErrorString(rc));
}

modalities::Status NativeThorFp8Forward::calibrate_decoder(
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    std::vector<float>* sample_scales,
    std::uintptr_t stream) const {
    if (!driver_ || !driver_->status().ok_status() || !workspace ||
        !sample_scales ||
        workspace->flavor() != NativeWorkspaceFlavor::kThorFp8) {
        return invalid("Thor decoder calibration owner is invalid");
    }
    const std::uint64_t sequence = workspace->chunk_size();
    const std::uint64_t steps = workspace->num_steps();
    const std::uint64_t keys = workspace->total_keys();
    const NativeWorkspaceBuffer* noise = buffer(
        *workspace, "diffusion_noise", modalities::DType::kFloat16,
        {sequence, 32});
    const NativeWorkspaceBuffer* x = buffer(
        *workspace, "decoder_x", modalities::DType::kFloat16,
        {sequence, kDecoderWidth});
    const NativeWorkspaceBuffer* xn = buffer(
        *workspace, "x_normed_buf", modalities::DType::kFloat16,
        {sequence, kDecoderWidth});
    const NativeWorkspaceBuffer* gate = buffer(
        *workspace, "gate_buf", modalities::DType::kFloat16,
        {sequence, kDecoderWidth});
    const NativeWorkspaceBuffer* qkv = buffer(
        *workspace, "decoder_QKV", modalities::DType::kFloat16,
        {sequence, 2560});
    const NativeWorkspaceBuffer* logits = buffer(
        *workspace, "decoder_logits", modalities::DType::kFloat16,
        {sequence * kHeads, keys});
    const NativeWorkspaceBuffer* attention = buffer(
        *workspace, "decoder_attn", modalities::DType::kFloat16,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* hidden = buffer(
        *workspace, "decoder_hidden", modalities::DType::kFloat16,
        {sequence, 2 * kDecoderHidden});
    const NativeWorkspaceBuffer* fg = buffer(
        *workspace, "decoder_fg", modalities::DType::kFloat16,
        {sequence, 2 * kDecoderHidden});
    const NativeWorkspaceBuffer* action_f32 = buffer(
        *workspace, "decoder_action_f32", modalities::DType::kFloat32,
        {sequence, 32});
    const NativeWorkspaceBuffer* xn_fp8 = buffer(
        *workspace, "decoder_x_fp8", modalities::DType::kUInt8,
        {sequence, kDecoderWidth});
    const NativeWorkspaceBuffer* hidden_fp8 = buffer(
        *workspace, "decoder_hidden_fp8", modalities::DType::kUInt8,
        {sequence, kDecoderHidden});
    const NativeWorkspaceBuffer* context_fp8 = buffer(
        *workspace, "decoder_context_fp8", modalities::DType::kUInt8,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* fp8_scratch = buffer(
        *workspace, "decoder_fp8_scratch", modalities::DType::kUInt8,
        {sequence, kDecoderHidden});
    const NativeWorkspaceBuffer* sample_scale_buffer = buffer(
        *workspace, "decoder_sample_scales", modalities::DType::kFloat32,
        {steps, kLayers, 4});
    const NativeWorkspaceBuffer* dynamic_scale = buffer(
        *workspace, "calibration_scale", modalities::DType::kFloat32, {1});
    const NativeWorkspaceBuffer* rope = buffer(
        *workspace, "decoder_rope_weights", modalities::DType::kFloat16,
        {sequence, kHeadDimension});
    const NativeWorkspaceBuffer* style_attn = buffer(
        *workspace, "decoder_style_attn", modalities::DType::kFloat16,
        {steps, kLayers, sequence, 3 * kDecoderWidth});
    const NativeWorkspaceBuffer* style_ffn = buffer(
        *workspace, "decoder_style_ffn", modalities::DType::kFloat16,
        {steps, kLayers, sequence, 3 * kDecoderWidth});
    const NativeWorkspaceBuffer* style_final = buffer(
        *workspace, "decoder_style_final", modalities::DType::kFloat16,
        {steps, sequence, 3 * kDecoderWidth});
    const NativeWorkspaceBuffer* key_cache = buffer(
        *workspace, "encoder_k_cache", modalities::DType::kFloat16,
        {kLayers, keys, kHeadDimension});
    const NativeWorkspaceBuffer* value_cache = buffer(
        *workspace, "encoder_v_cache", modalities::DType::kFloat16,
        {kLayers, keys, kHeadDimension});
    const NativeWorkspaceBuffer* valid_keys = buffer(
        *workspace, "attn_dec_seqused", modalities::DType::kUInt8,
        {sizeof(std::int32_t)});
    const NativeWorkspaceBuffer* device_position = buffer(
        *workspace, "attn_dec_devpos", modalities::DType::kUInt8,
        {sizeof(std::int32_t)});
    const NativeDeviceWeight* weight_scales = weight(
        weights, "decoder_weight_scales", NativeWeightDType::kFloat32,
        {kLayers * 4});
    const NativeDeviceWeight* input_w = weight(
        weights, "decoder_action_in_proj_w", NativeWeightDType::kFloat16,
        {32, kDecoderWidth});
    const NativeDeviceWeight* input_b = weight(
        weights, "decoder_action_in_proj_b", NativeWeightDType::kFloat16,
        {kDecoderWidth});
    const NativeDeviceWeight* output_w = weight(
        weights, "decoder_action_out_proj_w", NativeWeightDType::kFloat16,
        {kDecoderWidth, 32});
    const NativeDeviceWeight* output_b = weight(
        weights, "decoder_action_out_proj_b", NativeWeightDType::kFloat16,
        {32});
    if (!noise || !x || !xn || !gate || !qkv || !logits || !attention ||
        !hidden || !fg || !action_f32 || !xn_fp8 || !hidden_fp8 ||
        !context_fp8 || !fp8_scratch || !sample_scale_buffer ||
        !dynamic_scale || !rope || !style_attn || !style_ffn ||
        !style_final || !key_cache || !value_cache ||
        !device_position || !weight_scales || !input_w || !input_b ||
        !output_w || !output_b) {
        return invalid("Thor decoder calibration workspace is incomplete");
    }

    const std::size_t scale_count = steps * kLayers * 4;
    cudaError_t rc = cudaMemsetAsync(
        dptr(sample_scale_buffer), 0, scale_count * sizeof(float),
        reinterpret_cast<cudaStream_t>(stream));
    if (rc != cudaSuccess) return backend(cudaGetErrorString(rc));

    const float attention_scale =
        1.0f / std::sqrt(static_cast<float>(kHeadDimension));
    const float dt = -1.0f / static_cast<float>(steps);
    const std::size_t cache_layer_elements = keys * kHeadDimension;
    const std::size_t style_row_elements = sequence * 3 * kDecoderWidth;
    for (int step = 0; step < static_cast<int>(steps); ++step) {
        modalities::Status st;
        st = driver_->gmm_fp16(
            dptr(noise), dptr(input_w), dptr(x), static_cast<int>(sequence),
            kDecoderWidth, 32, 0.0f, stream);
        if (!st.ok_status()) return st;
        st = driver_->add_bias_fp16(
            dptr(x), dptr(input_b), static_cast<int>(sequence),
            kDecoderWidth, stream);
        if (!st.ok_status()) return st;

        for (int layer = 0; layer < kLayers; ++layer) {
            const std::string suffix = std::to_string(layer);
            const NativeDeviceWeight* qkv_w = weight(
                weights, "decoder_attn_qkv_w_" + suffix,
                NativeWeightDType::kFp8E4M3, {kDecoderWidth, 2560});
            const NativeDeviceWeight* o_w = weight(
                weights, "decoder_attn_o_w_" + suffix,
                NativeWeightDType::kFp8E4M3,
                {kEncoderWidth, kDecoderWidth});
            const NativeDeviceWeight* gate_w = weight(
                weights, "decoder_ffn_gate_up_w_" + suffix,
                NativeWeightDType::kFp8E4M3,
                {kDecoderWidth, 2 * kDecoderHidden});
            const NativeDeviceWeight* down_w = weight(
                weights, "decoder_ffn_down_w_" + suffix,
                NativeWeightDType::kFp8E4M3,
                {kDecoderHidden, kDecoderWidth});
            if (!qkv_w || !o_w || !gate_w || !down_w) {
                return invalid("Thor decoder calibration weights are incomplete");
            }
            const std::size_t site =
                (static_cast<std::size_t>(step) * kLayers + layer) * 4;
            const std::size_t style_site =
                (static_cast<std::size_t>(step) * kLayers + layer) *
                style_row_elements;
            const void* attn_style =
                offset_bytes(dptr(style_attn), style_site * 2);
            const void* ffn_style =
                offset_bytes(dptr(style_ffn), style_site * 2);

            if (layer == 0) {
                st = driver_->adarms_fp16(
                    dptr(x), attn_style, dptr(xn), dptr(gate),
                    static_cast<int>(sequence), kDecoderWidth, stream);
                if (!st.ok_status()) return st;
                st = measure_scale(
                    *driver_, dptr(xn), dptr(fp8_scratch),
                    static_cast<float*>(dptr(dynamic_scale)),
                    scale_ptr(sample_scale_buffer, site),
                    sequence * kDecoderWidth, stream);
                if (!st.ok_status()) return st;
                st = driver_->fused_adarms_fp8(
                    dptr(x), attn_style, dptr(xn_fp8), dptr(gate),
                    static_cast<int>(sequence), kDecoderWidth,
                    scale_ptr(sample_scale_buffer, site), stream);
                if (!st.ok_status()) return st;
            }

            st = driver_->fp8_descale(
                dptr(xn_fp8), dptr(qkv_w), dptr(qkv),
                static_cast<int>(sequence), 2560, kDecoderWidth,
                scale_ptr(sample_scale_buffer, site),
                scale_ptr(weight_scales, static_cast<std::size_t>(layer) * 4),
                stream);
            if (!st.ok_status()) return st;
            st = driver_->qkv_rope_cache_devpos_fp16(
                dptr(qkv), dptr(rope), dptr(attention), dptr(key_cache),
                dptr(value_cache), static_cast<const int*>(dptr(device_position)),
                static_cast<int>(sequence), kEncoderWidth, kHeadDimension,
                kHeadDimension, 2560,
                static_cast<int>(static_cast<std::size_t>(layer) *
                                 cache_layer_elements),
                kHeadDimension, stream);
            if (!st.ok_status()) return st;
            void* layer_key = offset_bytes(
                dptr(key_cache), static_cast<std::size_t>(layer) *
                                     cache_layer_elements * 2);
            void* layer_value = offset_bytes(
                dptr(value_cache), static_cast<std::size_t>(layer) *
                                       cache_layer_elements * 2);
            st = driver_->attention_seqused_fp16(
                dptr(attention), layer_key, layer_value, dptr(logits),
                dptr(attention), static_cast<int>(sequence),
                static_cast<int>(keys), kHeads, kHeadDimension,
                static_cast<const int*>(dptr(valid_keys)), attention_scale,
                stream);
            if (!st.ok_status()) return st;

            st = measure_scale(
                *driver_, dptr(attention), dptr(fp8_scratch),
                static_cast<float*>(dptr(dynamic_scale)),
                scale_ptr(sample_scale_buffer, site + 1),
                sequence * kEncoderWidth, stream);
            if (!st.ok_status()) return st;
            st = driver_->quantize_fp8_static(
                dptr(attention), dptr(context_fp8),
                scale_ptr(sample_scale_buffer, site + 1),
                sequence * kEncoderWidth, stream);
            if (!st.ok_status()) return st;
            st = driver_->fp8_descale(
                dptr(context_fp8), dptr(o_w), dptr(fg),
                static_cast<int>(sequence), kDecoderWidth, kEncoderWidth,
                scale_ptr(sample_scale_buffer, site + 1),
                scale_ptr(weight_scales,
                          static_cast<std::size_t>(layer) * 4 + 1),
                stream);
            if (!st.ok_status()) return st;

            st = driver_->gate_res_fp16(
                dptr(fg), dptr(gate), dptr(x),
                sequence * kDecoderWidth, stream);
            if (!st.ok_status()) return st;
            st = driver_->adarms_fp16(
                dptr(x), ffn_style, dptr(xn), dptr(gate),
                static_cast<int>(sequence), kDecoderWidth, stream);
            if (!st.ok_status()) return st;
            st = measure_scale(
                *driver_, dptr(xn), dptr(fp8_scratch),
                static_cast<float*>(dptr(dynamic_scale)),
                scale_ptr(sample_scale_buffer, site + 2),
                sequence * kDecoderWidth, stream);
            if (!st.ok_status()) return st;
            st = driver_->quantize_fp8_static(
                dptr(xn), dptr(xn_fp8),
                scale_ptr(sample_scale_buffer, site + 2),
                sequence * kDecoderWidth, stream);
            if (!st.ok_status()) return st;
            st = driver_->fp8_descale(
                dptr(xn_fp8), dptr(gate_w), dptr(fg),
                static_cast<int>(sequence), 2 * kDecoderHidden,
                kDecoderWidth, scale_ptr(sample_scale_buffer, site + 2),
                scale_ptr(weight_scales,
                          static_cast<std::size_t>(layer) * 4 + 2),
                stream);
            if (!st.ok_status()) return st;

            st = driver_->gate_gelu_fp16(
                dptr(fg), dptr(hidden), static_cast<int>(sequence),
                kDecoderHidden, stream);
            if (!st.ok_status()) return st;
            st = measure_scale(
                *driver_, dptr(hidden), dptr(fp8_scratch),
                static_cast<float*>(dptr(dynamic_scale)),
                scale_ptr(sample_scale_buffer, site + 3),
                sequence * kDecoderHidden, stream);
            if (!st.ok_status()) return st;
            st = driver_->gate_gelu_fp8(
                dptr(fg), dptr(hidden_fp8), static_cast<int>(sequence),
                kDecoderHidden,
                scale_ptr(sample_scale_buffer, site + 3), stream);
            if (!st.ok_status()) return st;
            st = driver_->fp8_descale(
                dptr(hidden_fp8), dptr(down_w), dptr(fg),
                static_cast<int>(sequence), kDecoderWidth, kDecoderHidden,
                scale_ptr(sample_scale_buffer, site + 3),
                scale_ptr(weight_scales,
                          static_cast<std::size_t>(layer) * 4 + 3),
                stream);
            if (!st.ok_status()) return st;

            if (layer + 1 < kLayers) {
                const std::size_t next_style_site =
                    style_site + style_row_elements;
                const void* next_attn_style = offset_bytes(
                    dptr(style_attn), next_style_site * 2);
                st = driver_->gate_res_fp16(
                    dptr(fg), dptr(gate), dptr(x),
                    sequence * kDecoderWidth, stream);
                if (!st.ok_status()) return st;
                st = driver_->adarms_fp16(
                    dptr(x), next_attn_style, dptr(xn), dptr(gate),
                    static_cast<int>(sequence), kDecoderWidth, stream);
                if (!st.ok_status()) return st;
                st = measure_scale(
                    *driver_, dptr(xn), dptr(fp8_scratch),
                    static_cast<float*>(dptr(dynamic_scale)),
                    scale_ptr(sample_scale_buffer, site + 4),
                    sequence * kDecoderWidth, stream);
                if (!st.ok_status()) return st;
                st = driver_->quantize_fp8_static(
                    dptr(xn), dptr(xn_fp8),
                    scale_ptr(sample_scale_buffer, site + 4),
                    sequence * kDecoderWidth, stream);
            } else {
                st = driver_->gate_res_fp16(
                    dptr(fg), dptr(gate), dptr(x),
                    sequence * kDecoderWidth, stream);
            }
            if (!st.ok_status()) return st;
        }

        const void* final_style = offset_bytes(
            dptr(style_final), static_cast<std::size_t>(step) *
                                   style_row_elements * 2);
        st = driver_->adarms_fp16(
            dptr(x), final_style, dptr(xn), dptr(gate),
            static_cast<int>(sequence), kDecoderWidth, stream);
        if (!st.ok_status()) return st;
        st = driver_->gmm_fp16_out_fp32(
            dptr(xn), dptr(output_w), static_cast<float*>(dptr(action_f32)),
            static_cast<int>(sequence), 32, kDecoderWidth, stream);
        if (!st.ok_status()) return st;
        st = driver_->action_update_fp16(
            static_cast<const float*>(dptr(action_f32)), dptr(output_b),
            dptr(noise), static_cast<int>(sequence), 32, dt, stream);
        if (!st.ok_status()) return st;
    }

    modalities::Status st = synchronize(stream);
    if (!st.ok_status()) return st;
    sample_scales->resize(scale_count);
    rc = cudaMemcpy(sample_scales->data(), dptr(sample_scale_buffer),
                    scale_count * sizeof(float), cudaMemcpyDeviceToHost);
    return rc == cudaSuccess ? modalities::Status::ok()
                             : backend(cudaGetErrorString(rc));
}

modalities::Status NativeThorFp8Forward::diffusion(
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    std::uintptr_t stream) const {
    if (!driver_ || !driver_->status().ok_status() || !workspace ||
        workspace->flavor() != NativeWorkspaceFlavor::kThorFp8) {
        return invalid("Thor decoder forward owner is invalid");
    }
    const std::uint64_t sequence = workspace->chunk_size();
    const std::uint64_t steps = workspace->num_steps();
    const std::uint64_t keys = workspace->total_keys();
    const NativeWorkspaceBuffer* noise = buffer(
        *workspace, "diffusion_noise", modalities::DType::kFloat16,
        {sequence, 32});
    const NativeWorkspaceBuffer* x = buffer(
        *workspace, "decoder_x", modalities::DType::kFloat16,
        {sequence, kDecoderWidth});
    const NativeWorkspaceBuffer* xn = buffer(
        *workspace, "x_normed_buf", modalities::DType::kFloat16,
        {sequence, kDecoderWidth});
    const NativeWorkspaceBuffer* gate = buffer(
        *workspace, "gate_buf", modalities::DType::kFloat16,
        {sequence, kDecoderWidth});
    const NativeWorkspaceBuffer* qkv = buffer(
        *workspace, "decoder_QKV", modalities::DType::kFloat16,
        {sequence, 2560});
    const NativeWorkspaceBuffer* logits = buffer(
        *workspace, "decoder_logits", modalities::DType::kFloat16,
        {sequence * kHeads, keys});
    const NativeWorkspaceBuffer* attention = buffer(
        *workspace, "decoder_attn", modalities::DType::kFloat16,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* hidden = buffer(
        *workspace, "decoder_hidden", modalities::DType::kFloat16,
        {sequence, 2 * kDecoderHidden});
    const NativeWorkspaceBuffer* fg = buffer(
        *workspace, "decoder_fg", modalities::DType::kFloat16,
        {sequence, 2 * kDecoderHidden});
    const NativeWorkspaceBuffer* action_f32 = buffer(
        *workspace, "decoder_action_f32", modalities::DType::kFloat32,
        {sequence, 32});
    const NativeWorkspaceBuffer* xn_fp8 = buffer(
        *workspace, "decoder_x_fp8", modalities::DType::kUInt8,
        {sequence, kDecoderWidth});
    const NativeWorkspaceBuffer* hidden_fp8 = buffer(
        *workspace, "decoder_hidden_fp8", modalities::DType::kUInt8,
        {sequence, kDecoderHidden});
    const NativeWorkspaceBuffer* context_fp8 = buffer(
        *workspace, "decoder_context_fp8", modalities::DType::kUInt8,
        {sequence, kEncoderWidth});
    const NativeWorkspaceBuffer* rope = buffer(
        *workspace, "decoder_rope_weights", modalities::DType::kFloat16,
        {sequence, kHeadDimension});
    const NativeWorkspaceBuffer* style_attn = buffer(
        *workspace, "decoder_style_attn", modalities::DType::kFloat16,
        {steps, kLayers, sequence, 3 * kDecoderWidth});
    const NativeWorkspaceBuffer* style_ffn = buffer(
        *workspace, "decoder_style_ffn", modalities::DType::kFloat16,
        {steps, kLayers, sequence, 3 * kDecoderWidth});
    const NativeWorkspaceBuffer* style_final = buffer(
        *workspace, "decoder_style_final", modalities::DType::kFloat16,
        {steps, sequence, 3 * kDecoderWidth});
    const NativeWorkspaceBuffer* activation_scales = buffer(
        *workspace, "decoder_activation_scales",
        modalities::DType::kFloat32, {steps, kLayers, 4});
    const NativeWorkspaceBuffer* key_cache = buffer(
        *workspace, "encoder_k_cache", modalities::DType::kFloat16,
        {kLayers, keys, kHeadDimension});
    const NativeWorkspaceBuffer* value_cache = buffer(
        *workspace, "encoder_v_cache", modalities::DType::kFloat16,
        {kLayers, keys, kHeadDimension});
    const NativeWorkspaceBuffer* valid_keys = buffer(
        *workspace, "attn_dec_seqused", modalities::DType::kUInt8,
        {sizeof(std::int32_t)});
    const NativeWorkspaceBuffer* device_position = buffer(
        *workspace, "attn_dec_devpos", modalities::DType::kUInt8,
        {sizeof(std::int32_t)});
    const NativeDeviceWeight* weight_scales = weight(
        weights, "decoder_weight_scales", NativeWeightDType::kFloat32,
        {kLayers * 4});
    const NativeDeviceWeight* input_w = weight(
        weights, "decoder_action_in_proj_w", NativeWeightDType::kFloat16,
        {32, kDecoderWidth});
    const NativeDeviceWeight* input_b = weight(
        weights, "decoder_action_in_proj_b", NativeWeightDType::kFloat16,
        {kDecoderWidth});
    const NativeDeviceWeight* output_w = weight(
        weights, "decoder_action_out_proj_w", NativeWeightDType::kFloat16,
        {kDecoderWidth, 32});
    const NativeDeviceWeight* output_b = weight(
        weights, "decoder_action_out_proj_b", NativeWeightDType::kFloat16,
        {32});
    if (!noise || !x || !xn || !gate || !qkv || !logits || !attention ||
        !hidden || !fg || !action_f32 || !xn_fp8 || !hidden_fp8 ||
        !context_fp8 || !rope || !style_attn || !style_ffn || !style_final ||
        !activation_scales || !key_cache || !value_cache || !valid_keys ||
        !device_position || !weight_scales || !input_w || !input_b ||
        !output_w || !output_b) {
        return invalid("Thor decoder workspace or global weights are incomplete");
    }

    const float attention_scale =
        1.0f / std::sqrt(static_cast<float>(kHeadDimension));
    const float dt = -1.0f / static_cast<float>(steps);
    const std::size_t cache_layer_elements = keys * kHeadDimension;
    const std::size_t style_row_elements = sequence * 3 * kDecoderWidth;
    for (int step = 0; step < static_cast<int>(steps); ++step) {
        modalities::Status st = driver_->gmm_fp16(
            dptr(noise), dptr(input_w), dptr(x), static_cast<int>(sequence),
            kDecoderWidth, 32, 0.0f, stream);
        if (!st.ok_status()) return st;
        st = driver_->add_bias_fp16(
            dptr(x), dptr(input_b), static_cast<int>(sequence),
            kDecoderWidth, stream);
        if (!st.ok_status()) return st;

        for (int layer = 0; layer < kLayers; ++layer) {
            const std::string suffix = std::to_string(layer);
            const NativeDeviceWeight* qkv_w = weight(
                weights, "decoder_attn_qkv_w_" + suffix,
                NativeWeightDType::kFp8E4M3, {kDecoderWidth, 2560});
            const NativeDeviceWeight* o_w = weight(
                weights, "decoder_attn_o_w_" + suffix,
                NativeWeightDType::kFp8E4M3,
                {kEncoderWidth, kDecoderWidth});
            const NativeDeviceWeight* gate_w = weight(
                weights, "decoder_ffn_gate_up_w_" + suffix,
                NativeWeightDType::kFp8E4M3,
                {kDecoderWidth, 2 * kDecoderHidden});
            const NativeDeviceWeight* down_w = weight(
                weights, "decoder_ffn_down_w_" + suffix,
                NativeWeightDType::kFp8E4M3,
                {kDecoderHidden, kDecoderWidth});
            if (!qkv_w || !o_w || !gate_w || !down_w) {
                return invalid("Thor decoder layer weights are incomplete");
            }
            const std::size_t site =
                (static_cast<std::size_t>(step) * kLayers + layer) * 4;
            const std::size_t style_site =
                (static_cast<std::size_t>(step) * kLayers + layer) *
                style_row_elements;
            const void* attn_style = offset_bytes(dptr(style_attn),
                                                   style_site * 2);
            const void* ffn_style = offset_bytes(dptr(style_ffn),
                                                  style_site * 2);
            if (layer == 0) {
                st = driver_->fused_adarms_fp8(
                    dptr(x), attn_style, dptr(xn_fp8), dptr(gate),
                    static_cast<int>(sequence), kDecoderWidth,
                    scale_ptr(activation_scales, site), stream);
                if (!st.ok_status()) return st;
            }
            st = driver_->fp8_descale(
                dptr(xn_fp8), dptr(qkv_w), dptr(qkv),
                static_cast<int>(sequence), 2560, kDecoderWidth,
                scale_ptr(activation_scales, site),
                scale_ptr(weight_scales, static_cast<std::size_t>(layer) * 4),
                stream);
            if (!st.ok_status()) return st;
            st = driver_->qkv_rope_cache_devpos_fp16(
                dptr(qkv), dptr(rope), dptr(attention), dptr(key_cache),
                dptr(value_cache), static_cast<const int*>(dptr(device_position)),
                static_cast<int>(sequence), kEncoderWidth, kHeadDimension,
                kHeadDimension, 2560,
                static_cast<int>(static_cast<std::size_t>(layer) *
                                 cache_layer_elements),
                kHeadDimension, stream);
            if (!st.ok_status()) return st;
            void* layer_key = offset_bytes(
                dptr(key_cache), static_cast<std::size_t>(layer) *
                                     cache_layer_elements * 2);
            void* layer_value = offset_bytes(
                dptr(value_cache), static_cast<std::size_t>(layer) *
                                       cache_layer_elements * 2);
            st = driver_->attention_seqused_fp16(
                dptr(attention), layer_key, layer_value, dptr(logits),
                dptr(attention), static_cast<int>(sequence),
                static_cast<int>(keys), kHeads, kHeadDimension,
                static_cast<const int*>(dptr(valid_keys)), attention_scale,
                stream);
            if (!st.ok_status()) return st;
            st = driver_->quantize_fp8_static(
                dptr(attention), dptr(context_fp8),
                scale_ptr(activation_scales, site + 1),
                sequence * kEncoderWidth, stream);
            if (!st.ok_status()) return st;
            st = driver_->fp8_descale(
                dptr(context_fp8), dptr(o_w), dptr(fg),
                static_cast<int>(sequence), kDecoderWidth, kEncoderWidth,
                scale_ptr(activation_scales, site + 1),
                scale_ptr(weight_scales,
                          static_cast<std::size_t>(layer) * 4 + 1),
                stream);
            if (!st.ok_status()) return st;
            st = driver_->gate_res_adarms_fp8(
                dptr(fg), dptr(gate), dptr(x), ffn_style, dptr(xn_fp8),
                dptr(gate), static_cast<int>(sequence), kDecoderWidth,
                scale_ptr(activation_scales, site + 2), stream);
            if (!st.ok_status()) return st;
            st = driver_->fp8_descale(
                dptr(xn_fp8), dptr(gate_w), dptr(fg),
                static_cast<int>(sequence), 2 * kDecoderHidden,
                kDecoderWidth, scale_ptr(activation_scales, site + 2),
                scale_ptr(weight_scales,
                          static_cast<std::size_t>(layer) * 4 + 2),
                stream);
            if (!st.ok_status()) return st;
            st = driver_->gate_gelu_fp8(
                dptr(fg), dptr(hidden_fp8), static_cast<int>(sequence),
                kDecoderHidden, scale_ptr(activation_scales, site + 3),
                stream);
            if (!st.ok_status()) return st;
            st = driver_->fp8_descale(
                dptr(hidden_fp8), dptr(down_w), dptr(fg),
                static_cast<int>(sequence), kDecoderWidth, kDecoderHidden,
                scale_ptr(activation_scales, site + 3),
                scale_ptr(weight_scales,
                          static_cast<std::size_t>(layer) * 4 + 3),
                stream);
            if (!st.ok_status()) return st;
            if (layer + 1 < kLayers) {
                const std::size_t next_style_site = style_site + style_row_elements;
                const void* next_attn_style = offset_bytes(
                    dptr(style_attn), next_style_site * 2);
                st = driver_->gate_res_adarms_fp8(
                    dptr(fg), dptr(gate), dptr(x), next_attn_style,
                    dptr(xn_fp8), dptr(gate), static_cast<int>(sequence),
                    kDecoderWidth, scale_ptr(activation_scales, site + 4),
                    stream);
            } else {
                st = driver_->gate_res_fp16(
                    dptr(fg), dptr(gate), dptr(x),
                    sequence * kDecoderWidth, stream);
            }
            if (!st.ok_status()) return st;
        }

        const void* final_style = offset_bytes(
            dptr(style_final), static_cast<std::size_t>(step) *
                                   style_row_elements * 2);
        st = driver_->adarms_fp16(
            dptr(x), final_style, dptr(xn), dptr(gate),
            static_cast<int>(sequence), kDecoderWidth, stream);
        if (!st.ok_status()) return st;
        st = driver_->gmm_fp16_out_fp32(
            dptr(xn), dptr(output_w), static_cast<float*>(dptr(action_f32)),
            static_cast<int>(sequence), 32, kDecoderWidth, stream);
        if (!st.ok_status()) return st;
        st = driver_->action_update_fp16(
            static_cast<const float*>(dptr(action_f32)), dptr(output_b),
            dptr(noise), static_cast<int>(sequence), 32, dt, stream);
        if (!st.ok_status()) return st;
    }
    return modalities::Status::ok();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

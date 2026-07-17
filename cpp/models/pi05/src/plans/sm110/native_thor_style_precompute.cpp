#include "flashrt/cpp/models/pi05/plans/sm110/native_thor_style_precompute.h"

#include <cuda_runtime_api.h>

#include <string>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

bool weight_shape(const NativeDeviceWeightStore& weights,
                  const std::string& name,
                  std::initializer_list<std::uint64_t> shape,
                  const NativeDeviceWeight** out) {
    const NativeDeviceWeight* weight = weights.find(name);
    if (!weight || weight->dtype != NativeWeightDType::kFloat16 ||
        weight->shape != std::vector<std::uint64_t>(shape)) {
        return false;
    }
    if (out) *out = weight;
    return true;
}

void* offset(void* base, std::size_t elements) {
    return static_cast<unsigned char*>(base) +
           elements * sizeof(std::uint16_t);
}

}  // namespace

modalities::Status NativeThorStylePrecomputer::run(
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    std::uintptr_t stream) const {
    if (!driver_ || !driver_->status().ok_status() || !workspace ||
        workspace->activation_dtype() != modalities::DType::kFloat16) {
        return invalid("Thor style precomputer is invalid");
    }
    const NativeWorkspaceBuffer* time_output =
        workspace->find("decoder_time_emb");
    const NativeWorkspaceBuffer* style_attn =
        workspace->find("decoder_style_attn");
    const NativeWorkspaceBuffer* style_ffn =
        workspace->find("decoder_style_ffn");
    const NativeWorkspaceBuffer* style_final =
        workspace->find("decoder_style_final");
    const NativeWorkspaceBuffer* scratch_a = workspace->find("decoder_x");
    const NativeWorkspaceBuffer* scratch_b = workspace->find("x_normed_buf");
    if (!time_output || !style_attn || !style_ffn || !style_final ||
        !scratch_a || !scratch_b ||
        time_output->dtype != modalities::DType::kFloat16 ||
        time_output->shape.size() != 3 || style_attn->shape.size() != 4 ||
        style_ffn->shape != style_attn->shape ||
        style_final->shape.size() != 3) {
        return invalid("Thor style workspace layout is invalid");
    }
    const int steps = static_cast<int>(time_output->shape[0]);
    const int chunk = static_cast<int>(time_output->shape[1]);
    if (time_output->shape[2] != 1024 || style_attn->shape[0] != steps ||
        style_attn->shape[1] != 18 || style_attn->shape[2] != chunk ||
        style_attn->shape[3] != 3072 ||
        style_final->shape !=
            std::vector<std::uint64_t>(
                {static_cast<std::uint64_t>(steps),
                 static_cast<std::uint64_t>(chunk), 3072})) {
        return invalid("Thor style workspace shape is invalid");
    }

    const NativeDeviceWeight* time_source = nullptr;
    const NativeDeviceWeight* time_in_w = nullptr;
    const NativeDeviceWeight* time_in_b = nullptr;
    const NativeDeviceWeight* time_out_w = nullptr;
    const NativeDeviceWeight* time_out_b = nullptr;
    const NativeDeviceWeight* final_w = nullptr;
    const NativeDeviceWeight* final_b = nullptr;
    if (!weight_shape(weights, "decoder_time_embeds",
                      {static_cast<std::uint64_t>(steps), 1024},
                      &time_source) ||
        !weight_shape(weights, "decoder_time_mlp_in_w", {1024, 1024},
                      &time_in_w) ||
        !weight_shape(weights, "decoder_time_mlp_in_b", {1024},
                      &time_in_b) ||
        !weight_shape(weights, "decoder_time_mlp_out_w", {1024, 1024},
                      &time_out_w) ||
        !weight_shape(weights, "decoder_time_mlp_out_b", {1024},
                      &time_out_b) ||
        !weight_shape(weights, "decoder_final_norm_mod_w", {1024, 3072},
                      &final_w) ||
        !weight_shape(weights, "decoder_final_norm_mod_b", {3072},
                      &final_b)) {
        return invalid("Thor style global weights are incomplete");
    }

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    for (int step = 0; step < steps; ++step) {
        void* time_row = offset(frt_buffer_dptr(time_source->buffer),
                                static_cast<std::size_t>(step) * 1024);
        modalities::Status st = driver_->fp16_nn(
            time_row, frt_buffer_dptr(time_in_w->buffer),
            frt_buffer_dptr(scratch_a->buffer), 1, 1024, 1024, stream);
        if (!st.ok_status()) return st;
        st = driver_->add_bias_fp16(
            frt_buffer_dptr(scratch_a->buffer),
            frt_buffer_dptr(time_in_b->buffer), 1, 1024, stream);
        if (!st.ok_status()) return st;
        st = driver_->precise_silu_fp16(
            frt_buffer_dptr(scratch_a->buffer), 1024, stream);
        if (!st.ok_status()) return st;
        st = driver_->fp16_nn(
            frt_buffer_dptr(scratch_a->buffer),
            frt_buffer_dptr(time_out_w->buffer),
            frt_buffer_dptr(scratch_b->buffer), 1, 1024, 1024, stream);
        if (!st.ok_status()) return st;
        st = driver_->add_bias_fp16(
            frt_buffer_dptr(scratch_b->buffer),
            frt_buffer_dptr(time_out_b->buffer), 1, 1024, stream);
        if (!st.ok_status()) return st;
        st = driver_->precise_silu_fp16(
            frt_buffer_dptr(scratch_b->buffer), 1024, stream);
        if (!st.ok_status()) return st;

        void* expanded = offset(
            frt_buffer_dptr(time_output->buffer),
            static_cast<std::size_t>(step) * chunk * 1024);
        for (int row = 0; row < chunk; ++row) {
            const cudaError_t rc = cudaMemcpyAsync(
                offset(expanded, static_cast<std::size_t>(row) * 1024),
                frt_buffer_dptr(scratch_b->buffer),
                1024 * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice,
                cuda_stream);
            if (rc != cudaSuccess) {
                return backend("Thor time style expansion failed");
            }
        }

        for (int layer = 0; layer < 18; ++layer) {
            const std::string suffix = std::to_string(layer);
            const NativeDeviceWeight* attn_w = nullptr;
            const NativeDeviceWeight* attn_b = nullptr;
            const NativeDeviceWeight* ffn_w = nullptr;
            const NativeDeviceWeight* ffn_b = nullptr;
            if (!weight_shape(weights,
                              "decoder_pre_attn_norm_mod_w_" + suffix,
                              {1024, 3072}, &attn_w) ||
                !weight_shape(weights,
                              "decoder_pre_attn_norm_mod_b_" + suffix,
                              {3072}, &attn_b) ||
                !weight_shape(weights,
                              "decoder_pre_ffn_norm_mod_w_" + suffix,
                              {1024, 3072}, &ffn_w) ||
                !weight_shape(weights,
                              "decoder_pre_ffn_norm_mod_b_" + suffix,
                              {3072}, &ffn_b)) {
                return invalid("Thor style layer weights are incomplete");
            }
            const std::size_t style_offset =
                (static_cast<std::size_t>(step) * 18 + layer) * chunk * 3072;
            void* attn_target =
                offset(frt_buffer_dptr(style_attn->buffer), style_offset);
            void* ffn_target =
                offset(frt_buffer_dptr(style_ffn->buffer), style_offset);
            st = driver_->fp16_nn(
                expanded, frt_buffer_dptr(attn_w->buffer), attn_target,
                chunk, 3072, 1024, stream);
            if (!st.ok_status()) return st;
            st = driver_->add_bias_fp16(
                attn_target, frt_buffer_dptr(attn_b->buffer), chunk, 3072,
                stream);
            if (!st.ok_status()) return st;
            st = driver_->fp16_nn(
                expanded, frt_buffer_dptr(ffn_w->buffer), ffn_target,
                chunk, 3072, 1024, stream);
            if (!st.ok_status()) return st;
            st = driver_->add_bias_fp16(
                ffn_target, frt_buffer_dptr(ffn_b->buffer), chunk, 3072,
                stream);
            if (!st.ok_status()) return st;
        }
        void* final_target = offset(
            frt_buffer_dptr(style_final->buffer),
            static_cast<std::size_t>(step) * chunk * 3072);
        st = driver_->fp16_nn(
            expanded, frt_buffer_dptr(final_w->buffer), final_target,
            chunk, 3072, 1024, stream);
        if (!st.ok_status()) return st;
        st = driver_->add_bias_fp16(
            final_target, frt_buffer_dptr(final_b->buffer), chunk, 3072,
            stream);
        if (!st.ok_status()) return st;
    }
    const cudaError_t rc = cudaStreamSynchronize(cuda_stream);
    return rc == cudaSuccess
               ? modalities::Status::ok()
               : backend("Thor style precompute synchronization failed");
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

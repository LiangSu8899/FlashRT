#include "flashrt/cpp/models/pi05/backends/sm120/native_rtx_autotune.h"

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

struct Shape {
    const char* weight;
    NativeRtxScaleSite site;
    const char* output;
    int m;
    int n;
    int k;
};

}  // namespace

modalities::Status autotune_native_rtx_fp8(
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    const NativeRtxLinear& linear,
    int num_views,
    int chunk_size) {
    if (!workspace || !linear.fp8() || num_views < 1 || num_views > 3 ||
        chunk_size <= 0 || workspace->num_views() != num_views ||
        workspace->chunk_size() != chunk_size) {
        return invalid("native RTX FP8 autotune configuration is invalid");
    }
    const int vision_sequence = num_views * 256;
    const int encoder_vision_sequence = workspace->encoder_vision_sequence();
    const int encoder_sequence = workspace->encoder_sequence();
    const Shape shapes[] = {
        {"vision_attn_qkv_w_0", {NativeRtxScaleDomain::kVision, 0},
         "vision_QKV", vision_sequence, 3456, 1152},
        {"vision_attn_o_w_0", {NativeRtxScaleDomain::kVision, 1},
         "vision_x_norm", vision_sequence, 1152, 1152},
        {"vision_ffn_up_w_0", {NativeRtxScaleDomain::kVision, 2},
         "vision_hidden", vision_sequence, 4304, 1152},
        {"vision_ffn_down_w_0", {NativeRtxScaleDomain::kVision, 3},
         "vision_x_norm", vision_sequence, 1152, 4304},
        {"encoder_multi_modal_projector_w",
         {NativeRtxScaleDomain::kVision, 108}, "encoder_x",
         encoder_vision_sequence, 2048, 1152},
        {"encoder_attn_qkv_w_0", {NativeRtxScaleDomain::kEncoder, 0},
         "encoder_QKV", encoder_sequence, 2560, 2048},
        {"encoder_attn_o_w_0", {NativeRtxScaleDomain::kEncoder, 1},
         "encoder_x_norm", encoder_sequence, 2048, 2048},
        {"encoder_ffn_gate_up_w_0", {NativeRtxScaleDomain::kEncoder, 2},
         "encoder_gate_merged", encoder_sequence, 32768, 2048},
        {"encoder_ffn_down_w_0", {NativeRtxScaleDomain::kEncoder, 3},
         "encoder_x_norm", encoder_sequence, 2048, 16384},
        {"decoder_attn_qkv_w_0", {NativeRtxScaleDomain::kDecoder, 0},
         "decoder_QKV", chunk_size, 2560, 1024},
        {"decoder_attn_o_w_0", {NativeRtxScaleDomain::kDecoder, 1},
         "x_normed_buf", chunk_size, 1024, 2048},
        {"decoder_ffn_gate_up_w_0", {NativeRtxScaleDomain::kDecoder, 2},
         "decoder_gate_merged", chunk_size, 8192, 1024},
        {"decoder_ffn_down_w_0", {NativeRtxScaleDomain::kDecoder, 3},
         "x_normed_buf", chunk_size, 1024, 4096},
    };
    for (const Shape& shape : shapes) {
        const NativeWorkspaceBuffer* output = workspace->find(shape.output);
        if (!output) return invalid("native FP8 autotune output is missing");
        modalities::Status st = linear.autotune(
            weights, workspace, shape.weight, shape.site,
            frt_buffer_dptr(output->buffer), shape.m, shape.n, shape.k);
        if (!st.ok_status()) return st;
    }
    return modalities::Status::ok();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

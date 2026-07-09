#include "flashrt/cpp/models/pi05/native_weight_materializer.h"

#include <string>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

std::string encoder_prefix(int layer) {
    return "paligemma_with_expert.paligemma.model.language_model.layers." +
           std::to_string(layer);
}

std::string layer_name(const char* stem, int layer) {
    return std::string(stem) + std::to_string(layer);
}

}  // namespace

modalities::Status NativeWeightMaterializer::load(
    const std::string& key,
    NativeFloatTensor* out) {
    return load_native_float_tensor(source_, key, out);
}

modalities::Status NativeWeightMaterializer::upload(
    const std::string& name,
    const NativeFloatTensor& tensor) {
    if (!destination_) return invalid("native weight destination is null");
    NativeBf16Tensor bf16;
    modalities::Status st = native_to_bf16(tensor, &bf16);
    if (!st.ok_status()) return st;
    return destination_->upload(name, bf16);
}

modalities::Status NativeWeightMaterializer::upload_rounded_transpose(
    const std::string& source_key,
    const std::string& destination_name) {
    NativeFloatTensor source;
    NativeFloatTensor rounded;
    NativeFloatTensor transposed;
    modalities::Status st = load(source_key, &source);
    if (!st.ok_status()) return st;
    st = native_round_to_bf16_float(source, &rounded);
    if (!st.ok_status()) return st;
    st = native_transpose_2d(rounded, &transposed);
    if (!st.ok_status()) return st;
    return upload(destination_name, transposed);
}

modalities::Status NativeWeightMaterializer::upload_folded_transpose(
    const std::string& source_key,
    const NativeFloatTensor& norm,
    const std::string& destination_name) {
    NativeFloatTensor source;
    NativeFloatTensor folded;
    NativeFloatTensor transposed;
    modalities::Status st = load(source_key, &source);
    if (!st.ok_status()) return st;
    st = native_fold_rms_columns(source, norm, &folded);
    if (!st.ok_status()) return st;
    st = native_transpose_2d(folded, &transposed);
    if (!st.ok_status()) return st;
    return upload(destination_name, transposed);
}

modalities::Status NativeWeightMaterializer::materialize_encoder_layer(
    int layer) {
    if (layer < 0 || layer >= 18 || !destination_) {
        return invalid("Pi0.5 encoder layer index is invalid");
    }
    const std::string prefix = encoder_prefix(layer);
    NativeFloatTensor norm;
    modalities::Status st = load(prefix + ".input_layernorm.weight", &norm);
    if (!st.ok_status()) return st;

    NativeFloatTensor q;
    NativeFloatTensor k;
    NativeFloatTensor v;
    NativeFloatTensor qi;
    NativeFloatTensor ki;
    NativeFloatTensor qf;
    NativeFloatTensor kf;
    NativeFloatTensor vf;
    NativeFloatTensor qkv;
    st = load(prefix + ".self_attn.q_proj.weight", &q);
    if (!st.ok_status()) return st;
    st = load(prefix + ".self_attn.k_proj.weight", &k);
    if (!st.ok_status()) return st;
    st = load(prefix + ".self_attn.v_proj.weight", &v);
    if (!st.ok_status()) return st;
    st = native_interleave_qk_rows(q, 8, &qi);
    if (!st.ok_status()) return st;
    st = native_interleave_qk_rows(k, 1, &ki);
    if (!st.ok_status()) return st;
    st = native_fold_rms_columns(qi, norm, &qf);
    if (!st.ok_status()) return st;
    st = native_fold_rms_columns(ki, norm, &kf);
    if (!st.ok_status()) return st;
    st = native_fold_rms_columns(v, norm, &vf);
    if (!st.ok_status()) return st;
    st = native_concat_rows_transpose({&qf, &kf, &vf}, &qkv);
    if (!st.ok_status()) return st;
    st = upload(layer_name("encoder_attn_qkv_w_", layer), qkv);
    if (!st.ok_status()) return st;

    st = upload_rounded_transpose(
        prefix + ".self_attn.o_proj.weight",
        layer_name("encoder_attn_o_w_", layer));
    if (!st.ok_status()) return st;

    st = load(prefix + ".post_attention_layernorm.weight", &norm);
    if (!st.ok_status()) return st;
    st = upload_folded_transpose(
        prefix + ".mlp.gate_proj.weight", norm,
        layer_name("encoder_ffn_gate_w_", layer));
    if (!st.ok_status()) return st;
    st = upload_folded_transpose(
        prefix + ".mlp.up_proj.weight", norm,
        layer_name("encoder_ffn_up_w_", layer));
    if (!st.ok_status()) return st;
    return upload_rounded_transpose(
        prefix + ".mlp.down_proj.weight",
        layer_name("encoder_ffn_down_w_", layer));
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

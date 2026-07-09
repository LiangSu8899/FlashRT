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

std::string decoder_prefix(int layer) {
    return "paligemma_with_expert.gemma_expert.model.layers." +
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

modalities::Status NativeWeightMaterializer::upload_rounded_copy(
    const std::string& source_key,
    const std::string& destination_name) {
    NativeFloatTensor source;
    NativeFloatTensor rounded;
    modalities::Status st = load(source_key, &source);
    if (!st.ok_status()) return st;
    st = native_round_to_bf16_float(source, &rounded);
    if (!st.ok_status()) return st;
    return upload(destination_name, rounded);
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

modalities::Status NativeWeightMaterializer::materialize_decoder_layer(
    int layer,
    bool merge_gate_up) {
    if (layer < 0 || layer >= 18 || !destination_) {
        return invalid("Pi0.5 decoder layer index is invalid");
    }
    const std::string prefix = decoder_prefix(layer);
    NativeFloatTensor q;
    NativeFloatTensor k;
    NativeFloatTensor v;
    NativeFloatTensor qr;
    NativeFloatTensor kr;
    NativeFloatTensor vr;
    NativeFloatTensor qi;
    NativeFloatTensor ki;
    NativeFloatTensor qkv;
    modalities::Status st = load(prefix + ".self_attn.q_proj.weight", &q);
    if (!st.ok_status()) return st;
    st = load(prefix + ".self_attn.k_proj.weight", &k);
    if (!st.ok_status()) return st;
    st = load(prefix + ".self_attn.v_proj.weight", &v);
    if (!st.ok_status()) return st;
    st = native_round_to_bf16_float(q, &qr);
    if (!st.ok_status()) return st;
    st = native_round_to_bf16_float(k, &kr);
    if (!st.ok_status()) return st;
    st = native_round_to_bf16_float(v, &vr);
    if (!st.ok_status()) return st;
    st = native_interleave_qk_rows(qr, 8, &qi);
    if (!st.ok_status()) return st;
    st = native_interleave_qk_rows(kr, 1, &ki);
    if (!st.ok_status()) return st;
    st = native_concat_rows_transpose({&qi, &ki, &vr}, &qkv);
    if (!st.ok_status()) return st;
    st = upload(layer_name("decoder_attn_qkv_w_", layer), qkv);
    if (!st.ok_status()) return st;

    st = upload_rounded_transpose(
        prefix + ".self_attn.o_proj.weight",
        layer_name("decoder_attn_o_w_", layer));
    if (!st.ok_status()) return st;

    NativeFloatTensor gate;
    NativeFloatTensor up;
    NativeFloatTensor gate_rounded;
    NativeFloatTensor up_rounded;
    NativeFloatTensor gate_t;
    NativeFloatTensor up_t;
    st = load(prefix + ".mlp.gate_proj.weight", &gate);
    if (!st.ok_status()) return st;
    st = load(prefix + ".mlp.up_proj.weight", &up);
    if (!st.ok_status()) return st;
    st = native_round_to_bf16_float(gate, &gate_rounded);
    if (!st.ok_status()) return st;
    st = native_round_to_bf16_float(up, &up_rounded);
    if (!st.ok_status()) return st;
    st = native_transpose_2d(gate_rounded, &gate_t);
    if (!st.ok_status()) return st;
    st = native_transpose_2d(up_rounded, &up_t);
    if (!st.ok_status()) return st;
    st = upload(layer_name("decoder_ffn_gate_w_", layer), gate_t);
    if (!st.ok_status()) return st;
    st = upload(layer_name("decoder_ffn_up_w_", layer), up_t);
    if (!st.ok_status()) return st;
    if (merge_gate_up) {
        NativeFloatTensor gate_up;
        st = native_concat_columns(gate_t, up_t, &gate_up);
        if (!st.ok_status()) return st;
        st = upload(layer_name("decoder_ffn_gate_up_w_", layer), gate_up);
        if (!st.ok_status()) return st;
    }
    st = upload_rounded_transpose(
        prefix + ".mlp.down_proj.weight",
        layer_name("decoder_ffn_down_w_", layer));
    if (!st.ok_status()) return st;

    st = upload_rounded_transpose(
        prefix + ".input_layernorm.dense.weight",
        layer_name("decoder_pre_attn_norm_mod_w_", layer));
    if (!st.ok_status()) return st;
    st = upload_rounded_copy(
        prefix + ".input_layernorm.dense.bias",
        layer_name("decoder_pre_attn_norm_mod_b_", layer));
    if (!st.ok_status()) return st;
    st = upload_rounded_transpose(
        prefix + ".post_attention_layernorm.dense.weight",
        layer_name("decoder_pre_ffn_norm_mod_w_", layer));
    if (!st.ok_status()) return st;
    return upload_rounded_copy(
        prefix + ".post_attention_layernorm.dense.bias",
        layer_name("decoder_pre_ffn_norm_mod_b_", layer));
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

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

const std::string& vision_prefix() {
    static const std::string prefix =
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model";
    return prefix;
}

const loader::SafetensorInfo* source_tensor(
    const loader::SafetensorsFile& source,
    const std::string& key) {
    const loader::SafetensorInfo* tensor = source.find(key);
    if (!tensor) tensor = source.find(std::string("model.") + key);
    return tensor;
}

const float* f32_data(const loader::SafetensorsFile& source,
                      const loader::SafetensorInfo* tensor) {
    if (!tensor || tensor->dtype != "F32") return nullptr;
    const void* data = source.data(*tensor);
    if (reinterpret_cast<std::uintptr_t>(data) % alignof(float) != 0) {
        return nullptr;
    }
    return static_cast<const float*>(data);
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
    const loader::SafetensorInfo* info = source_tensor(source_, source_key);
    if (destination_ && info && info->shape.size() == 2) {
        const float* data = f32_data(source_, info);
        if (data) {
            NativeBf16Tensor converted;
            modalities::Status st = native_f32_to_bf16(
                data, info->shape, true, &converted);
            if (!st.ok_status()) return st;
            return destination_->upload(destination_name, converted);
        }
    }
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
    const loader::SafetensorInfo* info = source_tensor(source_, source_key);
    if (destination_ && info) {
        const float* data = f32_data(source_, info);
        if (data) {
            NativeBf16Tensor converted;
            modalities::Status st = native_f32_to_bf16(
                data, info->shape, false, &converted);
            if (!st.ok_status()) return st;
            return destination_->upload(destination_name, converted);
        }
    }
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
    const loader::SafetensorInfo* info = source_tensor(source_, source_key);
    if (destination_ && info && info->shape.size() == 2) {
        const float* data = f32_data(source_, info);
        if (data) {
            NativeBf16Tensor converted;
            modalities::Status st = native_f32_fold_rms_columns_transpose(
                data, info->shape[0], info->shape[1], norm, &converted);
            if (!st.ok_status()) return st;
            return destination_->upload(destination_name, converted);
        }
    }
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

modalities::Status NativeWeightMaterializer::upload_rounded_scaled(
    const std::string& source_key,
    const std::string& destination_name,
    float scale,
    bool transpose) {
    const loader::SafetensorInfo* info = source_tensor(source_, source_key);
    if (destination_ && info) {
        const float* data = f32_data(source_, info);
        if (data) {
            NativeBf16Tensor converted;
            modalities::Status st = native_f32_round_scale_to_bf16(
                data, info->shape, scale, transpose, &converted);
            if (!st.ok_status()) return st;
            return destination_->upload(destination_name, converted);
        }
    }
    NativeFloatTensor source;
    NativeFloatTensor rounded;
    NativeFloatTensor arranged;
    NativeFloatTensor scaled;
    modalities::Status st = load(source_key, &source);
    if (!st.ok_status()) return st;
    st = native_round_to_bf16_float(source, &rounded);
    if (!st.ok_status()) return st;
    if (transpose) {
        st = native_transpose_2d(rounded, &arranged);
        if (!st.ok_status()) return st;
    } else {
        arranged = std::move(rounded);
    }
    st = native_scale(arranged, scale, &scaled);
    if (!st.ok_status()) return st;
    return upload(destination_name, scaled);
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

modalities::Status NativeWeightMaterializer::materialize_vision_layer(
    int layer) {
    if (layer < 0 || layer >= 27 || !destination_) {
        return invalid("Pi0.5 vision layer index is invalid");
    }
    const std::string prefix = vision_prefix() + ".encoder.layers." +
                               std::to_string(layer);
    NativeFloatTensor q;
    NativeFloatTensor k;
    NativeFloatTensor v;
    NativeFloatTensor qr;
    NativeFloatTensor kr;
    NativeFloatTensor vr;
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
    st = native_concat_rows_transpose({&qr, &kr, &vr}, &qkv);
    if (!st.ok_status()) return st;
    st = upload(layer_name("vision_attn_qkv_w_", layer), qkv);
    if (!st.ok_status()) return st;

    st = load(prefix + ".self_attn.q_proj.bias", &q);
    if (!st.ok_status()) return st;
    st = load(prefix + ".self_attn.k_proj.bias", &k);
    if (!st.ok_status()) return st;
    st = load(prefix + ".self_attn.v_proj.bias", &v);
    if (!st.ok_status()) return st;
    st = native_round_to_bf16_float(q, &qr);
    if (!st.ok_status()) return st;
    st = native_round_to_bf16_float(k, &kr);
    if (!st.ok_status()) return st;
    st = native_round_to_bf16_float(v, &vr);
    if (!st.ok_status()) return st;
    st = native_concat_vectors({&qr, &kr, &vr}, &qkv);
    if (!st.ok_status()) return st;
    st = upload(layer_name("vision_attn_qkv_b_", layer), qkv);
    if (!st.ok_status()) return st;

    const struct {
        const char* source;
        const char* destination;
        bool transpose;
    } entries[] = {
        {"self_attn.out_proj.weight", "vision_attn_o_w_", true},
        {"self_attn.out_proj.bias", "vision_attn_o_b_", false},
        {"mlp.fc1.weight", "vision_ffn_up_w_", true},
        {"mlp.fc1.bias", "vision_ffn_up_b_", false},
        {"mlp.fc2.weight", "vision_ffn_down_w_", true},
        {"mlp.fc2.bias", "vision_ffn_down_b_", false},
        {"layer_norm1.weight", "vision_pre_attn_norm_w_", false},
        {"layer_norm1.bias", "vision_pre_attn_norm_b_", false},
        {"layer_norm2.weight", "vision_pre_ffn_norm_w_", false},
        {"layer_norm2.bias", "vision_pre_ffn_norm_b_", false},
    };
    for (const auto& entry : entries) {
        st = entry.transpose
                 ? upload_rounded_transpose(
                       prefix + "." + entry.source,
                       layer_name(entry.destination, layer))
                 : upload_rounded_copy(
                       prefix + "." + entry.source,
                       layer_name(entry.destination, layer));
        if (!st.ok_status()) return st;
    }
    return modalities::Status::ok();
}

modalities::Status NativeWeightMaterializer::materialize_vision_globals() {
    if (!destination_) return invalid("native weight destination is null");
    const std::string prefix = vision_prefix();
    NativeFloatTensor patch;
    NativeFloatTensor rounded;
    NativeFloatTensor permuted;
    modalities::Status st = load(
        prefix + ".embeddings.patch_embedding.weight", &patch);
    if (!st.ok_status()) return st;
    st = native_round_to_bf16_float(patch, &rounded);
    if (!st.ok_status()) return st;
    st = native_patch_oihw_to_hwio(rounded, &permuted);
    if (!st.ok_status()) return st;
    st = upload("vision_patch_embedding_w", permuted);
    if (!st.ok_status()) return st;
    st = upload_rounded_copy(prefix + ".embeddings.patch_embedding.bias",
                             "vision_patch_embedding_b");
    if (!st.ok_status()) return st;
    st = upload_rounded_copy(prefix + ".embeddings.position_embedding.weight",
                             "vision_position_embedding");
    if (!st.ok_status()) return st;
    st = upload_rounded_copy(prefix + ".post_layernorm.weight",
                             "vision_final_norm_w");
    if (!st.ok_status()) return st;
    st = upload_rounded_copy(prefix + ".post_layernorm.bias",
                             "vision_final_norm_b");
    if (!st.ok_status()) return st;

    const std::string projector =
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear";
    st = upload_rounded_transpose(projector + ".weight",
                                  "encoder_multi_modal_projector_w");
    if (!st.ok_status()) return st;
    return upload_rounded_copy(projector + ".bias",
                               "encoder_multi_modal_projector_b");
}

modalities::Status NativeWeightMaterializer::materialize_decoder_globals(
    int num_steps) {
    if (!destination_ || num_steps <= 0) {
        return invalid("Pi0.5 decoder global configuration is invalid");
    }
    const struct {
        const char* source;
        const char* destination;
        bool transpose;
    } entries[] = {
        {"paligemma_with_expert.gemma_expert.model.norm.dense.weight",
         "decoder_final_norm_mod_w", true},
        {"paligemma_with_expert.gemma_expert.model.norm.dense.bias",
         "decoder_final_norm_mod_b", false},
        {"time_mlp_in.weight", "decoder_time_mlp_in_w", true},
        {"time_mlp_in.bias", "decoder_time_mlp_in_b", false},
        {"time_mlp_out.weight", "decoder_time_mlp_out_w", true},
        {"time_mlp_out.bias", "decoder_time_mlp_out_b", false},
        {"action_in_proj.weight", "decoder_action_in_proj_w", true},
        {"action_in_proj.bias", "decoder_action_in_proj_b", false},
    };
    for (const auto& entry : entries) {
        const modalities::Status st =
            entry.transpose
                ? upload_rounded_transpose(entry.source, entry.destination)
                : upload_rounded_copy(entry.source, entry.destination);
        if (!st.ok_status()) return st;
    }

    NativeFloatTensor time_embeddings;
    modalities::Status st =
        native_pi05_time_embeddings(num_steps, 1024, &time_embeddings);
    if (!st.ok_status()) return st;
    st = upload("decoder_time_embeds", time_embeddings);
    if (!st.ok_status()) return st;

    const float step_scale = -1.0f / static_cast<float>(num_steps);
    st = upload_rounded_scaled(
        "action_out_proj.weight", "decoder_action_out_proj_w", step_scale,
        true);
    if (!st.ok_status()) return st;
    return upload_rounded_scaled(
        "action_out_proj.bias", "decoder_action_out_proj_b", step_scale,
        false);
}

modalities::Status NativeWeightMaterializer::materialize_embedding() {
    if (!destination_) return invalid("native weight destination is null");
    return upload_rounded_copy(
        "paligemma_with_expert.paligemma.lm_head.weight",
        "embedding_weight");
}

modalities::Status NativeWeightMaterializer::materialize_all(
    const NativeMaterializationOptions& options) {
    if (!destination_ || options.num_steps <= 0) {
        return invalid("Pi0.5 materialization options are invalid");
    }
    modalities::Status st = materialize_vision_globals();
    if (!st.ok_status()) return st;
    for (int layer = 0; layer < 27; ++layer) {
        st = materialize_vision_layer(layer);
        if (!st.ok_status()) return st;
    }
    for (int layer = 0; layer < 18; ++layer) {
        st = materialize_encoder_layer(layer);
        if (!st.ok_status()) return st;
    }
    for (int layer = 0; layer < 18; ++layer) {
        st = materialize_decoder_layer(
            layer, options.merge_decoder_gate_up);
        if (!st.ok_status()) return st;
    }
    st = materialize_decoder_globals(options.num_steps);
    if (!st.ok_status() || !options.include_embedding) return st;
    return materialize_embedding();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

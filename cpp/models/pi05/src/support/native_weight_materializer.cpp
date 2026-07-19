#include "flashrt/cpp/models/pi05/support/native_weight_materializer.h"

#include "flashrt/cpp/models/pi05/model/dims.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>
#include <vector>

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

int materialization_workers(int layers) {
    int workers = std::min(layers, 8);
    const char* setting = std::getenv("FLASHRT_NATIVE_WEIGHT_WORKERS");
    if (!setting || !setting[0]) return workers;
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(setting, &end, 10);
    if (errno || !end || *end || parsed < 1 || parsed > 64) return workers;
    return std::min(layers, static_cast<int>(parsed));
}

template <typename Materialize>
modalities::Status materialize_layers_parallel(int layers,
                                               Materialize materialize) {
    if (layers <= 0) return invalid("parallel materialization is empty");
    std::vector<modalities::Status> statuses(
        static_cast<std::size_t>(layers), modalities::Status::ok());
    std::atomic<int> next{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(
        materialization_workers(layers)));
    try {
        for (int worker = 0; worker < materialization_workers(layers);
             ++worker) {
            threads.emplace_back([&] {
                while (!stop.load(std::memory_order_relaxed)) {
                    const int layer = next.fetch_add(1);
                    if (layer >= layers) break;
                    try {
                        statuses[static_cast<std::size_t>(layer)] =
                            materialize(layer);
                    } catch (const std::exception& error) {
                        statuses[static_cast<std::size_t>(layer)] = backend(
                            std::string("weight worker failed: ") +
                            error.what());
                    } catch (...) {
                        statuses[static_cast<std::size_t>(layer)] =
                            backend("weight worker failed");
                    }
                    if (!statuses[static_cast<std::size_t>(layer)]
                             .ok_status()) {
                        stop.store(true, std::memory_order_relaxed);
                    }
                }
            });
        }
    } catch (const std::exception& error) {
        stop.store(true, std::memory_order_relaxed);
        for (auto& thread : threads) thread.join();
        return backend(std::string("weight worker creation failed: ") +
                       error.what());
    }
    for (auto& thread : threads) thread.join();
    for (const modalities::Status& status : statuses) {
        if (!status.ok_status()) return status;
    }
    return modalities::Status::ok();
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
    if (logical_scalar_ == modalities::DType::kBFloat16) {
        NativeBf16Tensor converted;
        modalities::Status status = native_to_bf16(tensor, &converted);
        return status.ok_status() ? destination_->upload(name, converted)
                                  : status;
    }
    if (logical_scalar_ == modalities::DType::kFloat16) {
        NativeF16Tensor converted;
        modalities::Status status = native_to_f16(tensor, &converted);
        return status.ok_status() ? destination_->upload(name, converted)
                                  : status;
    }
    return invalid("native logical weight scalar is unsupported");
}

modalities::Status NativeWeightMaterializer::upload_source(
    const std::string& source_key,
    const std::string& destination_name,
    bool transpose) {
    if (!destination_) return invalid("native weight destination is null");
    NativeSourceTensorView source;
    modalities::Status st =
        load_native_source_tensor(source_, source_key, &source);
    if (!st.ok_status()) return st;
    if (logical_scalar_ == modalities::DType::kBFloat16) {
        NativeBf16Tensor converted;
        st = native_source_to_bf16(source, transpose, &converted);
        return st.ok_status()
                   ? destination_->upload(destination_name, converted)
                   : st;
    }
    if (logical_scalar_ == modalities::DType::kFloat16) {
        NativeF16Tensor converted;
        st = native_source_to_f16(source, transpose, &converted);
        return st.ok_status()
                   ? destination_->upload(destination_name, converted)
                   : st;
    }
    return invalid("native logical weight scalar is unsupported");
}

modalities::Status NativeWeightMaterializer::upload_folded_transpose(
    const std::string& source_key,
    const NativeFloatTensor& norm,
    const std::string& destination_name) {
    if (!destination_) return invalid("native weight destination is null");
    NativeSourceTensorView source;
    modalities::Status st =
        load_native_source_tensor(source_, source_key, &source);
    if (!st.ok_status()) return st;
    if (logical_scalar_ == modalities::DType::kBFloat16) {
        NativeBf16Tensor converted;
        st = native_source_fold_rms_columns_transpose(
            source, norm, &converted);
        return st.ok_status()
                   ? destination_->upload(destination_name, converted)
                   : st;
    }
    if (logical_scalar_ == modalities::DType::kFloat16) {
        NativeF16Tensor converted;
        st = native_source_fold_rms_columns_to_f16(
            source, norm, true, &converted);
        return st.ok_status()
                   ? destination_->upload(destination_name, converted)
                   : st;
    }
    return invalid("native logical weight scalar is unsupported");
}

modalities::Status NativeWeightMaterializer::materialize_encoder_layer(
    int layer) {
    if (layer < 0 || layer >= kPi05ModelDims.encoder_layers ||
        !destination_) {
        return invalid("Pi0.5 encoder layer index is invalid");
    }
    const std::string prefix = encoder_prefix(layer);
    NativeFloatTensor norm;
    modalities::Status st = load(prefix + ".input_layernorm.weight", &norm);
    if (!st.ok_status()) return st;

    NativeSourceTensorView q;
    NativeSourceTensorView k;
    NativeSourceTensorView v;
    st = load_native_source_tensor(
        source_, prefix + ".self_attn.q_proj.weight", &q);
    if (!st.ok_status()) return st;
    st = load_native_source_tensor(
        source_, prefix + ".self_attn.k_proj.weight", &k);
    if (!st.ok_status()) return st;
    st = load_native_source_tensor(
        source_, prefix + ".self_attn.v_proj.weight", &v);
    if (!st.ok_status()) return st;
    const std::string qkv_name = layer_name("encoder_attn_qkv_w_", layer);
    if (logical_scalar_ == modalities::DType::kBFloat16) {
        NativeBf16Tensor qkv;
        st = native_source_qkv_to_bf16(
            q, k, v, kPi05ModelDims.encoder_heads,
            kPi05ModelDims.encoder_kv_heads, &norm, &qkv);
        if (st.ok_status()) st = destination_->upload(qkv_name, qkv);
    } else if (logical_scalar_ == modalities::DType::kFloat16) {
        NativeF16Tensor qkv;
        st = native_source_qkv_to_f16(
            q, k, v, kPi05ModelDims.encoder_heads,
            kPi05ModelDims.encoder_kv_heads, &norm, true, &qkv);
        if (st.ok_status()) st = destination_->upload(qkv_name, qkv);
    } else {
        return invalid("native logical weight scalar is unsupported");
    }
    if (!st.ok_status()) return st;

    st = upload_source(
        prefix + ".self_attn.o_proj.weight",
        layer_name("encoder_attn_o_w_", layer), true);
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
    return upload_source(
        prefix + ".mlp.down_proj.weight",
        layer_name("encoder_ffn_down_w_", layer), true);
}

modalities::Status NativeWeightMaterializer::materialize_decoder_layer(
    int layer) {
    if (layer < 0 || layer >= kPi05ModelDims.decoder_layers ||
        !destination_) {
        return invalid("Pi0.5 decoder layer index is invalid");
    }
    const std::string prefix = decoder_prefix(layer);
    NativeSourceTensorView q;
    NativeSourceTensorView k;
    NativeSourceTensorView v;
    modalities::Status st = load_native_source_tensor(
        source_, prefix + ".self_attn.q_proj.weight", &q);
    if (!st.ok_status()) return st;
    st = load_native_source_tensor(
        source_, prefix + ".self_attn.k_proj.weight", &k);
    if (!st.ok_status()) return st;
    st = load_native_source_tensor(
        source_, prefix + ".self_attn.v_proj.weight", &v);
    if (!st.ok_status()) return st;
    const std::string qkv_name = layer_name("decoder_attn_qkv_w_", layer);
    if (logical_scalar_ == modalities::DType::kBFloat16) {
        NativeBf16Tensor qkv;
        st = native_source_qkv_to_bf16(
            q, k, v, kPi05ModelDims.decoder_heads,
            kPi05ModelDims.decoder_kv_heads, nullptr, &qkv);
        if (st.ok_status()) st = destination_->upload(qkv_name, qkv);
    } else if (logical_scalar_ == modalities::DType::kFloat16) {
        NativeF16Tensor qkv;
        st = native_source_qkv_to_f16(
            q, k, v, kPi05ModelDims.decoder_heads,
            kPi05ModelDims.decoder_kv_heads, nullptr, true, &qkv);
        if (st.ok_status()) st = destination_->upload(qkv_name, qkv);
    } else {
        return invalid("native logical weight scalar is unsupported");
    }
    if (!st.ok_status()) return st;

    st = upload_source(
        prefix + ".self_attn.o_proj.weight",
        layer_name("decoder_attn_o_w_", layer), true);
    if (!st.ok_status()) return st;

    st = upload_source(
        prefix + ".mlp.gate_proj.weight",
        layer_name("decoder_ffn_gate_w_", layer), true);
    if (!st.ok_status()) return st;
    st = upload_source(
        prefix + ".mlp.up_proj.weight",
        layer_name("decoder_ffn_up_w_", layer), true);
    if (!st.ok_status()) return st;
    st = upload_source(
        prefix + ".mlp.down_proj.weight",
        layer_name("decoder_ffn_down_w_", layer), true);
    if (!st.ok_status()) return st;

    st = upload_source(
        prefix + ".input_layernorm.dense.weight",
        layer_name("decoder_pre_attn_norm_mod_w_", layer), true);
    if (!st.ok_status()) return st;
    st = upload_source(
        prefix + ".input_layernorm.dense.bias",
        layer_name("decoder_pre_attn_norm_mod_b_", layer), false);
    if (!st.ok_status()) return st;
    st = upload_source(
        prefix + ".post_attention_layernorm.dense.weight",
        layer_name("decoder_pre_ffn_norm_mod_w_", layer), true);
    if (!st.ok_status()) return st;
    return upload_source(
        prefix + ".post_attention_layernorm.dense.bias",
        layer_name("decoder_pre_ffn_norm_mod_b_", layer), false);
}

modalities::Status NativeWeightMaterializer::materialize_vision_layer(
    int layer) {
    if (layer < 0 || layer >= kPi05ModelDims.vision_layers ||
        !destination_) {
        return invalid("Pi0.5 vision layer index is invalid");
    }
    const std::string prefix = vision_prefix() + ".encoder.layers." +
                               std::to_string(layer);
    NativeSourceTensorView q;
    NativeSourceTensorView k;
    NativeSourceTensorView v;
    modalities::Status st = load_native_source_tensor(
        source_, prefix + ".self_attn.q_proj.weight", &q);
    if (!st.ok_status()) return st;
    st = load_native_source_tensor(
        source_, prefix + ".self_attn.k_proj.weight", &k);
    if (!st.ok_status()) return st;
    st = load_native_source_tensor(
        source_, prefix + ".self_attn.v_proj.weight", &v);
    if (!st.ok_status()) return st;
    const std::string qkv_weight_name =
        layer_name("vision_attn_qkv_w_", layer);
    if (logical_scalar_ == modalities::DType::kBFloat16) {
        NativeBf16Tensor qkv;
        st = native_source_qkv_to_bf16(q, k, v, 0, 0, nullptr, &qkv);
        if (st.ok_status()) st = destination_->upload(qkv_weight_name, qkv);
    } else if (logical_scalar_ == modalities::DType::kFloat16) {
        NativeF16Tensor qkv;
        st = native_source_qkv_to_f16(
            q, k, v, 0, 0, nullptr, true, &qkv);
        if (st.ok_status()) st = destination_->upload(qkv_weight_name, qkv);
    } else {
        return invalid("native logical weight scalar is unsupported");
    }
    if (!st.ok_status()) return st;

    st = load_native_source_tensor(
        source_, prefix + ".self_attn.q_proj.bias", &q);
    if (!st.ok_status()) return st;
    st = load_native_source_tensor(
        source_, prefix + ".self_attn.k_proj.bias", &k);
    if (!st.ok_status()) return st;
    st = load_native_source_tensor(
        source_, prefix + ".self_attn.v_proj.bias", &v);
    if (!st.ok_status()) return st;
    const std::string qkv_bias_name =
        layer_name("vision_attn_qkv_b_", layer);
    if (logical_scalar_ == modalities::DType::kBFloat16) {
        NativeBf16Tensor qkv;
        st = native_source_concat_vectors_to_bf16({&q, &k, &v}, &qkv);
        if (st.ok_status()) st = destination_->upload(qkv_bias_name, qkv);
    } else {
        NativeF16Tensor qkv;
        st = native_source_concat_vectors_to_f16({&q, &k, &v}, &qkv);
        if (st.ok_status()) st = destination_->upload(qkv_bias_name, qkv);
    }
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
        st = upload_source(
            prefix + "." + entry.source,
            layer_name(entry.destination, layer), entry.transpose);
        if (!st.ok_status()) return st;
    }
    return modalities::Status::ok();
}

modalities::Status NativeWeightMaterializer::materialize_vision_globals() {
    if (!destination_) return invalid("native weight destination is null");
    const std::string prefix = vision_prefix();
    NativeSourceTensorView patch;
    modalities::Status st = load_native_source_tensor(
        source_, prefix + ".embeddings.patch_embedding.weight", &patch);
    if (!st.ok_status()) return st;
    if (logical_scalar_ == modalities::DType::kBFloat16) {
        NativeBf16Tensor permuted;
        st = native_source_patch_oihw_to_hwio_bf16(patch, &permuted);
        if (st.ok_status()) {
            st = destination_->upload("vision_patch_embedding_w", permuted);
        }
    } else if (logical_scalar_ == modalities::DType::kFloat16) {
        NativeF16Tensor permuted;
        st = native_source_patch_oihw_to_hwio_f16(patch, &permuted);
        if (st.ok_status()) {
            st = destination_->upload("vision_patch_embedding_w", permuted);
        }
    } else {
        return invalid("native logical weight scalar is unsupported");
    }
    if (!st.ok_status()) return st;
    st = upload_source(prefix + ".embeddings.patch_embedding.bias",
                       "vision_patch_embedding_b", false);
    if (!st.ok_status()) return st;
    st = upload_source(prefix + ".embeddings.position_embedding.weight",
                       "vision_position_embedding", false);
    if (!st.ok_status()) return st;
    st = upload_source(prefix + ".post_layernorm.weight",
                       "vision_final_norm_w", false);
    if (!st.ok_status()) return st;
    st = upload_source(prefix + ".post_layernorm.bias",
                       "vision_final_norm_b", false);
    if (!st.ok_status()) return st;

    const std::string projector =
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear";
    st = upload_source(projector + ".weight",
                       "encoder_multi_modal_projector_w", true);
    if (!st.ok_status()) return st;
    return upload_source(projector + ".bias",
                         "encoder_multi_modal_projector_b", false);
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
        const modalities::Status st = upload_source(
            entry.source, entry.destination, entry.transpose);
        if (!st.ok_status()) return st;
    }

    modalities::Status st;
    if (logical_scalar_ == modalities::DType::kBFloat16) {
        NativeFloatTensor time_embeddings;
        st = native_pi05_time_embeddings(
            num_steps, kPi05ModelDims.decoder_width, &time_embeddings);
        if (st.ok_status()) st = upload("decoder_time_embeds", time_embeddings);
    } else if (logical_scalar_ == modalities::DType::kFloat16) {
        NativeF16Tensor time_embeddings;
        st = native_pi05_time_embeddings_f16(
            num_steps, kPi05ModelDims.decoder_width, &time_embeddings);
        if (st.ok_status()) {
            st = destination_->upload("decoder_time_embeds", time_embeddings);
        }
    } else {
        return invalid("native logical weight scalar is unsupported");
    }
    if (!st.ok_status()) return st;

    st = upload_source(
        "action_out_proj.weight", "decoder_action_out_proj_w", true);
    if (!st.ok_status()) return st;
    return upload_source(
        "action_out_proj.bias", "decoder_action_out_proj_b", false);
}

modalities::Status NativeWeightMaterializer::materialize_embedding() {
    if (!destination_) return invalid("native weight destination is null");
    return upload_source(
        "paligemma_with_expert.paligemma.lm_head.weight",
        "embedding_weight", false);
}

modalities::Status NativeWeightMaterializer::materialize_all(
    const NativeMaterializationOptions& options) {
    if (!destination_ || options.num_steps <= 0 ||
        (logical_scalar_ != modalities::DType::kBFloat16 &&
         logical_scalar_ != modalities::DType::kFloat16)) {
        return invalid("Pi0.5 materialization options are invalid");
    }
    const bool profile = std::getenv("FLASHRT_PROFILE_NATIVE_SETUP");
    auto checkpoint = std::chrono::steady_clock::now();
    const auto report = [&](const char* phase) {
        const auto now = std::chrono::steady_clock::now();
        if (profile) {
            std::fprintf(stderr, "native_weights %s_ms=%.3f\n", phase,
                         std::chrono::duration<double, std::milli>(
                             now - checkpoint).count());
        }
        checkpoint = now;
    };
    modalities::Status st = materialize_vision_globals();
    if (!st.ok_status()) return st;
    st = materialize_layers_parallel(
        kPi05ModelDims.vision_layers,
        [this](int layer) { return materialize_vision_layer(layer); });
    if (!st.ok_status()) return st;
    report("vision");
    st = materialize_layers_parallel(
        kPi05ModelDims.encoder_layers,
        [this](int layer) { return materialize_encoder_layer(layer); });
    if (!st.ok_status()) return st;
    report("encoder");
    st = materialize_layers_parallel(
        kPi05ModelDims.decoder_layers,
        [this](int layer) { return materialize_decoder_layer(layer); });
    if (!st.ok_status()) return st;
    report("decoder");
    st = materialize_decoder_globals(options.num_steps);
    if (!st.ok_status() || !options.include_embedding) return st;
    report("globals");
    st = materialize_embedding();
    report("embedding");
    return st;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

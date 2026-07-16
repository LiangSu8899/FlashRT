#include "flashrt/cpp/models/pi05/native_thor_weight_materializer.h"

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <thread>

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
modalities::Status materialize_layers_parallel(
    int layers,
    Materialize materialize,
    std::vector<float>* scales) {
    if (layers <= 0 || !scales) {
        return invalid("Thor parallel materialization input is invalid");
    }
    std::vector<modalities::Status> statuses(
        static_cast<std::size_t>(layers), modalities::Status::ok());
    std::vector<std::vector<float>> layer_scales(
        static_cast<std::size_t>(layers));
    std::atomic<int> next{0};
    std::atomic<bool> stop{false};
    const int workers = materialization_workers(layers);
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers));
    try {
        for (int worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&] {
                while (!stop.load(std::memory_order_relaxed)) {
                    const int layer = next.fetch_add(1);
                    if (layer >= layers) break;
                    try {
                        statuses[static_cast<std::size_t>(layer)] = materialize(
                            layer,
                            &layer_scales[static_cast<std::size_t>(layer)]);
                    } catch (const std::exception& error) {
                        statuses[static_cast<std::size_t>(layer)] =
                            backend(std::string("Thor weight worker failed: ") +
                                    error.what());
                    } catch (...) {
                        statuses[static_cast<std::size_t>(layer)] =
                            backend("Thor weight worker failed");
                    }
                    if (!statuses[static_cast<std::size_t>(layer)].ok_status()) {
                        stop.store(true, std::memory_order_relaxed);
                    }
                }
            });
        }
    } catch (const std::exception& error) {
        stop.store(true, std::memory_order_relaxed);
        for (auto& thread : threads) thread.join();
        return backend(std::string("Thor worker creation failed: ") +
                       error.what());
    }
    for (auto& thread : threads) thread.join();
    for (int layer = 0; layer < layers; ++layer) {
        const auto& status = statuses[static_cast<std::size_t>(layer)];
        if (!status.ok_status()) return status;
        const auto& values = layer_scales[static_cast<std::size_t>(layer)];
        scales->insert(scales->end(), values.begin(), values.end());
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

modalities::Status NativeThorWeightMaterializer::upload_f16(
    const std::string& source_key,
    const std::string& destination_name,
    bool transpose) {
    if (!destination_) return invalid("Thor weight destination is null");
    NativeSourceTensorView source;
    NativeF16Tensor converted;
    modalities::Status st =
        load_native_source_tensor(source_, source_key, &source);
    if (!st.ok_status()) return st;
    st = native_source_to_f16(source, transpose, &converted);
    if (!st.ok_status()) return st;
    return destination_->upload(destination_name, converted);
}

modalities::Status NativeThorWeightMaterializer::upload_f16(
    const std::string& destination_name,
    const NativeF16Tensor& tensor) {
    if (!destination_) return invalid("Thor weight destination is null");
    return destination_->upload(destination_name, tensor);
}

modalities::Status NativeThorWeightMaterializer::upload_fp8(
    const std::string& destination_name,
    const NativeF16Tensor& tensor,
    std::vector<float>* scales) {
    if (!destination_ || !scales) {
        return invalid("Thor FP8 weight destination is invalid");
    }
    NativeFp8Tensor quantized;
    modalities::Status st =
        native_quantize_fp8_e4m3(tensor, false, &quantized);
    if (!st.ok_status()) return st;
    st = destination_->upload_bytes(
        destination_name, quantized.shape, NativeWeightDType::kFp8E4M3,
        quantized.values.data(), quantized.values.size());
    if (!st.ok_status()) return st;
    scales->push_back(quantized.scale);
    return modalities::Status::ok();
}

modalities::Status NativeThorWeightMaterializer::upload_scale_vector(
    const std::string& name,
    const std::vector<float>& values) {
    if (!destination_ || values.empty()) {
        return invalid("Thor weight scale vector is invalid");
    }
    return destination_->upload_bytes(
        name, {static_cast<std::uint64_t>(values.size())},
        NativeWeightDType::kFloat32, values.data(),
        values.size() * sizeof(float));
}

modalities::Status NativeThorWeightMaterializer::materialize_vision_globals() {
    if (!destination_) return invalid("Thor weight destination is null");
    const std::string prefix = vision_prefix();
    NativeSourceTensorView patch;
    NativeF16Tensor permuted;
    modalities::Status st = load_native_source_tensor(
        source_, prefix + ".embeddings.patch_embedding.weight", &patch);
    if (!st.ok_status()) return st;
    st = native_source_patch_oihw_to_hwio_f16(patch, &permuted);
    if (!st.ok_status()) return st;
    st = upload_f16("vision_patch_embedding_w", permuted);
    if (!st.ok_status()) return st;

    const struct {
        const char* source;
        const char* destination;
        bool transpose;
    } entries[] = {
        {"embeddings.patch_embedding.bias", "vision_patch_embedding_b", false},
        {"embeddings.position_embedding.weight", "vision_position_embedding", false},
        {"post_layernorm.weight", "vision_final_norm_w", false},
        {"post_layernorm.bias", "vision_final_norm_b", false},
    };
    for (const auto& entry : entries) {
        st = upload_f16(prefix + "." + entry.source, entry.destination,
                        entry.transpose);
        if (!st.ok_status()) return st;
    }
    const std::string projector =
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear";
    st = upload_f16(projector + ".weight",
                    "encoder_multi_modal_projector_w", true);
    if (!st.ok_status()) return st;
    return upload_f16(projector + ".bias",
                      "encoder_multi_modal_projector_b", false);
}

modalities::Status NativeThorWeightMaterializer::materialize_vision_layer(
    int layer,
    std::vector<float>* scales) {
    if (layer < 0 || layer >= 27 || !destination_ || !scales) {
        return invalid("Thor vision layer index is invalid");
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
    NativeF16Tensor joined;
    st = native_source_qkv_to_f16(q, k, v, 0, 0, nullptr, true, &joined);
    if (!st.ok_status()) return st;
    st = upload_fp8(layer_name("vision_attn_qkv_w_", layer), joined, scales);
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
    st = native_source_concat_vectors_to_f16({&q, &k, &v}, &joined);
    if (!st.ok_status()) return st;
    st = upload_f16(layer_name("vision_attn_qkv_b_", layer), joined);
    if (!st.ok_status()) return st;

    const struct {
        const char* source;
        const char* destination;
    } quantized[] = {
        {"self_attn.out_proj.weight", "vision_attn_o_w_"},
        {"mlp.fc1.weight", "vision_ffn_up_w_"},
        {"mlp.fc2.weight", "vision_ffn_down_w_"},
    };
    for (const auto& entry : quantized) {
        NativeSourceTensorView source;
        NativeF16Tensor converted;
        st = load_native_source_tensor(source_, prefix + "." + entry.source,
                                       &source);
        if (!st.ok_status()) return st;
        st = native_source_to_f16(source, true, &converted);
        if (!st.ok_status()) return st;
        st = upload_fp8(layer_name(entry.destination, layer), converted, scales);
        if (!st.ok_status()) return st;
    }

    const struct {
        const char* source;
        const char* destination;
    } fp16[] = {
        {"self_attn.out_proj.bias", "vision_attn_o_b_"},
        {"mlp.fc1.bias", "vision_ffn_up_b_"},
        {"mlp.fc2.bias", "vision_ffn_down_b_"},
        {"layer_norm1.weight", "vision_pre_attn_norm_w_"},
        {"layer_norm1.bias", "vision_pre_attn_norm_b_"},
        {"layer_norm2.weight", "vision_pre_ffn_norm_w_"},
        {"layer_norm2.bias", "vision_pre_ffn_norm_b_"},
    };
    for (const auto& entry : fp16) {
        st = upload_f16(prefix + "." + entry.source,
                        layer_name(entry.destination, layer), false);
        if (!st.ok_status()) return st;
    }
    return modalities::Status::ok();
}

modalities::Status NativeThorWeightMaterializer::materialize_encoder_layer(
    int layer,
    std::vector<float>* scales) {
    if (layer < 0 || layer >= 18 || !destination_ || !scales) {
        return invalid("Thor encoder layer index is invalid");
    }
    const std::string prefix = encoder_prefix(layer);
    NativeFloatTensor norm;
    modalities::Status st = load_native_float_tensor(
        source_, prefix + ".input_layernorm.weight", &norm);
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
    NativeF16Tensor converted;
    st = native_source_qkv_to_f16(
        q, k, v, 8, 1, &norm, false, &converted);
    if (!st.ok_status()) return st;
    st = upload_fp8(layer_name("encoder_attn_qkv_w_", layer),
                    converted, scales);
    if (!st.ok_status()) return st;

    NativeSourceTensorView source;
    st = load_native_source_tensor(
        source_, prefix + ".self_attn.o_proj.weight", &source);
    if (!st.ok_status()) return st;
    st = native_source_to_f16(source, false, &converted);
    if (!st.ok_status()) return st;
    st = upload_fp8(layer_name("encoder_attn_o_w_", layer), converted, scales);
    if (!st.ok_status()) return st;

    st = load_native_float_tensor(
        source_, prefix + ".post_attention_layernorm.weight", &norm);
    if (!st.ok_status()) return st;
    NativeSourceTensorView gate;
    NativeSourceTensorView up;
    st = load_native_source_tensor(
        source_, prefix + ".mlp.gate_proj.weight", &gate);
    if (!st.ok_status()) return st;
    st = load_native_source_tensor(
        source_, prefix + ".mlp.up_proj.weight", &up);
    if (!st.ok_status()) return st;
    st = native_source_pair_to_f16(gate, up, &norm, false, &converted);
    if (!st.ok_status()) return st;
    st = upload_fp8(layer_name("encoder_ffn_gate_up_w_", layer),
                    converted, scales);
    if (!st.ok_status()) return st;

    st = load_native_source_tensor(
        source_, prefix + ".mlp.down_proj.weight", &source);
    if (!st.ok_status()) return st;
    st = native_source_to_f16(source, false, &converted);
    if (!st.ok_status()) return st;
    return upload_fp8(layer_name("encoder_ffn_down_w_", layer),
                      converted, scales);
}

modalities::Status NativeThorWeightMaterializer::materialize_decoder_layer(
    int layer,
    std::vector<float>* scales) {
    if (layer < 0 || layer >= 18 || !destination_ || !scales) {
        return invalid("Thor decoder layer index is invalid");
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
    NativeF16Tensor converted;
    st = native_source_qkv_to_f16(
        q, k, v, 8, 1, nullptr, true, &converted);
    if (!st.ok_status()) return st;
    st = upload_fp8(layer_name("decoder_attn_qkv_w_", layer),
                    converted, scales);
    if (!st.ok_status()) return st;

    NativeSourceTensorView source;
    st = load_native_source_tensor(
        source_, prefix + ".self_attn.o_proj.weight", &source);
    if (!st.ok_status()) return st;
    st = native_source_to_f16(source, true, &converted);
    if (!st.ok_status()) return st;
    st = upload_fp8(layer_name("decoder_attn_o_w_", layer), converted, scales);
    if (!st.ok_status()) return st;

    NativeSourceTensorView gate;
    NativeSourceTensorView up;
    st = load_native_source_tensor(
        source_, prefix + ".mlp.gate_proj.weight", &gate);
    if (!st.ok_status()) return st;
    st = load_native_source_tensor(
        source_, prefix + ".mlp.up_proj.weight", &up);
    if (!st.ok_status()) return st;
    st = native_source_pair_to_f16(gate, up, nullptr, true, &converted);
    if (!st.ok_status()) return st;
    st = upload_fp8(layer_name("decoder_ffn_gate_up_w_", layer),
                    converted, scales);
    if (!st.ok_status()) return st;

    st = load_native_source_tensor(
        source_, prefix + ".mlp.down_proj.weight", &source);
    if (!st.ok_status()) return st;
    st = native_source_to_f16(source, true, &converted);
    if (!st.ok_status()) return st;
    st = upload_fp8(layer_name("decoder_ffn_down_w_", layer),
                    converted, scales);
    if (!st.ok_status()) return st;

    const struct {
        const char* source;
        const char* destination;
        bool transpose;
    } fp16[] = {
        {"input_layernorm.dense.weight", "decoder_pre_attn_norm_mod_w_", true},
        {"input_layernorm.dense.bias", "decoder_pre_attn_norm_mod_b_", false},
        {"post_attention_layernorm.dense.weight", "decoder_pre_ffn_norm_mod_w_", true},
        {"post_attention_layernorm.dense.bias", "decoder_pre_ffn_norm_mod_b_", false},
    };
    for (const auto& entry : fp16) {
        st = upload_f16(prefix + "." + entry.source,
                        layer_name(entry.destination, layer), entry.transpose);
        if (!st.ok_status()) return st;
    }
    return modalities::Status::ok();
}

modalities::Status NativeThorWeightMaterializer::materialize_decoder_globals(
    int num_steps) {
    if (!destination_ || num_steps <= 0) {
        return invalid("Thor decoder global configuration is invalid");
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
        {"action_out_proj.weight", "decoder_action_out_proj_w", true},
        {"action_out_proj.bias", "decoder_action_out_proj_b", false},
    };
    for (const auto& entry : entries) {
        modalities::Status st = upload_f16(
            entry.source, entry.destination, entry.transpose);
        if (!st.ok_status()) return st;
    }
    NativeF16Tensor time_embeddings;
    modalities::Status st = native_pi05_time_embeddings_f16(
        num_steps, 1024, &time_embeddings);
    if (!st.ok_status()) return st;
    return upload_f16("decoder_time_embeds", time_embeddings);
}

modalities::Status NativeThorWeightMaterializer::materialize_embedding() {
    return upload_f16(
        "paligemma_with_expert.paligemma.lm_head.weight",
        "embedding_weight", false);
}

modalities::Status NativeThorWeightMaterializer::materialize_all(
    const NativeThorMaterializationOptions& options,
    NativeThorWeightScales* scales) {
    if (!destination_ || !scales || options.num_steps <= 0) {
        return invalid("Thor materialization options are invalid");
    }
    scales->vision.clear();
    scales->encoder.clear();
    scales->decoder.clear();
    scales->vision.reserve(27 * 4);
    scales->encoder.reserve(18 * 4);
    scales->decoder.reserve(18 * 4);

    const bool profile = std::getenv("FLASHRT_PROFILE_NATIVE_SETUP");
    auto checkpoint = std::chrono::steady_clock::now();
    const auto report = [&](const char* phase) {
        const auto now = std::chrono::steady_clock::now();
        if (profile) {
            std::fprintf(stderr, "native_thor_weights %s_ms=%.3f\n", phase,
                         std::chrono::duration<double, std::milli>(
                             now - checkpoint).count());
        }
        checkpoint = now;
    };

    modalities::Status st = materialize_vision_globals();
    if (!st.ok_status()) return st;
    st = materialize_layers_parallel(
        27,
        [this](int layer, std::vector<float>* layer_scales) {
            return materialize_vision_layer(layer, layer_scales);
        },
        &scales->vision);
    if (!st.ok_status()) return st;
    report("vision");
    st = materialize_layers_parallel(
        18,
        [this](int layer, std::vector<float>* layer_scales) {
            return materialize_encoder_layer(layer, layer_scales);
        },
        &scales->encoder);
    if (!st.ok_status()) return st;
    report("encoder");
    st = materialize_layers_parallel(
        18,
        [this](int layer, std::vector<float>* layer_scales) {
            return materialize_decoder_layer(layer, layer_scales);
        },
        &scales->decoder);
    if (!st.ok_status()) return st;
    report("decoder");
    st = materialize_decoder_globals(options.num_steps);
    if (!st.ok_status()) return st;
    if (options.include_embedding) {
        st = materialize_embedding();
        if (!st.ok_status()) return st;
    }
    report("globals");

    if (scales->vision.size() != 108 || scales->encoder.size() != 72 ||
        scales->decoder.size() != 72) {
        return invalid("Thor materialized weight scale count is invalid");
    }
    st = upload_scale_vector("vision_weight_scales", scales->vision);
    if (!st.ok_status()) return st;
    st = upload_scale_vector("encoder_weight_scales", scales->encoder);
    if (!st.ok_status()) return st;
    return upload_scale_vector("decoder_weight_scales", scales->decoder);
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

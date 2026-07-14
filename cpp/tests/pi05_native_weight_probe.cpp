#include "flashrt/cpp/models/pi05/native_weight_ops.h"

#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using flashrt::loader::SafetensorsFile;
using flashrt::models::pi05::NativeBf16Tensor;
using flashrt::models::pi05::NativeFloatTensor;
using flashrt::models::pi05::NativeSourceTensorView;
using flashrt::modalities::Status;

constexpr const char* kVision =
    "paligemma_with_expert.paligemma.model.vision_tower.vision_model";
constexpr const char* kEncoder =
    "paligemma_with_expert.paligemma.model.language_model.layers.0";
constexpr const char* kDecoder =
    "paligemma_with_expert.gemma_expert.model.layers.0";

bool source_view(const SafetensorsFile& file, const std::string& key,
                 NativeSourceTensorView* out) {
    const Status st =
        flashrt::models::pi05::load_native_source_tensor(file, key, out);
    if (!st.ok_status()) std::cerr << st.message << '\n';
    return st.ok_status();
}

bool load(const SafetensorsFile& file, const std::string& key,
          NativeFloatTensor* out) {
    const Status st =
        flashrt::models::pi05::load_native_float_tensor(file, key, out);
    if (!st.ok_status()) std::cerr << st.message << '\n';
    return st.ok_status();
}

bool finish(const NativeFloatTensor& input, NativeBf16Tensor* out) {
    return flashrt::models::pi05::native_to_bf16(input, out).ok_status();
}

bool patch(const SafetensorsFile& file, NativeBf16Tensor* out) {
    NativeSourceTensorView source;
    return source_view(file, std::string(kVision) +
                                 ".embeddings.patch_embedding.weight",
                       &source) &&
           flashrt::models::pi05::native_source_patch_oihw_to_hwio_bf16(
               source, out).ok_status();
}

bool qkv(const SafetensorsFile& file, const std::string& prefix,
         bool fold_rms, NativeBf16Tensor* out) {
    NativeSourceTensorView q;
    NativeSourceTensorView k;
    NativeSourceTensorView v;
    if (!source_view(file, prefix + ".self_attn.q_proj.weight", &q) ||
        !source_view(file, prefix + ".self_attn.k_proj.weight", &k) ||
        !source_view(file, prefix + ".self_attn.v_proj.weight", &v)) {
        return false;
    }
    NativeFloatTensor norm;
    const NativeFloatTensor* norm_ptr = nullptr;
    if (fold_rms) {
        if (!load(file, prefix + ".input_layernorm.weight", &norm)) return false;
        norm_ptr = &norm;
    }
    return flashrt::models::pi05::native_source_qkv_to_bf16(
               q, k, v, 8, 1, norm_ptr, out).ok_status();
}

bool gate_up(const SafetensorsFile& file, NativeBf16Tensor* out) {
    NativeSourceTensorView gate;
    NativeSourceTensorView up;
    return source_view(file, std::string(kDecoder) +
                                 ".mlp.gate_proj.weight", &gate) &&
           source_view(file, std::string(kDecoder) +
                                 ".mlp.up_proj.weight", &up) &&
           flashrt::models::pi05::native_source_pair_transpose_concat_bf16(
               gate, up, out).ok_status();
}

bool action_out(const SafetensorsFile& file, int num_steps,
                NativeBf16Tensor* out) {
    NativeSourceTensorView source;
    return source_view(file, "action_out_proj.weight", &source) &&
           flashrt::models::pi05::native_source_round_scale_to_bf16(
               source, -1.0f / static_cast<float>(num_steps),
               true, out).ok_status();
}

bool rounded_transpose(const SafetensorsFile& file,
                       const std::string& key,
                       NativeBf16Tensor* out) {
    NativeSourceTensorView source;
    return source_view(file, key, &source) &&
           flashrt::models::pi05::native_source_to_bf16(source, true, out)
               .ok_status();
}

bool rounded_copy(const SafetensorsFile& file,
                  const std::string& key,
                  NativeBf16Tensor* out) {
    NativeSourceTensorView source;
    return source_view(file, key, &source) &&
           flashrt::models::pi05::native_source_to_bf16(source, false, out)
               .ok_status();
}

bool folded_transpose(const SafetensorsFile& file,
                      const std::string& key,
                      const std::string& norm_key,
                      NativeBf16Tensor* out) {
    NativeSourceTensorView source;
    NativeFloatTensor norm;
    return source_view(file, key, &source) && load(file, norm_key, &norm) &&
           flashrt::models::pi05::native_source_fold_rms_columns_transpose(
               source, norm, out)
               .ok_status();
}

bool time_embeds(int num_steps, NativeBf16Tensor* out) {
    NativeFloatTensor generated;
    return flashrt::models::pi05::native_pi05_time_embeddings(num_steps, 1024,
                                                              &generated)
               .ok_status() &&
           finish(generated, out);
}

std::uint64_t fnv1a(const std::vector<std::uint16_t>& values) {
    std::uint64_t hash = 14695981039346656037ull;
    const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
    for (std::size_t i = 0; i < values.size() * sizeof(std::uint16_t); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: pi05_native_weight_probe CHECKPOINT OP\n";
        return 2;
    }
    SafetensorsFile file;
    if (!file.open(std::string(argv[1]) + "/model.safetensors")) {
        std::cerr << file.error() << '\n';
        return 2;
    }
    NativeBf16Tensor output;
    const std::string op = argv[2];
    bool ok = false;
    if (op == "patch") {
        ok = patch(file, &output);
    } else if (op == "encoder_qkv0") {
        ok = qkv(file, kEncoder, true, &output);
    } else if (op == "decoder_qkv0") {
        ok = qkv(file, kDecoder, false, &output);
    } else if (op == "decoder_gate_up0") {
        ok = gate_up(file, &output);
    } else if (op == "encoder_o0_fast") {
        ok = rounded_transpose(
            file, std::string(kEncoder) + ".self_attn.o_proj.weight",
            &output);
    } else if (op == "encoder_gate0_fast") {
        ok = folded_transpose(
            file, std::string(kEncoder) + ".mlp.gate_proj.weight",
            std::string(kEncoder) + ".post_attention_layernorm.weight",
            &output);
    } else if (op == "decoder_mod_bias0_fast") {
        ok = rounded_copy(
            file, std::string(kDecoder) +
                      ".input_layernorm.dense.bias",
            &output);
    } else if (op == "action_out10") {
        ok = action_out(file, 10, &output);
    } else if (op == "action_out5") {
        ok = action_out(file, 5, &output);
    } else if (op == "time_embeds10") {
        ok = time_embeds(10, &output);
    } else if (op == "time_embeds5") {
        ok = time_embeds(5, &output);
    }
    if (!ok) {
        std::cerr << "weight probe operation failed: " << op << '\n';
        return 1;
    }
    std::cout << "shape=";
    for (std::size_t i = 0; i < output.shape.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << output.shape[i];
    }
    std::cout << " fnv=" << std::hex << std::setw(16) << std::setfill('0')
              << fnv1a(output.values) << '\n';
    return 0;
}

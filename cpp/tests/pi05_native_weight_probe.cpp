#include "flashrt/cpp/models/pi05/native_weight_ops.h"

#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using flashrt::loader::SafetensorsFile;
using flashrt::models::pi05::NativeBf16Tensor;
using flashrt::models::pi05::NativeFloatTensor;
using flashrt::modalities::Status;

constexpr const char* kVision =
    "paligemma_with_expert.paligemma.model.vision_tower.vision_model";
constexpr const char* kEncoder =
    "paligemma_with_expert.paligemma.model.language_model.layers.0";
constexpr const char* kDecoder =
    "paligemma_with_expert.gemma_expert.model.layers.0";

bool load(const SafetensorsFile& file, const std::string& key,
          NativeFloatTensor* out) {
    const Status st =
        flashrt::models::pi05::load_native_float_tensor(file, key, out);
    if (!st.ok_status()) std::cerr << st.message << '\n';
    return st.ok_status();
}

bool round_bf16(const NativeFloatTensor& input, NativeFloatTensor* out) {
    return flashrt::models::pi05::native_round_to_bf16_float(input, out)
        .ok_status();
}

bool finish(const NativeFloatTensor& input, NativeBf16Tensor* out) {
    return flashrt::models::pi05::native_to_bf16(input, out).ok_status();
}

bool patch(const SafetensorsFile& file, NativeBf16Tensor* out) {
    NativeFloatTensor source;
    NativeFloatTensor rounded;
    NativeFloatTensor transformed;
    return load(file, std::string(kVision) +
                          ".embeddings.patch_embedding.weight",
                &source) &&
           round_bf16(source, &rounded) &&
           flashrt::models::pi05::native_patch_oihw_to_hwio(
               rounded, &transformed).ok_status() &&
           finish(transformed, out);
}

bool qkv(const SafetensorsFile& file, const std::string& prefix,
         bool fold_rms, NativeBf16Tensor* out) {
    NativeFloatTensor q;
    NativeFloatTensor k;
    NativeFloatTensor v;
    if (!load(file, prefix + ".self_attn.q_proj.weight", &q) ||
        !load(file, prefix + ".self_attn.k_proj.weight", &k) ||
        !load(file, prefix + ".self_attn.v_proj.weight", &v)) {
        return false;
    }
    NativeFloatTensor q_input;
    NativeFloatTensor k_input;
    NativeFloatTensor v_input;
    if (fold_rms) {
        q_input = std::move(q);
        k_input = std::move(k);
        v_input = std::move(v);
    } else if (!round_bf16(q, &q_input) || !round_bf16(k, &k_input) ||
               !round_bf16(v, &v_input)) {
        return false;
    }

    NativeFloatTensor qi;
    NativeFloatTensor ki;
    if (!flashrt::models::pi05::native_interleave_qk_rows(q_input, 8, &qi)
             .ok_status() ||
        !flashrt::models::pi05::native_interleave_qk_rows(k_input, 1, &ki)
             .ok_status()) {
        return false;
    }
    if (fold_rms) {
        NativeFloatTensor norm;
        NativeFloatTensor qf;
        NativeFloatTensor kf;
        NativeFloatTensor vf;
        if (!load(file, prefix + ".input_layernorm.weight", &norm) ||
            !flashrt::models::pi05::native_fold_rms_columns(qi, norm, &qf)
                 .ok_status() ||
            !flashrt::models::pi05::native_fold_rms_columns(ki, norm, &kf)
                 .ok_status() ||
            !flashrt::models::pi05::native_fold_rms_columns(v_input, norm, &vf)
                 .ok_status()) {
            return false;
        }
        qi = std::move(qf);
        ki = std::move(kf);
        v_input = std::move(vf);
    }
    NativeFloatTensor joined;
    return flashrt::models::pi05::native_concat_rows_transpose(
               {&qi, &ki, &v_input}, &joined).ok_status() &&
           finish(joined, out);
}

bool gate_up(const SafetensorsFile& file, NativeBf16Tensor* out) {
    NativeFloatTensor gate;
    NativeFloatTensor up;
    NativeFloatTensor gate_rounded;
    NativeFloatTensor up_rounded;
    NativeFloatTensor gate_t;
    NativeFloatTensor up_t;
    NativeFloatTensor joined;
    return load(file, std::string(kDecoder) + ".mlp.gate_proj.weight",
                &gate) &&
           load(file, std::string(kDecoder) + ".mlp.up_proj.weight", &up) &&
           round_bf16(gate, &gate_rounded) &&
           round_bf16(up, &up_rounded) &&
           flashrt::models::pi05::native_transpose_2d(gate_rounded, &gate_t)
               .ok_status() &&
           flashrt::models::pi05::native_transpose_2d(up_rounded, &up_t)
               .ok_status() &&
           flashrt::models::pi05::native_concat_columns(gate_t, up_t, &joined)
               .ok_status() &&
           finish(joined, out);
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

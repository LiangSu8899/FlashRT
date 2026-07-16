#include "flashrt/cpp/models/pi05/native_quantization.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using flashrt::loader::SafetensorsFile;
using flashrt::models::pi05::NativeF16Tensor;
using flashrt::models::pi05::NativeFloatTensor;
using flashrt::models::pi05::NativeFp8Tensor;
using flashrt::models::pi05::NativeInt8Tensor;
using flashrt::models::pi05::NativeSourceTensorView;
using flashrt::modalities::Status;

constexpr const char* kDecoder =
    "paligemma_with_expert.gemma_expert.model.layers.0";
constexpr const char* kEncoder =
    "paligemma_with_expert.paligemma.model.language_model.layers.0";

bool load(const SafetensorsFile& file, const std::string& key,
          NativeFloatTensor* out) {
    const Status st =
        flashrt::models::pi05::load_native_float_tensor(file, key, out);
    if (!st.ok_status()) std::cerr << st.message << '\n';
    return st.ok_status();
}

bool decoder_qkv(const SafetensorsFile& file, NativeFloatTensor* out) {
    NativeFloatTensor q;
    NativeFloatTensor k;
    NativeFloatTensor v;
    NativeFloatTensor qr;
    NativeFloatTensor kr;
    NativeFloatTensor vr;
    NativeFloatTensor qi;
    NativeFloatTensor ki;
    return load(file, std::string(kDecoder) + ".self_attn.q_proj.weight",
                &q) &&
           load(file, std::string(kDecoder) + ".self_attn.k_proj.weight",
                &k) &&
           load(file, std::string(kDecoder) + ".self_attn.v_proj.weight",
                &v) &&
           flashrt::models::pi05::native_round_to_bf16_float(q, &qr)
               .ok_status() &&
           flashrt::models::pi05::native_round_to_bf16_float(k, &kr)
               .ok_status() &&
           flashrt::models::pi05::native_round_to_bf16_float(v, &vr)
               .ok_status() &&
           flashrt::models::pi05::native_interleave_qk_rows(qr, 8, &qi)
               .ok_status() &&
           flashrt::models::pi05::native_interleave_qk_rows(kr, 1, &ki)
               .ok_status() &&
           flashrt::models::pi05::native_concat_rows_transpose(
               {&qi, &ki, &vr}, out)
               .ok_status();
}

bool encoder_gate_up(const SafetensorsFile& file, NativeF16Tensor* out) {
    NativeSourceTensorView gate;
    NativeSourceTensorView up;
    NativeFloatTensor norm;
    return flashrt::models::pi05::load_native_source_tensor(
               file, std::string(kEncoder) + ".mlp.gate_proj.weight", &gate)
               .ok_status() &&
           flashrt::models::pi05::load_native_source_tensor(
               file, std::string(kEncoder) + ".mlp.up_proj.weight", &up)
               .ok_status() &&
           load(file, std::string(kEncoder) +
                          ".post_attention_layernorm.weight", &norm) &&
           flashrt::models::pi05::native_source_pair_to_f16(
               gate, up, &norm, false, out).ok_status();
}

std::uint64_t fnv1a(const void* data, std::size_t bytes) {
    std::uint64_t hash = 14695981039346656037ull;
    const auto* src = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= src[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

void print_shape(const std::vector<std::uint64_t>& shape) {
    std::cout << std::dec;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << shape[i];
    }
}

void print_result(const std::vector<std::uint64_t>& shape,
                  const void* values, std::size_t value_bytes,
                  const std::vector<float>& scales) {
    std::cout << "shape=";
    print_shape(shape);
    std::cout << " values_fnv=" << std::hex << std::setw(16)
              << std::setfill('0') << fnv1a(values, value_bytes)
              << " scale_shape=" << std::dec << scales.size()
              << " scales_fnv=" << std::hex << std::setw(16)
              << fnv1a(scales.data(), scales.size() * sizeof(float)) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: pi05_native_quant_probe CHECKPOINT OP [OUTPUT]\n";
        return 2;
    }
    SafetensorsFile file;
    if (!file.open(std::string(argv[1]) + "/model.safetensors")) {
        std::cerr << file.error() << '\n';
        return 2;
    }
    const std::string op = argv[2];
    if (op == "encoder_gate_up0_fp8") {
        NativeF16Tensor weight;
        NativeFp8Tensor output;
        if (!encoder_gate_up(file, &weight)) return 1;
        std::cout << "input_fnv=" << std::hex << std::setw(16)
                  << std::setfill('0')
                  << fnv1a(weight.values.data(),
                           weight.values.size() * sizeof(std::uint16_t))
                  << ' ';
        const Status st = flashrt::models::pi05::native_quantize_fp8_e4m3(
            weight, false, &output);
        if (!st.ok_status()) {
            std::cerr << st.message << '\n';
            return 1;
        }
        if (argc == 4) {
            std::ofstream file(argv[3], std::ios::binary | std::ios::trunc);
            file.write(reinterpret_cast<const char*>(output.values.data()),
                       static_cast<std::streamsize>(output.values.size()));
            if (!file) return 1;
        }
        print_result(output.shape, output.values.data(), output.values.size(),
                     {output.scale});
        return 0;
    }
    NativeFloatTensor weight;
    if (!decoder_qkv(file, &weight)) return 1;
    if (op == "decoder_qkv0_fp8_kn" || op == "decoder_qkv0_fp8_nk") {
        NativeFp8Tensor output;
        const bool transpose = op.back() == 'k';
        const Status st = flashrt::models::pi05::native_quantize_fp8_e4m3(
            weight, transpose, &output);
        if (!st.ok_status()) {
            std::cerr << st.message << '\n';
            return 1;
        }
        print_result(output.shape, output.values.data(), output.values.size(),
                     {output.scale});
        return 0;
    }
    if (op == "decoder_qkv0_int8") {
        NativeInt8Tensor output;
        const Status st =
            flashrt::models::pi05::native_quantize_int8_per_output(
                weight, &output);
        if (!st.ok_status()) {
            std::cerr << st.message << '\n';
            return 1;
        }
        print_result(output.shape, output.values.data(), output.values.size(),
                     output.scales);
        return 0;
    }
    std::cerr << "unknown quantization probe operation: " << op << '\n';
    return 2;
}

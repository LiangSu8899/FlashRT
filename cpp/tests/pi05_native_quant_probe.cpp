#include "flashrt/cpp/models/pi05/native_quantization.h"
#include "flashrt/cpp/models/pi05/native_device_weights.h"
#include "flashrt/cpp/models/pi05/native_kernel_driver.h"
#include "flashrt/cpp/models/pi05/native_rtx_weight_packer.h"
#include "flashrt/cpp/models/pi05/native_weight_materializer.h"
#include "flashrt/exec.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
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
    std::uint32_t first_scale_bits = 0;
    if (!scales.empty()) {
        static_assert(sizeof(first_scale_bits) == sizeof(scales.front()));
        std::memcpy(&first_scale_bits, scales.data(), sizeof(first_scale_bits));
    }
    std::cout << "shape=";
    print_shape(shape);
    std::cout << " values_fnv=" << std::hex << std::setw(16)
              << std::setfill('0') << fnv1a(values, value_bytes)
              << " scale_shape=" << std::dec << scales.size()
              << " scales_fnv=" << std::hex << std::setw(16)
              << fnv1a(scales.data(), scales.size() * sizeof(float))
              << " first_scale_bits=" << std::setw(8)
              << first_scale_bits << '\n';
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
    if (op == "decoder_qkv0_fp8_kn_gpu" ||
        op == "vision_attn_qkv0_fp8_kn_gpu" ||
        op == "vision_ffn_down4_fp8_kn_gpu") {
        frt_ctx ctx = frt_ctx_create();
        if (!ctx) {
            std::cerr << "failed to create FlashRT context\n";
            return 1;
        }
        std::vector<std::uint64_t> shape;
        std::vector<std::uint8_t> values;
        std::vector<float> scales;
        std::uint64_t input_hash = 0;
        std::string error;
        {
            flashrt::models::pi05::NativeDeviceWeightStore weights(ctx);
            flashrt::models::pi05::NativeWeightMaterializer materializer(
                file, &weights);
            const bool vision_qkv = op.find("vision_attn") == 0;
            const bool vision_down = op.find("vision_ffn") == 0;
            const std::string name =
                vision_qkv ? "vision_attn_qkv_w_0" :
                vision_down ? "vision_ffn_down_w_4" :
                              "decoder_attn_qkv_w_0";
            Status st = vision_qkv
                            ? materializer.materialize_vision_layer(0)
                        : vision_down
                            ? materializer.materialize_vision_layer(4)
                            : materializer.materialize_decoder_layer(0, true);
            flashrt::models::pi05::NativeKernelDriver driver;
            if (st.ok_status()) st = driver.status();
            flashrt::models::pi05::NativeRtxWeightPacker packer(
                &weights, &driver);
            if (st.ok_status()) st = packer.pack_weight(name);
            if (!st.ok_status()) {
                error = st.message;
            } else {
                const auto* input = weights.find(name);
                const auto* output = weights.find("fp8." + name);
                const auto* scale = weights.find("fp8." + name + ".scale");
                if (!input || !output || !scale) {
                    error = "GPU FP8 weight pack output is missing";
                } else {
                    std::vector<std::uint8_t> input_values(
                        frt_buffer_bytes(input->buffer));
                    shape = output->shape;
                    values.resize(frt_buffer_bytes(output->buffer));
                    scales.resize(1);
                    if (cudaMemcpy(
                            input_values.data(), frt_buffer_dptr(input->buffer),
                            input_values.size(), cudaMemcpyDeviceToHost) !=
                            cudaSuccess ||
                        cudaMemcpy(
                            values.data(), frt_buffer_dptr(output->buffer),
                            values.size(), cudaMemcpyDeviceToHost) !=
                            cudaSuccess ||
                        cudaMemcpy(
                            scales.data(), frt_buffer_dptr(scale->buffer),
                            sizeof(float), cudaMemcpyDeviceToHost) !=
                            cudaSuccess) {
                        error = "GPU FP8 weight download failed";
                    } else {
                        input_hash = fnv1a(
                            input_values.data(), input_values.size());
                    }
                }
            }
        }
        frt_ctx_destroy(ctx);
        if (!error.empty()) {
            std::cerr << error << '\n';
            return 1;
        }
        std::cout << "input_fnv=" << std::hex << std::setw(16)
                  << std::setfill('0') << input_hash << ' ';
        if (argc == 4) {
            std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
            output.write(
                reinterpret_cast<const char*>(values.data()),
                static_cast<std::streamsize>(values.size()));
            if (!output) return 1;
        }
        print_result(
            shape, values.data(), values.size(), scales);
        return 0;
    }
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

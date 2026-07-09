#include "flashrt/cpp/models/pi05/native_weight_packer.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

namespace {

bool has_cuda_device() {
    int count = 0;
    const cudaError_t rc = cudaGetDeviceCount(&count);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

template <typename T>
std::vector<T> download(const flashrt::models::pi05::NativeDeviceWeight& weight) {
    std::vector<T> result(frt_buffer_bytes(weight.buffer) / sizeof(T));
    assert(cudaMemcpy(result.data(), frt_buffer_dptr(weight.buffer),
                      result.size() * sizeof(T), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    return result;
}

void upload(flashrt::models::pi05::NativeDeviceWeightStore* store,
            const std::string& name,
            const flashrt::models::pi05::NativeBf16Tensor& tensor) {
    assert(store->upload(name, tensor).ok_status());
}

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    using namespace flashrt::models::pi05;
    NativeFloatTensor source{{2, 3}, {1, 2, 3, 4, 5, 6}};
    NativeBf16Tensor bf16;
    assert(native_to_bf16(source, &bf16).ok_status());
    NativeFloatTensor rounded;
    assert(native_round_to_bf16_float(source, &rounded).ok_status());

    frt_ctx ctx = frt_ctx_create();
    assert(ctx);
    {
        NativeDeviceWeightStore store(ctx);
        assert(store.upload("weight", bf16).ok_status());
        NativeBf16Tensor copied;
        assert(store.download_bf16("weight", &copied).ok_status());
        assert(copied.values == bf16.values);

        NativeWeightPacker packer(&store);
        assert(packer.pack_fp8("weight", false).ok_status());
        assert(packer.pack_fp8("weight", true).ok_status() == false);
        assert(packer.pack_int8("weight").ok_status());
        assert(store.size() == 5);

        NativeFp8Tensor expected_fp8;
        assert(native_quantize_fp8_e4m3(rounded, false, &expected_fp8)
                   .ok_status());
        const auto* fp8 = store.find("fp8.weight");
        assert(fp8 && fp8->dtype == NativeWeightDType::kFp8E4M3);
        assert(download<std::uint8_t>(*fp8) == expected_fp8.values);
        assert(download<float>(*store.find("fp8.weight.scale")) ==
               std::vector<float>({expected_fp8.scale}));

        NativeInt8Tensor expected_int8;
        assert(native_quantize_int8_per_output(rounded, &expected_int8)
                   .ok_status());
        const auto* int8 = store.find("int8.weight");
        assert(int8 && int8->dtype == NativeWeightDType::kInt8);
        assert(download<std::int8_t>(*int8) == expected_int8.values);
        assert(download<float>(*store.find("int8.weight.scale")) ==
               expected_int8.scales);
        assert(!packer.pack_int8("missing").ok_status());
    }
    frt_ctx_destroy(ctx);

    ctx = frt_ctx_create();
    assert(ctx);
    {
        NativeDeviceWeightStore store(ctx);
        NativeBf16Tensor tiny;
        tiny.shape = {1, 1};
        tiny.values = {flashrt::modalities::float_to_bfloat16(1.0f)};
        for (int layer = 0; layer < 27; ++layer) {
            for (const char* stem : {
                     "vision_attn_qkv_w_", "vision_attn_o_w_",
                     "vision_ffn_up_w_", "vision_ffn_down_w_"}) {
                upload(&store, std::string(stem) + std::to_string(layer),
                       tiny);
            }
        }
        upload(&store, "encoder_multi_modal_projector_w", tiny);
        for (int layer = 0; layer < 18; ++layer) {
            const std::string suffix = std::to_string(layer);
            for (const std::string& name : {
                     "encoder_attn_qkv_w_" + suffix,
                     "encoder_attn_o_w_" + suffix,
                     "encoder_ffn_gate_w_" + suffix,
                     "encoder_ffn_up_w_" + suffix,
                     "encoder_ffn_down_w_" + suffix,
                     "decoder_attn_qkv_w_" + suffix,
                     "decoder_attn_o_w_" + suffix,
                     "decoder_ffn_gate_w_" + suffix,
                     "decoder_ffn_up_w_" + suffix,
                     "decoder_ffn_gate_up_w_" + suffix,
                     "decoder_ffn_down_w_" + suffix}) {
                upload(&store, name, tiny);
            }
        }
        assert(store.size() == 307);
        NativeWeightPacker packer(&store);
        assert(packer.pack_all_fp8(false).ok_status());
        assert(packer.pack_vision_int8().ok_status());
        assert(packer.pack_encoder_int8().ok_status());
        assert(packer.pack_decoder_int8().ok_status());
        assert(store.size() == 1407);
        assert(store.find("fp8.vision_projector_w")->dtype ==
               NativeWeightDType::kFp8E4M3);
        assert(store.find("fp8.encoder_ffn_gate_up_w_17")->shape ==
               std::vector<std::uint64_t>({1, 2}));
        assert(store.find("int8.vision_ffn_down_w_26.scale")->dtype ==
               NativeWeightDType::kFloat32);
        assert(store.find("int8.encoder_ffn_up_w_17")->dtype ==
               NativeWeightDType::kInt8);
        assert(store.find("int8.decoder_ffn_down_w_17")->dtype ==
               NativeWeightDType::kInt8);
    }
    frt_ctx_destroy(ctx);
    std::printf("PASS - Pi0.5 native weight packer\n");
    return 0;
}

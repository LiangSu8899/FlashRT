#include "flashrt/cpp/models/pi05/native_rtx_attention.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdio>

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        cudaGetLastError();
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    using namespace flashrt::models::pi05;
    frt_ctx ctx = frt_ctx_create();
    assert(ctx);
    {
        NativeRtxAttentionWorkspace attention(ctx);
        NativeRtxAttentionConfig bad;
        bad.encoder_layers = 17;
        assert(!attention.allocate(bad).ok_status());
        NativeRtxAttentionConfig config;
        assert(attention.allocate(config).ok_status());
        assert(attention.size() == 22);
        assert(attention.allocated_bytes() > 0);
        assert(attention.encoder_splits() == 12);
        assert(attention.decoder_splits() == 12);
        assert(attention.kv_layer_stride_bytes() == 722 * 256 * 2);
        assert(attention.find("attn_enc_K")->shape ==
               std::vector<std::uint64_t>({18, 722, 1, 256}));
        assert(attention.find("attn_enc_lse")->shape ==
               std::vector<std::uint64_t>({1, 8, 768}));
        assert(attention.find("attn_dec_lse")->shape ==
               std::vector<std::uint64_t>({1, 8, 128}));
        void* base = frt_buffer_dptr(attention.find("attn_enc_K")->buffer);
        assert(attention.encoder_k_layer_dptr(0) == base);
        assert(static_cast<unsigned char*>(attention.encoder_k_layer_dptr(17)) ==
               static_cast<unsigned char*>(base) +
                   17 * attention.kv_layer_stride_bytes());
        assert(!attention.encoder_k_layer_dptr(18));

        void* seqused_ptr =
            frt_buffer_dptr(attention.find("attn_enc_seqused")->buffer);
        const std::size_t bytes = attention.allocated_bytes();
        for (int i = 0; i < 1000; ++i) {
            assert(attention.set_fixed_prompt_length(i % 201).ok_status());
            assert(frt_buffer_dptr(
                       attention.find("attn_enc_seqused")->buffer) ==
                   seqused_ptr);
            assert(attention.allocated_bytes() == bytes);
        }
        std::int32_t enc = 0;
        std::int32_t dec = 0;
        std::int32_t pos = 0;
        assert(cudaMemcpy(&enc, frt_buffer_dptr(
                                    attention.find("attn_enc_seqused")->buffer),
                          sizeof(enc), cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(cudaMemcpy(&dec, frt_buffer_dptr(
                                    attention.find("attn_dec_seqused")->buffer),
                          sizeof(dec), cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(cudaMemcpy(&pos, frt_buffer_dptr(
                                    attention.find("attn_dec_devpos")->buffer),
                          sizeof(pos), cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(enc == 512 + (999 % 201));
        assert(dec == enc + 10);
        assert(pos == enc);
        assert(!attention.set_fixed_prompt_length(201).ok_status());
    }
    frt_ctx_destroy(ctx);
    std::printf("PASS - Pi0.5 native RTX attention workspace\n");
    return 0;
}

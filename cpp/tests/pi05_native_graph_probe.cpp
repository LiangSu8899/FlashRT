#include "flashrt/cpp/models/pi05/native_graph_owner.h"
#include "flashrt/cpp/modalities/types.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<std::uint16_t> download(frt_buffer buffer) {
    std::vector<std::uint16_t> host(frt_buffer_bytes(buffer) /
                                    sizeof(std::uint16_t));
    if (cudaMemcpy(host.data(), frt_buffer_dptr(buffer),
                   frt_buffer_bytes(buffer), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        host.clear();
    }
    return host;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: pi05_native_graph_probe CHECKPOINT\n";
        return 2;
    }
    using namespace flashrt::models::pi05;
    flashrt::modalities::Status st;
    std::unique_ptr<NativeGraphOwner> owner = NativeGraphOwner::create(
        argv[1], NativeGraphConfig{}, &st);
    if (!owner) {
        std::cerr << st.message << '\n';
        return 1;
    }
    const NativeWorkspaceBuffer* images =
        owner->workspace().find("observation_images_normalized");
    const NativeWorkspaceBuffer* prompt =
        owner->workspace().find("prompt_embedding");
    const NativeWorkspaceBuffer* noise =
        owner->workspace().find("diffusion_noise");
    if (!images || !prompt || !noise || !owner->infer_graph() ||
        frt_graph_variant_count(owner->infer_graph()) != 1 ||
        owner->stream_id() < 0 || !owner->native_stream()) {
        return 1;
    }
    std::vector<std::uint16_t> host_images(
        frt_buffer_bytes(images->buffer) / sizeof(std::uint16_t));
    std::vector<std::uint16_t> host_prompt(
        frt_buffer_bytes(prompt->buffer) / sizeof(std::uint16_t));
    std::vector<std::uint16_t> host_noise(
        frt_buffer_bytes(noise->buffer) / sizeof(std::uint16_t));
    for (std::size_t i = 0; i < host_images.size(); ++i) {
        host_images[i] = flashrt::modalities::float_to_bfloat16(
            static_cast<float>(static_cast<int>(i % 257) - 128) / 128.0f);
    }
    for (std::size_t i = 0; i < host_prompt.size(); ++i) {
        host_prompt[i] = flashrt::modalities::float_to_bfloat16(
            static_cast<float>(static_cast<int>(i % 31) - 15) / 32.0f);
    }
    for (std::size_t i = 0; i < host_noise.size(); ++i) {
        host_noise[i] = flashrt::modalities::float_to_bfloat16(
            static_cast<float>(static_cast<int>(i % 23) - 11) / 12.0f);
    }
    if (cudaMemcpy(frt_buffer_dptr(images->buffer), host_images.data(),
                   frt_buffer_bytes(images->buffer),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(frt_buffer_dptr(prompt->buffer), host_prompt.data(),
                   frt_buffer_bytes(prompt->buffer),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        !owner->set_prompt_length(37).ok_status()) {
        return 1;
    }
    const std::size_t allocation_count = owner->workspace().allocation_count();
    if (cudaMemcpy(frt_buffer_dptr(noise->buffer), host_noise.data(),
                   frt_buffer_bytes(noise->buffer),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        owner->replay() != FRT_OK || !owner->synchronize().ok_status()) {
        return 1;
    }
    const std::vector<std::uint16_t> expected = download(noise->buffer);
    if (expected.empty()) return 1;
    for (int replay = 0; replay < 100; ++replay) {
        if (cudaMemcpyAsync(
                frt_buffer_dptr(noise->buffer), host_noise.data(),
                frt_buffer_bytes(noise->buffer), cudaMemcpyHostToDevice,
                static_cast<cudaStream_t>(owner->native_stream())) !=
                cudaSuccess ||
            owner->replay() != FRT_OK) {
            return 1;
        }
    }
    if (!owner->synchronize().ok_status() ||
        frt_graph_variant_count(owner->infer_graph()) != 1 ||
        owner->workspace().allocation_count() != allocation_count ||
        download(noise->buffer) != expected) {
        return 1;
    }
    std::cout << "PASS native full graph 100 replays\n";
    return 0;
}

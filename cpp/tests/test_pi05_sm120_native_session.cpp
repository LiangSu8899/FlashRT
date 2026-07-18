#include "flashrt/cpp/models/pi05/model/native_session.h"
#include "flashrt/cpp/models/pi05/targets/sm120/target.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace {

std::atomic<bool> count_allocations{false};
std::atomic<std::size_t> allocation_count{0};

[[noreturn]] void fail(const char* expression, int line) {
    std::fprintf(stderr, "FAIL line %d: %s\n", line, expression);
    std::abort();
}

#define CHECK(expression)                              \
    do {                                               \
        if (!(expression)) fail(#expression, __LINE__); \
    } while (false)

std::vector<std::uint16_t> download(frt_buffer buffer) {
    std::vector<std::uint16_t> output(
        frt_buffer_bytes(buffer) / sizeof(std::uint16_t));
    CHECK(cudaMemcpy(output.data(), frt_buffer_dptr(buffer),
                     frt_buffer_bytes(buffer), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
    return output;
}

}  // namespace

void* operator new(std::size_t bytes) {
    if (count_allocations.load(std::memory_order_relaxed)) {
        allocation_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (void* pointer = std::malloc(bytes)) return pointer;
    throw std::bad_alloc();
}

void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || !device_count) {
        cudaGetLastError();
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    int device = 0;
    cudaDeviceProp properties{};
    CHECK(cudaGetDevice(&device) == cudaSuccess);
    CHECK(cudaGetDeviceProperties(&properties, device) == cudaSuccess);
    if (properties.major != 12 || properties.minor != 0) {
        std::printf("SKIP - SM120 target needs compute capability 12.0\n");
        return 0;
    }
    const char* checkpoint = std::getenv("FLASHRT_PI05_CHECKPOINT");
    if (!checkpoint || !checkpoint[0]) {
        std::printf("SKIP - PI0.5 checkpoint is not configured\n");
        return 0;
    }

    using namespace flashrt::models::pi05;
    using flashrt::models::pi05::targets::sm120::Sm120TargetBundle;
    using flashrt::models::pi05::targets::sm120::Sm120Precision;
    using flashrt::models::pi05::targets::sm120::Sm120TargetConfig;
    Pi05ShapeConfig config;
    config.num_views = 3;
    config.max_prompt_tokens = 200;
    config.chunk = 10;
    config.num_steps = 10;
    config.vision_pool_factor = 1;
    config.state_dim = 8;
    config.robot_action_dim = 7;
    Pi05ResolvedShape shape;
    CHECK(resolve_pi05_shape(config, &shape).ok_status());

    frt_ctx context = frt_ctx_create();
    CHECK(context != nullptr);
    flashrt::modalities::Status status;
    Sm120TargetConfig target_config;
    target_config.checkpoint_path = checkpoint;
    const char* calibration = std::getenv("FLASHRT_PI05_CALIBRATION");
    if (calibration && calibration[0]) {
        NativeCalibrationArtifact artifact;
        CHECK(load_native_calibration_artifact(calibration, &artifact)
                  .ok_status());
        target_config.precision = Sm120Precision::kStaticFp8E4M3;
        target_config.calibration = std::move(artifact);
    }
    std::unique_ptr<Sm120TargetBundle> concrete = Sm120TargetBundle::create(
        context, shape, std::move(target_config), &status);
    CHECK(concrete && status.ok_status());
    std::unique_ptr<Pi05TargetBundle> target(std::move(concrete));
    std::unique_ptr<Pi05NativeSession> session = Pi05NativeSession::create(
        context, shape, std::move(target), &status);
    CHECK(session && status.ok_status());

    for (const Pi05GraphId id : {
             Pi05GraphId::kInfer,
             Pi05GraphId::kDecodeOnly,
             Pi05GraphId::kContext}) {
        CHECK(session->graph(id) != nullptr);
        CHECK(frt_graph_variant_count(session->graph(id)) == 1);
    }
    CHECK(session->stream_id() >= 0);
    CHECK(session->native_stream() != nullptr);
    CHECK(session->set_prompt_length(37).ok_status());

    const Pi05ResolvedBuffers& buffers = session->resources().buffers;
    std::vector<std::uint16_t> images(
        frt_buffer_bytes(buffers.images.buffer) / sizeof(std::uint16_t));
    std::vector<std::uint16_t> prompt(
        frt_buffer_bytes(buffers.prompt_embedding.buffer) /
        sizeof(std::uint16_t));
    std::vector<std::uint16_t> noise(
        frt_buffer_bytes(buffers.noise.buffer) / sizeof(std::uint16_t));
    for (std::size_t i = 0; i < images.size(); ++i) {
        images[i] = flashrt::modalities::float_to_bfloat16(
            static_cast<float>(static_cast<int>(i % 257) - 128) / 128.0f);
    }
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        prompt[i] = flashrt::modalities::float_to_bfloat16(
            static_cast<float>(static_cast<int>(i % 31) - 15) / 32.0f);
    }
    for (std::size_t i = 0; i < noise.size(); ++i) {
        noise[i] = flashrt::modalities::float_to_bfloat16(
            static_cast<float>(static_cast<int>(i % 23) - 11) / 12.0f);
    }
    CHECK(cudaMemcpy(frt_buffer_dptr(buffers.images.buffer), images.data(),
                     frt_buffer_bytes(buffers.images.buffer),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(frt_buffer_dptr(buffers.prompt_embedding.buffer),
                     prompt.data(),
                     frt_buffer_bytes(buffers.prompt_embedding.buffer),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(frt_buffer_dptr(buffers.noise.buffer), noise.data(),
                     frt_buffer_bytes(buffers.noise.buffer),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(session->replay() == FRT_OK);
    CHECK(session->synchronize().ok_status());
    const std::vector<std::uint16_t> expected = download(buffers.noise.buffer);

    const cudaStream_t stream =
        static_cast<cudaStream_t>(session->native_stream());
    CHECK(cudaMemcpyAsync(frt_buffer_dptr(buffers.noise.buffer), noise.data(),
                          frt_buffer_bytes(buffers.noise.buffer),
                          cudaMemcpyHostToDevice, stream) == cudaSuccess);
    CHECK(session->replay(Pi05GraphId::kContext) == FRT_OK);
    CHECK(session->replay(Pi05GraphId::kDecodeOnly) == FRT_OK);
    CHECK(session->synchronize().ok_status());
    CHECK(download(buffers.noise.buffer) == expected);

    allocation_count.store(0, std::memory_order_relaxed);
    count_allocations.store(true, std::memory_order_relaxed);
    for (int replay = 0; replay < 100; ++replay) {
        CHECK(cudaMemcpyAsync(
                  frt_buffer_dptr(buffers.noise.buffer), noise.data(),
                  frt_buffer_bytes(buffers.noise.buffer),
                  cudaMemcpyHostToDevice, stream) == cudaSuccess);
        if (replay % 2 == 0) {
            CHECK(session->replay(Pi05GraphId::kInfer) == FRT_OK);
        } else {
            CHECK(session->replay(Pi05GraphId::kContext) == FRT_OK);
            CHECK(session->replay(Pi05GraphId::kDecodeOnly) == FRT_OK);
        }
    }
    CHECK(session->synchronize().ok_status());
    count_allocations.store(false, std::memory_order_relaxed);
    CHECK(allocation_count.load(std::memory_order_relaxed) == 0);
    CHECK(download(buffers.noise.buffer) == expected);
    for (const Pi05GraphId id : {
             Pi05GraphId::kInfer,
             Pi05GraphId::kDecodeOnly,
             Pi05GraphId::kContext}) {
        CHECK(frt_graph_variant_count(session->graph(id)) == 1);
    }

    session.reset();
    std::printf("PASS - PI0.5 SM120 native session graphs\n");
    return 0;
}

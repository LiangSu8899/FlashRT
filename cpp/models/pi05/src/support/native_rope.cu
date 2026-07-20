#include "flashrt/cpp/models/pi05/support/native_rope.h"

#include "flashrt/cpp/models/pi05/model/dims.h"

#include "flashrt/native_cpp/operations.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

constexpr int kFrequencies = kPi05ModelDims.decoder_head_dim / 2;
static_assert(kPi05ModelDims.encoder_head_dim ==
                  kPi05ModelDims.decoder_head_dim,
              "PI0.5 encoder and decoder RoPE widths must match");

}  // namespace

modalities::Status generate_native_rope_f16(
    void* output, int start_position, int positions, std::uintptr_t stream) {
    constexpr int kMax = std::numeric_limits<int>::max();
    if (!output || start_position < 0 || positions <= 0 ||
        positions > kMax / kFrequencies ||
        start_position > kMax - (positions - 1)) {
        return modalities::Status::error(
            modalities::StatusCode::kInvalidArgument,
            "native RoPE generation arguments are invalid");
    }
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    ::flashrt_native_generate_rope_table_f16(
        static_cast<__half*>(output), start_position, positions,
        kFrequencies, 10000.0f, cuda_stream);
    cudaError_t rc = cudaGetLastError();
    if (rc == cudaSuccess) rc = cudaStreamSynchronize(cuda_stream);
    return rc == cudaSuccess
               ? modalities::Status::ok()
               : modalities::Status::error(
                     modalities::StatusCode::kBackend,
                     std::string("native RoPE generation failed: ") +
                         cudaGetErrorString(rc));
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

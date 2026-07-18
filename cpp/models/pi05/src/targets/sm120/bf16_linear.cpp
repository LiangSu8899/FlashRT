#include "flashrt/cpp/models/pi05/targets/sm120/bf16_linear.h"

#include "gemm_runner.h"

#include <cuda_runtime_api.h>

#include <exception>
#include <limits>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

bool valid_matrix_weight(const Pi05ResolvedWeight& weight,
                         int input_width,
                         int output_width) {
    if (!weight.device_data || weight.storage != Pi05WeightStorage::kBFloat16 ||
        !weight.shape.rank || weight.shape.rank > modalities::Shape::kMaxRank ||
        input_width <= 0 || output_width <= 0 ||
        weight.shape.dims[weight.shape.rank - 1] !=
            static_cast<std::uint64_t>(output_width)) {
        return false;
    }
    std::uint64_t elements = 1;
    for (std::size_t i = 0; i < weight.shape.rank; ++i) {
        const std::uint64_t dimension = weight.shape.dims[i];
        if (!dimension ||
            elements > std::numeric_limits<std::uint64_t>::max() / dimension) {
            return false;
        }
        elements *= dimension;
    }
    const std::uint64_t expected =
        static_cast<std::uint64_t>(input_width) *
        static_cast<std::uint64_t>(output_width);
    return elements == expected &&
           expected <= std::numeric_limits<std::uint64_t>::max() /
                           sizeof(std::uint16_t) &&
           weight.bytes == expected * sizeof(std::uint16_t);
}

}  // namespace

struct Sm120Bf16Linear::Impl final {
    GemmRunner gemm;
};

Sm120Bf16Linear::Sm120Bf16Linear() noexcept {
    try {
        impl_.reset(new Impl());
    } catch (const std::exception& error) {
        error_ = error.what();
    } catch (...) {
        error_ = "SM120 BF16 linear initialization failed";
    }
}

Sm120Bf16Linear::~Sm120Bf16Linear() = default;

modalities::Status Sm120Bf16Linear::status() const {
    return impl_ ? modalities::Status::ok() : backend(error_);
}

modalities::Status Sm120Bf16Linear::run(
    const Pi05ResolvedWeight& weight,
    const void* input,
    void* output,
    int rows,
    int input_width,
    int output_width,
    Pi05Stream stream) const {
    if (!impl_) return backend(error_);
    if (!input || !output || rows <= 0 ||
        !valid_matrix_weight(weight, input_width, output_width)) {
        return invalid("SM120 BF16 linear arguments are invalid");
    }
    try {
        impl_->gemm.bf16_nn(
            const_cast<void*>(input), const_cast<void*>(weight.device_data),
            output, rows, output_width, input_width,
            reinterpret_cast<cudaStream_t>(stream));
        return modalities::Status::ok();
    } catch (const std::exception& error) {
        return backend(error.what());
    } catch (...) {
        return backend("SM120 BF16 linear launch failed");
    }
}

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

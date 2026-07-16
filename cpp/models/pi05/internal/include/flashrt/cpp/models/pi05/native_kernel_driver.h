#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_KERNEL_DRIVER_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_KERNEL_DRIVER_H

#include "flashrt/cpp/modalities/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {

class NativeKernelDriver {
public:
    NativeKernelDriver() noexcept;
    ~NativeKernelDriver();

    NativeKernelDriver(const NativeKernelDriver&) = delete;
    NativeKernelDriver& operator=(const NativeKernelDriver&) = delete;

    modalities::Status status() const;
    modalities::Status bf16_nn(void* a, void* b, void* output,
                               int m, int n, int k,
                               std::uintptr_t stream) const;
    modalities::Status add_bias_bf16(void* values, const void* bias,
                                     int rows, int columns,
                                     std::uintptr_t stream) const;
    modalities::Status silu_bf16(void* values, std::size_t elements,
                                 std::uintptr_t stream) const;
    modalities::Status gelu_bf16(void* values, std::size_t elements,
                                 std::uintptr_t stream) const;
    modalities::Status gate_gelu_bf16(const void* gate, const void* up,
                                      void* output, std::size_t elements,
                                      std::uintptr_t stream) const;
    modalities::Status residual_add_bf16(void* residual, const void* values,
                                         std::size_t elements,
                                         std::uintptr_t stream) const;
    modalities::Status bias_residual_bf16(
        void* residual, const void* values, const void* bias,
        int rows, int columns, std::uintptr_t stream) const;
    modalities::Status gate_mul_residual_bf16(
        void* residual, const void* values, const void* gate,
        std::size_t elements, std::uintptr_t stream) const;
    modalities::Status rms_norm_bf16(
        const void* values, const void* weight, void* output,
        int rows, int columns, float epsilon, std::uintptr_t stream) const;
    modalities::Status layer_norm_bf16(
        const void* values, const void* weight, const void* bias, void* output,
        int rows, int columns, float epsilon, std::uintptr_t stream) const;
    modalities::Status ada_rms_norm_style_bf16(
        const void* values, const void* weight, const void* style,
        void* output, void* gate_output, int rows, int columns,
        float epsilon, std::uintptr_t stream) const;
    modalities::Status qkv_split_bf16(
        const void* qkv, void* query, void* key, void* value,
        int rows, int query_columns, int key_columns, int value_columns,
        std::uintptr_t stream) const;
    modalities::Status qkv_split_rope_bf16(
        const void* qkv, const void* rope, void* query, void* key, void* value,
        int rows, int query_columns, int key_columns, int value_columns,
        int head_dimension, std::uintptr_t stream) const;
    modalities::Status qkv_split_rope_devpos_bf16(
        const void* qkv, const void* rope, void* query, void* key, void* value,
        const void* device_position, int rows, int query_columns,
        int key_columns, int value_columns, int head_dimension,
        std::uintptr_t stream) const;
    modalities::Status patch_im2col_16bit(
        const void* images, void* patches, int num_views,
        std::uintptr_t stream) const;
    modalities::Status avg_pool_vision_bf16(
        const void* values, void* output, int num_views, int height, int width,
        int columns, int pool_factor, std::uintptr_t stream) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string error_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_KERNEL_DRIVER_H

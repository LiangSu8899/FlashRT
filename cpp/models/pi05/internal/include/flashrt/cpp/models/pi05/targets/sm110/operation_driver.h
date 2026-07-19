#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_OPERATION_DRIVER_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_OPERATION_DRIVER_H

#include "flashrt/cpp/modalities/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm110 {

enum class Sm110Fp8Tactic {
    kSquare,
    kT1,
    kWide,
    kPlain,
};

class Sm110OperationDriver {
public:
    Sm110OperationDriver() noexcept;
    ~Sm110OperationDriver();

    Sm110OperationDriver(const Sm110OperationDriver&) = delete;
    Sm110OperationDriver& operator=(const Sm110OperationDriver&) = delete;

    modalities::Status status() const;
    modalities::Status fp16_nn(void* a, void* b, void* output,
                               int m, int n, int k,
                               std::uintptr_t stream) const;
    modalities::Status fp8_nn_bias(
        void* a, void* b, void* output, void* bias,
        int m, int n, int k, float alpha, std::uintptr_t stream) const;
    modalities::Status fp8_nn_bias_residual(
        void* a, void* b, void* output, void* bias,
        int m, int n, int k, float alpha, std::uintptr_t stream) const;
    modalities::Status fp8_nn_gelu_bias(
        void* a, void* b, void* output, void* bias,
        int m, int n, int k, float alpha, std::uintptr_t stream) const;
    modalities::Status fp8_cutlass(
        void* a, void* b, void* output, int m, int n, int k,
        float alpha, float beta, Sm110Fp8Tactic tactic,
        std::uintptr_t stream) const;
    modalities::Status fp8_descale(
        void* a, void* b, void* output, int m, int n, int k,
        const float* activation_scale, const float* weight_scale,
        std::uintptr_t stream) const;

    modalities::Status add_bias_fp16(void* values, const void* bias,
                                     int rows, int columns,
                                     std::uintptr_t stream) const;
    modalities::Status layer_norm_fp16(
        const void* values, const void* weight, const void* bias, void* output,
        int rows, int columns, float epsilon, std::uintptr_t stream) const;
    modalities::Status layer_norm_fp8(
        const void* values, void* output, const void* weight, const void* bias,
        int rows, int columns, float epsilon, std::uintptr_t stream) const;
    modalities::Status rms_norm_fp16(
        const void* values, const void* weight, void* output,
        int rows, int columns, float epsilon, std::uintptr_t stream) const;
    modalities::Status rms_norm_fp8_noweight(
        const void* values, void* output, int rows, int columns,
        const float* scale, std::uintptr_t stream) const;
    modalities::Status residual_rms_norm_fp8_noweight(
        void* residual, const void* values, void* output,
        int rows, int columns, const float* scale,
        std::uintptr_t stream) const;
    modalities::Status quantize_fp8_static(
        const void* values, void* output, const float* scale,
        std::size_t elements, std::uintptr_t stream) const;
    modalities::Status quantize_fp8_dynamic(
        const void* values, void* output, float* scale,
        std::size_t elements, std::uintptr_t stream) const;

    modalities::Status gelu_fp16(void* values, std::size_t elements,
                                 std::uintptr_t stream) const;
    modalities::Status silu_fp16(void* values, std::size_t elements,
                                 std::uintptr_t stream) const;
    modalities::Status precise_silu_fp16(
        void* values, std::size_t elements, std::uintptr_t stream) const;
    modalities::Status gate_gelu_fp16(
        const void* merged, void* output, int rows, int hidden,
        std::uintptr_t stream) const;
    modalities::Status gate_gelu_fp8(
        const void* merged, void* output, int rows, int hidden,
        const float* scale, std::uintptr_t stream) const;
    modalities::Status residual_add_fp16(
        void* residual, const void* values, std::size_t elements,
        std::uintptr_t stream) const;

    modalities::Status fused_adarms_fp8(
        const void* values, const void* style, void* output, void* gate,
        int rows, int columns, const float* scale,
        std::uintptr_t stream) const;
    modalities::Status gate_res_adarms_fp8(
        const void* gemm_output, const void* previous_gate, void* residual,
        const void* style, void* output, void* gate,
        int rows, int columns, const float* scale,
        std::uintptr_t stream) const;
    modalities::Status gate_res_fp16(
        const void* gemm_output, const void* gate, void* residual,
        std::size_t elements, std::uintptr_t stream) const;
    modalities::Status adarms_fp16(
        const void* values, const void* style, void* output, void* gate,
        int rows, int columns, std::uintptr_t stream) const;

    modalities::Status qkv_rope_cache_fp16(
        const void* qkv, const void* rope, void* query, void* key_cache,
        void* value_cache, int rows, int query_columns, int key_columns,
        int head_dimension, int qkv_stride, int cache_offset,
        int cache_stride, std::uintptr_t stream) const;
    modalities::Status qkv_rope_cache_devpos_fp16(
        const void* qkv, const void* rope, void* query, void* key_cache,
        void* value_cache, const int* device_position, int rows,
        int query_columns, int key_columns, int head_dimension,
        int qkv_stride, int cache_offset, int cache_stride,
        std::uintptr_t stream) const;
    modalities::Status attention_seqused_fp16(
        const void* query, const void* key, const void* value, void* logits,
        void* output, int query_rows, int max_key_rows, int heads,
        int head_dimension, const int* valid_key_rows, float scale,
        std::uintptr_t stream) const;
    modalities::Status vision_fmha_fp16(
        const void* query, const void* key, const void* value, void* output,
        int batch, int query_rows, int key_rows, int query_heads,
        int key_heads, int head_dimension, int query_stride, int key_stride,
        std::uintptr_t stream) const;

    modalities::Status patch_im2col_fp16(
        const void* images, void* patches, int num_views,
        std::uintptr_t stream) const;
    modalities::Status patch_bias_position_fp16(
        void* output, const void* bias, const void* position,
        int sequence, int columns, int positions_per_view,
        std::uintptr_t stream) const;
    modalities::Status bias_residual_fp16(
        void* residual, const void* values, const void* bias,
        int rows, int columns, std::uintptr_t stream) const;

    modalities::Status gmm_fp16(
        const void* a, const void* b, void* output,
        int m, int n, int k, float beta, std::uintptr_t stream) const;
    modalities::Status gmm_fp16_out_fp32(
        const void* a, const void* b, float* output,
        int m, int n, int k, std::uintptr_t stream) const;
    modalities::Status action_update_fp16(
        const float* delta, const void* bias, void* noise,
        int rows, int columns, float dt, std::uintptr_t stream) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string error_;
};

}  // namespace sm110
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_OPERATION_DRIVER_H

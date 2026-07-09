#include "flashrt/cpp/models/pi05/native_kernel_driver.h"

#include "activation.cuh"
#include "elementwise.cuh"
#include "gemm_runner.h"
#include "norm.cuh"
#include "patch_embed.cuh"
#include "rope.cuh"

#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include <climits>
#include <exception>

void add_bias_bf16(__nv_bfloat16* x, const __nv_bfloat16* b,
                   int rows, int columns, cudaStream_t stream);

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

__global__ void native_silu_bf16_kernel(__nv_bfloat16* values,
                                        std::size_t elements) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float value = __bfloat162float(values[index]);
        values[index] =
            __float2bfloat16(value / (1.0f + expf(-value)));
    }
}

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

modalities::Status launch_status() {
    const cudaError_t rc = cudaGetLastError();
    return rc == cudaSuccess ? modalities::Status::ok()
                             : backend(cudaGetErrorString(rc));
}

}  // namespace

struct NativeKernelDriver::Impl {
    GemmRunner gemm;
};

NativeKernelDriver::NativeKernelDriver() noexcept {
    try {
        impl_.reset(new Impl());
    } catch (const std::exception& e) {
        error_ = e.what();
    } catch (...) {
        error_ = "native kernel driver initialization failed";
    }
}

NativeKernelDriver::~NativeKernelDriver() = default;

modalities::Status NativeKernelDriver::status() const {
    return impl_ ? modalities::Status::ok() : backend(error_);
}

modalities::Status NativeKernelDriver::bf16_nn(
    void* a,
    void* b,
    void* output,
    int m,
    int n,
    int k,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!a || !b || !output || m <= 0 || n <= 0 || k <= 0) {
        return invalid("native BF16 GEMM arguments are invalid");
    }
    try {
        impl_->gemm.bf16_nn(a, b, output, m, n, k,
                            reinterpret_cast<cudaStream_t>(stream));
        return modalities::Status::ok();
    } catch (const std::exception& e) {
        return backend(e.what());
    } catch (...) {
        return backend("native BF16 GEMM launch failed");
    }
}

modalities::Status NativeKernelDriver::add_bias_bf16(
    void* values,
    const void* bias,
    int rows,
    int columns,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!values || !bias || rows <= 0 || columns <= 0) {
        return invalid("native BF16 bias arguments are invalid");
    }
    ::add_bias_bf16(static_cast<__nv_bfloat16*>(values),
                    static_cast<const __nv_bfloat16*>(bias), rows, columns,
                    reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeKernelDriver::silu_bf16(
    void* values,
    std::size_t elements,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!values || !elements) {
        return invalid("native BF16 SiLU arguments are invalid");
    }
    native_silu_bf16_kernel<<<(elements + 255) / 256, 256, 0,
                              reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(values), elements);
    return launch_status();
}

modalities::Status NativeKernelDriver::gelu_bf16(
    void* values, std::size_t elements, std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!values || !elements || (elements & 1) ||
        elements > static_cast<std::size_t>(INT_MAX)) {
        return invalid("native BF16 GELU arguments are invalid");
    }
    ::gelu_inplace(static_cast<__nv_bfloat16*>(values),
                   static_cast<int>(elements),
                   reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeKernelDriver::gate_gelu_bf16(
    const void* gate, const void* up, void* output, std::size_t elements,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!gate || !up || !output || !elements ||
        elements > static_cast<std::size_t>(INT_MAX)) {
        return invalid("native BF16 gated GELU arguments are invalid");
    }
    ::gate_silu_mul(static_cast<const __nv_bfloat16*>(gate),
                    static_cast<const __nv_bfloat16*>(up),
                    static_cast<__nv_bfloat16*>(output),
                    static_cast<int>(elements),
                    reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeKernelDriver::residual_add_bf16(
    void* residual, const void* values, std::size_t elements,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!residual || !values || !elements || (elements & 1) ||
        elements > static_cast<std::size_t>(INT_MAX)) {
        return invalid("native BF16 residual arguments are invalid");
    }
    ::residual_add(static_cast<__nv_bfloat16*>(residual),
                   static_cast<const __nv_bfloat16*>(values),
                   static_cast<int>(elements),
                   reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeKernelDriver::bias_residual_bf16(
    void* residual, const void* values, const void* bias, int rows,
    int columns, std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!residual || !values || !bias || rows <= 0 || columns <= 0 ||
        (columns & 1)) {
        return invalid("native BF16 bias residual arguments are invalid");
    }
    ::bias_residual(static_cast<__nv_bfloat16*>(residual),
                    static_cast<const __nv_bfloat16*>(values),
                    static_cast<const __nv_bfloat16*>(bias), rows, columns,
                    reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeKernelDriver::gate_mul_residual_bf16(
    void* residual, const void* values, const void* gate,
    std::size_t elements, std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!residual || !values || !gate || !elements || (elements & 1) ||
        elements > static_cast<std::size_t>(INT_MAX)) {
        return invalid("native BF16 gated residual arguments are invalid");
    }
    ::gate_mul_residual(static_cast<__nv_bfloat16*>(residual),
                        static_cast<const __nv_bfloat16*>(values),
                        static_cast<const __nv_bfloat16*>(gate),
                        static_cast<int>(elements),
                        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeKernelDriver::rms_norm_bf16(
    const void* values, const void* weight, void* output, int rows,
    int columns, float epsilon, std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!values || !weight || !output || rows <= 0 || columns <= 0 ||
        (columns & 1) || !(epsilon > 0.0f)) {
        return invalid("native BF16 RMSNorm arguments are invalid");
    }
    ::rms_norm(static_cast<const __nv_bfloat16*>(values),
               static_cast<const __nv_bfloat16*>(weight),
               static_cast<__nv_bfloat16*>(output), rows, columns, epsilon,
               reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeKernelDriver::layer_norm_bf16(
    const void* values, const void* weight, const void* bias, void* output,
    int rows, int columns, float epsilon, std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!values || !weight || !bias || !output || rows <= 0 || columns <= 0 ||
        (columns & 1) || !(epsilon > 0.0f)) {
        return invalid("native BF16 LayerNorm arguments are invalid");
    }
    ::layer_norm(static_cast<const __nv_bfloat16*>(values),
                 static_cast<const __nv_bfloat16*>(weight),
                 static_cast<const __nv_bfloat16*>(bias),
                 static_cast<__nv_bfloat16*>(output), rows, columns, epsilon,
                 reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeKernelDriver::ada_rms_norm_style_bf16(
    const void* values, const void* weight, const void* style, void* output,
    void* gate_output, int rows, int columns, float epsilon,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!values || !weight || !style || !output || !gate_output || rows <= 0 ||
        columns <= 0 || (columns & 1) || !(epsilon > 0.0f)) {
        return invalid("native BF16 AdaRMSNorm arguments are invalid");
    }
    ::ada_rms_norm_style(
        static_cast<const __nv_bfloat16*>(values),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<const __nv_bfloat16*>(style),
        static_cast<__nv_bfloat16*>(output),
        static_cast<__nv_bfloat16*>(gate_output), rows, columns, epsilon,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeKernelDriver::qkv_split_bf16(
    const void* qkv, void* query, void* key, void* value, int rows,
    int query_columns, int key_columns, int value_columns,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!qkv || !query || !key || !value || rows <= 0 || query_columns <= 0 ||
        key_columns <= 0 || value_columns <= 0) {
        return invalid("native BF16 QKV split arguments are invalid");
    }
    ::qkv_split(static_cast<const __nv_bfloat16*>(qkv),
                static_cast<__nv_bfloat16*>(query),
                static_cast<__nv_bfloat16*>(key),
                static_cast<__nv_bfloat16*>(value), rows, query_columns,
                key_columns, value_columns,
                reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeKernelDriver::qkv_split_rope_bf16(
    const void* qkv, const void* rope, void* query, void* key, void* value,
    int rows, int query_columns, int key_columns, int value_columns,
    int head_dimension, std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!qkv || !rope || !query || !key || !value || rows <= 0 ||
        query_columns <= 0 || key_columns <= 0 || value_columns <= 0 ||
        head_dimension <= 0 || (head_dimension & 1) ||
        query_columns % head_dimension || key_columns % head_dimension) {
        return invalid("native BF16 QKV RoPE arguments are invalid");
    }
    ::qkv_split_rope(
        static_cast<const __nv_bfloat16*>(qkv),
        static_cast<const __nv_bfloat16*>(rope),
        static_cast<__nv_bfloat16*>(query),
        static_cast<__nv_bfloat16*>(key),
        static_cast<__nv_bfloat16*>(value), rows, query_columns, key_columns,
        value_columns, head_dimension,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeKernelDriver::qkv_split_rope_devpos_bf16(
    const void* qkv, const void* rope, void* query, void* key, void* value,
    const void* device_position, int rows, int query_columns, int key_columns,
    int value_columns, int head_dimension, std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!qkv || !rope || !query || !key || !value || !device_position ||
        rows <= 0 || query_columns <= 0 || key_columns <= 0 ||
        value_columns <= 0 || head_dimension <= 0 || (head_dimension & 1) ||
        query_columns % head_dimension || key_columns % head_dimension) {
        return invalid("native BF16 QKV devpos arguments are invalid");
    }
    ::qkv_split_rope_devpos(
        static_cast<const __nv_bfloat16*>(qkv),
        static_cast<const __nv_bfloat16*>(rope),
        static_cast<__nv_bfloat16*>(query),
        static_cast<__nv_bfloat16*>(key),
        static_cast<__nv_bfloat16*>(value),
        static_cast<const int*>(device_position), rows, query_columns,
        key_columns, value_columns, head_dimension,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeKernelDriver::patch_im2col_16bit(
    const void* images, void* patches, int num_views,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!images || !patches || num_views <= 0) {
        return invalid("native patch im2col arguments are invalid");
    }
    ::patch_im2col(static_cast<const __half*>(images),
                   static_cast<__half*>(patches), num_views,
                   reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status NativeKernelDriver::avg_pool_vision_bf16(
    const void* values, void* output, int num_views, int height, int width,
    int columns, int pool_factor, std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!values || !output || num_views <= 0 || height <= 0 || width <= 0 ||
        columns <= 0 || pool_factor <= 0 || height % pool_factor ||
        width % pool_factor) {
        return invalid("native vision pooling arguments are invalid");
    }
    ::avg_pool_vision_tokens(
        static_cast<const __nv_bfloat16*>(values),
        static_cast<__nv_bfloat16*>(output), num_views, height, width, columns,
        pool_factor, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#include "flashrt/cpp/models/pi05/native_rtx_linear.h"

#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

void* dptr(const NativeWorkspaceBuffer* buffer) {
    return buffer ? frt_buffer_dptr(buffer->buffer) : nullptr;
}

void* dptr(const NativeDeviceWeight* weight) {
    return weight ? frt_buffer_dptr(weight->buffer) : nullptr;
}

}  // namespace

const NativeDeviceWeight* NativeRtxLinear::find_weight(
    const NativeDeviceWeightStore& weights,
    const std::string& name) const {
    if (!fp8()) return weights.find(name);
    const std::string packed_name =
        name == "encoder_multi_modal_projector_w"
            ? "vision_projector_w"
            : name;
    return weights.find("fp8." + packed_name);
}

bool NativeRtxLinear::weight_shape_is(
    const NativeDeviceWeightStore& weights,
    const std::string& name,
    std::initializer_list<std::uint64_t> shape) const {
    const NativeDeviceWeight* weight = find_weight(weights, name);
    return weight &&
           weight->dtype == (fp8() ? NativeWeightDType::kFp8E4M3
                                   : NativeWeightDType::kBf16) &&
           weight->shape == std::vector<std::uint64_t>(shape);
}

const NativeWorkspaceBuffer* NativeRtxLinear::scale_buffer(
    const NativeWorkspace& workspace,
    NativeRtxScaleDomain domain) const {
    switch (domain) {
        case NativeRtxScaleDomain::kVision:
            return workspace.find("rtx_fp8_vision_scales");
        case NativeRtxScaleDomain::kEncoder:
            return workspace.find("rtx_fp8_encoder_scales");
        case NativeRtxScaleDomain::kDecoder:
            return workspace.find("rtx_fp8_decoder_scales");
    }
    return nullptr;
}

const float* NativeRtxLinear::scale(
    const NativeWorkspace& workspace,
    NativeRtxScaleSite site) const {
    if (!fp8() || site.index < 0) return nullptr;
    const NativeWorkspaceBuffer* scales = scale_buffer(workspace, site.domain);
    if (!scales || scales->dtype != modalities::DType::kFloat32 ||
        scales->shape.size() != 1 ||
        static_cast<std::uint64_t>(site.index) >= scales->shape[0]) {
        return nullptr;
    }
    return static_cast<const float*>(dptr(scales)) + site.index;
}

modalities::Status NativeRtxLinear::run(
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    const std::string& weight_name,
    NativeRtxScaleSite site,
    const void* input,
    void* output,
    int m,
    int n,
    int k,
    std::uintptr_t stream) const {
    if (!driver_ || !workspace || weight_name.empty() || !input || !output ||
        m <= 0 || n <= 0 || k <= 0) {
        return invalid("native RTX linear arguments are invalid");
    }
    const NativeDeviceWeight* weight = find_weight(weights, weight_name);
    if (!fp8()) {
        if (!weight || weight->dtype != NativeWeightDType::kBf16) {
            return invalid("native BF16 linear weight is invalid");
        }
        return driver_->bf16_nn(
            const_cast<void*>(input), dptr(weight), output, m, n, k, stream);
    }
    const std::string packed_name =
        weight_name == "encoder_multi_modal_projector_w"
            ? "vision_projector_w"
            : weight_name;
    const NativeDeviceWeight* weight_scale =
        weights.find("fp8." + packed_name + ".scale");
    const NativeWorkspaceBuffer* scratch =
        workspace->find("rtx_fp8_scratch");
    const NativeWorkspaceBuffer* scales = scale_buffer(*workspace, site.domain);
    if (!weight || weight->dtype != NativeWeightDType::kFp8E4M3 ||
        !weight_scale || weight_scale->dtype != NativeWeightDType::kFloat32 ||
        weight_scale->shape != std::vector<std::uint64_t>({1}) ||
        !scratch || scratch->dtype != modalities::DType::kUInt8 ||
        !scales || scales->dtype != modalities::DType::kFloat32 ||
        scales->shape.size() != 1 || site.index < 0 ||
        static_cast<std::uint64_t>(site.index) >= scales->shape[0] ||
        static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(k) >
            frt_buffer_bytes(scratch->buffer)) {
        return invalid("native FP8 linear storage is invalid");
    }
    auto* scale = static_cast<float*>(dptr(scales)) + site.index;
    modalities::Status st = dynamic_fp8()
        ? driver_->quantize_fp8_dynamic_bf16(
              input, dptr(scratch), scale,
              static_cast<std::size_t>(m) * k, stream)
        : driver_->quantize_fp8_static_bf16(
              input, dptr(scratch), scale,
              static_cast<std::size_t>(m) * k, stream);
    if (!st.ok_status()) return st;
    return driver_->fp8_nn_bf16(
        dptr(scratch), dptr(weight), output, m, n, k, scale,
        static_cast<const float*>(dptr(weight_scale)), stream);
}

modalities::Status NativeRtxLinear::autotune(
    const NativeDeviceWeightStore& weights,
    NativeWorkspace* workspace,
    const std::string& weight_name,
    NativeRtxScaleSite site,
    void* output,
    int m,
    int n,
    int k) const {
    if (!fp8() || !driver_ || !workspace || !output || m <= 0 || n <= 0 ||
        k <= 0) {
        return invalid("native FP8 autotune arguments are invalid");
    }
    const NativeDeviceWeight* weight = find_weight(weights, weight_name);
    const std::string packed_name =
        weight_name == "encoder_multi_modal_projector_w"
            ? "vision_projector_w"
            : weight_name;
    const NativeDeviceWeight* weight_scale =
        weights.find("fp8." + packed_name + ".scale");
    const NativeWorkspaceBuffer* scratch =
        workspace->find("rtx_fp8_scratch");
    const NativeWorkspaceBuffer* scales = scale_buffer(*workspace, site.domain);
    if (!weight || weight->dtype != NativeWeightDType::kFp8E4M3 ||
        weight->shape != std::vector<std::uint64_t>(
                             {static_cast<std::uint64_t>(k),
                              static_cast<std::uint64_t>(n)}) ||
        !weight_scale || weight_scale->dtype != NativeWeightDType::kFloat32 ||
        weight_scale->shape != std::vector<std::uint64_t>({1}) || !scratch ||
        !scales || scales->shape.size() != 1 || site.index < 0 ||
        static_cast<std::uint64_t>(site.index) >= scales->shape[0] ||
        static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(k) >
            frt_buffer_bytes(scratch->buffer)) {
        return invalid("native FP8 autotune storage is invalid");
    }
    const auto* scale = static_cast<float*>(dptr(scales)) + site.index;
    return driver_->autotune_fp8_nn_bf16(
        dptr(scratch), dptr(weight), output, m, n, k, scale,
        static_cast<const float*>(dptr(weight_scale)));
}

modalities::Status NativeRtxLinear::run_prequantized(
    const NativeDeviceWeightStore& weights,
    const std::string& weight_name,
    NativeRtxScaleSite site,
    const NativeWorkspace& workspace,
    const void* input,
    void* output,
    int m,
    int n,
    int k,
    std::uintptr_t stream) const {
    if (!fp8() || !driver_ || weight_name.empty() || !input || !output ||
        m <= 0 || n <= 0 || k <= 0) {
        return invalid("native prequantized FP8 linear arguments are invalid");
    }
    const NativeDeviceWeight* weight = find_weight(weights, weight_name);
    const std::string packed_name =
        weight_name == "encoder_multi_modal_projector_w"
            ? "vision_projector_w"
            : weight_name;
    const NativeDeviceWeight* weight_scale =
        weights.find("fp8." + packed_name + ".scale");
    const float* activation_scale = scale(workspace, site);
    if (!weight || weight->dtype != NativeWeightDType::kFp8E4M3 ||
        weight->shape != std::vector<std::uint64_t>(
                             {static_cast<std::uint64_t>(k),
                              static_cast<std::uint64_t>(n)}) ||
        !weight_scale || weight_scale->dtype != NativeWeightDType::kFloat32 ||
        weight_scale->shape != std::vector<std::uint64_t>({1}) ||
        !activation_scale) {
        return invalid("native prequantized FP8 linear storage is invalid");
    }
    return driver_->fp8_nn_bf16(
        const_cast<void*>(input), dptr(weight), output, m, n, k,
        activation_scale, static_cast<const float*>(dptr(weight_scale)),
        stream);
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

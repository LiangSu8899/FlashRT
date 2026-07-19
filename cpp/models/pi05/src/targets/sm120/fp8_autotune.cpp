#include "flashrt/cpp/models/pi05/targets/sm120/fp8_autotune.h"

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

void* data(const Pi05ResolvedBuffer& buffer) {
    return buffer.buffer ? frt_buffer_dptr(buffer.buffer) : nullptr;
}

bool bf16_span_fits(const void* pointer,
                    std::size_t bytes,
                    int rows,
                    int columns) {
    return pointer && rows > 0 && columns > 0 &&
           static_cast<std::size_t>(rows) <=
               std::numeric_limits<std::size_t>::max() /
                   static_cast<std::size_t>(columns) &&
           static_cast<std::size_t>(rows) *
                   static_cast<std::size_t>(columns) <=
               bytes / sizeof(std::uint16_t);
}

class Sm120Fp8AutotuneSink final : public Pi05LinearWeightGroupSink {
public:
    Sm120Fp8AutotuneSink(const Pi05ResolvedShape& shape,
                         Pi05ResolvedResources* resources,
                         const Sm120Bf16ScratchBacking& scratch,
                         Sm120Fp8Linear* linear)
        : shape_(shape),
          resources_(resources),
          scratch_(scratch),
          linear_(linear) {}

    modalities::Status record(
        const Pi05LinearWeightGroup& group) override {
        const Pi05ResolvedWeight* weight =
            group.fused && group.fused->device_data ? group.fused
                                                    : group.first;
        if (!resources_ || !linear_ || !weight || !weight->device_data ||
            weight->shape.rank != 2 ||
            weight->shape.dims[0] >
                static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
            weight->shape.dims[1] >
                static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            return invalid("SM120 FP8 autotune weight is invalid");
        }

        const void* input = nullptr;
        std::size_t input_bytes = 0;
        void* output = nullptr;
        std::size_t output_bytes = 0;
        int rows = 0;
        switch (group.key.domain) {
            case Pi05LinearDomain::kVision: {
                rows = shape_.vision_sequence;
                input = scratch_.vision().normalized.device_data();
                input_bytes = scratch_.vision().normalized.bytes();
                output = scratch_.vision().normalized.device_data();
                output_bytes = scratch_.vision().normalized.bytes();
                switch (group.key.role) {
                    case Pi05LinearRole::kAttentionQkv:
                        output = scratch_.vision().qkv.device_data();
                        output_bytes = scratch_.vision().qkv.bytes();
                        break;
                    case Pi05LinearRole::kMlpUp:
                        output = scratch_.vision().hidden.device_data();
                        output_bytes = scratch_.vision().hidden.bytes();
                        break;
                    case Pi05LinearRole::kMlpDown:
                        input = scratch_.vision().hidden.device_data();
                        input_bytes = scratch_.vision().hidden.bytes();
                        break;
                    case Pi05LinearRole::kProjector:
                        rows = shape_.encoder_vision_sequence;
                        output = data(resources_->buffers.encoder_state);
                        output_bytes = static_cast<std::size_t>(
                            resources_->buffers.encoder_state.physical_bytes);
                        break;
                    case Pi05LinearRole::kAttentionOutput: break;
                    case Pi05LinearRole::kMlpGateUpGroup:
                        return invalid("SM120 vision autotune role is invalid");
                }
                break;
            }
            case Pi05LinearDomain::kEncoder: {
                rows = shape_.encoder_sequence;
                input = scratch_.encoder().normalized.device_data();
                input_bytes = scratch_.encoder().normalized.bytes();
                output = scratch_.encoder().normalized.device_data();
                output_bytes = scratch_.encoder().normalized.bytes();
                switch (group.key.role) {
                    case Pi05LinearRole::kAttentionQkv:
                        output = scratch_.encoder().qkv.device_data();
                        output_bytes = scratch_.encoder().qkv.bytes();
                        break;
                    case Pi05LinearRole::kMlpGateUpGroup:
                        output = scratch_.encoder().gate.device_data();
                        output_bytes = scratch_.encoder().gate.bytes();
                        break;
                    case Pi05LinearRole::kMlpDown:
                        input = scratch_.encoder().hidden.device_data();
                        input_bytes = scratch_.encoder().hidden.bytes();
                        break;
                    case Pi05LinearRole::kAttentionOutput: break;
                    case Pi05LinearRole::kMlpUp:
                    case Pi05LinearRole::kProjector:
                        return invalid("SM120 encoder autotune role is invalid");
                }
                break;
            }
            case Pi05LinearDomain::kDecoder: {
                rows = shape_.chunk;
                input = scratch_.decoder().normalized.device_data();
                input_bytes = scratch_.decoder().normalized.bytes();
                output = scratch_.decoder().normalized.device_data();
                output_bytes = scratch_.decoder().normalized.bytes();
                switch (group.key.role) {
                    case Pi05LinearRole::kAttentionQkv:
                        output = scratch_.decoder().qkv.device_data();
                        output_bytes = scratch_.decoder().qkv.bytes();
                        break;
                    case Pi05LinearRole::kAttentionOutput:
                        input = scratch_.decoder().qkv.device_data();
                        input_bytes = scratch_.decoder().qkv.bytes();
                        break;
                    case Pi05LinearRole::kMlpGateUpGroup:
                        output =
                            scratch_.decoder().gate_projection.device_data();
                        output_bytes =
                            scratch_.decoder().gate_projection.bytes();
                        break;
                    case Pi05LinearRole::kMlpDown:
                        input = scratch_.decoder().hidden.device_data();
                        input_bytes = scratch_.decoder().hidden.bytes();
                        break;
                    case Pi05LinearRole::kMlpUp:
                    case Pi05LinearRole::kProjector:
                        return invalid("SM120 decoder autotune role is invalid");
                }
                break;
            }
        }

        const int input_width = static_cast<int>(weight->shape.dims[0]);
        const int output_width = static_cast<int>(weight->shape.dims[1]);
        if (!bf16_span_fits(input, input_bytes, rows, input_width) ||
            !bf16_span_fits(output, output_bytes, rows, output_width)) {
            return invalid("SM120 FP8 autotune scratch is too small");
        }
        Pi05LinearActivationSite site;
        const int step = group.key.domain == Pi05LinearDomain::kDecoder
                             ? 0
                             : -1;
        modalities::Status status = resolve_pi05_linear_activation_site(
            group.key, step, shape_, &site);
        return status.ok_status()
                   ? linear_->autotune(*weight, site, input, output, rows,
                                       input_width, output_width)
                   : status;
    }

private:
    const Pi05ResolvedShape& shape_;
    Pi05ResolvedResources* resources_ = nullptr;
    const Sm120Bf16ScratchBacking& scratch_;
    Sm120Fp8Linear* linear_ = nullptr;
};

}  // namespace

modalities::Status autotune_sm120_fp8(
    const Pi05ResolvedShape& shape,
    Pi05ResolvedResources* resources,
    const Sm120Bf16ScratchBacking& scratch,
    Sm120Fp8Linear* linear) {
    if (!resources || !linear || !scratch.allocated() ||
        !scratch.fused_gate_up() ||
        !pi05_resolved_shape_equal(shape, scratch.shape()) ||
        linear->autotune_frozen()) {
        return invalid("SM120 FP8 autotune state is invalid");
    }
    modalities::Status status =
        validate_pi05_resolved_resources(*resources, shape);
    if (!status.ok_status()) return status;
    Sm120Fp8AutotuneSink sink(shape, resources, scratch, linear);
    status = visit_pi05_linear_weight_groups(&resources->weights, &sink);
    return status.ok_status() ? linear->freeze_autotune() : status;
}

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

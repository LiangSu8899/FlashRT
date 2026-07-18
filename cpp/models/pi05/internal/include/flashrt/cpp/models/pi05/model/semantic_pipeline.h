#ifndef FLASHRT_CPP_MODELS_PI05_MODEL_SEMANTIC_PIPELINE_H
#define FLASHRT_CPP_MODELS_PI05_MODEL_SEMANTIC_PIPELINE_H

#include "flashrt/cpp/models/pi05/model/dims.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace flashrt {
namespace models {
namespace pi05 {

enum class Pi05ValueId : std::uint8_t {
    kImages = 0,
    kPromptEmbedding,
    kVisionState,
    kEncoderState,
    kKeyCache,
    kValueCache,
    kNoise,
    kDecoderState,
    kActionDelta,
    kCount,
};

enum class Pi05ScalarKind : std::uint8_t {
    kActivation = 0,
    kActionUpdate,
};

enum class Pi05OperationId : std::uint8_t {
    kComposePrompt = 0,
    kVisionEmbed,
    kVisionAttention,
    kVisionMlp,
    kVisionProject,
    kEncoderAttention,
    kEncoderMlp,
    kEncoderCacheFinalize,
    kDiffusionInputProject,
    kDecoderAttention,
    kDecoderMlp,
    kActionProject,
    kDiffusionUpdate,
    kCount,
};

enum class Pi05IndexDomain : std::uint8_t {
    kNone = 0,
    kVisionLayer,
    kEncoderLayer,
    kEncoderFinalLayer,
    kDiffusionStep,
    kDecoderLayer,
};

constexpr std::size_t kPi05MaxOperationValues = 3;
constexpr std::uint8_t kPi05NoAlias = 0xff;

struct Pi05TensorSpec final {
    Pi05ScalarKind scalar = Pi05ScalarKind::kActivation;
    std::uint8_t rank = 0;
    std::array<std::uint64_t, 4> dimensions{};
};

struct Pi05OperationContract final {
    Pi05OperationId id = Pi05OperationId::kCount;
    const char* name = nullptr;
    Pi05IndexDomain index_domain = Pi05IndexDomain::kNone;
    std::array<Pi05ValueId, kPi05MaxOperationValues> inputs{};
    std::uint8_t input_count = 0;
    std::array<Pi05ValueId, kPi05MaxOperationValues> outputs{};
    std::uint8_t output_count = 0;
    std::array<std::uint8_t, kPi05MaxOperationValues> output_alias_input{};
};

struct Pi05OperationCall final {
    Pi05OperationId id = Pi05OperationId::kCount;
    int layer = -1;
    int step = -1;
    std::array<std::uint64_t, kPi05MaxOperationValues> input_generation{};
    std::array<std::uint64_t, kPi05MaxOperationValues> output_generation{};
};

const char* pi05_value_name(Pi05ValueId id);
const char* pi05_operation_name(Pi05OperationId id);
const Pi05OperationContract* pi05_operation_contract(Pi05OperationId id);

modalities::Status pi05_value_spec(Pi05ValueId id,
                                   const Pi05ResolvedShape& shape,
                                   Pi05TensorSpec* out);

modalities::Status validate_pi05_operation_call(
    const Pi05OperationCall& call,
    const Pi05ResolvedShape& shape);

using Pi05Stream = std::uintptr_t;

class Pi05OperationSink {
public:
    virtual ~Pi05OperationSink() = default;
    virtual modalities::Status record(const Pi05OperationCall& call,
                                      const Pi05ResolvedShape& shape,
                                      Pi05Stream stream) = 0;
};

class Pi05SemanticPipeline final {
public:
    explicit Pi05SemanticPipeline(Pi05ResolvedShape shape)
        : shape_(shape) {}

    const Pi05ResolvedShape& shape() const { return shape_; }

    modalities::Status record_context(Pi05OperationSink& sink,
                                      Pi05Stream stream = 0) const;
    modalities::Status record_decode(Pi05OperationSink& sink,
                                     Pi05Stream stream = 0) const;
    modalities::Status record_full(Pi05OperationSink& sink,
                                   Pi05Stream stream = 0) const;

private:
    Pi05ResolvedShape shape_;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_MODEL_SEMANTIC_PIPELINE_H

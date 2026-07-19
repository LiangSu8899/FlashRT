#include "flashrt/cpp/models/pi05/model/native_session.h"

#include "flashrt/cpp/models/pi05/model/frontend_ops.h"

#include "pi05_resolved_fixture.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace modalities = flashrt::modalities;
namespace pi05 = flashrt::models::pi05;
namespace fixture = flashrt::tests::pi05_fixture;

namespace {

[[noreturn]] void fail(const char* expression, int line) {
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    std::abort();
}

#define CHECK(expression)                              \
    do {                                               \
        if (!(expression)) fail(#expression, __LINE__); \
    } while (false)

enum class FailPoint {
    kNone = 0,
    kInitialize,
    kResolve,
    kInvalidResources,
    kPrepareExecution,
    kCompletePrepare,
    kFinalize,
    kForwardExecution,
    kCaptureInputs,
    kReset,
    kSetPrompt,
};

enum class LifecycleStep {
    kInitialize = 0,
    kResolve,
    kPrepareExecution,
    kCompletePrepare,
    kFinalize,
    kForwardExecution,
    kCaptureInputs,
    kReset,
    kSetPrompt,
};

struct Probe final {
    std::vector<LifecycleStep> lifecycle;
    std::size_t record_calls = 0;
    std::size_t default_stream_calls = 0;
    std::size_t capture_stream_calls = 0;
    std::size_t fail_record_at = std::numeric_limits<std::size_t>::max();
    bool fail_frontend_primitive = false;
    std::size_t layer0_pending_updates = 0;
    int prompt_length = -1;
    bool target_destroyed = false;
    bool context_alive_at_target_destroy = false;
};

constexpr std::size_t kDecodeTargetOperations = 390;
constexpr std::size_t kContextTargetOperations = 0;
constexpr std::size_t kInferTargetOperations = kDecodeTargetOperations;
constexpr std::size_t kCapturedTargetOperations =
    kInferTargetOperations + kDecodeTargetOperations +
    kContextTargetOperations;

modalities::Status injected(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

modalities::Status noop_linear(
    void*, const pi05::Pi05ResolvedWeight&, const void*, void*, int, int, int,
    pi05::Pi05Stream) {
    return modalities::Status::ok();
}

modalities::Status noop_bias(
    void*, void*, const pi05::Pi05ResolvedWeight&, int, int,
    pi05::Pi05Stream) {
    return modalities::Status::ok();
}

modalities::Status noop_projected_linear(
    void*, const pi05::Pi05ResolvedWeight&, pi05::Pi05LinearWeightKey, int,
    const void*, void*, int, int, int, bool, pi05::Pi05Stream) {
    return modalities::Status::ok();
}

modalities::Status noop_unary(void*, void*, std::size_t,
                              pi05::Pi05Stream) {
    return modalities::Status::ok();
}

modalities::Status noop_copy(void* state, void*, const void*, std::size_t,
                             pi05::Pi05Stream) {
    Probe* probe = static_cast<Probe*>(state);
    if (probe && probe->fail_frontend_primitive) {
        return injected("injected frontend primitive failure");
    }
    return modalities::Status::ok();
}

modalities::Status noop_patchify(void*, const void*, void*, int,
                                 pi05::Pi05Stream) {
    return modalities::Status::ok();
}

modalities::Status noop_bias_residual(
    void*, void*, const void*, const pi05::Pi05ResolvedWeight&, int, int,
    pi05::Pi05Stream) {
    return modalities::Status::ok();
}

modalities::Status noop_layer_norm(
    void*, const void*, const pi05::Pi05ResolvedWeight&,
    const pi05::Pi05ResolvedWeight&, void*, int, int, float,
    pi05::Pi05Stream) {
    return modalities::Status::ok();
}

modalities::Status noop_qkv_split(void*, const void*, void*, void*, void*, int,
                                  int, int, int, pi05::Pi05Stream) {
    return modalities::Status::ok();
}

modalities::Status noop_normalize(
    void* state, void*, const void* update, const pi05::Pi05ResolvedBuffer&,
    void* output, pi05::Pi05LinearWeightKey key, int, int, int, float,
    const void** linear_input, bool* prequantized, pi05::Pi05Stream) {
    Probe* probe = static_cast<Probe*>(state);
    if (probe && update && key.domain == pi05::Pi05LinearDomain::kEncoder &&
        key.role == pi05::Pi05LinearRole::kAttentionQkv && key.layer == 0) {
        ++probe->layer0_pending_updates;
    }
    if (linear_input) *linear_input = output;
    if (prequantized) *prequantized = false;
    return modalities::Status::ok();
}

modalities::Status noop_qkv_rope(
    void*, const void*, const pi05::Pi05ResolvedBuffer&, void*, void*, void*,
    int, int, int, int, int, pi05::Pi05Stream) {
    return modalities::Status::ok();
}

modalities::Status noop_attention(void*, pi05::Pi05LinearDomain, int,
                                  const void*, const void*, const void*, void*,
                                  int, int, int, int, int, int,
                                  pi05::Pi05Stream) {
    return modalities::Status::ok();
}

modalities::Status noop_pool(void*, const void*, void*, int, int, int, int,
                             int, pi05::Pi05Stream) {
    return modalities::Status::ok();
}

modalities::Status noop_gate_up(
    void*, const pi05::Pi05FeedForwardWeights&, pi05::Pi05LinearWeightKey,
    int, const void*, bool, void*, void*, int, int, int, bool* merged,
    pi05::Pi05Stream) {
    if (merged) *merged = false;
    return modalities::Status::ok();
}

modalities::Status noop_gated_activation(
    void*, const void*, const void*, bool, void* output, int, int,
    pi05::Pi05LinearWeightKey, int, const void** linear_input,
    bool* prequantized, pi05::Pi05Stream) {
    if (linear_input) *linear_input = output;
    if (prequantized) *prequantized = false;
    return modalities::Status::ok();
}

pi05::Pi05PrimitiveSet fake_primitives(Probe* probe) {
    pi05::Pi05PrimitiveSet result;
    result.state = probe;
    result.linear = noop_linear;
    result.projected_linear = noop_projected_linear;
    result.add_bias = noop_bias;
    result.silu = noop_unary;
    result.copy = noop_copy;
    result.patchify = noop_patchify;
    result.bias_residual = noop_bias_residual;
    result.layer_norm = noop_layer_norm;
    result.qkv_split = noop_qkv_split;
    result.normalize_for_linear = noop_normalize;
    result.qkv_rope = noop_qkv_rope;
    result.attention = noop_attention;
    result.gate_up = noop_gate_up;
    result.gated_activation = noop_gated_activation;
    result.gelu = noop_unary;
    result.vision_pool = noop_pool;
    return result;
}

class FakeTarget final : public pi05::Pi05TargetBundle {
public:
    FakeTarget(frt_ctx context,
               bool warmup,
               FailPoint fail_point,
               Probe* probe)
        : Pi05TargetBundle(context, warmup),
          fail_point_(fail_point),
          probe_(probe) {
        CHECK(probe_ != nullptr);
        frontend_ops_.profile.activation_dtype =
            modalities::DType::kBFloat16;
        frontend_ops_.bf16 = fake_primitives(probe_);
    }

    ~FakeTarget() override {
        probe_->target_destroyed = true;
        if (anchor_) {
            probe_->context_alive_at_target_destroy =
                frt_buffer_dptr(anchor_) != nullptr &&
                frt_buffer_bytes(anchor_) >= prepare_storage_bytes_;
        } else {
            probe_->context_alive_at_target_destroy =
                frt_ctx_stream(context(), 0) >= 0;
        }
    }

    modalities::Status initialize_resources() override {
        probe_->lifecycle.push_back(LifecycleStep::kInitialize);
        anchor_ = frt_buffer_alloc(context(), "pi05_session_anchor",
                                   prepare_storage_bytes_);
        if (!anchor_) return injected("fake target anchor allocation failed");
        if (fail_point_ == FailPoint::kInitialize) {
            return injected("injected resource initialization failure");
        }
        resources_ = fixture::make_resources(
            context(), anchor_, modalities::DType::kBFloat16, false, false);
        return modalities::Status::ok();
    }

    modalities::Status resolve_resources(
        pi05::Pi05ResolvedResources* out) override {
        probe_->lifecycle.push_back(LifecycleStep::kResolve);
        if (!out || fail_point_ == FailPoint::kResolve) {
            return injected("injected resource resolution failure");
        }
        *out = resources_;
        if (fail_point_ == FailPoint::kInvalidResources) {
            out->buffers.noise.buffer = nullptr;
        }
        return modalities::Status::ok();
    }

    modalities::Status make_prepare_execution(
        pi05::Pi05PrepareExecution* out) override {
        probe_->lifecycle.push_back(LifecycleStep::kPrepareExecution);
        if (!out || fail_point_ == FailPoint::kPrepareExecution) {
            return injected("injected prepare execution failure");
        }
        *out = {&resources_,
                &frontend_ops_,
                {frt_buffer_dptr(anchor_), frt_buffer_dptr(anchor_),
                 prepare_storage_bytes_}};
        return modalities::Status::ok();
    }

    modalities::Status complete_prepare() override {
        probe_->lifecycle.push_back(LifecycleStep::kCompletePrepare);
        return fail_point_ == FailPoint::kCompletePrepare
                   ? injected("injected prepare completion failure")
                   : modalities::Status::ok();
    }

    modalities::Status finalize_setup() override {
        probe_->lifecycle.push_back(LifecycleStep::kFinalize);
        return fail_point_ == FailPoint::kFinalize
                   ? injected("injected setup finalization failure")
                   : modalities::Status::ok();
    }

    modalities::Status make_forward_execution(
        pi05::Pi05ForwardExecution* out) override {
        probe_->lifecycle.push_back(LifecycleStep::kForwardExecution);
        if (!out || fail_point_ == FailPoint::kForwardExecution) {
            return injected("injected forward execution failure");
        }
        void* data = frt_buffer_dptr(anchor_);
        out->resources = &resources_;
        out->ops = &frontend_ops_;
        out->vision = {data, data, data, data, data, data,
                       data, data, data, data};
        out->encoder = {resources_.buffers.images, data, data, data,
                        data, data, data};
        out->fallback = this;
        return modalities::Status::ok();
    }

    modalities::Status initialize_capture_inputs() override {
        probe_->lifecycle.push_back(LifecycleStep::kCaptureInputs);
        if (fail_point_ == FailPoint::kCaptureInputs) {
            return injected("injected capture input failure");
        }
        return clear_anchor();
    }

    modalities::Status reset_after_warmup() override {
        probe_->lifecycle.push_back(LifecycleStep::kReset);
        if (fail_point_ == FailPoint::kReset) {
            return injected("injected warmup reset failure");
        }
        return clear_anchor();
    }

    modalities::Status set_prompt_length(int prompt_tokens) override {
        probe_->lifecycle.push_back(LifecycleStep::kSetPrompt);
        if (fail_point_ == FailPoint::kSetPrompt) {
            return injected("injected prompt update failure");
        }
        probe_->prompt_length = prompt_tokens;
        return modalities::Status::ok();
    }

    modalities::Status record(
        const pi05::Pi05OperationCall& call,
        const pi05::Pi05ResolvedShape& shape,
        pi05::Pi05Stream stream) override {
        modalities::Status status =
            pi05::validate_pi05_operation_call(call, shape);
        if (!status.ok_status()) return status;
        const std::size_t index = probe_->record_calls++;
        if (stream) {
            ++probe_->capture_stream_calls;
        } else {
            ++probe_->default_stream_calls;
        }
        if (index == probe_->fail_record_at) {
            return injected("injected operation record failure");
        }
        const cudaError_t result = cudaMemsetAsync(
            frt_buffer_dptr(anchor_), static_cast<int>(call.id) + 1, 1,
            reinterpret_cast<cudaStream_t>(stream));
        return result == cudaSuccess
                   ? modalities::Status::ok()
                   : injected(cudaGetErrorString(result));
    }

private:
    modalities::Status clear_anchor() {
        if (!anchor_ ||
            cudaMemset(frt_buffer_dptr(anchor_), 0, 1) != cudaSuccess) {
            return injected("fake target anchor reset failed");
        }
        return modalities::Status::ok();
    }

    FailPoint fail_point_ = FailPoint::kNone;
    Probe* probe_ = nullptr;
    static constexpr std::size_t prepare_storage_bytes_ = 16 * 1024 * 1024;
    frt_buffer anchor_ = nullptr;
    pi05::Pi05ResolvedResources resources_;
    pi05::Pi05FrontendOps frontend_ops_;
};

std::unique_ptr<pi05::Pi05NativeSession> create_session(
    bool warmup,
    FailPoint fail_point,
    Probe* probe,
    modalities::Status* status,
    pi05::Pi05ResolvedShape shape = fixture::canonical_shape()) {
    frt_ctx context = frt_ctx_create();
    CHECK(context != nullptr);
    std::unique_ptr<pi05::Pi05TargetBundle> target(
        new FakeTarget(context, warmup, fail_point, probe));
    return pi05::Pi05NativeSession::create(
        context, shape, std::move(target), status);
}

void check_graphs(const pi05::Pi05NativeSession& session) {
    for (const pi05::Pi05GraphId id : {
             pi05::Pi05GraphId::kInfer,
             pi05::Pi05GraphId::kDecodeOnly,
             pi05::Pi05GraphId::kContext}) {
        CHECK(session.graph(id) != nullptr);
        CHECK(frt_graph_variant_count(session.graph(id)) == 1);
    }
    CHECK(session.graph(pi05::Pi05GraphId::kCount) == nullptr);
    CHECK(session.stream_id() >= 0);
    CHECK(session.native_stream() != nullptr);
}

void test_success_without_warmup() {
    Probe probe;
    modalities::Status status;
    std::unique_ptr<pi05::Pi05NativeSession> session =
        create_session(false, FailPoint::kNone, &probe, &status);
    CHECK(session != nullptr);
    CHECK(status.ok_status());
    CHECK(probe.lifecycle == std::vector<LifecycleStep>({
        LifecycleStep::kInitialize,
        LifecycleStep::kResolve,
        LifecycleStep::kPrepareExecution,
        LifecycleStep::kCompletePrepare,
        LifecycleStep::kFinalize,
        LifecycleStep::kForwardExecution,
        LifecycleStep::kCaptureInputs}));
    CHECK(probe.record_calls == kCapturedTargetOperations);
    CHECK(probe.default_stream_calls == 0);
    CHECK(probe.capture_stream_calls == kCapturedTargetOperations);
    CHECK(probe.layer0_pending_updates == 0);
    check_graphs(*session);

    CHECK(session->set_prompt_length(43).ok_status());
    CHECK(probe.prompt_length == 43);
    const std::size_t prompt_events = probe.lifecycle.size();
    CHECK(!session->set_prompt_length(-1).ok_status());
    CHECK(!session->set_prompt_length(65).ok_status());
    CHECK(probe.lifecycle.size() == prompt_events);

    CHECK(session->replay(pi05::Pi05GraphId::kContext) == FRT_OK);
    CHECK(session->replay(pi05::Pi05GraphId::kDecodeOnly) == FRT_OK);
    CHECK(session->replay() == FRT_OK);
    CHECK(session->synchronize().ok_status());
    CHECK(session->replay(pi05::Pi05GraphId::kCount) == FRT_ERR_INVALID);

    session.reset();
    CHECK(probe.target_destroyed);
    CHECK(probe.context_alive_at_target_destroy);
}

void test_success_with_warmup() {
    Probe probe;
    modalities::Status status;
    std::unique_ptr<pi05::Pi05NativeSession> session =
        create_session(true, FailPoint::kNone, &probe, &status);
    CHECK(session != nullptr);
    CHECK(status.ok_status());
    CHECK(probe.lifecycle == std::vector<LifecycleStep>({
        LifecycleStep::kInitialize,
        LifecycleStep::kResolve,
        LifecycleStep::kPrepareExecution,
        LifecycleStep::kCompletePrepare,
        LifecycleStep::kFinalize,
        LifecycleStep::kForwardExecution,
        LifecycleStep::kCaptureInputs,
        LifecycleStep::kReset}));
    CHECK(probe.record_calls ==
          kInferTargetOperations + kCapturedTargetOperations);
    CHECK(probe.default_stream_calls == kInferTargetOperations);
    CHECK(probe.capture_stream_calls == kCapturedTargetOperations);
    CHECK(probe.layer0_pending_updates == 0);
    check_graphs(*session);
    session.reset();
    CHECK(probe.target_destroyed);
    CHECK(probe.context_alive_at_target_destroy);
}

void expect_create_failure(
    bool warmup,
    FailPoint fail_point,
    std::size_t fail_record_at = std::numeric_limits<std::size_t>::max()) {
    Probe probe;
    probe.fail_record_at = fail_record_at;
    modalities::Status status;
    std::unique_ptr<pi05::Pi05NativeSession> session =
        create_session(warmup, fail_point, &probe, &status);
    CHECK(session == nullptr);
    CHECK(!status.ok_status());
    CHECK(probe.target_destroyed);
    CHECK(probe.context_alive_at_target_destroy);
    if (fail_record_at != std::numeric_limits<std::size_t>::max()) {
        CHECK(probe.record_calls == fail_record_at + 1);
    }
    switch (fail_point) {
        case FailPoint::kInitialize:
            CHECK(probe.lifecycle.back() == LifecycleStep::kInitialize);
            break;
        case FailPoint::kResolve:
        case FailPoint::kInvalidResources:
            CHECK(probe.lifecycle.back() == LifecycleStep::kResolve);
            break;
        case FailPoint::kPrepareExecution:
            CHECK(probe.lifecycle.back() == LifecycleStep::kPrepareExecution);
            break;
        case FailPoint::kCompletePrepare:
            CHECK(probe.lifecycle.back() == LifecycleStep::kCompletePrepare);
            break;
        case FailPoint::kFinalize:
            CHECK(probe.lifecycle.back() == LifecycleStep::kFinalize);
            break;
        case FailPoint::kForwardExecution:
            CHECK(probe.lifecycle.back() == LifecycleStep::kForwardExecution);
            break;
        case FailPoint::kCaptureInputs:
            CHECK(probe.lifecycle.back() == LifecycleStep::kCaptureInputs);
            break;
        case FailPoint::kReset:
            CHECK(probe.lifecycle.back() == LifecycleStep::kReset);
            break;
        case FailPoint::kNone:
        case FailPoint::kSetPrompt:
            break;
    }
}

void test_failure_matrix() {
    expect_create_failure(false, FailPoint::kInitialize);
    expect_create_failure(false, FailPoint::kResolve);
    expect_create_failure(false, FailPoint::kInvalidResources);
    expect_create_failure(false, FailPoint::kPrepareExecution);
    expect_create_failure(false, FailPoint::kCompletePrepare);
    expect_create_failure(false, FailPoint::kFinalize);
    expect_create_failure(false, FailPoint::kForwardExecution);
    expect_create_failure(false, FailPoint::kCaptureInputs);
    expect_create_failure(true, FailPoint::kNone, 11);
    expect_create_failure(true, FailPoint::kReset);
    expect_create_failure(false, FailPoint::kNone, 11);
    expect_create_failure(
        false, FailPoint::kNone, kInferTargetOperations + 11);

    Probe primitive_probe;
    primitive_probe.fail_frontend_primitive = true;
    modalities::Status primitive_status;
    CHECK(create_session(false, FailPoint::kNone,
                         &primitive_probe, &primitive_status) == nullptr);
    CHECK(!primitive_status.ok_status());
    CHECK(primitive_probe.record_calls == 0);
    CHECK(primitive_probe.target_destroyed);

    pi05::Pi05ResolvedShape invalid_shape = fixture::canonical_shape();
    invalid_shape.chunk = 0;
    Probe probe;
    modalities::Status status;
    CHECK(create_session(false, FailPoint::kNone, &probe, &status,
                         invalid_shape) == nullptr);
    CHECK(!status.ok_status());
    CHECK(probe.lifecycle.empty());
    CHECK(probe.target_destroyed);
    CHECK(probe.context_alive_at_target_destroy);
}

void test_owner_validation() {
    modalities::Status status;
    frt_ctx context = frt_ctx_create();
    CHECK(context != nullptr);
    CHECK(pi05::Pi05NativeSession::create(
              context, fixture::canonical_shape(), nullptr, &status) ==
          nullptr);
    CHECK(!status.ok_status());

    frt_ctx target_context = frt_ctx_create();
    frt_ctx supplied_context = frt_ctx_create();
    CHECK(target_context != nullptr);
    CHECK(supplied_context != nullptr);
    Probe probe;
    std::unique_ptr<pi05::Pi05TargetBundle> target(
        new FakeTarget(target_context, false, FailPoint::kNone, &probe));
    CHECK(pi05::Pi05NativeSession::create(
              supplied_context, fixture::canonical_shape(),
              std::move(target), &status) == nullptr);
    CHECK(!status.ok_status());
    CHECK(probe.target_destroyed);
    CHECK(probe.context_alive_at_target_destroy);
    frt_ctx_destroy(target_context);
}

void test_prompt_failure_does_not_invalidate_session() {
    Probe probe;
    modalities::Status status;
    std::unique_ptr<pi05::Pi05NativeSession> session =
        create_session(false, FailPoint::kSetPrompt, &probe, &status);
    CHECK(session != nullptr);
    CHECK(!session->set_prompt_length(32).ok_status());
    CHECK(session->replay() == FRT_OK);
    CHECK(session->synchronize().ok_status());
}

}  // namespace

int main() {
    test_success_without_warmup();
    test_success_with_warmup();
    test_failure_matrix();
    test_owner_validation();
    test_prompt_failure_does_not_invalidate_session();
    std::cout << "PASS - PI0.5 native session lifecycle\n";
    return 0;
}

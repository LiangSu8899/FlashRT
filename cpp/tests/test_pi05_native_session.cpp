#include "flashrt/cpp/models/pi05/model/native_session.h"

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
    kFinalize,
    kCaptureInputs,
    kReset,
    kSetPrompt,
};

enum class LifecycleStep {
    kInitialize = 0,
    kResolve,
    kFinalize,
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
    int prompt_length = -1;
    bool target_destroyed = false;
    bool context_alive_at_target_destroy = false;
};

modalities::Status injected(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
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
    }

    ~FakeTarget() override {
        probe_->target_destroyed = true;
        if (anchor_) {
            probe_->context_alive_at_target_destroy =
                frt_buffer_dptr(anchor_) != nullptr &&
                frt_buffer_bytes(anchor_) == 1;
        } else {
            probe_->context_alive_at_target_destroy =
                frt_ctx_stream(context(), 0) >= 0;
        }
    }

    modalities::Status initialize_resources() override {
        probe_->lifecycle.push_back(LifecycleStep::kInitialize);
        anchor_ = frt_buffer_alloc(context(), "pi05_session_anchor", 1);
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

    modalities::Status finalize_setup() override {
        probe_->lifecycle.push_back(LifecycleStep::kFinalize);
        return fail_point_ == FailPoint::kFinalize
                   ? injected("injected setup finalization failure")
                   : modalities::Status::ok();
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
    frt_buffer anchor_ = nullptr;
    pi05::Pi05ResolvedResources resources_;
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
        LifecycleStep::kFinalize,
        LifecycleStep::kCaptureInputs}));
    CHECK(probe.record_calls == 380 + 482 + 390 + 92);
    CHECK(probe.default_stream_calls == 380);
    CHECK(probe.capture_stream_calls == 482 + 390 + 92);
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
        LifecycleStep::kFinalize,
        LifecycleStep::kCaptureInputs,
        LifecycleStep::kReset}));
    CHECK(probe.record_calls == 380 + 482 + 482 + 390 + 92);
    CHECK(probe.default_stream_calls == 380 + 482);
    CHECK(probe.capture_stream_calls == 482 + 390 + 92);
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
        case FailPoint::kFinalize:
            CHECK(probe.lifecycle.back() == LifecycleStep::kFinalize);
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
    expect_create_failure(false, FailPoint::kNone, 11);
    expect_create_failure(false, FailPoint::kFinalize);
    expect_create_failure(false, FailPoint::kCaptureInputs);
    expect_create_failure(true, FailPoint::kNone, 380 + 11);
    expect_create_failure(true, FailPoint::kReset);
    expect_create_failure(false, FailPoint::kNone, 380 + 11);
    expect_create_failure(false, FailPoint::kNone, 380 + 482 + 11);
    expect_create_failure(
        false, FailPoint::kNone, 380 + 482 + 390 + 11);

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

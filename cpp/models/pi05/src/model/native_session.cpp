#include "flashrt/cpp/models/pi05/model/native_session.h"

#include <new>
#include <utility>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

void set_status(modalities::Status* destination,
                modalities::Status status) {
    if (destination) *destination = std::move(status);
}

}  // namespace

Pi05NativeSession::Pi05NativeSession(
    frt_ctx context,
    Pi05ResolvedShape shape,
    std::unique_ptr<Pi05TargetBundle> target)
    : pipeline_(shape), program_(context), target_(std::move(target)) {}

std::unique_ptr<Pi05NativeSession> Pi05NativeSession::create(
    frt_ctx context,
    Pi05ResolvedShape shape,
    std::unique_ptr<Pi05TargetBundle> target,
    modalities::Status* status) {
    if (!context || !target || target->context() != context) {
        target.reset();
        if (context) frt_ctx_destroy(context);
        set_status(status, invalid("PI0.5 native session owner is invalid"));
        return nullptr;
    }

    Pi05NativeSession* allocated = nullptr;
    try {
        allocated = new (std::nothrow) Pi05NativeSession(
            context, shape, std::move(target));
    } catch (const std::bad_alloc&) {
        target.reset();
        frt_ctx_destroy(context);
        set_status(status,
                   backend("PI0.5 native session allocation failed"));
        return nullptr;
    }
    std::unique_ptr<Pi05NativeSession> session(allocated);
    if (!session) {
        target.reset();
        frt_ctx_destroy(context);
        set_status(status,
                   backend("PI0.5 native session allocation failed"));
        return nullptr;
    }
    modalities::Status result = session->initialize();
    if (!result.ok_status()) {
        session.reset();
        set_status(status, std::move(result));
        return nullptr;
    }
    set_status(status, modalities::Status::ok());
    return session;
}

modalities::Status Pi05NativeSession::initialize() {
    if (!program_.context() || !target_ ||
        target_->context() != program_.context()) {
        return invalid("PI0.5 native session state is invalid");
    }
    modalities::Status result = validate_pi05_resolved_shape(shape());
    if (!result.ok_status()) return result;

    result = target_->initialize_resources();
    if (!result.ok_status()) return result;

    Pi05ResolvedResources resolved;
    result = target_->resolve_resources(&resolved);
    if (!result.ok_status()) return result;
    result = validate_pi05_resolved_resources(resolved, shape());
    if (!result.ok_status()) return result;
    resources_ = resolved;

    result = pipeline_.record_prepare(*target_);
    if (!result.ok_status()) return result;
    result = target_->finalize_setup();
    if (!result.ok_status()) return result;
    result = target_->initialize_capture_inputs();
    if (!result.ok_status()) return result;

    if (target_->warmup_before_capture()) {
        result = program_.warmup(pipeline_, *target_);
        if (!result.ok_status()) return result;
        result = target_->reset_after_warmup();
        if (!result.ok_status()) return result;
    }

    Pi05ResolvedGraphBindings bindings;
    result = make_pi05_graph_bindings(resources_.buffers, &bindings);
    if (!result.ok_status()) return result;
    return program_.capture(pipeline_, *target_, bindings);
}

modalities::Status Pi05NativeSession::set_prompt_length(
    int prompt_tokens) {
    if (!target_ || prompt_tokens < 0 ||
        prompt_tokens > shape().max_prompt_tokens) {
        return invalid("PI0.5 prompt length is out of range");
    }
    return target_->set_prompt_length(prompt_tokens);
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

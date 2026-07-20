#include "abi/model_runtime_v1_abi_baseline.h"
#include "abi/model_runtime_v1_abi_baseline_producer.h"

namespace {

int unsupported_set_input(void*, uint32_t, const void*, uint64_t, int) {
    return -3;
}
int unsupported_get_output(void*, uint32_t, void*, uint64_t, uint64_t*, int) {
    return -3;
}
int unsupported_prepare(void*, uint32_t, frt_shape_key) { return -3; }
int unsupported_step(void*) { return -3; }
const char* unsupported_last_error(void*) {
    return "verb not provided by the ABI baseline producer";
}

}  // namespace

namespace flashrt::model_runtime_v1_abi {

void* create_baseline(const frt_runtime_export_v1* exp, void* owner,
                      void (*retain_owner)(void*),
                      void (*release_owner)(void*)) {
    auto* model = new frt_model_runtime_v1{};
    model->abi_version = FRT_MODEL_RUNTIME_ABI_VERSION;
    model->struct_size = (uint32_t)sizeof(*model);
    model->exp = exp;
    static const frt_runtime_stage_desc stage = {0, 0, nullptr};
    model->stages = &stage;
    model->n_stages = 1;
    model->verbs.struct_size = (uint32_t)sizeof(model->verbs);
    model->verbs.set_input = unsupported_set_input;
    model->verbs.get_output = unsupported_get_output;
    model->verbs.prepare = unsupported_prepare;
    model->verbs.step = unsupported_step;
    model->verbs.last_error = unsupported_last_error;
    model->owner = owner;
    model->retain = retain_owner;
    model->release = release_owner;
    return model;
}

void destroy_baseline(void* model) {
    delete static_cast<frt_model_runtime_v1*>(model);
}

}  // namespace flashrt::model_runtime_v1_abi

extern "C" void* flashrt_model_runtime_v1_abi_baseline_create(
        const frt_runtime_export_v1* exp, void* owner,
        void (*retain_owner)(void*), void (*release_owner)(void*)) {
    return flashrt::model_runtime_v1_abi::create_baseline(
        exp, owner, retain_owner, release_owner);
}

extern "C" void flashrt_model_runtime_v1_abi_baseline_destroy(void* model) {
    flashrt::model_runtime_v1_abi::destroy_baseline(model);
}

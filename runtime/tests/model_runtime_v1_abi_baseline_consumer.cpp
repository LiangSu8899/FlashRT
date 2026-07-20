#include "abi/model_runtime_v1_abi_baseline.h"

extern "C" int flashrt_model_runtime_v1_prefix_consume(const void* object) {
    auto* model = static_cast<const frt_model_runtime_v1*>(object);
    if (!model || model->abi_version != FRT_MODEL_RUNTIME_ABI_VERSION ||
        model->struct_size < sizeof(frt_model_runtime_v1) || !model->exp ||
        !model->retain || !model->release) return -1;
    model->retain(model->owner);
    model->release(model->owner);
    return 0;
}

extern "C" int flashrt_model_runtime_v1_exact_size_consume(
        const void* object) {
    auto* model = static_cast<const frt_model_runtime_v1*>(object);
    if (!model || model->abi_version != FRT_MODEL_RUNTIME_ABI_VERSION ||
        model->struct_size != sizeof(frt_model_runtime_v1)) return -1;
    return 0;
}

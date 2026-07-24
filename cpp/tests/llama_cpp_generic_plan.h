#ifndef FLASHRT_TESTS_LLAMA_CPP_GENERIC_PLAN_H
#define FLASHRT_TESTS_LLAMA_CPP_GENERIC_PLAN_H

#include "flashrt/model_runtime.h"

#include <cstring>

inline const frt_generic_stage_plan_ext_v1* llama_cpp_generic_plan(
        const frt_model_runtime_v1* model) {
    if (!model ||
        model->struct_size < FRT_MODEL_RUNTIME_V1_QUERY_EXTENSION_SIZE ||
        !model->query_extension) {
        return nullptr;
    }
    const void* extension = nullptr;
    if (model->query_extension(
            model, FRT_EXT_GENERIC_STAGE_PLAN_V1,
            FRT_GENERIC_STAGE_PLAN_ABI_VERSION, &extension) != 0) {
        return nullptr;
    }
    return static_cast<const frt_generic_stage_plan_ext_v1*>(extension);
}

inline int llama_cpp_run_generic_stage(
        const frt_model_runtime_v1* model, const char* name) {
    const auto* plan = llama_cpp_generic_plan(model);
    if (!plan || !name || !plan->run_opaque) return -3;
    for (uint64_t i = 0; i < plan->n_stages; ++i) {
        const auto& stage = plan->stages[i];
        if (stage.name && std::strcmp(stage.name, name) == 0) {
            if (stage.executor_kind != FRT_GENERIC_STAGE_OPAQUE) return -3;
            return plan->run_opaque(plan->stage_self, stage.executor_ref);
        }
    }
    return -3;
}

#endif

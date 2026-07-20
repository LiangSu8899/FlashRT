#ifndef FLASHRT_MODEL_RUNTIME_V1_ABI_BASELINE_PRODUCER_H
#define FLASHRT_MODEL_RUNTIME_V1_ABI_BASELINE_PRODUCER_H

#include "flashrt/runtime.h"

namespace flashrt::model_runtime_v1_abi {

void* create_baseline(const frt_runtime_export_v1* exp, void* owner,
                      void (*retain_owner)(void*),
                      void (*release_owner)(void*));
void destroy_baseline(void* model);

}  // namespace flashrt::model_runtime_v1_abi

#endif  // FLASHRT_MODEL_RUNTIME_V1_ABI_BASELINE_PRODUCER_H

/* Pi0.5 as an frt_model_runtime_v1 producer (the generic model-runtime face).
 *
 * The standard hand-off for hosts: instead of the model-specific
 * frt_pi05_runtime_* verbs, a host receives one frt_model_runtime_v1 and
 * drives it through the generic port/stage/verb contract
 * (flashrt/model_runtime.h). Pi0.5 semantics map onto it as:
 *
 *   port "images"  IN  STAGED  IMAGE   set_input <- frt_image_view[] in the
 *                                      declared camera-view order
 *   no "prompt" port                  adopted-export path: prompt embedding
 *                                      is prepared by the producer before
 *                                      capture. A native tokenizer producer
 *                                      adds a real STAGED TEXT port later.
 *   port "noise"   IN  SWAP    TENSOR  the diffusion seed window — the host
 *                                      writes raw bytes directly
 *   port "actions" OUT STAGED  ACTION  get_output -> unnormalized f32 robot
 *                                      actions (capacity/written in bytes)
 *   stage 0                            the configured infer graph
 *
 * This is the model-specific ADAPTER construction path
 * (frt_model_runtime_wrap): identity/fingerprint inherit from the export.
 */
#ifndef FLASHRT_CPP_MODELS_PI05_MODEL_RUNTIME_H
#define FLASHRT_CPP_MODELS_PI05_MODEL_RUNTIME_H

#include "flashrt/model_runtime.h"
#include "flashrt/cpp/models/pi05/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Build a retained frt_model_runtime_v1 over an adopted export. `config`
 * follows the same rules as frt_pi05_runtime_create. Release the returned
 * object via its own release(owner) — that destroys the internal Pi0.5
 * runtime and drops its export references. Returns 0 or a negative status
 * (same codes as the pi05 C API). */
int frt_pi05_model_runtime_create(const frt_runtime_export_v1* exp,
                                  const frt_pi05_runtime_config* config,
                                  frt_model_runtime_v1** out);

#ifdef __cplusplus
}
#endif

#endif  /* FLASHRT_CPP_MODELS_PI05_MODEL_RUNTIME_H */

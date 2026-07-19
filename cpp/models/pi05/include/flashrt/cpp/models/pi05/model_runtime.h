/* Pi0.5 adapters for the generic frt_model_runtime_v1 face.
 *
 * The complete native_v2 producer returned by frt_model_runtime_open_v1
 * declares prompt, state, images, noise, actions, and actions_raw ports. Its
 * graph catalog is infer, decode_only, and context. The selected stage plan is
 * either one infer stage or context followed by an action stage backed by the
 * decode_only graph. Hosts discover those declarations from the returned
 * model; this header does not define a second execution contract.
 *
 * Two lower-level construction paths remain for producer integration:
 *   - create(exp, ...): legacy adapter for an export without a model-runtime
 *     declaration. It creates images/noise/actions ports and one infer stage;
 *     prompt embedding remains producer setup state.
 *   - create_over(model, ...): verb overlay for a producer-owned declaration.
 *     Ports, stage DAG, identity, and fingerprint are inherited exactly. The
 *     complete native_v2 factory uses this path after installing real prompt,
 *     state, image, inference, and action behavior.
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

/* Build a retained Pi0.5 native verb overlay over an existing model-runtime
 * declaration. Ports/stages/identity/fingerprint are inherited exactly from
 * `model`; the returned object replaces only set_input/get_output/prepare/step.
 * Required ports by name: "images" (IMAGE IN STAGED) and "actions" (ACTION OUT
 * STAGED). Optional "noise"/"actions_raw" must be matching TENSOR SWAP ports;
 * optional "prompt"/"state" must be TEXT/STATE IN STAGED and require a fully
 * configured tokenizer/embedding/state-normalization implementation. */
int frt_pi05_model_runtime_create_over(const frt_model_runtime_v1* model,
                                       const frt_pi05_runtime_config* config,
                                       frt_model_runtime_v1** out);

#ifdef __cplusplus
}
#endif

#endif  /* FLASHRT_CPP_MODELS_PI05_MODEL_RUNTIME_H */

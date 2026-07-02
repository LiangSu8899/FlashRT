/* FlashRT Model Runtime — public C ABI (v1).
 *
 * The standard face of one DEPLOYED, TICKABLE model. It wraps the runtime
 * export (flashrt/runtime.h — the frozen execution/state kernel) and adds the
 * dynamic-IO contract a production tick needs:
 *
 *   dynamic inputs -> standardized update -> replay -> standardized outputs
 *
 * The contract is DATA FIRST, VERBS AS SUGAR:
 *   - `ports`  declare every dynamic input/output: modality, dtype, shape,
 *     layout, direction, and — the load-bearing part — the UPDATE CLASS.
 *   - `stages` declare the subgraph DAG (indices into the export's graphs +
 *     dependency edges). A white-box host schedules stages itself with the
 *     export handles; `step` merely fires them in declared order.
 *   - four verbs cover what data alone cannot: staged input transform,
 *     transformed output readback, warm-phase variant preparation, and the
 *     one-call tick.
 *
 * Update classes (the two-speed hot path):
 *   FRT_RT_PORT_SWAP   : the port IS a device-buffer window. The host writes
 *                        raw bytes directly (its own copy verb / cap_swap) —
 *                        zero model code in the loop. Microsecond lane.
 *   FRT_RT_PORT_STAGED : the model runtime's `set_input` transforms host data
 *                        (tokenize / resize / normalize / embed) into bound
 *                        buffers, optionally firing a micro-graph.
 *   FRT_RT_PORT_SETUP  : legal only outside the tick (weights, calibration).
 * A STAGED declaration is a PROMISE: the port accepts hot updates. A producer
 * that cannot update an input in the hot phase declares SETUP or omits the
 * port — never advertise-and-refuse.
 *
 * Production contract for BOTH hot classes (SWAP, STAGED) — conformance
 * suites pin these down:
 *   - never recapture a graph, never allocate, never rebind graph pointers;
 *     only buffer CONTENTS change (the graph-safe mutation discipline);
 *   - replay graphs are fixed-shape or shape-bucket-keyed; a shape bucket
 *     miss is handled by `prepare` in the WARM phase, never inside a tick.
 *
 * Two producers, one struct (mirroring the export):
 *   today : assembled by the export builder (Python setup or native C++);
 *   later : a native model-runtime .so exports FRT_MODEL_RUNTIME_OPEN_V1_SYMBOL.
 * Consumers (e.g. a capsule/state host) never learn the model, the producer
 * language, or the transform internals.
 *
 * Design rationale: docs/runtime_contract.md
 */
#ifndef FLASHRT_MODEL_RUNTIME_H
#define FLASHRT_MODEL_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "flashrt/runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FRT_MODEL_RUNTIME_ABI_VERSION 1u

/* ------------------------------------------------------------------ */
/* Enums — values are ABI-frozen after v1 (append-only).               */
/* ------------------------------------------------------------------ */

enum frt_rt_modality {
    FRT_RT_MOD_TENSOR = 0,   /* raw tensor per declared dtype/shape        */
    FRT_RT_MOD_IMAGE  = 1,   /* payload: frt_image_view[]                  */
    FRT_RT_MOD_TEXT   = 2,   /* payload: UTF-8 bytes (no NUL required)     */
    FRT_RT_MOD_STATE  = 3,   /* proprioception / numeric state (as TENSOR) */
    FRT_RT_MOD_ACTION = 4,   /* action chunk (as TENSOR)                   */
    FRT_RT_MOD_AUDIO  = 5,   /* PCM per declared dtype/shape               */
    FRT_RT_MOD_DEPTH  = 6,   /* payload: frt_image_view[] (single channel) */
    FRT_RT_MOD_FORCE  = 7    /* force/torque (as TENSOR)                   */
};

enum frt_rt_dtype {
    FRT_RT_DTYPE_U8   = 0,
    FRT_RT_DTYPE_F32  = 1,
    FRT_RT_DTYPE_F16  = 2,
    FRT_RT_DTYPE_BF16 = 3,
    FRT_RT_DTYPE_I32  = 4,
    FRT_RT_DTYPE_I64  = 5
};

enum frt_rt_layout {
    FRT_RT_LAYOUT_FLAT = 0,
    FRT_RT_LAYOUT_HWC  = 1,
    FRT_RT_LAYOUT_NHWC = 2,
    FRT_RT_LAYOUT_CHW  = 3,
    FRT_RT_LAYOUT_NCHW = 4
};

enum frt_rt_pixel_format {
    FRT_RT_PIXEL_RGB8  = 0,
    FRT_RT_PIXEL_BGR8  = 1,
    FRT_RT_PIXEL_RGBA8 = 2,
    FRT_RT_PIXEL_BGRA8 = 3,
    FRT_RT_PIXEL_GRAY8 = 4
};

enum frt_rt_port_direction { FRT_RT_PORT_IN = 0, FRT_RT_PORT_OUT = 1 };

enum frt_rt_port_update {
    FRT_RT_PORT_SWAP   = 0,
    FRT_RT_PORT_STAGED = 1,
    FRT_RT_PORT_SETUP  = 2
};

/* ------------------------------------------------------------------ */
/* Payload types (STAGED lane).                                        */
/* ------------------------------------------------------------------ */

/* One sensor frame handed to an IMAGE/DEPTH port. `set_input` receives an
 * array of these; `bytes` of the call = n_frames * sizeof(frt_image_view).
 * Frames are matched to the model's camera views POSITIONALLY, in the view
 * order the producer declared (see the port's manifest entry). */
typedef struct frt_image_view {
    uint32_t struct_size;      /* = sizeof(frt_image_view)                 */
    uint32_t pixel_format;     /* enum frt_rt_pixel_format                 */
    const void* data;          /* host pixels                              */
    uint64_t bytes;
    int32_t width, height, stride_bytes;
    uint32_t reserved;
    uint64_t timestamp_ns;
} frt_image_view;

/* ------------------------------------------------------------------ */
/* Descriptors. Strings/arrays are owned by the runtime object and     */
/* stay valid while the consumer holds a reference.                    */
/* ------------------------------------------------------------------ */

typedef struct frt_runtime_port_desc {
    const char* name;          /* "images", "prompt", "state", "actions"   */
    uint32_t modality;         /* frt_rt_modality                          */
    uint32_t dtype;            /* frt_rt_dtype (of the DEVICE-side tensor) */
    uint32_t layout;           /* frt_rt_layout                            */
    uint32_t direction;        /* frt_rt_port_direction                    */
    uint32_t update;           /* frt_rt_port_update                       */
    uint32_t required;         /* must be written before the first tick    */
    const int64_t* shape;      /* declared port tensor dims; for STAGED
                                * outputs this is the host-visible payload,
                                * not necessarily the raw bound buffer shape;
                                * -1 = bucket-variable                     */
    uint32_t rank;
    uint32_t cadence_hint_hz;  /* expected update rate; 0 = unknown. Hint,
                                * not contract — scheduling stays host-side */
    /* SWAP fast lane: the device window the host writes/reads directly.
     * Null buffer = STAGED-only port (no raw window is exposed). */
    frt_buffer buffer;
    uint64_t offset, bytes;
} frt_runtime_port_desc;

/* One schedulable stage = one export graph + dependency edges. Declared
 * array order is the sequential firing order `step` uses; `after` lists
 * stage indices that must complete first (for hosts that overlap stages
 * across streams). */
typedef struct frt_runtime_stage_desc {
    uint32_t graph;            /* index into exp->graphs                   */
    uint32_t n_after;
    const uint32_t* after;     /* stage indices                            */
} frt_runtime_stage_desc;

/* ------------------------------------------------------------------ */
/* Verbs — implemented by the producer, called by the host.            */
/* set_input / get_output are HOT (contract above); prepare is WARM;   */
/* step is sugar over the stage list. The construction paths fill any  */
/* verb the producer omits with an unsupported stub (returns -3), so   */
/* every entry is always callable — never a null pointer.              */
/* ------------------------------------------------------------------ */
typedef struct frt_model_runtime_verbs {
    uint32_t struct_size;      /* = sizeof(frt_model_runtime_verbs)        */
    uint32_t reserved;

    /* Write one input port. `data` is interpreted per the port's modality
     * (see payload conventions above). `stream` = an exp stream_id, or -1
     * for the port's default. Never recaptures/allocates/rebinds. */
    int (*set_input)(void* self, uint32_t port,
                     const void* data, uint64_t bytes, int stream);

    /* Read one output port through the producer's postprocess (e.g. action
     * unnormalize). `capacity`/`written` are BYTES. Raw readback needs no
     * verb — use the port's buffer. */
    int (*get_output)(void* self, uint32_t port,
                      void* out, uint64_t capacity, uint64_t* written,
                      int stream);

    /* WARM phase only: ensure graph `graph` (exp index) has a variant for
     * `key` (capture-on-miss for shape buckets). Never call inside a tick. */
    int (*prepare)(void* self, uint32_t graph, frt_shape_key key);

    /* Sugar: fire all stages in declared order on their declared streams.
     * Hosts that schedule/overlap/interrupt fire stages themselves. */
    int (*step)(void* self);

    const char* (*last_error)(void* self);
} frt_model_runtime_verbs;

/* ------------------------------------------------------------------ */
/* The model runtime object.                                           */
/* ------------------------------------------------------------------ */
typedef struct frt_model_runtime_v1 {
    uint32_t abi_version;      /* = FRT_MODEL_RUNTIME_ABI_VERSION          */
    uint32_t struct_size;      /* = sizeof(frt_model_runtime_v1)           */

    /* The execution/state kernel. Snapshot/restore/replay/regions all live
     * here, unchanged. */
    const frt_runtime_export_v1* exp;

    const frt_runtime_port_desc*  ports;  uint64_t n_ports;
    const frt_runtime_stage_desc* stages; uint64_t n_stages;

    void* self;                /* passed to every verb                     */
    frt_model_runtime_verbs verbs;

    /* Lifetime. The consumer retains/releases ONLY this object; the owner
     * holds one export reference internally. Thread-safe; a Python producer
     * handles GIL acquisition inside release. */
    void* owner;
    void (*retain)(void* owner);
    void (*release)(void* owner);
} frt_model_runtime_v1;

/* Factory symbol convention for NATIVE model runtimes: a model-runtime .so
 * exports exactly this symbol. Returns a retained object (caller releases). */
#define FRT_MODEL_RUNTIME_OPEN_V1_SYMBOL "frt_model_runtime_open_v1"
typedef int (*frt_model_runtime_open_v1_fn)(const char* config_json,
                                            frt_model_runtime_v1** out);

/* ------------------------------------------------------------------ */
/* Construction path 1 — INTEGRATED (preferred): the export builder    */
/* assembles export + ports + stages in one identity. Port and stage   */
/* records join the canonical identity string, so a port-schema change */
/* changes the fingerprint (a schema change means the captured IO      */
/* surface changed; stored state must be refused). A port's identity   */
/* covers its schema AND its bound window (buffer index/offset/bytes); */
/* only cadence_hint_hz stays out — it is advisory, not contract.      */
/* ------------------------------------------------------------------ */
int frt_runtime_builder_add_port(frt_runtime_builder, const char* name,
                                 uint32_t modality, uint32_t dtype,
                                 uint32_t layout, uint32_t direction,
                                 uint32_t update, uint32_t required,
                                 const int64_t* shape, uint32_t rank,
                                 uint32_t cadence_hint_hz,
                                 frt_buffer buffer, uint64_t offset,
                                 uint64_t bytes);
int frt_runtime_builder_add_stage(frt_runtime_builder, uint32_t graph,
                                  const uint32_t* after, uint32_t n_after);

/* Like frt_runtime_builder_finish, but returns the model runtime whose
 * `exp` is the internally-built export (one object, one refcount). `verbs`
 * is copied; entries may be null (the runtime then reports them
 * unsupported). Consumes the builder. */
frt_model_runtime_v1* frt_runtime_builder_finish_model(
    frt_runtime_builder,
    const frt_model_runtime_verbs* verbs, void* verbs_self,
    void* owner, void (*retain_owner)(void*), void (*release_owner)(void*));

/* ------------------------------------------------------------------ */
/* Construction path 2 — ADAPTER: wrap an EXISTING export with ports   */
/* and verbs (e.g. a native C++ model runtime over a Python-built      */
/* export). Identity/fingerprint are inherited from the export — ports */
/* are NOT re-fingerprinted on this path; prefer path 1 when the same  */
/* producer builds both. Descriptor arrays are copied. The wrapper     */
/* takes one export reference and calls `wrapper_release(wrapper_owner)`*/
/* exactly once when its refcount hits zero (use it to destroy the     */
/* producer instance behind `verbs_self`).                             */
/* ------------------------------------------------------------------ */
frt_model_runtime_v1* frt_model_runtime_wrap(
    const frt_runtime_export_v1* exp,
    const frt_runtime_port_desc* ports, uint64_t n_ports,
    const frt_runtime_stage_desc* stages, uint64_t n_stages,
    const frt_model_runtime_verbs* verbs, void* verbs_self,
    void* wrapper_owner, void (*wrapper_release)(void*));

/* ------------------------------------------------------------------ */
/* Construction path 3 — VERB OVERRIDE: keep an existing model runtime */
/* declaration (export + ports + stages) and replace only the verbs.   */
/* This is the clean hand-off when one producer owns capture/schema and */
/* a native runtime owns hot-path transforms. The override retains `in` */
/* so all inherited descriptor pointers stay valid; consumers release   */
/* only the returned object. `retain_owner`/`release_owner` manage the  */
/* native verb object, called once at construction/destruction.         */
/* ------------------------------------------------------------------ */
frt_model_runtime_v1* frt_model_runtime_override_verbs(
    const frt_model_runtime_v1* in,
    const frt_model_runtime_verbs* verbs, void* verbs_self,
    void* owner, void (*retain_owner)(void*), void (*release_owner)(void*));

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* FLASHRT_MODEL_RUNTIME_H */

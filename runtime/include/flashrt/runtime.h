/* FlashRT Runtime Export — public C ABI (v1).
 *
 * The single hand-off surface between a FlashRT model runtime (the PRODUCER:
 * whatever captured the graphs and owns the weights) and a host / serving
 * layer (the CONSUMER: e.g. a capsule/state host, a robot loop, a server
 * shell). One captured, replay-ready model is packaged as one POD struct:
 * context + streams + graphs + buffers + restorable state regions + identity.
 *
 * Neither side sees the other's internals. The producer knows nothing about
 * scheduling or sessions; the consumer knows nothing about Python, torch, or
 * model code — it sees only exec-contract handles (flashrt/exec.h) and this
 * struct.
 *
 * Two producers, one struct (the point of this layer):
 *   today : Python setup/capture fills the struct in-process
 *           (flash_rt/runtime/export.py over the builder below)
 *   later : a native C++ model runtime .so exports FRT_RUNTIME_OPEN_V1_SYMBOL
 *           and fills the SAME struct — consumers do not change.
 *
 * Mechanism only, like exec.h: no schedule, no session, no policy. Plans are
 * deliberately NOT exported — DAG orchestration belongs to the consumer.
 *
 * Design rationale: docs/runtime_contract.md
 */
#ifndef FLASHRT_RUNTIME_H
#define FLASHRT_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "flashrt/exec.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FRT_RUNTIME_ABI_VERSION 1u

/* ------------------------------------------------------------------ */
/* Enums — values are ABI-frozen after v1 (append-only).               */
/* ------------------------------------------------------------------ */

/* Buffer role: a BITMASK, not an enum value — a buffer can be both input and
 * output (e.g. an in-place diffusion noise/action buffer). */
enum frt_runtime_role {
    FRT_RT_ROLE_INPUT   = 1u << 0,
    FRT_RT_ROLE_OUTPUT  = 1u << 1,
    FRT_RT_ROLE_STATE   = 1u << 2,
    FRT_RT_ROLE_SCRATCH = 1u << 3
};

/* Capsule-region flags (bitmask). A region normally carries both bits: it is
 * included in snapshots AND written back on restore. Single-bit uses exist
 * (e.g. snapshot-only telemetry state). The byte LAYOUT of every exported
 * region (name/offset/bytes/flags/order) is ALWAYS part of the identity
 * fingerprint — there is no opt-out, because restore matches regions by
 * position. */
enum frt_runtime_region_flags {
    FRT_RT_REGION_SNAPSHOT = 1u << 0,
    FRT_RT_REGION_RESTORE  = 1u << 1
};

/* ------------------------------------------------------------------ */
/* Descriptors — POD views. All strings/arrays are owned by the export */
/* object and stay valid while the consumer holds a reference.         */
/* ------------------------------------------------------------------ */

typedef struct frt_runtime_stream_desc {
    const char* name;      /* e.g. "main", "vision", "asr"                    */
    int   stream_id;       /* frt_ctx-scoped id (usable with frt_graph_replay) */
    int   priority;        /* 0 = normal; more-negative = higher (CUDA conv.) */
    void* native_handle;   /* raw backend stream (e.g. cudaStream_t). Lets a
                            * consumer drive its own events/waits on this
                            * stream without an frt accessor. Borrowed: valid
                            * while the export is retained; never destroy it. */
} frt_runtime_stream_desc;

typedef struct frt_runtime_graph_desc {
    const char* name;            /* e.g. "infer", "decode_only", "preprocess" */
    frt_graph   handle;
    frt_shape_key default_key;   /* the key a plain host fires                */
    const frt_shape_key* keys;   /* captured variant keys at export time      */
    uint64_t    n_keys;          /* (discovery aid; the live table may grow — */
                                 /*  frt_graph_has_variant is the truth)      */
    int         stream_id;       /* default replay stream (host may override) */
} frt_runtime_graph_desc;

typedef struct frt_runtime_buffer_desc {
    const char* name;
    frt_buffer  handle;
    uint64_t    bytes;
    uint32_t    role;            /* frt_runtime_role bitmask                  */
    uint32_t    reserved;        /* zero for v1                               */
} frt_runtime_buffer_desc;

/* A restorable state region: what a capsule/state host snapshots + restores.
 * Regions are matched BY POSITION on restore, so array order is contractual
 * (and fingerprinted). */
typedef struct frt_runtime_region_desc {
    const char* name;
    frt_buffer  buffer;
    uint64_t    offset;
    uint64_t    bytes;
    uint32_t    flags;           /* frt_runtime_region_flags bitmask          */
    uint32_t    reserved;        /* zero for v1                               */
} frt_runtime_region_desc;

/* ------------------------------------------------------------------ */
/* The export object.                                                  */
/* ------------------------------------------------------------------ */
typedef struct frt_runtime_export_v1 {
    uint32_t abi_version;        /* = FRT_RUNTIME_ABI_VERSION                 */
    uint32_t struct_size;        /* = sizeof(frt_runtime_export_v1)           */

    frt_ctx ctx;                 /* the exec context everything lives in      */

    const frt_runtime_stream_desc* streams;         uint64_t n_streams;
    const frt_runtime_graph_desc*  graphs;          uint64_t n_graphs;
    const frt_runtime_buffer_desc* buffers;         uint64_t n_buffers;
    const frt_runtime_region_desc* capsule_regions; uint64_t n_capsule_regions;

    /* Identity vs discovery — deliberately SPLIT:
     *   identity    : the canonical identity string (human-readable). The
     *                 fingerprint is a hash of exactly this string, computed
     *                 ONLY by frt_runtime_builder_finish — one implementation,
     *                 one hashing rule. Covers user identity pairs (weights
     *                 digest, quant, kernel version, arch), graph names, and
     *                 the full capsule-region layout. On a fingerprint
     *                 mismatch, print both identity strings to see WHY.
     *   fingerprint : FNV-1a 64 of `identity` (frt_runtime_fingerprint).
     *   manifest    : free-form JSON for discovery/tooling (shapes, dtypes,
     *                 semantic docs). NOT part of identity — editing the
     *                 manifest never invalidates stored state.               */
    uint64_t    fingerprint;
    const char* identity;
    const char* manifest_json;   /* may be NULL */

    /* Lifetime. The consumer MUST call retain(owner) when adopting and
     * release(owner) when done; every handle above (ctx, graphs, buffers,
     * native stream handles, strings) is valid only while it holds a
     * reference. retain/release are thread-safe; a Python producer handles
     * GIL acquisition internally — a consumer may release from any thread. */
    void* owner;
    void (*retain)(void* owner);
    void (*release)(void* owner);
} frt_runtime_export_v1;

/* Factory symbol convention for NATIVE runtimes (phase 2). A model runtime
 * shared object exports exactly this symbol; a host dlopens the .so, dlsyms
 * the symbol, and receives the same struct Python produces today:
 *
 *   int frt_runtime_open_v1(const char* config_json,
 *                           frt_runtime_export_v1** out);
 *   -> 0 on success (out points to a retained export; caller must release),
 *      negative on failure.
 */
#define FRT_RUNTIME_OPEN_V1_SYMBOL "frt_runtime_open_v1"
typedef int (*frt_runtime_open_v1_fn)(const char* config_json,
                                      frt_runtime_export_v1** out);

/* ------------------------------------------------------------------ */
/* Builder — the producer-side helper (libflashrt_runtime).            */
/* Consumers do NOT need it: they only read the struct above.          */
/* Setup-time only; allocates freely. Handles passed in are borrowed   */
/* (kept alive by the producer via the owner reference).               */
/* ------------------------------------------------------------------ */
typedef struct frt_runtime_builder_s* frt_runtime_builder;

frt_runtime_builder frt_runtime_builder_create(frt_ctx);

/* All add_* return 0 on success, negative on bad args. Order of add_* calls
 * defines descriptor array order (regions: contractual + fingerprinted). */
int frt_runtime_builder_add_stream(frt_runtime_builder, const char* name,
                                   int stream_id, int priority,
                                   void* native_handle);
int frt_runtime_builder_add_graph (frt_runtime_builder, const char* name,
                                   frt_graph, frt_shape_key default_key,
                                   const frt_shape_key* keys, uint64_t n_keys,
                                   int stream_id);
int frt_runtime_builder_add_buffer(frt_runtime_builder, const char* name,
                                   frt_buffer, uint64_t bytes, uint32_t role);
int frt_runtime_builder_add_region(frt_runtime_builder, const char* name,
                                   frt_buffer, uint64_t offset, uint64_t bytes,
                                   uint32_t flags);

/* Append one canonical identity pair (e.g. "weights_sha256", "quant",
 * "kernel_version", "arch"). Insertion order is canonical — a producer must
 * emit pairs in a deterministic order. */
int frt_runtime_builder_add_identity(frt_runtime_builder, const char* key,
                                     const char* value);
int frt_runtime_builder_set_manifest(frt_runtime_builder, const char* json);

/* Finish: builds the canonical identity string, computes the fingerprint,
 * flattens everything into one export object, and consumes the builder
 * (valid or not — never use the builder again after this call).
 *
 * `owner` + callbacks anchor the producer's world (in phase 1: the Python
 * objects behind every handle). `retain_owner` may be NULL; `release_owner`
 * (if non-NULL) is called exactly once when the export refcount hits zero.
 * The returned export starts with ONE reference held by the caller. */
frt_runtime_export_v1* frt_runtime_builder_finish(frt_runtime_builder,
                                                  void* owner,
                                                  void (*retain_owner)(void*),
                                                  void (*release_owner)(void*));

/* The one hashing rule (FNV-1a 64). Exposed so tests and mismatch tooling can
 * recompute fingerprints from identity strings. */
uint64_t frt_runtime_fingerprint(const void* data, size_t len);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* FLASHRT_RUNTIME_H */

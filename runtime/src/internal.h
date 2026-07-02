/* internal.h — shared builder/holder machinery for the runtime-export and
 * model-runtime C ABIs. Not installed; the public surface is
 * include/flashrt/runtime.h + include/flashrt/model_runtime.h.
 */
#ifndef FLASHRT_RUNTIME_INTERNAL_H
#define FLASHRT_RUNTIME_INTERNAL_H

#include "flashrt/runtime.h"
#include "flashrt/model_runtime.h"

#include <atomic>
#include <deque>
#include <string>
#include <vector>

namespace frt_rt {

/* One block that owns every array/string the export (and, when built via
 * finish_model, the model runtime) points into. Freed when the reference
 * count drops to zero. std::deque: element addresses are stable under
 * push_back, so descriptors can point at .c_str() / .data() safely. */
struct Holder {
    std::atomic<int> refs{1};
    void* user_owner = nullptr;
    void (*user_release)(void*) = nullptr;

    std::deque<std::string> names;
    std::deque<std::vector<frt_shape_key>> key_arrays;
    std::string identity;
    std::string manifest;
    bool has_manifest = false;

    std::vector<frt_runtime_stream_desc> streams;
    std::vector<frt_runtime_graph_desc>  graphs;
    std::vector<frt_runtime_buffer_desc> buffers;
    std::vector<frt_runtime_region_desc> regions;

    /* model-runtime additions (empty for plain exports) */
    std::deque<std::vector<int64_t>>  shape_arrays;
    std::deque<std::vector<uint32_t>> after_arrays;
    std::vector<frt_runtime_port_desc>  ports;
    std::vector<frt_runtime_stage_desc> stages;

    frt_runtime_export_v1 exp{};
    frt_model_runtime_v1  model{};
};

extern "C" void frt_rt_holder_retain(void* owner);
extern "C" void frt_rt_holder_release(void* owner);

const char* stored(Holder* h, const char* s);

}  // namespace frt_rt

struct frt_runtime_builder_s {
    frt_ctx ctx = nullptr;
    frt_rt::Holder* h = nullptr;  /* built up in place; adopted by finish */
    std::string identity_pairs;
};

namespace frt_rt {

/* Canonical identity + export fill, shared by finish and finish_model.
 * Appends port/stage records when present (restore matches regions by
 * position and replay depends on the declared IO surface, so both are
 * identity). Consumes nothing; the caller consumes the builder. */
void finish_export_into(Holder* h, frt_runtime_builder_s* b,
                        void* owner, void (*retain_owner)(void*),
                        void (*release_owner)(void*));

}  // namespace frt_rt

#endif /* FLASHRT_RUNTIME_INTERNAL_H */

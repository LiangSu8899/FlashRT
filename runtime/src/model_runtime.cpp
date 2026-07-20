/* model_runtime.cpp — builder extensions + adapter wrap for
 * flashrt/model_runtime.h. Like the export builder, this layer only records
 * declarations and manages lifetime; every transform stays behind the
 * producer's verbs.
 */
#include "internal.h"

#include <cstring>

using frt_rt::Holder;
using frt_rt::stored;

namespace {

bool valid_port_args(const char* name, uint32_t direction, uint32_t update,
                     const int64_t* shape, uint32_t rank) {
    if (!name || !name[0]) return false;
    if (direction > FRT_RT_PORT_OUT) return false;
    if (update > FRT_RT_PORT_SETUP) return false;
    if (rank && !shape) return false;
    return true;
}

bool has_complete_verbs(const frt_model_runtime_verbs* verbs) {
    return verbs &&
           verbs->struct_size >= (uint32_t)sizeof(frt_model_runtime_verbs);
}

bool valid_staged_verbs(const frt_runtime_port_desc* ports, uint64_t n_ports,
                        const frt_model_runtime_verbs* verbs) {
    bool needs_set_input = false;
    bool needs_get_output = false;
    for (uint64_t i = 0; i < n_ports; ++i) {
        if (ports[i].update != FRT_RT_PORT_STAGED) continue;
        if (ports[i].direction == FRT_RT_PORT_IN) needs_set_input = true;
        if (ports[i].direction == FRT_RT_PORT_OUT) needs_get_output = true;
    }
    if (!needs_set_input && !needs_get_output) return true;
    if (!has_complete_verbs(verbs)) return false;
    return (!needs_set_input || verbs->set_input) &&
           (!needs_get_output || verbs->get_output);
}

bool valid_utf8_stage_name(const char* name) {
    if (!name) return false;
    const auto* p = reinterpret_cast<const unsigned char*>(name);
    const size_t n = std::strlen(name);
    if (n == 0 || n > FRT_GENERIC_STAGE_NAME_MAX_BYTES) return false;
    for (size_t i = 0; i < n;) {
        const unsigned char c = p[i];
        if (c < 0x20 || c == 0x7f) return false;
        if (c < 0x80) { ++i; continue; }
        size_t more = 0;
        uint32_t cp = 0;
        if ((c & 0xe0) == 0xc0) { more = 1; cp = c & 0x1f; }
        else if ((c & 0xf0) == 0xe0) { more = 2; cp = c & 0x0f; }
        else if ((c & 0xf8) == 0xf0) { more = 3; cp = c & 0x07; }
        else return false;
        if (i + more >= n) return false;
        for (size_t j = 1; j <= more; ++j) {
            if ((p[i + j] & 0xc0) != 0x80) return false;
            cp = (cp << 6) | (p[i + j] & 0x3f);
        }
        if ((more == 1 && cp < 0x80) || (more == 2 && cp < 0x800) ||
            (more == 3 && cp < 0x10000) || cp > 0x10ffff ||
            (cp >= 0xd800 && cp <= 0xdfff)) return false;
        i += more + 1;
    }
    return true;
}

bool has_real_step(const frt_model_runtime_verbs* verbs) {
    return has_complete_verbs(verbs) && verbs->step;
}

bool has_real_last_error(const frt_model_runtime_verbs* verbs) {
    return has_complete_verbs(verbs) && verbs->last_error;
}

bool valid_authority(const Holder* h, const frt_model_runtime_verbs* verbs) {
    if (!h->stages.empty() && h->generic_plan_present) return false;
    if (h->generic_plan_present) {
        if (h->n_generic_stages == 0) return false;
        bool has_opaque = false;
        for (size_t i = 0; i < h->n_generic_stages; ++i)
            has_opaque |= h->generic_stages[i].executor_kind ==
                          FRT_GENERIC_STAGE_OPAQUE;
        if (!has_opaque) return false;
        if (!h->generic_runner_registered || !h->run_opaque ||
            !has_real_last_error(verbs)) return false;
        return true;
    }
    return !h->stages.empty() || has_real_step(verbs);
}

/* Default stubs for verbs a producer does not provide: report unsupported
 * (-3) instead of leaving null function pointers for consumers to crash on. */
int stub_set_input(void*, uint32_t, const void*, uint64_t, int) { return -3; }
int stub_get_output(void*, uint32_t, void*, uint64_t, uint64_t*, int) {
    return -3;
}
int stub_prepare(void*, uint32_t, frt_shape_key) { return -3; }
int stub_step(void*) { return -3; }
const char* stub_last_error(void*) {
    return "verb not provided by this producer";
}

int query_no_extensions(const frt_model_runtime_v1* runtime, uint64_t,
                        uint32_t min_version, const void** out_extension) {
    if (out_extension) *out_extension = nullptr;
    if (!runtime || !out_extension || min_version == 0) return -1;
    return -3;
}

int query_holder_extensions(const frt_model_runtime_v1* runtime,
                            uint64_t extension_id, uint32_t min_version,
                            const void** out_extension) {
    if (out_extension) *out_extension = nullptr;
    if (!runtime || !out_extension || min_version == 0) return -1;
    if (extension_id != FRT_EXT_GENERIC_STAGE_PLAN_V1 || min_version > 1)
        return -3;
    auto* h = static_cast<const Holder*>(runtime->owner);
    if (!h || !h->generic_plan_present) return -3;
    *out_extension = &h->generic_stage_plan;
    return 0;
}

void copy_verbs(frt_model_runtime_v1* m, const frt_model_runtime_verbs* verbs,
                void* verbs_self) {
    m->verbs.struct_size = (uint32_t)sizeof(frt_model_runtime_verbs);
    if (verbs && verbs->struct_size >= sizeof(frt_model_runtime_verbs)) {
        m->verbs.set_input = verbs->set_input;
        m->verbs.get_output = verbs->get_output;
        m->verbs.prepare = verbs->prepare;
        m->verbs.step = verbs->step;
        m->verbs.last_error = verbs->last_error;
    }
    if (!m->verbs.set_input) m->verbs.set_input = stub_set_input;
    if (!m->verbs.get_output) m->verbs.get_output = stub_get_output;
    if (!m->verbs.prepare) m->verbs.prepare = stub_prepare;
    if (!m->verbs.step) m->verbs.step = stub_step;
    if (!m->verbs.last_error) m->verbs.last_error = stub_last_error;
    m->self = verbs_self;
    m->query_extension = query_no_extensions;
}

}  // namespace

extern "C" int frt_runtime_builder_add_port(frt_runtime_builder b,
                                            const char* name,
                                            uint32_t modality, uint32_t dtype,
                                            uint32_t layout, uint32_t direction,
                                            uint32_t update, uint32_t required,
                                            const int64_t* shape, uint32_t rank,
                                            uint32_t cadence_hint_hz,
                                            frt_buffer buffer, uint64_t offset,
                                            uint64_t bytes) {
    if (!b || !valid_port_args(name, direction, update, shape, rank)) return -1;
    if (b->metadata_only &&
        (update == FRT_RT_PORT_SWAP || buffer || offset || bytes)) return -1;
    Holder* h = b->h;
    h->shape_arrays.emplace_back(shape, shape + rank);
    frt_runtime_port_desc d{};
    d.name = stored(h, name);
    d.modality = modality;
    d.dtype = dtype;
    d.layout = layout;
    d.direction = direction;
    d.update = update;
    d.required = required;
    d.shape = h->shape_arrays.back().data();
    d.rank = rank;
    d.cadence_hint_hz = cadence_hint_hz;
    d.buffer = buffer;
    d.offset = offset;
    d.bytes = bytes;
    h->ports.push_back(d);
    return 0;
}

extern "C" int frt_runtime_builder_add_stage(frt_runtime_builder b,
                                             uint32_t graph,
                                             const uint32_t* after,
                                             uint32_t n_after) {
    if (!b || b->metadata_only || b->h->generic_plan_present ||
        (n_after && !after)) return -1;
    Holder* h = b->h;
    if (graph >= h->graphs.size()) return -1;
    for (uint32_t i = 0; i < n_after; ++i)
        if (after[i] >= h->stages.size()) return -1;   /* only earlier stages */
    h->after_arrays.emplace_back(after, after + n_after);
    frt_runtime_stage_desc d{};
    d.graph = graph;
    d.after = h->after_arrays.back().data();
    d.n_after = n_after;
    h->stages.push_back(d);
    return 0;
}

extern "C" int frt_runtime_builder_add_generic_stage(
        frt_runtime_builder b, const char* name, uint32_t executor_kind,
        uint32_t executor_ref, const uint32_t* after, uint32_t n_after) {
    if (!b || !b->h->stages.empty() || !valid_utf8_stage_name(name) ||
        executor_kind > FRT_GENERIC_STAGE_OPAQUE || (n_after && !after))
        return -1;
    Holder* h = b->h;
    if (b->metadata_only && executor_kind != FRT_GENERIC_STAGE_OPAQUE)
        return -1;
    if (executor_kind == FRT_GENERIC_STAGE_GRAPH &&
        executor_ref >= h->graphs.size()) return -1;
    for (size_t i = 0; i < h->n_generic_stages; ++i)
        if (std::strcmp(h->generic_stages[i].name, name) == 0) return -1;
    uint32_t previous = 0;
    for (uint32_t i = 0; i < n_after; ++i) {
        if (after[i] >= h->n_generic_stages ||
            (i && after[i] <= previous)) return -1;
        previous = after[i];
    }
    h->after_arrays.emplace_back(after, after + n_after);
    frt_generic_stage_desc_v1 d{};
    d.name = stored(h, name);
    d.executor_kind = executor_kind;
    d.executor_ref = executor_ref;
    d.n_after = n_after;
    d.after = h->after_arrays.back().data();
    if (h->n_generic_stages == h->generic_stage_capacity) {
        const size_t next_capacity = h->generic_stage_capacity
            ? h->generic_stage_capacity * 2 : 4;
        auto next = std::make_unique<frt_generic_stage_desc_v1[]>(
            next_capacity);
        for (size_t i = 0; i < h->n_generic_stages; ++i)
            next[i] = h->generic_stages[i];
        h->generic_stages = std::move(next);
        h->generic_stage_capacity = next_capacity;
    }
    h->generic_stages[h->n_generic_stages++] = d;
    h->generic_plan_present = true;
    return 0;
}

extern "C" int frt_runtime_builder_set_generic_stage_runner(
        frt_runtime_builder b, void* stage_self,
        int (*run_opaque)(void*, uint32_t)) {
    if (!b || !run_opaque || b->h->generic_runner_registered) return -1;
    b->h->generic_plan_present = true;
    b->h->generic_runner_registered = true;
    b->h->generic_stage_self = stage_self;
    b->h->run_opaque = run_opaque;
    return 0;
}

extern "C" frt_model_runtime_v1* frt_runtime_builder_finish_model(
        frt_runtime_builder b,
        const frt_model_runtime_verbs* verbs, void* verbs_self,
        void* owner, void (*retain_owner)(void*),
        void (*release_owner)(void*)) {
    if (!b) return nullptr;
    Holder* h = b->h;
    if (!valid_staged_verbs(h->ports.data(), h->ports.size(), verbs))
        return nullptr;
    if (!valid_authority(h, verbs)) return nullptr;
    frt_rt::finish_export_into(h, b, owner, retain_owner, release_owner);

    if (h->generic_plan_present) {
        h->generic_stage_plan.abi_version =
            FRT_GENERIC_STAGE_PLAN_ABI_VERSION;
        h->generic_stage_plan.struct_size =
            (uint32_t)sizeof(frt_generic_stage_plan_ext_v1);
        h->generic_stage_plan.stages = h->generic_stages.get();
        h->generic_stage_plan.n_stages = h->n_generic_stages;
        h->generic_stage_plan.stage_self = h->generic_stage_self;
        h->generic_stage_plan.run_opaque = h->run_opaque;
    }

    frt_model_runtime_v1& m = h->model;
    m.abi_version = FRT_MODEL_RUNTIME_ABI_VERSION;
    m.struct_size = (uint32_t)sizeof(frt_model_runtime_v1);
    m.exp = &h->exp;
    m.ports = h->ports.data();   m.n_ports = h->ports.size();
    m.stages = h->stages.data(); m.n_stages = h->stages.size();
    copy_verbs(&m, verbs, verbs_self);
    m.query_extension = query_holder_extensions;
    m.owner = h;
    m.retain = frt_rt::frt_rt_holder_retain;
    m.release = frt_rt::frt_rt_holder_release;

    delete b;  /* h lives on inside the model runtime */
    return &h->model;
}

/* ---- adapter path: wrap an existing export -------------------------------- */

namespace {

struct Wrapper {
    std::atomic<int> refs{1};
    const frt_runtime_export_v1* exp = nullptr;
    void* wrapper_owner = nullptr;
    void (*wrapper_release)(void*) = nullptr;

    std::deque<std::string> names;
    std::deque<std::vector<int64_t>>  shape_arrays;
    std::deque<std::vector<uint32_t>> after_arrays;
    std::vector<frt_runtime_port_desc>  ports;
    std::vector<frt_runtime_stage_desc> stages;
    frt_model_runtime_v1 model{};
};

extern "C" void wrapper_retain(void* owner) {
    static_cast<Wrapper*>(owner)->refs.fetch_add(1, std::memory_order_relaxed);
}

extern "C" void wrapper_release(void* owner) {
    Wrapper* w = static_cast<Wrapper*>(owner);
    if (w->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (w->wrapper_release) w->wrapper_release(w->wrapper_owner);
        if (w->exp && w->exp->release) w->exp->release(w->exp->owner);
        delete w;
    }
}

}  // namespace

extern "C" frt_model_runtime_v1* frt_model_runtime_wrap(
        const frt_runtime_export_v1* exp,
        const frt_runtime_port_desc* ports, uint64_t n_ports,
        const frt_runtime_stage_desc* stages, uint64_t n_stages,
        const frt_model_runtime_verbs* verbs, void* verbs_self,
        void* wrapper_owner, void (*wrapper_release_fn)(void*)) {
    if (!exp || !exp->ctx || exp->abi_version != FRT_RUNTIME_ABI_VERSION ||
        exp->struct_size < sizeof(frt_runtime_export_v1) ||
        !exp->retain || !exp->release || n_stages == 0) return nullptr;
    if ((n_ports && !ports) || (n_stages && !stages)) return nullptr;
    for (uint64_t i = 0; i < n_ports; ++i)
        if (!valid_port_args(ports[i].name, ports[i].direction,
                             ports[i].update, ports[i].shape, ports[i].rank))
            return nullptr;
    if (!valid_staged_verbs(ports, n_ports, verbs)) return nullptr;
    for (uint64_t i = 0; i < n_stages; ++i) {
        if (stages[i].graph >= exp->n_graphs) return nullptr;
        if (stages[i].n_after && !stages[i].after) return nullptr;
        for (uint32_t d = 0; d < stages[i].n_after; ++d)
            if (stages[i].after[d] >= i) return nullptr;
    }

    auto* w = new Wrapper();
    w->exp = exp;
    w->wrapper_owner = wrapper_owner;
    w->wrapper_release = wrapper_release_fn;
    exp->retain(exp->owner);

    for (uint64_t i = 0; i < n_ports; ++i) {
        frt_runtime_port_desc d = ports[i];
        w->names.emplace_back(d.name);
        d.name = w->names.back().c_str();
        w->shape_arrays.emplace_back(d.shape, d.shape + d.rank);
        d.shape = w->shape_arrays.back().data();
        w->ports.push_back(d);
    }
    for (uint64_t i = 0; i < n_stages; ++i) {
        frt_runtime_stage_desc d = stages[i];
        w->after_arrays.emplace_back(d.after, d.after + d.n_after);
        d.after = w->after_arrays.back().data();
        w->stages.push_back(d);
    }

    frt_model_runtime_v1& m = w->model;
    m.abi_version = FRT_MODEL_RUNTIME_ABI_VERSION;
    m.struct_size = (uint32_t)sizeof(frt_model_runtime_v1);
    m.exp = exp;
    m.ports = w->ports.data();   m.n_ports = w->ports.size();
    m.stages = w->stages.data(); m.n_stages = w->stages.size();
    copy_verbs(&m, verbs, verbs_self);
    m.owner = w;
    m.retain = wrapper_retain;
    m.release = wrapper_release;
    return &w->model;
}

/* ---- verb override path: inherit declarations, replace verbs -------------- */

namespace {

struct VerbOverride {
    std::atomic<int> refs{1};
    const frt_model_runtime_v1* base = nullptr;
    void* owner = nullptr;
    void (*release_owner)(void*) = nullptr;
    frt_model_runtime_verbs target_verbs{};
    void* target_self = nullptr;
    bool forward_base_error = false;
    frt_model_runtime_v1 model{};
};

int override_set_input(void* self, uint32_t port, const void* data,
                       uint64_t bytes, int stream) {
    auto* o = static_cast<VerbOverride*>(self);
    return o->target_verbs.set_input(o->target_self, port, data, bytes, stream);
}
int override_get_output(void* self, uint32_t port, void* out,
                        uint64_t capacity, uint64_t* written, int stream) {
    auto* o = static_cast<VerbOverride*>(self);
    return o->target_verbs.get_output(o->target_self, port, out, capacity,
                                      written, stream);
}
int override_prepare(void* self, uint32_t graph, frt_shape_key key) {
    auto* o = static_cast<VerbOverride*>(self);
    return o->target_verbs.prepare(o->target_self, graph, key);
}
int override_step(void* self) {
    auto* o = static_cast<VerbOverride*>(self);
    return o->target_verbs.step(o->target_self);
}
const char* override_last_error(void* self) {
    auto* o = static_cast<VerbOverride*>(self);
    if (o->forward_base_error)
        return o->base->verbs.last_error(o->base->self);
    return o->target_verbs.last_error(o->target_self);
}

int override_query_extensions(const frt_model_runtime_v1* runtime,
                              uint64_t extension_id, uint32_t min_version,
                              const void** out_extension) {
    if (out_extension) *out_extension = nullptr;
    if (!runtime || !out_extension || min_version == 0) return -1;
    auto* o = static_cast<const VerbOverride*>(runtime->owner);
    if (!o || !o->base ||
        o->base->struct_size < FRT_MODEL_RUNTIME_V1_QUERY_EXTENSION_SIZE ||
        !o->base->query_extension) return -3;
    return o->base->query_extension(o->base, extension_id, min_version,
                                    out_extension);
}

extern "C" void override_retain(void* owner) {
    static_cast<VerbOverride*>(owner)->refs.fetch_add(1,
                                                      std::memory_order_relaxed);
}

extern "C" void override_release(void* owner) {
    VerbOverride* o = static_cast<VerbOverride*>(owner);
    if (o->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (o->release_owner) o->release_owner(o->owner);
        if (o->base && o->base->release) o->base->release(o->base->owner);
        delete o;
    }
}

bool valid_model_runtime(const frt_model_runtime_v1* m) {
    if (!m || m->abi_version != FRT_MODEL_RUNTIME_ABI_VERSION ||
        m->struct_size < FRT_MODEL_RUNTIME_V1_BASE_SIZE) return false;
    if (!m->exp || !m->retain || !m->release) return false;
    if ((m->n_ports && !m->ports) || (m->n_stages && !m->stages)) return false;
    return true;
}

int get_generic_plan(const frt_model_runtime_v1* model,
                     const frt_generic_stage_plan_ext_v1** out) {
    *out = nullptr;
    if (model->struct_size < FRT_MODEL_RUNTIME_V1_QUERY_EXTENSION_SIZE ||
        !model->query_extension) return -3;
    const void* extension = nullptr;
    const int rc = model->query_extension(
        model, FRT_EXT_GENERIC_STAGE_PLAN_V1, 1, &extension);
    if (rc != 0) return rc;
    auto* plan = static_cast<const frt_generic_stage_plan_ext_v1*>(extension);
    if (!plan || plan->abi_version < FRT_GENERIC_STAGE_PLAN_ABI_VERSION ||
        plan->struct_size < FRT_GENERIC_STAGE_PLAN_EXT_V1_SIZE ||
        !plan->stages || plan->n_stages == 0) return -1;
    *out = plan;
    return 0;
}

}  // namespace

extern "C" frt_model_runtime_v1* frt_model_runtime_override_verbs(
        const frt_model_runtime_v1* in,
        const frt_model_runtime_verbs* verbs, void* verbs_self,
        void* owner, void (*retain_owner)(void*),
        void (*release_owner)(void*)) {
    if (!valid_model_runtime(in)) return nullptr;
    if (!valid_staged_verbs(in->ports, in->n_ports, verbs)) return nullptr;
    const frt_generic_stage_plan_ext_v1* generic_plan = nullptr;
    const int generic_rc = get_generic_plan(in, &generic_plan);
    if (generic_rc != 0 && generic_rc != -3) return nullptr;
    if (!in->n_stages && !generic_plan) return nullptr;  /* step-only */
    if (generic_plan) {
        for (uint64_t i = 0; i < generic_plan->n_stages; ++i)
            if (generic_plan->stages[i].executor_kind !=
                FRT_GENERIC_STAGE_OPAQUE) return nullptr;
    }

    auto* o = new VerbOverride();
    o->base = in;
    o->owner = owner;
    o->release_owner = release_owner;

    in->retain(in->owner);
    if (retain_owner) retain_owner(owner);

    frt_model_runtime_v1& m = o->model;
    m.abi_version = FRT_MODEL_RUNTIME_ABI_VERSION;
    m.struct_size = (uint32_t)sizeof(frt_model_runtime_v1);
    m.exp = in->exp;
    m.ports = in->ports;     m.n_ports = in->n_ports;
    m.stages = in->stages;   m.n_stages = in->n_stages;
    copy_verbs(&m, verbs, verbs_self);
    o->target_verbs = m.verbs;
    o->target_self = verbs_self;
    o->forward_base_error = generic_plan != nullptr;
    m.verbs.set_input = override_set_input;
    m.verbs.get_output = override_get_output;
    m.verbs.prepare = override_prepare;
    m.verbs.step = override_step;
    m.verbs.last_error = override_last_error;
    m.self = o;
    m.owner = o;
    m.retain = override_retain;
    m.release = override_release;
    m.query_extension = override_query_extensions;
    return &o->model;
}

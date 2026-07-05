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
    if (!b || (n_after && !after)) return -1;
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

extern "C" frt_model_runtime_v1* frt_runtime_builder_finish_model(
        frt_runtime_builder b,
        const frt_model_runtime_verbs* verbs, void* verbs_self,
        void* owner, void (*retain_owner)(void*),
        void (*release_owner)(void*)) {
    if (!b) return nullptr;
    Holder* h = b->h;
    frt_rt::finish_export_into(h, b, owner, retain_owner, release_owner);

    frt_model_runtime_v1& m = h->model;
    m.abi_version = FRT_MODEL_RUNTIME_ABI_VERSION;
    m.struct_size = (uint32_t)sizeof(frt_model_runtime_v1);
    m.exp = &h->exp;
    m.ports = h->ports.data();   m.n_ports = h->ports.size();
    m.stages = h->stages.data(); m.n_stages = h->stages.size();
    copy_verbs(&m, verbs, verbs_self);
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
    if (!exp || exp->abi_version != FRT_RUNTIME_ABI_VERSION ||
        exp->struct_size < sizeof(frt_runtime_export_v1) ||
        !exp->retain || !exp->release) return nullptr;
    if ((n_ports && !ports) || (n_stages && !stages)) return nullptr;
    for (uint64_t i = 0; i < n_ports; ++i)
        if (!valid_port_args(ports[i].name, ports[i].direction,
                             ports[i].update, ports[i].shape, ports[i].rank))
            return nullptr;
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
    frt_model_runtime_v1 model{};
};

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
        m->struct_size < sizeof(frt_model_runtime_v1)) return false;
    if (!m->exp || !m->retain || !m->release) return false;
    if ((m->n_ports && !m->ports) || (m->n_stages && !m->stages)) return false;
    return true;
}

}  // namespace

extern "C" frt_model_runtime_v1* frt_model_runtime_override_verbs(
        const frt_model_runtime_v1* in,
        const frt_model_runtime_verbs* verbs, void* verbs_self,
        void* owner, void (*retain_owner)(void*),
        void (*release_owner)(void*)) {
    if (!valid_model_runtime(in)) return nullptr;

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
    m.owner = o;
    m.retain = override_retain;
    m.release = override_release;
    return &o->model;
}

/* runtime_export.cpp — builder + export lifetime for flashrt/runtime.h.
 *
 * Zero backend dependency on purpose: the builder only records handles and
 * strings, computes the identity fingerprint, and flattens everything into a
 * single refcounted export object. It never calls into the exec library or
 * the GPU — capture/allocation intelligence stays with the producer.
 */
#include "flashrt/runtime.h"

#include <atomic>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

namespace {

/* One block that owns every array/string the export points into. Freed when
 * the reference count drops to zero. */
struct Holder {
    std::atomic<int> refs{1};
    void* user_owner = nullptr;
    void (*user_release)(void*) = nullptr;

    /* std::deque: element addresses are stable under push_back, so the
     * descriptors can point at .c_str() / .data() safely. */
    std::deque<std::string> names;
    std::deque<std::vector<frt_shape_key>> key_arrays;
    std::string identity;
    std::string manifest;
    bool has_manifest = false;

    std::vector<frt_runtime_stream_desc> streams;
    std::vector<frt_runtime_graph_desc>  graphs;
    std::vector<frt_runtime_buffer_desc> buffers;
    std::vector<frt_runtime_region_desc> regions;

    frt_runtime_export_v1 exp{};
};

extern "C" void holder_retain(void* owner) {
    static_cast<Holder*>(owner)->refs.fetch_add(1, std::memory_order_relaxed);
}

extern "C" void holder_release(void* owner) {
    Holder* h = static_cast<Holder*>(owner);
    if (h->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (h->user_release) h->user_release(h->user_owner);
        delete h;
    }
}

const char* stored(Holder* h, const char* s) {
    h->names.emplace_back(s ? s : "");
    return h->names.back().c_str();
}

}  // namespace

struct frt_runtime_builder_s {
    frt_ctx ctx = nullptr;
    Holder* h = nullptr;  /* built up in place; adopted by finish */
    std::string identity_pairs;
};

extern "C" frt_runtime_builder frt_runtime_builder_create(frt_ctx ctx) {
    if (!ctx) return nullptr;
    auto* b = new frt_runtime_builder_s();
    b->ctx = ctx;
    b->h = new Holder();
    return b;
}

extern "C" int frt_runtime_builder_add_stream(frt_runtime_builder b,
                                              const char* name, int stream_id,
                                              int priority, void* native_handle) {
    if (!b || !name || stream_id < 0) return -1;
    frt_runtime_stream_desc d{};
    d.name = stored(b->h, name);
    d.stream_id = stream_id;
    d.priority = priority;
    d.native_handle = native_handle;
    b->h->streams.push_back(d);
    return 0;
}

extern "C" int frt_runtime_builder_add_graph(frt_runtime_builder b,
                                             const char* name, frt_graph g,
                                             frt_shape_key default_key,
                                             const frt_shape_key* keys,
                                             uint64_t n_keys, int stream_id) {
    if (!b || !name || !g || (n_keys && !keys)) return -1;
    b->h->key_arrays.emplace_back(keys, keys + n_keys);
    frt_runtime_graph_desc d{};
    d.name = stored(b->h, name);
    d.handle = g;
    d.default_key = default_key;
    d.keys = b->h->key_arrays.back().data();
    d.n_keys = n_keys;
    d.stream_id = stream_id;
    b->h->graphs.push_back(d);
    return 0;
}

extern "C" int frt_runtime_builder_add_buffer(frt_runtime_builder b,
                                              const char* name, frt_buffer buf,
                                              uint64_t bytes, uint32_t role) {
    if (!b || !name || !buf) return -1;
    frt_runtime_buffer_desc d{};
    d.name = stored(b->h, name);
    d.handle = buf;
    d.bytes = bytes;
    d.role = role;
    b->h->buffers.push_back(d);
    return 0;
}

extern "C" int frt_runtime_builder_add_region(frt_runtime_builder b,
                                              const char* name, frt_buffer buf,
                                              uint64_t offset, uint64_t bytes,
                                              uint32_t flags) {
    if (!b || !name || !buf || !bytes) return -1;
    frt_runtime_region_desc d{};
    d.name = stored(b->h, name);
    d.buffer = buf;
    d.offset = offset;
    d.bytes = bytes;
    d.flags = flags;
    b->h->regions.push_back(d);
    return 0;
}

extern "C" int frt_runtime_builder_add_identity(frt_runtime_builder b,
                                                const char* key,
                                                const char* value) {
    if (!b || !key || !value) return -1;
    b->identity_pairs += key;
    b->identity_pairs += '=';
    b->identity_pairs += value;
    b->identity_pairs += '\n';
    return 0;
}

extern "C" int frt_runtime_builder_set_manifest(frt_runtime_builder b,
                                                const char* json) {
    if (!b || !json) return -1;
    b->h->manifest = json;
    b->h->has_manifest = true;
    return 0;
}

extern "C" uint64_t frt_runtime_fingerprint(const void* data, size_t len) {
    /* FNV-1a 64 — deterministic, dependency-free. An identity guard against
     * accidental mismatch, not an adversarial hash. */
    const unsigned char* p = static_cast<const unsigned char*>(data);
    uint64_t hash = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < len; ++i) {
        hash ^= p[i];
        hash *= 0x100000001b3ull;
    }
    return hash;
}

extern "C" frt_runtime_export_v1* frt_runtime_builder_finish(
        frt_runtime_builder b, void* owner,
        void (*retain_owner)(void*), void (*release_owner)(void*)) {
    if (!b) return nullptr;
    Holder* h = b->h;

    /* Canonical identity: version header, producer pairs (insertion order),
     * graph names, then the full region layout. Restore matches regions by
     * position, so name/offset/bytes/flags/order are all identity. */
    std::string id = "frt-runtime-identity-v1\n";
    id += b->identity_pairs;
    for (const auto& g : h->graphs) {
        id += "graph:";
        id += g.name;
        id += '\n';
    }
    char line[160];
    for (size_t i = 0; i < h->regions.size(); ++i) {
        const auto& r = h->regions[i];
        std::snprintf(line, sizeof(line), "region:%zu:%s:%llu:%llu:%u\n",
                      i, r.name,
                      (unsigned long long)r.offset,
                      (unsigned long long)r.bytes, r.flags);
        id += line;
    }
    h->identity = std::move(id);

    h->user_owner = owner;
    h->user_release = release_owner;
    if (retain_owner) retain_owner(owner);

    frt_runtime_export_v1& e = h->exp;
    e.abi_version = FRT_RUNTIME_ABI_VERSION;
    e.struct_size = (uint32_t)sizeof(frt_runtime_export_v1);
    e.ctx = b->ctx;
    e.streams = h->streams.data();          e.n_streams = h->streams.size();
    e.graphs = h->graphs.data();            e.n_graphs = h->graphs.size();
    e.buffers = h->buffers.data();          e.n_buffers = h->buffers.size();
    e.capsule_regions = h->regions.data();  e.n_capsule_regions = h->regions.size();
    e.identity = h->identity.c_str();
    e.fingerprint = frt_runtime_fingerprint(h->identity.data(), h->identity.size());
    e.manifest_json = h->has_manifest ? h->manifest.c_str() : nullptr;
    e.owner = h;
    e.retain = holder_retain;
    e.release = holder_release;

    delete b;  /* h lives on inside the export */
    return &h->exp;
}

/* FlashRT runtime — pybind module (`_flashrt_runtime`).
 *
 * Setup/dev bridge only: lets the Python frontend (phase-1 producer) assemble
 * an frt_runtime_export_v1 from raw exec handles. The struct itself is the
 * deployment surface — consumers link nothing from this module.
 *
 * Handles cross as integers (uintptr). This module deliberately does NOT
 * import the exec pybind types, so the two dev modules stay decoupled; the
 * exec wrappers expose .raw() for exactly this hand-off.
 *
 * Ownership: finish(owner) boxes the Python owner in a heap py::object whose
 * destruction is the export's release path. Release acquires the GIL first,
 * so a native consumer may drop its reference from any thread.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "flashrt/runtime.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

void release_py_owner(void* owner) {
    py::gil_scoped_acquire gil;
    delete static_cast<py::object*>(owner);
}

void check(int rc, const char* what) {
    if (rc < 0) throw std::runtime_error(std::string(what) + " failed: rc=" + std::to_string(rc));
}

struct PyRtBuilder {
    frt_runtime_builder b;

    explicit PyRtBuilder(std::uintptr_t ctx_raw) {
        b = frt_runtime_builder_create(reinterpret_cast<frt_ctx>(ctx_raw));
        if (!b) throw std::runtime_error("frt_runtime_builder_create failed (null ctx?)");
    }
    ~PyRtBuilder() {
        /* A never-finished builder leaks its holder by design tradeoff: the
         * builder is consumed by finish(); reaching here without finish() is
         * a setup-path error, not a hot-path concern. */
    }

    void need() const {
        if (!b) throw std::runtime_error("builder already finished");
    }

    std::uintptr_t finish(py::object owner) {
        need();
        auto* boxed = new py::object(std::move(owner));
        frt_runtime_export_v1* e = frt_runtime_builder_finish(
            b, boxed, /*retain_owner=*/nullptr, &release_py_owner);
        b = nullptr;
        if (!e) { release_py_owner(boxed); throw std::runtime_error("builder_finish failed"); }
        return reinterpret_cast<std::uintptr_t>(e);
    }
};

frt_runtime_export_v1* as_export(std::uintptr_t p) {
    auto* e = reinterpret_cast<frt_runtime_export_v1*>(p);
    if (!e || e->abi_version != FRT_RUNTIME_ABI_VERSION ||
        e->struct_size != sizeof(frt_runtime_export_v1))
        throw std::runtime_error("not a valid frt_runtime_export_v1 pointer");
    return e;
}

}  // namespace

PYBIND11_MODULE(_flashrt_runtime, m) {
    m.doc() = "FlashRT runtime-export ABI (setup/dev binding)";

    m.attr("ABI_VERSION") = FRT_RUNTIME_ABI_VERSION;
    m.attr("ROLE_INPUT") = (unsigned)FRT_RT_ROLE_INPUT;
    m.attr("ROLE_OUTPUT") = (unsigned)FRT_RT_ROLE_OUTPUT;
    m.attr("ROLE_STATE") = (unsigned)FRT_RT_ROLE_STATE;
    m.attr("ROLE_SCRATCH") = (unsigned)FRT_RT_ROLE_SCRATCH;
    m.attr("REGION_SNAPSHOT") = (unsigned)FRT_RT_REGION_SNAPSHOT;
    m.attr("REGION_RESTORE") = (unsigned)FRT_RT_REGION_RESTORE;

    py::class_<PyRtBuilder>(m, "Builder")
        .def(py::init<std::uintptr_t>(), py::arg("ctx_raw"))
        .def("add_stream", [](PyRtBuilder& s, const std::string& name, int stream_id,
                              int priority, std::uintptr_t native_handle) {
            s.need();
            check(frt_runtime_builder_add_stream(s.b, name.c_str(), stream_id, priority,
                                                 reinterpret_cast<void*>(native_handle)),
                  "add_stream");
        }, py::arg("name"), py::arg("stream_id"), py::arg("priority") = 0,
           py::arg("native_handle") = 0)
        .def("add_graph", [](PyRtBuilder& s, const std::string& name, std::uintptr_t graph_raw,
                             std::uint64_t default_key, const std::vector<std::uint64_t>& keys,
                             int stream_id) {
            s.need();
            check(frt_runtime_builder_add_graph(s.b, name.c_str(),
                                                reinterpret_cast<frt_graph>(graph_raw),
                                                default_key, keys.data(), keys.size(),
                                                stream_id),
                  "add_graph");
        }, py::arg("name"), py::arg("graph_raw"), py::arg("default_key") = 0,
           py::arg("keys") = std::vector<std::uint64_t>{}, py::arg("stream_id") = 0)
        .def("add_buffer", [](PyRtBuilder& s, const std::string& name,
                              std::uintptr_t buffer_raw, std::uint64_t bytes, unsigned role) {
            s.need();
            check(frt_runtime_builder_add_buffer(s.b, name.c_str(),
                                                 reinterpret_cast<frt_buffer>(buffer_raw),
                                                 bytes, role),
                  "add_buffer");
        }, py::arg("name"), py::arg("buffer_raw"), py::arg("bytes"), py::arg("role"))
        .def("add_region", [](PyRtBuilder& s, const std::string& name,
                              std::uintptr_t buffer_raw, std::uint64_t offset,
                              std::uint64_t bytes, unsigned flags) {
            s.need();
            check(frt_runtime_builder_add_region(s.b, name.c_str(),
                                                 reinterpret_cast<frt_buffer>(buffer_raw),
                                                 offset, bytes, flags),
                  "add_region");
        }, py::arg("name"), py::arg("buffer_raw"), py::arg("offset"), py::arg("bytes"),
           py::arg("flags"))
        .def("add_identity", [](PyRtBuilder& s, const std::string& k, const std::string& v) {
            s.need();
            check(frt_runtime_builder_add_identity(s.b, k.c_str(), v.c_str()), "add_identity");
        })
        .def("set_manifest", [](PyRtBuilder& s, const std::string& json) {
            s.need();
            check(frt_runtime_builder_set_manifest(s.b, json.c_str()), "set_manifest");
        })
        .def("finish", &PyRtBuilder::finish, py::arg("owner"),
             "Consume the builder; returns the export pointer (uintptr). The export "
             "holds one reference; hand the pointer to a native consumer, which must "
             "retain/release per the ABI.");

    /* Introspection over a raw export pointer (tests / mismatch tooling). */
    m.def("export_fingerprint", [](std::uintptr_t p) { return as_export(p)->fingerprint; });
    m.def("export_identity", [](std::uintptr_t p) { return std::string(as_export(p)->identity); });
    m.def("export_manifest", [](std::uintptr_t p) {
        const char* j = as_export(p)->manifest_json;
        return j ? py::object(py::str(j)) : py::object(py::none());
    });
    m.def("export_counts", [](std::uintptr_t p) {
        auto* e = as_export(p);
        py::dict d;
        d["streams"] = e->n_streams; d["graphs"] = e->n_graphs;
        d["buffers"] = e->n_buffers; d["capsule_regions"] = e->n_capsule_regions;
        return d;
    });
    m.def("export_retain", [](std::uintptr_t p) { auto* e = as_export(p); e->retain(e->owner); });
    m.def("export_release", [](std::uintptr_t p) {
        auto* e = as_export(p);
        /* The release path may destroy a boxed py::object; it re-acquires the
         * GIL itself, so drop it here to avoid a deadlock-by-convention. */
        py::gil_scoped_release nogil;
        e->release(e->owner);
    });
    m.def("fingerprint", [](py::bytes data) {
        std::string s = data;
        return frt_runtime_fingerprint(s.data(), s.size());
    }, "Recompute the identity hash (FNV-1a 64) — the one hashing rule.");
}

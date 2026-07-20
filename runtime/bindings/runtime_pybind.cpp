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
#include "flashrt/model_runtime.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

void release_py_owner(void* owner) {
    py::gil_scoped_acquire gil;
    delete static_cast<py::object*>(owner);
}

/* Model-runtime verbs implemented by Python callables. Trampolines acquire
 * the GIL (a native consumer calls these fn pointers from its own threads)
 * and translate exceptions into negative status + last_error. */
struct PyVerbs;

struct PyOpaqueRunner {
    py::object callback;
    PyVerbs* error_sink = nullptr;
};

struct PyVerbs {
    py::object set_input, get_output, prepare, step;
    std::string last_error;
    py::object owner;   /* the producer object the export anchors */
    std::unique_ptr<PyOpaqueRunner> opaque_runner;
};

void release_py_verbs(void* p) {
    py::gil_scoped_acquire gil;
    delete static_cast<PyVerbs*>(p);
}

int verb_set_input(void* self, uint32_t port, const void* data,
                   uint64_t bytes, int stream) {
    auto* v = static_cast<PyVerbs*>(self);
    py::gil_scoped_acquire gil;
    if (!v->set_input || v->set_input.is_none()) {
        v->last_error = "set_input is not provided by this producer";
        return -3;
    }
    try {
        py::bytes payload(static_cast<const char*>(data),
                          static_cast<size_t>(bytes));
        return py::cast<int>(v->set_input(port, payload, stream));
    } catch (const std::exception& e) {
        v->last_error = e.what();
        return -1;
    }
}

int verb_get_output(void* self, uint32_t port, void* out, uint64_t capacity,
                    uint64_t* written, int stream) {
    auto* v = static_cast<PyVerbs*>(self);
    py::gil_scoped_acquire gil;
    if (!v->get_output || v->get_output.is_none()) {
        v->last_error = "get_output is not provided by this producer";
        return -3;
    }
    try {
        py::bytes result = v->get_output(port, stream);
        std::string s = result;
        if (written) *written = s.size();
        if (capacity < s.size()) {
            v->last_error = "output buffer is too small";
            return -5;
        }
        std::memcpy(out, s.data(), s.size());
        return 0;
    } catch (const std::exception& e) {
        v->last_error = e.what();
        return -1;
    }
}

int verb_prepare(void* self, uint32_t graph, frt_shape_key key) {
    auto* v = static_cast<PyVerbs*>(self);
    py::gil_scoped_acquire gil;
    if (!v->prepare || v->prepare.is_none()) {
        v->last_error = "prepare is not provided by this producer";
        return -3;
    }
    try {
        return py::cast<int>(v->prepare(graph, (std::uint64_t)key));
    } catch (const std::exception& e) {
        v->last_error = e.what();
        return -1;
    }
}

int verb_step(void* self) {
    auto* v = static_cast<PyVerbs*>(self);
    py::gil_scoped_acquire gil;
    if (!v->step || v->step.is_none()) {
        v->last_error = "step is not provided by this producer";
        return -3;
    }
    try {
        return py::cast<int>(v->step());
    } catch (const std::exception& e) {
        v->last_error = e.what();
        return -1;
    }
}

int verb_run_opaque(void* self, uint32_t executor_ref) {
    auto* runner = static_cast<PyOpaqueRunner*>(self);
    py::gil_scoped_acquire gil;
    try {
        return py::cast<int>(runner->callback(executor_ref));
    } catch (const std::exception& e) {
        if (runner->error_sink) runner->error_sink->last_error = e.what();
        return -1;
    }
}

const char* verb_last_error(void* self) {
    return static_cast<PyVerbs*>(self)->last_error.c_str();
}

void check(int rc, const char* what) {
    if (rc < 0) throw std::runtime_error(std::string(what) + " failed: rc=" + std::to_string(rc));
}

struct PyRtBuilder {
    frt_runtime_builder b;
    std::unique_ptr<PyOpaqueRunner> opaque_runner;

    explicit PyRtBuilder(std::uintptr_t ctx_raw) {
        b = frt_runtime_builder_create(reinterpret_cast<frt_ctx>(ctx_raw));
        if (!b) throw std::runtime_error("frt_runtime_builder_create failed (null ctx?)");
    }
    PyRtBuilder() {
        b = frt_model_runtime_builder_create_metadata();
        if (!b) throw std::runtime_error(
            "frt_model_runtime_builder_create_metadata failed");
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

    std::uintptr_t finish_model(py::object owner, py::object set_input,
                                py::object get_output, py::object prepare,
                                py::object step) {
        need();
        auto* pv = new PyVerbs();
        pv->set_input = std::move(set_input);
        pv->get_output = std::move(get_output);
        pv->prepare = std::move(prepare);
        pv->step = std::move(step);
        pv->owner = std::move(owner);
        if (opaque_runner) opaque_runner->error_sink = pv;

        frt_model_runtime_verbs verbs{};
        verbs.struct_size = sizeof(verbs);
        verbs.set_input = pv->set_input.is_none() ? nullptr : &verb_set_input;
        verbs.get_output =
            pv->get_output.is_none() ? nullptr : &verb_get_output;
        verbs.prepare = pv->prepare.is_none() ? nullptr : &verb_prepare;
        verbs.step = pv->step.is_none() ? nullptr : &verb_step;
        verbs.last_error = &verb_last_error;

        frt_model_runtime_v1* mr = frt_runtime_builder_finish_model(
            b, &verbs, pv, pv, /*retain_owner=*/nullptr, &release_py_verbs);
        if (!mr) {
            if (opaque_runner) opaque_runner->error_sink = nullptr;
            release_py_verbs(pv);
            throw std::runtime_error("finish_model failed");
        }
        pv->opaque_runner = std::move(opaque_runner);
        b = nullptr;
        return reinterpret_cast<std::uintptr_t>(mr);
    }
};

const frt_generic_stage_plan_ext_v1* generic_plan(frt_model_runtime_v1* model) {
    if (model->struct_size < FRT_MODEL_RUNTIME_V1_QUERY_EXTENSION_SIZE ||
        !model->query_extension) return nullptr;
    const void* extension = nullptr;
    if (model->query_extension(model, FRT_EXT_GENERIC_STAGE_PLAN_V1, 1,
                               &extension) != 0) return nullptr;
    auto* plan = static_cast<const frt_generic_stage_plan_ext_v1*>(extension);
    if (!plan || plan->abi_version < FRT_GENERIC_STAGE_PLAN_ABI_VERSION ||
        plan->struct_size < FRT_GENERIC_STAGE_PLAN_EXT_V1_SIZE)
        throw std::runtime_error("invalid generic stage-plan extension");
    return plan;
}

frt_runtime_export_v1* as_export(std::uintptr_t p) {
    auto* e = reinterpret_cast<frt_runtime_export_v1*>(p);
    if (!e || e->abi_version != FRT_RUNTIME_ABI_VERSION ||
        e->struct_size != sizeof(frt_runtime_export_v1))
        throw std::runtime_error("not a valid frt_runtime_export_v1 pointer");
    return e;
}

frt_model_runtime_v1* as_model(std::uintptr_t p) {
    auto* m = reinterpret_cast<frt_model_runtime_v1*>(p);
    if (!m || m->abi_version != FRT_MODEL_RUNTIME_ABI_VERSION ||
        m->struct_size < FRT_MODEL_RUNTIME_V1_BASE_SIZE)
        throw std::runtime_error("not a valid frt_model_runtime_v1 pointer");
    return m;
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

    m.attr("MODEL_ABI_VERSION") = FRT_MODEL_RUNTIME_ABI_VERSION;
    m.attr("MODEL_V1_BASE_SIZE") = FRT_MODEL_RUNTIME_V1_BASE_SIZE;
    m.attr("MODEL_V1_QUERY_EXTENSION_SIZE") =
        FRT_MODEL_RUNTIME_V1_QUERY_EXTENSION_SIZE;
    m.attr("MOD_TENSOR") = (unsigned)FRT_RT_MOD_TENSOR;
    m.attr("MOD_IMAGE") = (unsigned)FRT_RT_MOD_IMAGE;
    m.attr("MOD_TEXT") = (unsigned)FRT_RT_MOD_TEXT;
    m.attr("MOD_STATE") = (unsigned)FRT_RT_MOD_STATE;
    m.attr("MOD_ACTION") = (unsigned)FRT_RT_MOD_ACTION;
    m.attr("MOD_AUDIO") = (unsigned)FRT_RT_MOD_AUDIO;
    m.attr("MOD_DEPTH") = (unsigned)FRT_RT_MOD_DEPTH;
    m.attr("MOD_FORCE") = (unsigned)FRT_RT_MOD_FORCE;
    m.attr("DTYPE_U8") = (unsigned)FRT_RT_DTYPE_U8;
    m.attr("DTYPE_F32") = (unsigned)FRT_RT_DTYPE_F32;
    m.attr("DTYPE_F16") = (unsigned)FRT_RT_DTYPE_F16;
    m.attr("DTYPE_BF16") = (unsigned)FRT_RT_DTYPE_BF16;
    m.attr("DTYPE_I32") = (unsigned)FRT_RT_DTYPE_I32;
    m.attr("DTYPE_I64") = (unsigned)FRT_RT_DTYPE_I64;
    m.attr("LAYOUT_FLAT") = (unsigned)FRT_RT_LAYOUT_FLAT;
    m.attr("LAYOUT_HWC") = (unsigned)FRT_RT_LAYOUT_HWC;
    m.attr("LAYOUT_NHWC") = (unsigned)FRT_RT_LAYOUT_NHWC;
    m.attr("LAYOUT_CHW") = (unsigned)FRT_RT_LAYOUT_CHW;
    m.attr("LAYOUT_NCHW") = (unsigned)FRT_RT_LAYOUT_NCHW;
    m.attr("PORT_IN") = (unsigned)FRT_RT_PORT_IN;
    m.attr("PORT_OUT") = (unsigned)FRT_RT_PORT_OUT;
    m.attr("PORT_SWAP") = (unsigned)FRT_RT_PORT_SWAP;
    m.attr("PORT_STAGED") = (unsigned)FRT_RT_PORT_STAGED;
    m.attr("PORT_SETUP") = (unsigned)FRT_RT_PORT_SETUP;
    m.attr("GENERIC_STAGE_GRAPH") = (unsigned)FRT_GENERIC_STAGE_GRAPH;
    m.attr("GENERIC_STAGE_OPAQUE") = (unsigned)FRT_GENERIC_STAGE_OPAQUE;
    m.attr("GENERIC_STAGE_NAME_MAX_BYTES") =
        (unsigned)FRT_GENERIC_STAGE_NAME_MAX_BYTES;
    m.attr("GENERIC_STAGE_DESC_V1_SIZE") =
        (unsigned)sizeof(frt_generic_stage_desc_v1);
    m.attr("GENERIC_STAGE_PLAN_EXT_V1_SIZE") =
        (unsigned)FRT_GENERIC_STAGE_PLAN_EXT_V1_SIZE;
    m.attr("EXT_GENERIC_STAGE_PLAN_V1") = FRT_EXT_GENERIC_STAGE_PLAN_V1;

    py::class_<PyRtBuilder>(m, "Builder")
        .def(py::init<std::uintptr_t>(), py::arg("ctx_raw"))
        .def_static("metadata", []() { return std::make_unique<PyRtBuilder>(); })
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
        .def("add_port", [](PyRtBuilder& s, const std::string& name, unsigned modality,
                            unsigned dtype, unsigned layout, unsigned direction,
                            unsigned update, unsigned required,
                            const std::vector<std::int64_t>& shape,
                            unsigned cadence_hint_hz, std::uintptr_t buffer_raw,
                            std::uint64_t offset, std::uint64_t bytes) {
            s.need();
            check(frt_runtime_builder_add_port(
                      s.b, name.c_str(), modality, dtype, layout, direction,
                      update, required, shape.data(), (uint32_t)shape.size(),
                      cadence_hint_hz, reinterpret_cast<frt_buffer>(buffer_raw),
                      offset, bytes),
                  "add_port");
        }, py::arg("name"), py::arg("modality"), py::arg("dtype"),
           py::arg("layout"), py::arg("direction"), py::arg("update"),
           py::arg("required") = 0,
           py::arg("shape") = std::vector<std::int64_t>{},
           py::arg("cadence_hint_hz") = 0, py::arg("buffer_raw") = 0,
           py::arg("offset") = 0, py::arg("bytes") = 0)
        .def("add_stage", [](PyRtBuilder& s, unsigned graph,
                             const std::vector<unsigned>& after) {
            s.need();
            check(frt_runtime_builder_add_stage(s.b, graph, after.data(),
                                                (uint32_t)after.size()),
                  "add_stage");
        }, py::arg("graph"), py::arg("after") = std::vector<unsigned>{})
        .def("add_generic_stage", [](PyRtBuilder& s, const std::string& name,
                                     unsigned executor_kind,
                                     unsigned executor_ref,
                                     const std::vector<unsigned>& after) {
            s.need();
            check(frt_runtime_builder_add_generic_stage(
                      s.b, name.c_str(), executor_kind, executor_ref,
                      after.data(), (uint32_t)after.size()),
                  "add_generic_stage");
        }, py::arg("name"), py::arg("executor_kind"),
           py::arg("executor_ref"),
           py::arg("after") = std::vector<unsigned>{})
        .def("set_generic_stage_runner", [](PyRtBuilder& s,
                                             py::object callback) {
            s.need();
            if (callback.is_none())
                throw std::invalid_argument("opaque runner must be callable");
            auto runner = std::make_unique<PyOpaqueRunner>();
            runner->callback = std::move(callback);
            check(frt_runtime_builder_set_generic_stage_runner(
                      s.b, runner.get(), verb_run_opaque),
                  "set_generic_stage_runner");
            s.opaque_runner = std::move(runner);
        }, py::arg("callback"))
        .def("finish", &PyRtBuilder::finish, py::arg("owner"),
             "Consume the builder; returns the export pointer (uintptr). The export "
             "holds one reference; hand the pointer to a native consumer, which must "
             "retain/release per the ABI.")
        .def("finish_model", &PyRtBuilder::finish_model, py::arg("owner"),
             py::arg("set_input") = py::none(), py::arg("get_output") = py::none(),
             py::arg("prepare") = py::none(), py::arg("step") = py::none(),
             "Consume the builder; returns the frt_model_runtime_v1 pointer "
             "(uintptr). Verb callables run under trampolines that acquire the "
             "GIL, so a native consumer may call them from any thread. "
             "set_input(port, payload: bytes, stream) -> int; "
             "get_output(port, stream) -> bytes; prepare(graph, key) -> int; "
             "step() -> int.");

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

    /* Introspection over a raw model-runtime pointer. */
    m.def("model_export_ptr", [](std::uintptr_t p) {
        return reinterpret_cast<std::uintptr_t>(as_model(p)->exp);
    });
    m.def("model_ports", [](std::uintptr_t p) {
        auto* mr = as_model(p);
        py::list out;
        for (std::uint64_t i = 0; i < mr->n_ports; ++i) {
            const frt_runtime_port_desc& d = mr->ports[i];
            py::dict e;
            e["name"] = std::string(d.name);
            e["modality"] = d.modality; e["dtype"] = d.dtype;
            e["layout"] = d.layout; e["direction"] = d.direction;
            e["update"] = d.update; e["required"] = d.required;
            e["shape"] = std::vector<std::int64_t>(d.shape, d.shape + d.rank);
            e["cadence_hint_hz"] = d.cadence_hint_hz;
            e["buffer"] = reinterpret_cast<std::uintptr_t>(d.buffer);
            e["offset"] = d.offset; e["bytes"] = d.bytes;
            out.append(e);
        }
        return out;
    });
    m.def("model_stages", [](std::uintptr_t p) {
        auto* mr = as_model(p);
        py::list out;
        for (std::uint64_t i = 0; i < mr->n_stages; ++i) {
            const frt_runtime_stage_desc& d = mr->stages[i];
            py::dict e;
            e["graph"] = d.graph;
            e["after"] = std::vector<unsigned>(d.after, d.after + d.n_after);
            out.append(e);
        }
        return out;
    });
    m.def("model_generic_stages", [](std::uintptr_t p) {
        auto* plan = generic_plan(as_model(p));
        py::list out;
        if (!plan) return out;
        for (std::uint64_t i = 0; i < plan->n_stages; ++i) {
            const frt_generic_stage_desc_v1& d = plan->stages[i];
            py::dict e;
            e["name"] = std::string(d.name);
            e["executor_kind"] = d.executor_kind;
            e["executor_ref"] = d.executor_ref;
            e["after"] = std::vector<unsigned>(d.after, d.after + d.n_after);
            out.append(e);
        }
        return out;
    });
    m.def("model_run_opaque", [](std::uintptr_t p, unsigned executor_ref) {
        auto* plan = generic_plan(as_model(p));
        if (!plan || !plan->run_opaque)
            throw std::runtime_error("model has no OPAQUE executor");
        py::gil_scoped_release nogil;
        return plan->run_opaque(plan->stage_self, executor_ref);
    });
    m.def("model_retain", [](std::uintptr_t p) { auto* mr = as_model(p); mr->retain(mr->owner); });
    m.def("model_release", [](std::uintptr_t p) {
        auto* mr = as_model(p);
        py::gil_scoped_release nogil;   /* release re-acquires internally */
        mr->release(mr->owner);
    });
    /* Drive the verbs THROUGH the C fn pointers (tests exercise the same
     * entry a native consumer uses). */
    m.def("model_set_input", [](std::uintptr_t p, unsigned port, py::bytes data,
                                int stream) {
        auto* mr = as_model(p);
        std::string s = data;
        py::gil_scoped_release nogil;   /* trampoline re-acquires */
        return mr->verbs.set_input(mr->self, port, s.data(), s.size(), stream);
    }, py::arg("ptr"), py::arg("port"), py::arg("data"), py::arg("stream") = -1);
    m.def("model_get_output", [](std::uintptr_t p, unsigned port,
                                 std::uint64_t capacity, int stream) {
        auto* mr = as_model(p);
        std::string buf(capacity, '\0');
        std::uint64_t written = 0;
        int rc;
        {
            py::gil_scoped_release nogil;
            rc = mr->verbs.get_output(mr->self, port, buf.data(), capacity,
                                      &written, stream);
        }
        return py::make_tuple(rc, py::bytes(buf.data(),
                                            rc == 0 ? written : 0), written);
    }, py::arg("ptr"), py::arg("port"), py::arg("capacity"), py::arg("stream") = -1);
    m.def("model_step", [](std::uintptr_t p) {
        auto* mr = as_model(p);
        py::gil_scoped_release nogil;
        return mr->verbs.step(mr->self);
    });
    m.def("model_last_error", [](std::uintptr_t p) {
        auto* mr = as_model(p);
        return std::string(mr->verbs.last_error(mr->self));
    });
}

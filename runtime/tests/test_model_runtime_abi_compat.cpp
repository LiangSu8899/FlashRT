#include "flashrt/model_runtime.h"

#include <dlfcn.h>

#include <cstdio>

namespace {

using BaselineCreate = void* (*)(const frt_runtime_export_v1*, void*,
                                  void (*)(void*), void (*)(void*));
using BaselineDestroy = void (*)(void*);
using Consume = int (*)(const void*);

int failures = 0;
#define CHECK(condition, message) do {                                      \
    if (!(condition)) { std::printf("FAIL: %s\n", message); failures = 1; } \
} while (0)

void retain_noop(void*) {}
void release_noop(void*) {}
int step_noop(void*) { return 0; }
const char* no_error(void*) { return ""; }

void* open_local(const char* path) {
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) std::printf("dlopen failed: %s\n", dlerror());
    return handle;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    void* producer_dso = open_local(argv[1]);
    void* consumer_dso = open_local(argv[2]);
    CHECK(producer_dso && consumer_dso, "fixtures load with RTLD_LOCAL");
    if (!producer_dso || !consumer_dso) return 1;

    auto create = reinterpret_cast<BaselineCreate>(
        dlsym(producer_dso, "flashrt_model_runtime_v1_abi_baseline_create"));
    auto destroy = reinterpret_cast<BaselineDestroy>(
        dlsym(producer_dso, "flashrt_model_runtime_v1_abi_baseline_destroy"));
    auto prefix_consume = reinterpret_cast<Consume>(
        dlsym(consumer_dso, "flashrt_model_runtime_v1_prefix_consume"));
    auto exact_consume = reinterpret_cast<Consume>(
        dlsym(consumer_dso,
              "flashrt_model_runtime_v1_exact_size_consume"));
    CHECK(create && destroy && prefix_consume && exact_consume,
          "fixture symbols resolve through their own handles");

    frt_runtime_export_v1 fake_export{};
    fake_export.abi_version = FRT_RUNTIME_ABI_VERSION;
    fake_export.struct_size = sizeof(fake_export);
    void* baseline = create(&fake_export, nullptr, retain_noop, release_noop);
    auto* baseline_view = static_cast<frt_model_runtime_v1*>(baseline);
    CHECK(baseline_view &&
              baseline_view->struct_size == FRT_MODEL_RUNTIME_V1_BASE_SIZE,
          "baseline producer publishes exactly the required prefix");
    CHECK(prefix_consume(baseline) == 0 && exact_consume(baseline) == 0,
          "baseline producer is accepted by baseline consumers");
    CHECK(baseline_view->struct_size <
              FRT_MODEL_RUNTIME_V1_QUERY_EXTENSION_SIZE,
          "tail-aware consumer treats baseline extension as absent");

    frt_runtime_builder builder =
        frt_runtime_builder_create(reinterpret_cast<frt_ctx>(0x10));
    frt_runtime_builder_add_stream(builder, "main", 0, 0, nullptr);
    const frt_shape_key key = 0;
    frt_runtime_builder_add_graph(
        builder, "infer", reinterpret_cast<frt_graph>(0x20), 0, &key, 1, 0);
    frt_runtime_builder_add_stage(builder, 0, nullptr, 0);
    frt_runtime_builder_add_identity(builder, "fixture", "tail");
    frt_model_runtime_verbs verbs{};
    verbs.struct_size = sizeof(verbs);
    verbs.step = step_noop;
    verbs.last_error = no_error;
    frt_model_runtime_v1* tail = frt_runtime_builder_finish_model(
        builder, &verbs, nullptr, nullptr, nullptr, nullptr);
    CHECK(tail && tail->struct_size >=
                      FRT_MODEL_RUNTIME_V1_QUERY_EXTENSION_SIZE,
          "tail producer publishes the additive query field");
    CHECK(prefix_consume(tail) == 0,
          "prefix-compliant consumer ignores the additive tail");
    CHECK(exact_consume(tail) == -1,
          "archived exact-size consumer safely rejects and requires rebuild");
    const void* extension = reinterpret_cast<const void*>(0x1);
    CHECK(tail->query_extension(tail, UINT64_C(0xffff), 1, &extension) == -3 &&
              extension == nullptr,
          "tail-aware consumer calls the stable unsupported query stub");

    tail->release(tail->owner);
    destroy(baseline);
    dlclose(consumer_dso);
    dlclose(producer_dso);
    return failures;
}

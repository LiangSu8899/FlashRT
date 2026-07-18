#include "internal.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>

namespace {

std::atomic<bool> count_allocations{false};
std::atomic<std::size_t> allocation_count{0};

[[noreturn]] void fail(const char* expression, int line) {
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    std::abort();
}

#define CHECK(expression)                              \
    do {                                               \
        if (!(expression)) fail(#expression, __LINE__); \
    } while (false)

}  // namespace

void* operator new(std::size_t bytes) {
    if (count_allocations.load(std::memory_order_relaxed)) {
        allocation_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (void* pointer = std::malloc(bytes)) return pointer;
    throw std::bad_alloc();
}

void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

int main() {
    frt_graph_s graph;
    graph.lru = {1, 2, 3};

    allocation_count.store(0, std::memory_order_relaxed);
    count_allocations.store(true, std::memory_order_relaxed);
    for (int i = 0; i < 1000; ++i) graph.touch((i % 3) + 1);
    count_allocations.store(false, std::memory_order_relaxed);

    CHECK(allocation_count.load(std::memory_order_relaxed) == 0);
    CHECK(graph.lru.size() == 3);
    CHECK(graph.lru.back() == 1);

    allocation_count.store(0, std::memory_order_relaxed);
    count_allocations.store(true, std::memory_order_relaxed);
    graph.touch(4);
    count_allocations.store(false, std::memory_order_relaxed);
    CHECK(allocation_count.load(std::memory_order_relaxed) > 0);
    CHECK(graph.lru.size() == 4);
    CHECK(graph.lru.back() == 4);

    std::cout << "PASS - graph LRU touch allocation contract\n";
    return 0;
}

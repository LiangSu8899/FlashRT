#include "internal.h"

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <new>

namespace {

std::atomic<bool> count_allocations{false};
std::atomic<std::size_t> allocation_count{0};

}  // namespace

void* operator new(std::size_t bytes) {
    if (count_allocations.load(std::memory_order_relaxed)) {
        allocation_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (void* p = std::malloc(bytes)) return p;
    throw std::bad_alloc();
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

int main() {
    frt_graph_s graph;
    graph.lru = {1, 2, 3};

    allocation_count.store(0, std::memory_order_relaxed);
    count_allocations.store(true, std::memory_order_relaxed);
    for (int i = 0; i < 1000; ++i) graph.touch((i % 3) + 1);
    count_allocations.store(false, std::memory_order_relaxed);

    assert(allocation_count.load(std::memory_order_relaxed) == 0);
    assert(graph.lru.size() == 3);
    assert(graph.lru.back() == 1);

    allocation_count.store(0, std::memory_order_relaxed);
    count_allocations.store(true, std::memory_order_relaxed);
    graph.touch(4);
    count_allocations.store(false, std::memory_order_relaxed);
    assert(allocation_count.load(std::memory_order_relaxed) > 0);
    assert(graph.lru.back() == 4);
    return 0;
}

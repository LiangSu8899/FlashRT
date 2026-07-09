#include "flashrt/cpp/modalities/text.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using flashrt::modalities::DType;
using flashrt::modalities::EmbeddingGatherSpec;
using flashrt::modalities::Layout;
using flashrt::modalities::MemoryPlace;
using flashrt::modalities::Shape;
using flashrt::modalities::StatusCode;
using flashrt::modalities::TensorView;
using flashrt::modalities::bfloat16_to_float;
using flashrt::modalities::float_to_bfloat16;
using flashrt::modalities::gather_token_embeddings_cpu;

namespace {

void test_f32_embedding_gather() {
    std::vector<float> table = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };
    std::int32_t ids[] = {2, 0};
    std::vector<float> out(2 * 4, 0.0f);

    TensorView src{table.data(), static_cast<std::uint64_t>(table.size() * 4),
                   DType::kFloat32, MemoryPlace::kHost, Layout::kFlat,
                   Shape{3, 4}};
    TensorView dst{out.data(), static_cast<std::uint64_t>(out.size() * 4),
                   DType::kFloat32, MemoryPlace::kHost, Layout::kFlat,
                   Shape{2, 4}};
    EmbeddingGatherSpec spec{3, 4, 2.0f};
    auto st = gather_token_embeddings_cpu(spec, ids, 2, src, dst);
    assert(st.ok_status());
    const std::vector<float> expected = {
        18.0f, 20.0f, 22.0f, 24.0f,
        2.0f, 4.0f, 6.0f, 8.0f,
    };
    assert(out == expected);
}

void test_bf16_embedding_gather() {
    std::vector<std::uint16_t> table = {
        float_to_bfloat16(0.5f), float_to_bfloat16(-1.0f),
        float_to_bfloat16(2.0f), float_to_bfloat16(3.0f),
    };
    std::int32_t ids[] = {1};
    std::vector<std::uint16_t> out(2);

    TensorView src{table.data(), static_cast<std::uint64_t>(table.size() * 2),
                   DType::kBFloat16, MemoryPlace::kHost, Layout::kFlat,
                   Shape{2, 2}};
    TensorView dst{out.data(), static_cast<std::uint64_t>(out.size() * 2),
                   DType::kBFloat16, MemoryPlace::kHost, Layout::kFlat,
                   Shape{1, 2}};
    EmbeddingGatherSpec spec{2, 2, 1.5f};
    auto st = gather_token_embeddings_cpu(spec, ids, 1, src, dst);
    assert(st.ok_status());
    assert(std::fabs(bfloat16_to_float(out[0]) - 3.0f) < 0.01f);
    assert(std::fabs(bfloat16_to_float(out[1]) - 4.5f) < 0.01f);
}

void test_invalid_token_rejected() {
    std::vector<float> table(4, 0.0f);
    std::vector<float> out(2, 0.0f);
    std::int32_t ids[] = {2};
    TensorView src{table.data(), static_cast<std::uint64_t>(table.size() * 4),
                   DType::kFloat32, MemoryPlace::kHost, Layout::kFlat,
                   Shape{2, 2}};
    TensorView dst{out.data(), static_cast<std::uint64_t>(out.size() * 4),
                   DType::kFloat32, MemoryPlace::kHost, Layout::kFlat,
                   Shape{1, 2}};
    EmbeddingGatherSpec spec{2, 2, 1.0f};
    auto st = gather_token_embeddings_cpu(spec, ids, 1, src, dst);
    assert(!st.ok_status());
    assert(st.code == StatusCode::kInvalidArgument);
}

}  // namespace

int main() {
    test_f32_embedding_gather();
    test_bf16_embedding_gather();
    test_invalid_token_rejected();
    std::cout << "PASS - text modality contracts\n";
    return 0;
}

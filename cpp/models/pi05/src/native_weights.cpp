#include "flashrt/cpp/models/pi05/native_weights.h"

#include <initializer_list>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

using Requirement = NativeTensorRequirement;

void add(std::vector<Requirement>* out, const std::string& key,
         std::initializer_list<std::uint64_t> shape) {
    out->push_back(Requirement{key, shape});
}

std::vector<Requirement> build_requirements() {
    std::vector<Requirement> out;
    out.reserve(820);

    const std::string vision =
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model";
    add(&out, vision + ".embeddings.patch_embedding.weight", {1152, 3, 14, 14});
    add(&out, vision + ".embeddings.patch_embedding.bias", {1152});
    add(&out, vision + ".embeddings.position_embedding.weight", {256, 1152});
    for (int layer = 0; layer < 27; ++layer) {
        const std::string p = vision + ".encoder.layers." +
                              std::to_string(layer);
        for (const char* projection : {"q_proj", "k_proj", "v_proj",
                                       "out_proj"}) {
            add(&out, p + ".self_attn." + projection + ".weight",
                {1152, 1152});
            add(&out, p + ".self_attn." + projection + ".bias", {1152});
        }
        add(&out, p + ".mlp.fc1.weight", {4304, 1152});
        add(&out, p + ".mlp.fc1.bias", {4304});
        add(&out, p + ".mlp.fc2.weight", {1152, 4304});
        add(&out, p + ".mlp.fc2.bias", {1152});
        for (const char* norm : {"layer_norm1", "layer_norm2"}) {
            add(&out, p + "." + norm + ".weight", {1152});
            add(&out, p + "." + norm + ".bias", {1152});
        }
    }
    add(&out, vision + ".post_layernorm.weight", {1152});
    add(&out, vision + ".post_layernorm.bias", {1152});

    const std::string projector =
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear";
    add(&out, projector + ".weight", {2048, 1152});
    add(&out, projector + ".bias", {2048});

    const std::string encoder =
        "paligemma_with_expert.paligemma.model.language_model.layers.";
    for (int layer = 0; layer < 18; ++layer) {
        const std::string p = encoder + std::to_string(layer);
        add(&out, p + ".self_attn.q_proj.weight", {2048, 2048});
        add(&out, p + ".self_attn.k_proj.weight", {256, 2048});
        add(&out, p + ".self_attn.v_proj.weight", {256, 2048});
        add(&out, p + ".self_attn.o_proj.weight", {2048, 2048});
        add(&out, p + ".mlp.gate_proj.weight", {16384, 2048});
        add(&out, p + ".mlp.up_proj.weight", {16384, 2048});
        add(&out, p + ".mlp.down_proj.weight", {2048, 16384});
        add(&out, p + ".input_layernorm.weight", {2048});
        add(&out, p + ".post_attention_layernorm.weight", {2048});
    }
    add(&out, "paligemma_with_expert.paligemma.model.language_model.norm.weight",
        {2048});
    add(&out, "paligemma_with_expert.paligemma.lm_head.weight",
        {257152, 2048});

    const std::string decoder =
        "paligemma_with_expert.gemma_expert.model.layers.";
    for (int layer = 0; layer < 18; ++layer) {
        const std::string p = decoder + std::to_string(layer);
        add(&out, p + ".self_attn.q_proj.weight", {2048, 1024});
        add(&out, p + ".self_attn.k_proj.weight", {256, 1024});
        add(&out, p + ".self_attn.v_proj.weight", {256, 1024});
        add(&out, p + ".self_attn.o_proj.weight", {1024, 2048});
        add(&out, p + ".mlp.gate_proj.weight", {4096, 1024});
        add(&out, p + ".mlp.up_proj.weight", {4096, 1024});
        add(&out, p + ".mlp.down_proj.weight", {1024, 4096});
        for (const char* norm : {"input_layernorm", "post_attention_layernorm"}) {
            add(&out, p + "." + norm + ".dense.weight", {3072, 1024});
            add(&out, p + "." + norm + ".dense.bias", {3072});
        }
    }
    add(&out, "paligemma_with_expert.gemma_expert.model.norm.dense.weight",
        {3072, 1024});
    add(&out, "paligemma_with_expert.gemma_expert.model.norm.dense.bias",
        {3072});
    add(&out, "paligemma_with_expert.gemma_expert.lm_head.weight",
        {257152, 1024});

    add(&out, "action_in_proj.weight", {1024, 32});
    add(&out, "action_in_proj.bias", {1024});
    add(&out, "action_out_proj.weight", {32, 1024});
    add(&out, "action_out_proj.bias", {32});
    add(&out, "time_mlp_in.weight", {1024, 1024});
    add(&out, "time_mlp_in.bias", {1024});
    add(&out, "time_mlp_out.weight", {1024, 1024});
    add(&out, "time_mlp_out.bias", {1024});
    return out;
}

}  // namespace

const std::vector<NativeTensorRequirement>& native_tensor_requirements() {
    static const std::vector<NativeTensorRequirement> requirements =
        build_requirements();
    return requirements;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

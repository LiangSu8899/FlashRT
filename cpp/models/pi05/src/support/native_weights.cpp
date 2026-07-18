#include "flashrt/cpp/models/pi05/support/native_weights.h"

#include "flashrt/cpp/models/pi05/model/dims.h"

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

    constexpr Pi05ModelDims d = kPi05ModelDims;
    constexpr int encoder_kv_width = d.encoder_kv_heads * d.encoder_head_dim;
    constexpr int decoder_q_width = d.decoder_heads * d.decoder_head_dim;
    constexpr int decoder_kv_width = d.decoder_kv_heads * d.decoder_head_dim;
    constexpr int decoder_modulation_width = 3 * d.decoder_width;

    const std::string vision =
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model";
    add(&out, vision + ".embeddings.patch_embedding.weight",
        {d.vision_width, d.image_channels, d.vision_patch, d.vision_patch});
    add(&out, vision + ".embeddings.patch_embedding.bias", {d.vision_width});
    add(&out, vision + ".embeddings.position_embedding.weight",
        {d.vision_tokens_per_view, d.vision_width});
    for (int layer = 0; layer < d.vision_layers; ++layer) {
        const std::string p = vision + ".encoder.layers." +
                              std::to_string(layer);
        for (const char* projection : {"q_proj", "k_proj", "v_proj",
                                       "out_proj"}) {
            add(&out, p + ".self_attn." + projection + ".weight",
                {d.vision_width, d.vision_width});
            add(&out, p + ".self_attn." + projection + ".bias",
                {d.vision_width});
        }
        add(&out, p + ".mlp.fc1.weight", {d.vision_hidden, d.vision_width});
        add(&out, p + ".mlp.fc1.bias", {d.vision_hidden});
        add(&out, p + ".mlp.fc2.weight", {d.vision_width, d.vision_hidden});
        add(&out, p + ".mlp.fc2.bias", {d.vision_width});
        for (const char* norm : {"layer_norm1", "layer_norm2"}) {
            add(&out, p + "." + norm + ".weight", {d.vision_width});
            add(&out, p + "." + norm + ".bias", {d.vision_width});
        }
    }
    add(&out, vision + ".post_layernorm.weight", {d.vision_width});
    add(&out, vision + ".post_layernorm.bias", {d.vision_width});

    const std::string projector =
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear";
    add(&out, projector + ".weight", {d.encoder_width, d.vision_width});
    add(&out, projector + ".bias", {d.encoder_width});

    const std::string encoder =
        "paligemma_with_expert.paligemma.model.language_model.layers.";
    for (int layer = 0; layer < d.encoder_layers; ++layer) {
        const std::string p = encoder + std::to_string(layer);
        add(&out, p + ".self_attn.q_proj.weight",
            {d.encoder_width, d.encoder_width});
        add(&out, p + ".self_attn.k_proj.weight",
            {encoder_kv_width, d.encoder_width});
        add(&out, p + ".self_attn.v_proj.weight",
            {encoder_kv_width, d.encoder_width});
        add(&out, p + ".self_attn.o_proj.weight",
            {d.encoder_width, d.encoder_width});
        add(&out, p + ".mlp.gate_proj.weight",
            {d.encoder_hidden, d.encoder_width});
        add(&out, p + ".mlp.up_proj.weight",
            {d.encoder_hidden, d.encoder_width});
        add(&out, p + ".mlp.down_proj.weight",
            {d.encoder_width, d.encoder_hidden});
        add(&out, p + ".input_layernorm.weight", {d.encoder_width});
        add(&out, p + ".post_attention_layernorm.weight",
            {d.encoder_width});
    }
    add(&out, "paligemma_with_expert.paligemma.model.language_model.norm.weight",
        {d.encoder_width});
    add(&out, "paligemma_with_expert.paligemma.lm_head.weight",
        {d.embedding_vocab, d.embedding_width});

    const std::string decoder =
        "paligemma_with_expert.gemma_expert.model.layers.";
    for (int layer = 0; layer < d.decoder_layers; ++layer) {
        const std::string p = decoder + std::to_string(layer);
        add(&out, p + ".self_attn.q_proj.weight",
            {decoder_q_width, d.decoder_width});
        add(&out, p + ".self_attn.k_proj.weight",
            {decoder_kv_width, d.decoder_width});
        add(&out, p + ".self_attn.v_proj.weight",
            {decoder_kv_width, d.decoder_width});
        add(&out, p + ".self_attn.o_proj.weight",
            {d.decoder_width, decoder_q_width});
        add(&out, p + ".mlp.gate_proj.weight",
            {d.decoder_hidden, d.decoder_width});
        add(&out, p + ".mlp.up_proj.weight",
            {d.decoder_hidden, d.decoder_width});
        add(&out, p + ".mlp.down_proj.weight",
            {d.decoder_width, d.decoder_hidden});
        for (const char* norm : {"input_layernorm", "post_attention_layernorm"}) {
            add(&out, p + "." + norm + ".dense.weight",
                {decoder_modulation_width, d.decoder_width});
            add(&out, p + "." + norm + ".dense.bias",
                {decoder_modulation_width});
        }
    }
    add(&out, "paligemma_with_expert.gemma_expert.model.norm.dense.weight",
        {decoder_modulation_width, d.decoder_width});
    add(&out, "paligemma_with_expert.gemma_expert.model.norm.dense.bias",
        {decoder_modulation_width});
    add(&out, "paligemma_with_expert.gemma_expert.lm_head.weight",
        {d.embedding_vocab, d.decoder_width});

    add(&out, "action_in_proj.weight", {d.decoder_width, d.action_width});
    add(&out, "action_in_proj.bias", {d.decoder_width});
    add(&out, "action_out_proj.weight", {d.action_width, d.decoder_width});
    add(&out, "action_out_proj.bias", {d.action_width});
    add(&out, "time_mlp_in.weight", {d.decoder_width, d.decoder_width});
    add(&out, "time_mlp_in.bias", {d.decoder_width});
    add(&out, "time_mlp_out.weight", {d.decoder_width, d.decoder_width});
    add(&out, "time_mlp_out.bias", {d.decoder_width});
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

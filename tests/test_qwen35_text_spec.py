"""The checkpoint contract: geometry derived, quantization refused by name.

No checkpoint and no GPU. The configs here are built in the test, which is the
point -- what is under test is that the geometry follows the config rather
than a size someone wrote down, and that each property the packed 4-bit decode
depends on is refused when absent.
"""

from __future__ import annotations

import json

import pytest

from flash_rt.frontends.torch._qwen35_text_spec import (
    PACKED_SUFFIX,
    SCALE_SUFFIX,
    CheckpointContractError,
    QuantSpec,
    TextDims,
    compressed_sites,
    plain_sites,
    validate_checkpoint,
)


def _text_config(**overrides):
    config = {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 8,
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "vocab_size": 248320,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"]
                       + ["linear_attention"] * 3 + ["full_attention"],
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25},
    }
    config.update(overrides)
    return {"text_config": config}


def _quant_config(**overrides):
    weights = {
        "num_bits": 4, "type": "int", "symmetric": True, "strategy": "group",
        "group_size": 32, "actorder": None, "block_structure": None,
    }
    weights.update(overrides)
    return {
        "format": "pack-quantized",
        "config_groups": {"group_0": {"weights": weights,
                                      "input_activations": None}},
    }


def test_q_projection_is_twice_the_query_width_when_the_gate_rides_along():
    # A checkpoint with attn_output_gate carries the output gate in the same
    # projection, so q_proj has twice the rows the head count suggests. Sizing
    # it from heads alone reads half a tensor and is finite and wrong.
    gated = TextDims.from_config(_text_config())
    plain = TextDims.from_config(_text_config(attn_output_gate=False))

    assert gated.attn_width == 16 * 256
    assert gated.q_width == 2 * gated.attn_width
    assert plain.q_width == plain.attn_width


def test_gated_deltanet_projection_is_two_keys_and_a_value():
    dims = TextDims.from_config(_text_config())

    assert dims.lin_key_width == 16 * 128
    assert dims.lin_value_width == 32 * 128
    assert dims.lin_qkv_width == 2 * dims.lin_key_width + dims.lin_value_width


def test_rotary_covers_only_its_fraction_of_the_head():
    dims = TextDims.from_config(_text_config())

    assert dims.rotary_dim == 64
    assert dims.rotary_dim < dims.head_dim


def test_geometry_follows_the_config_rather_than_a_remembered_size():
    small = TextDims.from_config(_text_config())
    large = TextDims.from_config(
        _text_config(hidden_size=5120, num_hidden_layers=4,
                     intermediate_size=17408,
                     layer_types=["linear_attention"] * 3
                                 + ["full_attention"]))

    assert small.hidden != large.hidden
    sites = compressed_sites(large)
    assert sites["model.language_model.layers.0.mlp.gate_proj"] == (17408, 5120)
    assert sites["model.language_model.layers.0.mlp.down_proj"] == (5120, 17408)


def test_layer_types_must_cover_every_layer():
    with pytest.raises(CheckpointContractError, match="layer_types lists"):
        TextDims.from_config(_text_config(layer_types=["full_attention"]))


def test_an_unimplemented_layer_type_is_named():
    with pytest.raises(CheckpointContractError, match="sliding_attention"):
        TextDims.from_config(
            _text_config(layer_types=["sliding_attention"] * 8))


def test_the_recurrence_decay_parameters_stay_uncompressed():
    # A_log and dt_bias set how the state forgets rather than contributing to
    # a product, so they belong with the plain tensors and not the packed ones.
    dims = TextDims.from_config(_text_config())
    plain = plain_sites(dims)
    packed = compressed_sites(dims)

    assert "model.language_model.layers.0.linear_attn.A_log" in plain
    assert "model.language_model.layers.0.linear_attn.dt_bias" in plain
    assert not any("A_log" in name for name in packed)


@pytest.mark.parametrize("override,message", [
    ({"num_bits": 8}, "num_bits=8"),
    ({"type": "float"}, "type='float'"),
    ({"symmetric": False}, "no zero point"),
    ({"strategy": "channel"}, "strategy='channel'"),
    ({"actorder": "group"}, "no column permutation"),
    ({"group_size": 16}, "group_size=16"),
])
def test_each_decode_assumption_is_refused_by_name(override, message):
    # Every one of these would load, run, and produce plausible numbers.
    with pytest.raises(CheckpointContractError, match=message):
        QuantSpec.from_config({"quantization_config": _quant_config(**override)})


def test_an_unpacked_format_is_refused():
    quant = _quant_config()
    quant["format"] = "int-quantized"
    with pytest.raises(CheckpointContractError, match="pack-quantized"):
        QuantSpec.from_config({"quantization_config": quant})


def test_a_second_quantization_group_is_refused():
    quant = _quant_config()
    quant["config_groups"]["group_1"] = quant["config_groups"]["group_0"]
    with pytest.raises(CheckpointContractError, match="quantization groups"):
        QuantSpec.from_config({"quantization_config": quant})


@pytest.mark.parametrize("group", [32, 64, 128])
def test_the_published_group_sizes_are_accepted(group):
    assert QuantSpec.from_config(
        {"quantization_config": _quant_config(group_size=group)}
    ).group_size == group


def test_validate_reports_what_is_missing(tmp_path):
    config = _text_config()
    config["quantization_config"] = _quant_config()
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.language_model.norm.weight": "a"}}))

    with pytest.raises(CheckpointContractError, match="missing text backbone"):
        validate_checkpoint(str(tmp_path))


def test_validate_accepts_a_complete_index(tmp_path):
    config = _text_config()
    config["quantization_config"] = _quant_config()
    dims = TextDims.from_config(config)
    names = set(plain_sites(dims))
    for site in compressed_sites(dims):
        names.add(site + PACKED_SUFFIX)
        names.add(site + SCALE_SUFFIX)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "a" for name in names}}))

    contract = validate_checkpoint(str(tmp_path))

    assert contract["dims"].num_layers == 8
    assert contract["quant"].group_size == 32

"""Loading a packed checkpoint without unpacking it.

The checkpoint here is synthesized, small and complete, so the properties
under test are the loader's and not any particular file's: that the packed
weights stay packed, that what the decode step receives is addresses rather
than tensors, and that a shape the geometry does not predict is refused.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from flash_rt.frontends.torch._qwen35_text_spec import (  # noqa: E402
    PACKED_SUFFIX,
    SCALE_SUFFIX,
    compressed_sites,
    plain_sites,
    validate_checkpoint,
)
from flash_rt.frontends.torch._qwen35_text_weights import (  # noqa: E402
    load_text_weights,
)

GROUP = 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _config():
    return {
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 4,
            "hidden_size": 128,
            "intermediate_size": 256,
            "vocab_size": 512,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "attn_output_gate": True,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {"rope_theta": 1e6,
                                "partial_rotary_factor": 0.25},
        },
        "quantization_config": {
            "format": "pack-quantized",
            "config_groups": {"group_0": {"weights": {
                "num_bits": 4, "type": "int", "symmetric": True,
                "strategy": "group", "group_size": GROUP,
                "actorder": None, "block_structure": None,
            }, "input_activations": None}},
        },
    }


def _write_checkpoint(directory, mangle=None):
    from flash_rt.frontends.torch._qwen35_text_spec import TextDims

    config = _config()
    (directory / "config.json").write_text(json.dumps(config))
    dims = TextDims.from_config(config)

    tensors = {}
    for name, shape in plain_sites(dims).items():
        tensors[name] = torch.zeros(*shape, dtype=torch.bfloat16)
    for site, (rows, columns) in compressed_sites(dims).items():
        packed_columns = columns // 8
        if mangle == site:
            packed_columns += 1           # a shape the geometry does not make
        tensors[site + PACKED_SUFFIX] = torch.randint(
            -(2 ** 31), 2 ** 31 - 1, (rows, packed_columns), dtype=torch.int32)
        tensors[site + SCALE_SUFFIX] = torch.ones(
            rows, columns // GROUP, dtype=torch.bfloat16)
    safetensors_torch.save_file(
        tensors, str(directory / "model.safetensors"))
    return dims, tensors


def test_packed_weights_are_not_unpacked_on_the_way_in(tmp_path):
    # Half a byte per value on the file is half a byte per value resident. A
    # loader that dequantized would still work and would cost eight times the
    # memory, which on the target is the whole question.
    dims, tensors = _write_checkpoint(tmp_path)
    contract = validate_checkpoint(str(tmp_path))

    weights = load_text_weights(str(tmp_path), contract, device=DEVICE)

    on_file = sum(t.numel() * t.element_size() for t in tensors.values())
    # The embedding is counted once even though lm_head is tied to it.
    assert weights.resident_bytes == pytest.approx(on_file, rel=0.01)
    weights.close()


def test_the_decode_surface_is_addresses_not_tensors(tmp_path):
    # Holding tensors is how a hot path ends up asking Torch for shapes and
    # slices. Every value handed to the decode step is an int, and the tensors
    # live only in anchors.
    _write_checkpoint(tmp_path)
    contract = validate_checkpoint(str(tmp_path))

    weights = load_text_weights(str(tmp_path), contract, device=DEVICE)

    for entry in weights.layers:
        for name, value in entry.items():
            assert isinstance(value, int), f"layer field {name} is {type(value)}"
    for name, value in weights.top.items():
        assert isinstance(value, int), f"top field {name} is {type(value)}"
    assert weights.anchors, "nothing is keeping the weights alive"
    weights.close()


def test_tied_embeddings_are_loaded_once(tmp_path):
    _write_checkpoint(tmp_path)
    contract = validate_checkpoint(str(tmp_path))

    weights = load_text_weights(str(tmp_path), contract, device=DEVICE)

    assert weights.top["lm_head_tied"] == 1
    assert weights.top["lm_head"] == weights.top["embed"]
    weights.close()


def test_every_layer_gets_the_sites_its_type_needs(tmp_path):
    dims, _ = _write_checkpoint(tmp_path)
    contract = validate_checkpoint(str(tmp_path))

    weights = load_text_weights(str(tmp_path), contract, device=DEVICE)

    for index, entry in enumerate(weights.layers):
        assert {"gate_up_packed", "down_packed"} <= set(entry)
        if dims.layer_types[index] == "linear_attention":
            assert {"in_proj_packed", "out_packed", "in_ab",
                    "neg_exp_a_log", "dt_bias", "conv"} <= set(entry)
            assert "qkv_packed" not in entry
        else:
            assert {"qkv_packed", "o_packed",
                    "q_norm", "k_norm"} <= set(entry)
            assert "in_qkv_packed" not in entry
    weights.close()


def test_a_packed_shape_the_geometry_does_not_make_is_refused(tmp_path):
    site = "model.language_model.layers.0.mlp.down_proj"
    _write_checkpoint(tmp_path, mangle=site)
    contract = validate_checkpoint(str(tmp_path))

    with pytest.raises(ValueError, match="packed is"):
        load_text_weights(str(tmp_path), contract, device=DEVICE)


def test_the_recorded_widths_are_the_logical_ones(tmp_path):
    # The kernel takes K in values, not in packed words; handing it the packed
    # width would read an eighth of the weight and still return numbers.
    dims, _ = _write_checkpoint(tmp_path)
    contract = validate_checkpoint(str(tmp_path))

    weights = load_text_weights(str(tmp_path), contract, device=DEVICE)

    entry = weights.layers[0]
    # gate and up are one weight now; the fused N is their sum.
    assert entry["gate_up_n"] == 2 * dims.intermediate
    assert entry["gate_up_k"] == dims.hidden
    assert entry["gate_up_gate_offset"] == 0
    assert entry["gate_up_up_offset"] == dims.intermediate
    assert entry["down_n"] == dims.hidden
    assert entry["down_k"] == dims.intermediate
    weights.close()


def test_the_plain_norms_are_offset_and_the_gated_one_is_not(tmp_path):
    # This family's plain RMSNorm scales by 1 + weight, and stores the
    # parameter centred on zero. Placing it unchanged is not an error anything
    # reports: it scales the hidden state down at every layer and the model
    # still emits fluent tokens unrelated to the input, which is expensive to
    # attribute later. The gated norm inside the recurrence is an ordinary
    # scale and must not be offset -- so both halves are checked here.
    _write_checkpoint(tmp_path)
    contract = validate_checkpoint(str(tmp_path))
    weights = load_text_weights(str(tmp_path), contract, device=DEVICE)
    dims = weights.dims

    by_address = {int(t.data_ptr()): t for t in weights.anchors}
    for key in ("input_norm", "post_norm"):
        for entry in weights.layers:
            assert torch.all(by_address[entry[key]] == 1.0), key
    assert torch.all(by_address[weights.top["final_norm"]] == 1.0)

    for index, entry in enumerate(weights.layers):
        if dims.layer_types[index] == "full_attention":
            for key in ("q_norm", "k_norm"):
                assert torch.all(by_address[entry[key]] == 1.0), key
        else:
            assert torch.all(by_address[entry["gdn_norm"]] == 0.0)


def test_the_decay_constant_is_exponentiated_once_at_load(tmp_path):
    # The recurrence wants -exp(A_log), never A_log, and wants it in float32:
    # it multiplies the state every step and never re-derives it, so a
    # bfloat16 decay makes a long memory and a permanent one the same number.
    _write_checkpoint(tmp_path)
    contract = validate_checkpoint(str(tmp_path))
    weights = load_text_weights(str(tmp_path), contract, device=DEVICE)

    by_address = {int(t.data_ptr()): t for t in weights.anchors}
    for index, entry in enumerate(weights.layers):
        if weights.dims.layer_types[index] != "linear_attention":
            continue
        decay = by_address[entry["neg_exp_a_log"]]
        assert decay.dtype is torch.float32
        # the synthetic checkpoint stores A_log = 0, so -exp(0) = -1
        assert torch.all(decay == -1.0)
        assert by_address[entry["dt_bias"]].dtype is torch.float32

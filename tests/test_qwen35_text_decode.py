"""The MLP block: three launches, fixed addresses, no tensors in the step.

Correctness is checked against the block written out in Torch over the
dequantized weight, which is the only reference that does not share the
kernel's assumptions about the layout.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

if not torch.cuda.is_available():                        # pragma: no cover
    pytest.skip("needs a GPU", allow_module_level=True)

try:
    from flash_rt import flash_rt_kernels as fvk
except ImportError:                                      # pragma: no cover
    pytest.skip("flash_rt_kernels is not built", allow_module_level=True)

if not hasattr(fvk, "w4a16_packed_matvec_bf16"):         # pragma: no cover
    pytest.skip("built without the packed 4-bit kernels",
                allow_module_level=True)

from flash_rt.frontends.torch._qwen35_text_decode import (  # noqa: E402
    Workspace,
    mlp_block,
)
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
DEVICE = "cuda:0"


def _config(hidden=128, intermediate=256):
    return {
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 2,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "vocab_size": 512,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "layer_types": ["linear_attention", "full_attention"],
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


def _pack(values):
    rows, columns = values.shape
    nibbles = (values.to(torch.int32) + 8) & 0xF
    packed = torch.zeros(rows, columns // 8, dtype=torch.int32)
    for i in range(8):
        packed |= nibbles[:, i::8] << (4 * i)
    return packed


def _write(directory):
    from flash_rt.frontends.torch._qwen35_text_spec import TextDims

    config = _config()
    (directory / "config.json").write_text(json.dumps(config))
    dims = TextDims.from_config(config)
    torch.manual_seed(0)

    tensors = {}
    plain = {}
    for name, shape in plain_sites(dims).items():
        tensors[name] = torch.zeros(*shape, dtype=torch.bfloat16)
    for site, (rows, columns) in compressed_sites(dims).items():
        values = torch.randint(-8, 8, (rows, columns), dtype=torch.int32)
        scale = (torch.rand(rows, columns // GROUP) * 0.05 + 1e-3).to(
            torch.bfloat16)
        tensors[site + PACKED_SUFFIX] = _pack(values)
        tensors[site + SCALE_SUFFIX] = scale
        plain[site] = (values.float()
                       * scale.float().repeat_interleave(GROUP, dim=1))
    safetensors_torch.save_file(tensors, str(directory / "model.safetensors"))
    return dims, plain


def test_mlp_block_matches_the_block_written_out(tmp_path):
    dims, plain = _write(tmp_path)
    contract = validate_checkpoint(str(tmp_path))
    weights = load_text_weights(str(tmp_path), contract, device=DEVICE)
    work = Workspace(weights, device=DEVICE)
    stream = torch.cuda.current_stream(DEVICE).cuda_stream

    x = torch.randn(1, dims.hidden, dtype=torch.bfloat16, device=DEVICE)
    out = torch.empty(1, dims.hidden, dtype=torch.bfloat16, device=DEVICE)
    mlp_block(weights.layers[0], work, fvk, x.data_ptr(), out.data_ptr(),
              rows=1, stream=stream)
    torch.cuda.synchronize(DEVICE)

    prefix = "model.language_model.layers.0."
    gate_w = plain[prefix + "mlp.gate_proj"].to(DEVICE)
    up_w = plain[prefix + "mlp.up_proj"].to(DEVICE)
    down_w = plain[prefix + "mlp.down_proj"].to(DEVICE)
    gate = x.float() @ gate_w.T
    up = x.float() @ up_w.T
    want = (torch.nn.functional.silu(gate.bfloat16().float())
            * up.bfloat16().float()) @ down_w.T

    error = (out.float() - want).abs().max() / want.abs().max()
    assert error < 2e-2, f"relative error {error:.3g}"
    work.close()
    weights.close()


def test_the_step_issues_two_launches_and_allocates_nothing(tmp_path):
    # The point of the workspace is that a step is launches and nothing else.
    # Anything allocated per call is both a dispatch and an address a captured
    # graph could not replay.
    _write(tmp_path)
    contract = validate_checkpoint(str(tmp_path))
    weights = load_text_weights(str(tmp_path), contract, device=DEVICE)
    work = Workspace(weights, device=DEVICE)
    stream = torch.cuda.current_stream(DEVICE).cuda_stream
    x = torch.randn(1, weights.dims.hidden, dtype=torch.bfloat16,
                    device=DEVICE)
    out = torch.empty_like(x)

    watched = ("w4a16_packed_matvec_gated_bf16",
               "w4a16_packed_matvec_bf16")
    originals = {name: getattr(fvk, name) for name in watched}
    calls = []
    for name in watched:
        def counted(*args, _name=name, _original=originals[name]):
            calls.append(_name)
            return _original(*args)

        setattr(fvk, name, counted)
    try:
        torch.cuda.synchronize(DEVICE)
        before = torch.cuda.memory_allocated(DEVICE)
        mlp_block(weights.layers[0], work, fvk, x.data_ptr(), out.data_ptr(),
                  rows=1, stream=stream)
        torch.cuda.synchronize(DEVICE)
        after = torch.cuda.memory_allocated(DEVICE)
    finally:
        for name, original in originals.items():
            setattr(fvk, name, original)

    assert calls == ["w4a16_packed_matvec_gated_bf16",
                     "w4a16_packed_matvec_bf16"]
    assert after == before, f"the step allocated {after - before} bytes"
    work.close()
    weights.close()


def test_the_workspace_addresses_do_not_move(tmp_path):
    # A captured graph replays the addresses it saw, so they have to be the
    # same ones on the second call as on the first.
    _write(tmp_path)
    contract = validate_checkpoint(str(tmp_path))
    weights = load_text_weights(str(tmp_path), contract, device=DEVICE)
    work = Workspace(weights, device=DEVICE)

    before = (work.fused.address, work.gated.address, work.normed.address)
    stream = torch.cuda.current_stream(DEVICE).cuda_stream
    x = torch.randn(1, weights.dims.hidden, dtype=torch.bfloat16,
                    device=DEVICE)
    out = torch.empty_like(x)
    for _ in range(3):
        mlp_block(weights.layers[0], work, fvk, x.data_ptr(), out.data_ptr(),
                  rows=1, stream=stream)
    torch.cuda.synchronize(DEVICE)

    assert (work.fused.address, work.gated.address,
            work.normed.address) == before
    work.close()
    weights.close()

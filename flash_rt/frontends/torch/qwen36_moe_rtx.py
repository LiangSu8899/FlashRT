"""Qwen3.6-35B-A3B text inference on RTX SM120.

The language backbone is the same ``qwen3_5_moe`` architecture used by
Nex-N2-mini, so this frontend reuses that implementation.  The official
Qwen3.6 checkpoint also contains a vision tower and an MTP head; this entry
validates those weights but intentionally exposes only text prefill and greedy
decode.  Vision and speculative decoding are separate integration surfaces.
"""

from __future__ import annotations

import json
import os
from typing import Any

from flash_rt.frontends.torch.nexn2_rtx import Nexn2TorchFrontendRtx


_EXPECTED_LAYER_TYPES = tuple(
    "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
    for i in range(40)
)

_EXPECTED_TEXT_CONFIG = {
    "model_type": "qwen3_5_moe_text",
    "num_hidden_layers": 40,
    "hidden_size": 2048,
    "vocab_size": 248320,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "full_attention_interval": 4,
    "mtp_num_hidden_layers": 1,
}

_MTP_KEYS = {
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "mtp.layers.0.mlp.experts.down_proj",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
}


def _required_text_keys(layer_types: tuple[str, ...]) -> set[str]:
    keys = {
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    }
    mlp_suffixes = (
        "mlp.gate.weight",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    )
    linear_suffixes = (
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.conv1d.weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
    )
    full_suffixes = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    )
    for i, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{i}."
        keys.add(prefix + "input_layernorm.weight")
        keys.add(prefix + "post_attention_layernorm.weight")
        keys.update(prefix + suffix for suffix in mlp_suffixes)
        suffixes = (
            full_suffixes
            if layer_type == "full_attention"
            else linear_suffixes
        )
        keys.update(prefix + suffix for suffix in suffixes)
    return keys


def validate_qwen36_moe_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """Validate the official BF16 checkpoint before allocating GPU weights."""
    checkpoint_path = os.path.abspath(os.fspath(checkpoint_path))
    config_path = os.path.join(checkpoint_path, "config.json")
    index_path = os.path.join(
        checkpoint_path, "model.safetensors.index.json")

    for path in (config_path, index_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Qwen3.6-35B-A3B checkpoint is missing {path!r}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("model_type") != "qwen3_5_moe":
        raise ValueError(
            "Qwen3.6-35B-A3B text frontend requires "
            "model_type='qwen3_5_moe'; "
            f"got {config.get('model_type')!r} in {config_path}")

    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError(f"missing text_config object in {config_path}")

    mismatches = []
    for name, expected in _EXPECTED_TEXT_CONFIG.items():
        actual = text_config.get(name)
        if actual != expected:
            mismatches.append(f"{name}={actual!r} (expected {expected!r})")
    layer_types = tuple(text_config.get("layer_types") or ())
    if layer_types != _EXPECTED_LAYER_TYPES:
        mismatches.append(
            "layer_types does not match the 30-linear/10-full attention "
            "qwen3_5_moe schedule")
    if mismatches:
        raise ValueError(
            "checkpoint is not compatible with the Qwen3.6-35B-A3B "
            "SM120 text pipeline: " + "; ".join(mismatches))

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"missing weight_map object in {index_path}")

    required_text = _required_text_keys(layer_types)
    missing_text = sorted(required_text.difference(weight_map))
    if missing_text:
        preview = ", ".join(missing_text[:8])
        if len(missing_text) > 8:
            preview += f", ... ({len(missing_text)} missing)"
        raise ValueError(
            "Qwen3.6-35B-A3B checkpoint is missing text backbone tensors: "
            + preview)

    missing_mtp = sorted(_MTP_KEYS.difference(weight_map))
    if missing_mtp:
        preview = ", ".join(missing_mtp[:8])
        if len(missing_mtp) > 8:
            preview += f", ... ({len(missing_mtp)} missing)"
        raise ValueError(
            "Qwen3.6-35B-A3B checkpoint is missing its MTP tensor group: "
            + preview)

    shards = sorted(set(weight_map.values()))
    bad_shards = []
    for shard in shards:
        path = os.path.join(checkpoint_path, shard)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            bad_shards.append(shard)
    if bad_shards:
        preview = ", ".join(bad_shards[:8])
        if len(bad_shards) > 8:
            preview += f", ... ({len(bad_shards)} missing or empty)"
        raise FileNotFoundError(
            "Qwen3.6-35B-A3B checkpoint has missing or empty shards: "
            + preview)

    return {
        "checkpoint_path": checkpoint_path,
        "text_tensor_count": len(required_text),
        "mtp_tensor_count": len(_MTP_KEYS),
        "vision_tensor_count": sum(
            ".visual." in name for name in weight_map),
        "tensor_count": len(weight_map),
        "shard_count": len(shards),
    }


class Qwen36MoeTextFrontendRtx(Nexn2TorchFrontendRtx):
    """Qwen3.6-35B-A3B text-only frontend for RTX SM120."""

    _MODEL_LABEL = "Qwen3.6-35B-A3B text"
    _USAGE_DOC = "docs/qwen36_moe_usage.md"

    def __init__(self, checkpoint_path: str, *,
                 device: str = "cuda:0",
                 max_seq: int = 2048,
                 quant: str = "nvfp4",
                 kernelized: bool = False,
                 quant_scope: str = "experts") -> None:
        if quant != "nvfp4":
            raise NotImplementedError(
                f"quant={quant!r} is not implemented; only 'nvfp4' is "
                "supported")
        contract = validate_qwen36_moe_checkpoint(checkpoint_path)
        super().__init__(
            checkpoint_path,
            device=device,
            max_seq=max_seq,
            quant=quant,
            kernelized=kernelized,
            quant_scope=quant_scope,
        )
        self._checkpoint_contract = contract

    def generate(self, max_new_tokens: int, *, do_sample: bool = False):
        """Generate tokens with the shared qwen3_5_moe CUDA Graph path."""
        if not self._kernelized:
            return super().generate(
                max_new_tokens=max_new_tokens, do_sample=do_sample)
        if self._prompt_ids is None:
            raise ValueError("call set_prompt(...) before generate()")
        if do_sample:
            raise NotImplementedError(
                "the Qwen3.6-35B-A3B kernelized path supports greedy "
                "decoding only")

        from flash_rt.frontends.torch._nexn2_rtx_decode import (
            Nexn2DecodeState,
            generate_greedy_graph,
        )

        if self._decode_state is None:
            self._decode_state = Nexn2DecodeState(
                self._weights, self._user_max_seq, self.device)
        return generate_greedy_graph(
            self._decode_state,
            self._prompt_ids,
            max_new_tokens,
            self._fvk,
            self.device,
        )

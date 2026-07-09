#!/usr/bin/env python3
import argparse
import subprocess

import torch
from safetensors import safe_open


VISION = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
ENCODER = "paligemma_with_expert.paligemma.model.language_model.layers.0"
DECODER = "paligemma_with_expert.gemma_expert.model.layers.0"


def interleave_qk(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    out_dim, in_dim = weight.shape
    head_dim = out_dim // num_heads
    return (
        weight.reshape(num_heads, head_dim, in_dim)
        .reshape(num_heads, 2, head_dim // 2, in_dim)
        .permute(0, 2, 1, 3)
        .reshape(out_dim, in_dim)
    )


def fnv1a(data: bytes) -> int:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def digest(tensor: torch.Tensor) -> tuple[tuple[int, ...], int]:
    tensor = tensor.contiguous().view(torch.uint16).cpu()
    return tuple(tensor.shape), fnv1a(tensor.numpy().tobytes())


def parse_probe(text: str) -> tuple[tuple[int, ...], int]:
    fields = dict(field.split("=", 1) for field in text.strip().split())
    shape = tuple(int(dim) for dim in fields["shape"].split(","))
    return shape, int(fields["fnv"], 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    args = parser.parse_args()

    file = safe_open(f"{args.checkpoint}/model.safetensors", framework="pt")
    keys = set(file.keys())
    prefix = "model." if "model.action_in_proj.weight" in keys else ""

    def raw(key: str) -> torch.Tensor:
        return file.get_tensor(prefix + key)

    def bf16(key: str) -> torch.Tensor:
        return raw(key).to(torch.bfloat16)

    patch = bf16(f"{VISION}.embeddings.patch_embedding.weight")
    expected = {
        "patch": patch.permute(2, 3, 1, 0).contiguous(),
    }

    q = interleave_qk(raw(f"{ENCODER}.self_attn.q_proj.weight").float(), 8)
    k = interleave_qk(raw(f"{ENCODER}.self_attn.k_proj.weight").float(), 1)
    v = raw(f"{ENCODER}.self_attn.v_proj.weight").float()
    norm = 1.0 + raw(f"{ENCODER}.input_layernorm.weight").float()
    expected["encoder_qkv0"] = torch.cat(
        [q * norm.unsqueeze(0), k * norm.unsqueeze(0), v * norm.unsqueeze(0)],
        dim=0,
    ).t().to(torch.bfloat16).contiguous()

    q = interleave_qk(bf16(f"{DECODER}.self_attn.q_proj.weight").float(), 8)
    k = interleave_qk(bf16(f"{DECODER}.self_attn.k_proj.weight").float(), 1)
    v = bf16(f"{DECODER}.self_attn.v_proj.weight")
    expected["decoder_qkv0"] = torch.cat([q, k, v], dim=0).t().to(
        torch.bfloat16
    ).contiguous()

    gate = bf16(f"{DECODER}.mlp.gate_proj.weight").t()
    up = bf16(f"{DECODER}.mlp.up_proj.weight").t()
    expected["decoder_gate_up0"] = torch.cat([gate, up], dim=1).contiguous()

    for operation, tensor in expected.items():
        output = subprocess.check_output(
            [args.probe, args.checkpoint, operation], text=True
        )
        actual = parse_probe(output)
        reference = digest(tensor)
        if actual != reference:
            raise AssertionError(
                f"{operation}: C++ {actual} != PyTorch {reference}"
            )
        print(f"PASS {operation} shape={actual[0]} fnv={actual[1]:016x}")


if __name__ == "__main__":
    main()

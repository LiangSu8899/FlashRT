#!/usr/bin/env python3
import argparse
import gc
import pathlib
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


SEQUENCE = 712
WIDTH = 2048


def interleave_qk(weight: torch.Tensor, heads: int) -> torch.Tensor:
    output, inputs = weight.shape
    head_dim = output // heads
    return (
        weight.reshape(heads, head_dim, inputs)
        .reshape(heads, 2, head_dim // 2, inputs)
        .permute(0, 2, 1, 3)
        .reshape(output, inputs)
    )


def rms(values: torch.Tensor) -> torch.Tensor:
    source = values.float()
    return (source * torch.rsqrt(source.square().mean(-1, keepdim=True) + 1e-6)).to(
        torch.bfloat16
    )


def rotate(tensor: torch.Tensor, heads: int, rope: torch.Tensor) -> torch.Tensor:
    pairs = tensor.reshape(SEQUENCE, heads, 128, 2).float()
    cosine = rope[:, None, :, 0].float()
    sine = rope[:, None, :, 1].float()
    return torch.stack(
        [
            pairs[..., 0] * cosine - pairs[..., 1] * sine,
            pairs[..., 1] * cosine + pairs[..., 0] * sine,
        ],
        -1,
    ).to(torch.bfloat16).reshape(SEQUENCE, heads, 256)


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> str:
    cosine = float(
        F.cosine_similarity(
            actual.flatten().double(), expected.flatten().double(), dim=0
        )
    )
    maximum = float((actual - expected).abs().max())
    if cosine < 0.9999:
        raise AssertionError(f"{name}: cosine={cosine:.8f} max={maximum:.6f}")
    return f"{name} cosine={cosine:.8f} max={maximum:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    args = parser.parse_args()
    file = safe_open(f"{args.checkpoint}/model.safetensors", framework="pt")
    keys = set(file.keys())
    root = "model." if "model.action_in_proj.weight" in keys else ""

    def raw(name: str) -> torch.Tensor:
        return file.get_tensor(root + name)

    x = torch.zeros((SEQUENCE, WIDTH), device="cuda", dtype=torch.bfloat16)
    rows = torch.arange(SEQUENCE, device="cuda")[:, None]
    columns = torch.arange(512, device="cuda")[None, :]
    x[:, :512] = (((rows + columns) % 15 - 7).float() / 8).to(torch.bfloat16)
    positions = torch.arange(SEQUENCE, dtype=torch.float64)[:, None]
    pair = torch.arange(128, dtype=torch.float64)[None, :]
    phase = positions / torch.pow(10000.0, (2 * pair) / 256.0)
    rope = torch.stack([torch.cos(phase), torch.sin(phase)], -1).to(
        device="cuda", dtype=torch.bfloat16
    )

    final_q = final_k = final_v = None
    for index in range(18):
        layer = (
            "paligemma_with_expert.paligemma.model.language_model.layers."
            f"{index}"
        )
        input_norm = 1.0 + raw(f"{layer}.input_layernorm.weight").float()
        q = interleave_qk(raw(f"{layer}.self_attn.q_proj.weight").float(), 8)
        k = interleave_qk(raw(f"{layer}.self_attn.k_proj.weight").float(), 1)
        v = raw(f"{layer}.self_attn.v_proj.weight").float()
        qkv_weight = torch.cat(
            [
                q * input_norm[None, :],
                k * input_norm[None, :],
                v * input_norm[None, :],
            ],
            dim=0,
        ).t().to(device="cuda", dtype=torch.bfloat16).contiguous()
        qkv = rms(x) @ qkv_weight
        query, key, value = torch.split(qkv, [2048, 256, 256], dim=-1)
        query = rotate(query, 8, rope)
        key = rotate(key, 1, rope)
        value = value.reshape(SEQUENCE, 1, 256)
        if index == 17:
            final_q = query.reshape(SEQUENCE, 2048).cpu().float()
            final_k = key.reshape(SEQUENCE, 256).cpu().float()
            final_v = value.reshape(SEQUENCE, 256).cpu().float()
            break

        attended = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            scale=1.0 / 16.0,
            enable_gqa=True,
        ).squeeze(0).transpose(0, 1).reshape(SEQUENCE, 2048)
        output_weight = raw(f"{layer}.self_attn.o_proj.weight").to(
            device="cuda", dtype=torch.bfloat16
        ).t().contiguous()
        x = (x.float() + (attended @ output_weight).float()).to(torch.bfloat16)
        post_norm = 1.0 + raw(
            f"{layer}.post_attention_layernorm.weight"
        ).float()
        gate_weight = (
            raw(f"{layer}.mlp.gate_proj.weight").float() * post_norm[None, :]
        ).t().to(device="cuda", dtype=torch.bfloat16).contiguous()
        up_weight = (
            raw(f"{layer}.mlp.up_proj.weight").float() * post_norm[None, :]
        ).t().to(device="cuda", dtype=torch.bfloat16).contiguous()
        down_weight = raw(f"{layer}.mlp.down_proj.weight").to(
            device="cuda", dtype=torch.bfloat16
        ).t().contiguous()
        normalized = rms(x)
        gate = normalized @ gate_weight
        up = normalized @ up_weight
        gate_float = gate.float()
        activated = gate_float / (
            1.0
            + torch.exp(
                -1.5957691216057308
                * gate_float
                * (1.0 + 0.044715 * gate_float.square())
            )
        )
        hidden = (activated * up.float()).to(torch.bfloat16)
        x = (x.float() + (hidden @ down_weight).float()).to(torch.bfloat16)
        del q, k, v, qkv_weight, qkv, query, key, value
        del attended, output_weight, gate_weight, up_weight, down_weight
        del normalized, gate, up, gate_float, activated, hidden
        gc.collect()

    expected_x = x.cpu().float()
    del x, rope
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory() as directory:
        output = str(pathlib.Path(directory) / "encoder.bin")
        subprocess.check_call([args.probe, args.checkpoint, output])
        bits = np.fromfile(output, dtype=np.uint16)
    sizes = [SEQUENCE * 2048, SEQUENCE * 2048, SEQUENCE * 256, SEQUENCE * 256]
    if bits.size != sum(sizes):
        raise AssertionError(f"encoder probe output elements={bits.size}")
    tensors = []
    offset = 0
    for size in sizes:
        tensors.append(
            torch.from_numpy(bits[offset : offset + size].copy())
            .view(torch.bfloat16)
            .float()
        )
        offset += size
    messages = [
        compare("x", tensors[0].reshape(SEQUENCE, 2048), expected_x),
        compare("q", tensors[1].reshape(SEQUENCE, 2048), final_q),
        compare("k", tensors[2].reshape(SEQUENCE, 256), final_k),
        compare("v", tensors[3].reshape(SEQUENCE, 256), final_v),
    ]
    print("PASS encoder 18 layers " + "; ".join(messages))


if __name__ == "__main__":
    main()

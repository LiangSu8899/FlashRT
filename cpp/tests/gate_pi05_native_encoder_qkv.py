#!/usr/bin/env python3
import argparse
import pathlib
import subprocess
import tempfile

import numpy as np
import torch
from safetensors import safe_open


def interleave_qk(weight: torch.Tensor, heads: int) -> torch.Tensor:
    output, inputs = weight.shape
    head_dim = output // heads
    return (
        weight.reshape(heads, head_dim, inputs)
        .reshape(heads, 2, head_dim // 2, inputs)
        .permute(0, 2, 1, 3)
        .reshape(output, inputs)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the encoder QKV gate")

    file = safe_open(f"{args.checkpoint}/model.safetensors", framework="pt")
    keys = set(file.keys())
    prefix = "model." if "model.action_in_proj.weight" in keys else ""
    layer = "paligemma_with_expert.paligemma.model.language_model.layers.17"

    def raw(name: str) -> torch.Tensor:
        return file.get_tensor(prefix + name)

    q = interleave_qk(raw(f"{layer}.self_attn.q_proj.weight").float(), 8)
    k = interleave_qk(raw(f"{layer}.self_attn.k_proj.weight").float(), 1)
    v = raw(f"{layer}.self_attn.v_proj.weight").float()
    norm = 1.0 + raw(f"{layer}.input_layernorm.weight").float()
    weight = torch.cat(
        [q * norm[None, :], k * norm[None, :], v * norm[None, :]], dim=0
    ).t().to(device="cuda", dtype=torch.bfloat16).contiguous()

    x = torch.zeros((712, 2048), device="cuda", dtype=torch.bfloat16)
    rows = torch.arange(712, device="cuda")[:, None]
    columns = torch.arange(512, device="cuda")[None, :]
    x[:, :512] = (((rows + columns) % 15 - 7).float() / 8).to(torch.bfloat16)
    x_float = x.float()
    normalized = (
        x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    qkv = normalized @ weight
    query, key, value = torch.split(qkv, [2048, 256, 256], dim=-1)

    positions = torch.arange(712, dtype=torch.float64)[:, None]
    pair = torch.arange(128, dtype=torch.float64)[None, :]
    phase = positions / torch.pow(10000.0, (2 * pair) / 256.0)
    rope = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1).to(
        device="cuda", dtype=torch.bfloat16
    )

    def apply_rope(tensor: torch.Tensor, heads: int) -> torch.Tensor:
        pairs = tensor.reshape(712, heads, 128, 2).float()
        cosine = rope[:, None, :, 0].float()
        sine = rope[:, None, :, 1].float()
        even = pairs[..., 0] * cosine - pairs[..., 1] * sine
        odd = pairs[..., 1] * cosine + pairs[..., 0] * sine
        return torch.stack([even, odd], dim=-1).to(torch.bfloat16).reshape(
            712, heads * 256
        )

    expected = {
        "q": apply_rope(query, 8),
        "k": apply_rope(key, 1),
        "v": value.contiguous(),
    }
    with tempfile.TemporaryDirectory() as directory:
        output = str(pathlib.Path(directory) / "encoder")
        subprocess.check_call([args.probe, args.checkpoint, output])
        for name, reference in expected.items():
            actual_bits = np.fromfile(f"{output}.{name}.bin", dtype=np.uint16)
            actual = torch.from_numpy(actual_bits.copy()).view(torch.bfloat16)
            actual = actual.reshape(reference.shape).float()
            target = reference.cpu().float()
            cosine = float(torch.nn.functional.cosine_similarity(
                actual.flatten().double(), target.flatten().double(), dim=0
            ))
            maximum = float((actual - target).abs().max())
            if cosine < 0.9999:
                raise AssertionError(
                    f"{name}: cosine={cosine:.8f} max={maximum:.6f}"
                )
            print(f"PASS encoder17 {name} cosine={cosine:.8f} max={maximum:.6f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import math
import pathlib
import subprocess
import tempfile

import numpy as np
import torch
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the style precompute gate")

    file = safe_open(f"{args.checkpoint}/model.safetensors", framework="pt")
    keys = set(file.keys())
    prefix = "model." if "model.action_in_proj.weight" in keys else ""

    def bf16(key: str) -> torch.Tensor:
        return file.get_tensor(prefix + key).to(
            device="cuda", dtype=torch.bfloat16
        )

    decoder = "paligemma_with_expert.gemma_expert.model.layers"
    time_in_w = bf16("time_mlp_in.weight").t().contiguous()
    time_in_b = bf16("time_mlp_in.bias")
    time_out_w = bf16("time_mlp_out.weight").t().contiguous()
    time_out_b = bf16("time_mlp_out.bias")
    attn_w = torch.stack(
        [bf16(f"{decoder}.{i}.input_layernorm.dense.weight").t() for i in range(18)]
    )
    attn_b = torch.stack(
        [bf16(f"{decoder}.{i}.input_layernorm.dense.bias") for i in range(18)]
    )
    ffn_w = torch.stack(
        [
            bf16(f"{decoder}.{i}.post_attention_layernorm.dense.weight").t()
            for i in range(18)
        ]
    )
    ffn_b = torch.stack(
        [
            bf16(f"{decoder}.{i}.post_attention_layernorm.dense.bias")
            for i in range(18)
        ]
    )
    final_w = bf16(
        "paligemma_with_expert.gemma_expert.model.norm.dense.weight"
    ).t()
    final_b = bf16("paligemma_with_expert.gemma_expert.model.norm.dense.bias")

    fraction = torch.linspace(0.0, 1.0, 512)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    t = torch.tensor(1.0, dtype=torch.float32)
    rows = []
    for _ in range(10):
        angle = t * (1.0 / period) * 2 * math.pi
        rows.append(
            torch.cat([torch.sin(angle), torch.cos(angle)]).to(
                device="cuda", dtype=torch.bfloat16
            )
        )
        t = t - 0.1
    schedule = torch.stack(rows)
    expected = {
        "decoder_time_emb": torch.empty(
            10, 10, 1024, dtype=torch.bfloat16, device="cuda"
        ),
        "decoder_style_attn": torch.empty(
            10, 18, 10, 3072, dtype=torch.bfloat16, device="cuda"
        ),
        "decoder_style_ffn": torch.empty_like(
            torch.empty(10, 18, 10, 3072, dtype=torch.bfloat16, device="cuda")
        ),
        "decoder_style_final": torch.empty(
            10, 10, 3072, dtype=torch.bfloat16, device="cuda"
        ),
    }
    for step in range(10):
        value = schedule[step : step + 1]
        value = (value @ time_in_w + time_in_b[None, :]).float()
        value = (value * torch.sigmoid(value)).to(torch.bfloat16)
        value = (value @ time_out_w + time_out_b[None, :]).float()
        value = (value * torch.sigmoid(value)).to(torch.bfloat16)
        expanded = value.expand(10, -1).contiguous()
        expected["decoder_time_emb"][step] = expanded
        for layer in range(18):
            expected["decoder_style_attn"][step, layer] = (
                expanded @ attn_w[layer] + attn_b[layer][None, :]
            )
            expected["decoder_style_ffn"][step, layer] = (
                expanded @ ffn_w[layer] + ffn_b[layer][None, :]
            )
        expected["decoder_style_final"][step] = (
            expanded @ final_w + final_b[None, :]
        )

    with tempfile.TemporaryDirectory() as directory:
        output_prefix = str(pathlib.Path(directory) / "styles")
        subprocess.check_call([args.probe, args.checkpoint, output_prefix])
        for name, reference in expected.items():
            actual_bits = np.fromfile(
                f"{output_prefix}.{name}.bin", dtype=np.uint16
            ).reshape(tuple(reference.shape))
            reference_bits = reference.contiguous().view(torch.uint16).cpu().numpy()
            exact = float(np.mean(actual_bits == reference_bits))
            actual = torch.from_numpy(actual_bits.copy()).view(torch.bfloat16).float()
            target = reference.cpu().float()
            maximum = float((actual - target).abs().max())
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    actual.flatten().double(), target.flatten().double(), dim=0
                )
            )
            if cosine < 0.9999:
                raise AssertionError(
                    f"{name}: exact={exact} max={maximum} cosine={cosine}"
                )
            print(
                f"PASS {name} exact={exact:.6f} "
                f"max={maximum:.6f} cosine={cosine:.8f}"
            )


if __name__ == "__main__":
    main()

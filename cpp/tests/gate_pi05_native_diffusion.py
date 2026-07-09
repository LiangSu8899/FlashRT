#!/usr/bin/env python3
import argparse
import gc
import math
import pathlib
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


CHUNK = 10
PREFIX = 712


def interleave_qk(weight: torch.Tensor, heads: int) -> torch.Tensor:
    output, inputs = weight.shape
    head_dim = output // heads
    return (
        weight.reshape(heads, head_dim, inputs)
        .reshape(heads, 2, head_dim // 2, inputs)
        .permute(0, 2, 1, 3)
        .reshape(output, inputs)
    )


def ada_rms(values: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    source = values.float()
    normalized = source * torch.rsqrt(source.square().mean(-1, keepdim=True) + 1e-6)
    scale, shift, gate = style.float().chunk(3, dim=-1)
    output = (normalized * (1.0 + scale) + shift).to(torch.bfloat16)
    return output, gate.to(torch.bfloat16)


def rotate(tensor: torch.Tensor, heads: int, rope: torch.Tensor) -> torch.Tensor:
    pairs = tensor.reshape(CHUNK, heads, 128, 2).float()
    cosine = rope[:, None, :, 0].float()
    sine = rope[:, None, :, 1].float()
    return torch.stack(
        [
            pairs[..., 0] * cosine - pairs[..., 1] * sine,
            pairs[..., 1] * cosine + pairs[..., 0] * sine,
        ],
        -1,
    ).to(torch.bfloat16).reshape(CHUNK, heads, 256)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--start-step", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.start_step < 0 or args.start_step + args.steps > 10:
        raise ValueError("start-step and steps must select a subset of [0, 10)")
    file = safe_open(f"{args.checkpoint}/model.safetensors", framework="pt")
    keys = set(file.keys())
    root = "model." if "model.action_in_proj.weight" in keys else ""

    def raw(name: str) -> torch.Tensor:
        return file.get_tensor(root + name)

    def bf16(name: str) -> torch.Tensor:
        return raw(name).to(device="cuda", dtype=torch.bfloat16)

    decoder = "paligemma_with_expert.gemma_expert.model.layers"
    time_in_w = bf16("time_mlp_in.weight").t().contiguous()
    time_in_b = bf16("time_mlp_in.bias")
    time_out_w = bf16("time_mlp_out.weight").t().contiguous()
    time_out_b = bf16("time_mlp_out.bias")
    attn_mod_w = [
        bf16(f"{decoder}.{i}.input_layernorm.dense.weight").t().contiguous()
        for i in range(18)
    ]
    attn_mod_b = [
        bf16(f"{decoder}.{i}.input_layernorm.dense.bias") for i in range(18)
    ]
    ffn_mod_w = [
        bf16(f"{decoder}.{i}.post_attention_layernorm.dense.weight")
        .t()
        .contiguous()
        for i in range(18)
    ]
    ffn_mod_b = [
        bf16(f"{decoder}.{i}.post_attention_layernorm.dense.bias")
        for i in range(18)
    ]
    final_mod_w = bf16(
        "paligemma_with_expert.gemma_expert.model.norm.dense.weight"
    ).t().contiguous()
    final_mod_b = bf16(
        "paligemma_with_expert.gemma_expert.model.norm.dense.bias"
    )
    fraction = torch.linspace(0.0, 1.0, 512)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    current = torch.tensor(1.0, dtype=torch.float32)
    schedule = []
    for _ in range(10):
        angle = current * (1.0 / period) * 2 * math.pi
        schedule.append(
            torch.cat([torch.sin(angle), torch.cos(angle)]).to(
                device="cuda", dtype=torch.bfloat16
            )
        )
        current = current - 0.1
    styles_attn = torch.empty(
        10, 18, CHUNK, 3072, device="cuda", dtype=torch.bfloat16
    )
    styles_ffn = torch.empty_like(styles_attn)
    styles_final = torch.empty(
        10, CHUNK, 3072, device="cuda", dtype=torch.bfloat16
    )
    step_range = range(args.start_step, args.start_step + args.steps)
    for step in step_range:
        value = schedule[step][None, :] @ time_in_w
        value = (value.float() + time_in_b.float()).to(torch.bfloat16)
        value_float = value.float()
        value = (value_float * torch.sigmoid(value_float)).to(torch.bfloat16)
        value = value @ time_out_w
        value = (value.float() + time_out_b.float()).to(torch.bfloat16)
        value_float = value.float()
        value = (value_float * torch.sigmoid(value_float)).to(torch.bfloat16)
        expanded = value.expand(CHUNK, -1).contiguous()
        for layer in range(18):
            styles_attn[step, layer] = (
                (expanded @ attn_mod_w[layer]).float()
                + attn_mod_b[layer].float()
            ).to(torch.bfloat16)
            styles_ffn[step, layer] = (
                (expanded @ ffn_mod_w[layer]).float()
                + ffn_mod_b[layer].float()
            ).to(torch.bfloat16)
        styles_final[step] = (
            (expanded @ final_mod_w).float() + final_mod_b.float()
        ).to(torch.bfloat16)

    layers = []
    for index in range(18):
        prefix = f"{decoder}.{index}"
        q = interleave_qk(raw(f"{prefix}.self_attn.q_proj.weight"), 8)
        k = interleave_qk(raw(f"{prefix}.self_attn.k_proj.weight"), 1)
        v = raw(f"{prefix}.self_attn.v_proj.weight")
        layers.append(
            {
                "qkv": torch.cat([q, k, v], dim=0)
                .t()
                .to(device="cuda", dtype=torch.bfloat16)
                .contiguous(),
                "output": bf16(f"{prefix}.self_attn.o_proj.weight")
                .t()
                .contiguous(),
                "gate": bf16(f"{prefix}.mlp.gate_proj.weight").t().contiguous(),
                "up": bf16(f"{prefix}.mlp.up_proj.weight").t().contiguous(),
                "down": bf16(f"{prefix}.mlp.down_proj.weight").t().contiguous(),
            }
        )

    positions = torch.arange(PREFIX, PREFIX + CHUNK, dtype=torch.float64)[:, None]
    pair = torch.arange(128, dtype=torch.float64)[None, :]
    phase = positions / torch.pow(10000.0, (2 * pair) / 256.0)
    rope = torch.stack([torch.cos(phase), torch.sin(phase)], -1).to(
        device="cuda", dtype=torch.bfloat16
    )
    layer_index = torch.arange(18, device="cuda")[:, None, None]
    row_index = torch.arange(722, device="cuda")[None, :, None]
    column_index = torch.arange(256, device="cuda")[None, None, :]
    cache_k = (
        ((layer_index + row_index + column_index) % 17 - 8).float() / 16.0
    ).to(torch.bfloat16)
    cache_v = (
        ((2 * layer_index + row_index + 3 * column_index) % 19 - 9).float()
        / 16.0
    ).to(torch.bfloat16)
    flat = torch.arange(CHUNK * 32, device="cuda")
    noise = (((flat % 23) - 11).float() / 12.0).to(torch.bfloat16).reshape(
        CHUNK, 32
    )
    input_weight = bf16("action_in_proj.weight").t().contiguous()
    input_bias = bf16("action_in_proj.bias")
    output_weight = (
        bf16("action_out_proj.weight").float() * -0.1
    ).to(torch.bfloat16).t().contiguous()
    output_bias = (
        bf16("action_out_proj.bias").float() * -0.1
    ).to(torch.bfloat16)

    for step in step_range:
        x = noise @ input_weight
        x = (x.float() + input_bias.float()).to(torch.bfloat16)
        for index, weights in enumerate(layers):
            x_norm, residual_gate = ada_rms(x, styles_attn[step, index])
            qkv = x_norm @ weights["qkv"]
            query, key, value = torch.split(qkv, [2048, 256, 256], dim=-1)
            query = rotate(query, 8, rope)
            key = rotate(key, 1, rope).reshape(CHUNK, 256)
            value = value.reshape(CHUNK, 256)
            cache_k[index, PREFIX : PREFIX + CHUNK] = key
            cache_v[index, PREFIX : PREFIX + CHUNK] = value
            attended = F.scaled_dot_product_attention(
                query.transpose(0, 1).unsqueeze(0),
                cache_k[index].reshape(722, 1, 256).transpose(0, 1).unsqueeze(0),
                cache_v[index].reshape(722, 1, 256).transpose(0, 1).unsqueeze(0),
                scale=1.0 / 16.0,
                enable_gqa=True,
            ).squeeze(0).transpose(0, 1).reshape(CHUNK, 2048)
            projected = attended @ weights["output"]
            x = (
                x.float() + projected.float() * residual_gate.float()
            ).to(torch.bfloat16)
            x_norm, residual_gate = ada_rms(x, styles_ffn[step, index])
            gate = x_norm @ weights["gate"]
            up = x_norm @ weights["up"]
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
            down = hidden @ weights["down"]
            x = (x.float() + down.float() * residual_gate.float()).to(
                torch.bfloat16
            )
        x_norm, _ = ada_rms(x, styles_final[step])
        action = x_norm @ output_weight
        action = (action.float() + output_bias.float()).to(torch.bfloat16)
        noise = (noise.float() + action.float()).to(torch.bfloat16)

    expected = noise.cpu().float()
    del cache_k, cache_v, noise, x, x_norm, action
    gc.collect()
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory() as directory:
        output = str(pathlib.Path(directory) / "diffusion.bin")
        subprocess.check_call(
            [
                args.probe,
                args.checkpoint,
                output,
                str(args.steps),
                str(args.start_step),
            ]
        )
        bits = np.fromfile(output, dtype=np.uint16)
    if bits.size != CHUNK * 32:
        raise AssertionError(f"diffusion probe output elements={bits.size}")
    actual = torch.from_numpy(bits.copy()).view(torch.bfloat16).float().reshape(
        CHUNK, 32
    )
    cosine = float(
        F.cosine_similarity(
            actual.flatten().double(), expected.flatten().double(), dim=0
        )
    )
    maximum = float((actual - expected).abs().max())
    if cosine < 0.9999:
        raise AssertionError(f"cosine={cosine:.8f} max={maximum:.6f}")
    print(
        f"PASS diffusion steps {args.start_step}.."
        f"{args.start_step + args.steps - 1} "
        f"cosine={cosine:.8f} max={maximum:.6f}"
    )


if __name__ == "__main__":
    main()

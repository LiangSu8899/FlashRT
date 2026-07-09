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


NUM_VIEWS = 2
SEQUENCE = NUM_VIEWS * 256
WIDTH = 1152
HIDDEN = 4304


def layer_norm(
    values: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    source = values.float()
    mean = source.mean(-1, keepdim=True)
    variance = (source - mean).square().mean(-1, keepdim=True)
    return (
        (source - mean)
        * torch.rsqrt(variance + 1e-5)
        * weight.float()
        + bias.float()
    ).to(torch.bfloat16)


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
    vision = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"

    def raw(name: str) -> torch.Tensor:
        return file.get_tensor(root + name)

    def bf16(name: str) -> torch.Tensor:
        return raw(name).to(device="cuda", dtype=torch.bfloat16)

    flat = torch.arange(NUM_VIEWS * 224 * 224 * 3, device="cuda")
    images = (((flat % 257) - 128).float() / 128.0).to(torch.bfloat16).reshape(
        NUM_VIEWS, 224, 224, 3
    )
    patches = (
        images.reshape(NUM_VIEWS, 16, 14, 16, 14, 3)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(SEQUENCE, 588)
    )
    patch_weight = bf16(f"{vision}.embeddings.patch_embedding.weight")
    patch_weight = patch_weight.permute(2, 3, 1, 0).reshape(588, WIDTH)
    patch_bias = bf16(f"{vision}.embeddings.patch_embedding.bias")
    position = bf16(f"{vision}.embeddings.position_embedding.weight").repeat(
        NUM_VIEWS, 1
    )
    x = (patches @ patch_weight).to(torch.bfloat16)
    x = (x.float() + position.float() + patch_bias.float()).to(torch.bfloat16)
    first = f"{vision}.encoder.layers.0"
    x_norm = layer_norm(
        x, bf16(f"{first}.layer_norm1.weight"), bf16(f"{first}.layer_norm1.bias")
    )

    for index in range(27):
        layer = f"{vision}.encoder.layers.{index}"
        q_weight = bf16(f"{layer}.self_attn.q_proj.weight")
        k_weight = bf16(f"{layer}.self_attn.k_proj.weight")
        v_weight = bf16(f"{layer}.self_attn.v_proj.weight")
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0).t().contiguous()
        qkv_bias = torch.cat(
            [
                bf16(f"{layer}.self_attn.q_proj.bias"),
                bf16(f"{layer}.self_attn.k_proj.bias"),
                bf16(f"{layer}.self_attn.v_proj.bias"),
            ]
        )
        qkv = x_norm @ qkv_weight
        qkv = (qkv.float() + qkv_bias.float()).to(torch.bfloat16)
        query, key, value = qkv.reshape(NUM_VIEWS, 256, 3, 16, 72).unbind(2)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            scale=1.0 / np.sqrt(72.0),
        ).transpose(1, 2).reshape(SEQUENCE, WIDTH)
        output_weight = bf16(f"{layer}.self_attn.out_proj.weight").t().contiguous()
        output_bias = bf16(f"{layer}.self_attn.out_proj.bias")
        projected = attended @ output_weight
        x = (x.float() + projected.float() + output_bias.float()).to(torch.bfloat16)
        x_norm = layer_norm(
            x,
            bf16(f"{layer}.layer_norm2.weight"),
            bf16(f"{layer}.layer_norm2.bias"),
        )
        up_weight = bf16(f"{layer}.mlp.fc1.weight").t().contiguous()
        up_bias = bf16(f"{layer}.mlp.fc1.bias")
        hidden = x_norm @ up_weight
        hidden = (hidden.float() + up_bias.float()).to(torch.bfloat16)
        hidden_float = hidden.float()
        hidden = (
            hidden_float
            * 0.5
            * (
                1.0
                + torch.tanh(
                    0.7978845608
                    * (hidden_float + 0.044715 * hidden_float.pow(3))
                )
            )
        ).to(torch.bfloat16)
        down_weight = bf16(f"{layer}.mlp.fc2.weight").t().contiguous()
        down_bias = bf16(f"{layer}.mlp.fc2.bias")
        down = hidden @ down_weight
        x = (x.float() + down.float() + down_bias.float()).to(torch.bfloat16)
        if index != 26:
            next_layer = f"{vision}.encoder.layers.{index + 1}"
            x_norm = layer_norm(
                x,
                bf16(f"{next_layer}.layer_norm1.weight"),
                bf16(f"{next_layer}.layer_norm1.bias"),
            )
        del q_weight, k_weight, v_weight, qkv_weight, qkv_bias, qkv
        del query, key, value, attended, output_weight, output_bias, projected
        del up_weight, up_bias, hidden, hidden_float, down_weight, down_bias, down
        gc.collect()

    expected_vision = x.cpu().float()
    final_norm = layer_norm(
        x,
        bf16(f"{vision}.post_layernorm.weight"),
        bf16(f"{vision}.post_layernorm.bias"),
    )
    projector = (
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
    )
    projected = final_norm @ bf16(f"{projector}.weight").t().contiguous()
    expected_encoder = (
        projected.float() + bf16(f"{projector}.bias").float()
    ).to(torch.bfloat16).cpu().float()
    del x, x_norm, final_norm, projected, images, patches
    torch.cuda.empty_cache()

    with tempfile.TemporaryDirectory() as directory:
        output = str(pathlib.Path(directory) / "vision.bin")
        subprocess.check_call([args.probe, args.checkpoint, output])
        bits = np.fromfile(output, dtype=np.uint16)
    sizes = [SEQUENCE * WIDTH, SEQUENCE * 2048]
    if bits.size != sum(sizes):
        raise AssertionError(f"vision probe output elements={bits.size}")
    vision_bits = bits[: sizes[0]].copy()
    encoder_bits = bits[sizes[0] :].copy()
    actual_vision = (
        torch.from_numpy(vision_bits).view(torch.bfloat16).float().reshape(SEQUENCE, WIDTH)
    )
    actual_encoder = (
        torch.from_numpy(encoder_bits)
        .view(torch.bfloat16)
        .float()
        .reshape(SEQUENCE, 2048)
    )
    print(
        "PASS vision 27 layers "
        + compare("vision", actual_vision, expected_vision)
        + "; "
        + compare("encoder", actual_encoder, expected_encoder)
    )


if __name__ == "__main__":
    main()

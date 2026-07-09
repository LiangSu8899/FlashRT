#!/usr/bin/env python3
import argparse
import subprocess

import torch
from safetensors import safe_open


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


def digest(tensor: torch.Tensor) -> int:
    return fnv1a(tensor.contiguous().cpu().numpy().tobytes())


def parse_probe(text: str) -> tuple[tuple[int, ...], int, int, int]:
    fields = dict(field.split("=", 1) for field in text.strip().split())
    return (
        tuple(int(dim) for dim in fields["shape"].split(",")),
        int(fields["values_fnv"], 16),
        int(fields["scale_shape"]),
        int(fields["scales_fnv"], 16),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the producer quantization gate")
    file = safe_open(f"{args.checkpoint}/model.safetensors", framework="pt")
    keys = set(file.keys())
    prefix = "model." if "model.action_in_proj.weight" in keys else ""

    def bf16(key: str) -> torch.Tensor:
        return file.get_tensor(prefix + key).to(torch.bfloat16)

    q = interleave_qk(bf16(f"{DECODER}.self_attn.q_proj.weight").float(), 8)
    k = interleave_qk(bf16(f"{DECODER}.self_attn.k_proj.weight").float(), 1)
    v = bf16(f"{DECODER}.self_attn.v_proj.weight")
    weight = torch.cat([q, k, v], dim=0).t().to(
        device="cuda", dtype=torch.bfloat16
    ).contiguous()

    expected = {}
    for layout in ("kn", "nk"):
        arranged = weight.t().contiguous() if layout == "nk" else weight
        scale = max(arranged.float().abs().max().item() / 448.0, 1e-12)
        quantized = (arranged.float() / scale).clamp(-448.0, 448.0).to(
            torch.float8_e4m3fn
        )
        scale_tensor = torch.tensor([scale], dtype=torch.float32, device="cuda")
        expected[f"decoder_qkv0_fp8_{layout}"] = (
            tuple(quantized.shape),
            digest(quantized.view(torch.uint8)),
            1,
            digest(scale_tensor),
        )

    transposed = weight.float().transpose(0, 1).contiguous()
    scales = torch.clamp(
        transposed.abs().amax(dim=1) / 127.0, min=1e-12
    ).to(dtype=torch.float32).contiguous()
    quantized = torch.clamp(
        torch.round(transposed / scales[:, None]), -127, 127
    ).to(torch.int8).contiguous()
    expected["decoder_qkv0_int8"] = (
        tuple(quantized.shape),
        digest(quantized),
        scales.numel(),
        digest(scales),
    )

    for operation, reference in expected.items():
        output = subprocess.check_output(
            [args.probe, args.checkpoint, operation], text=True
        )
        actual = parse_probe(output)
        if actual != reference:
            raise AssertionError(
                f"{operation}: C++ {actual} != PyTorch {reference}"
            )
        print(
            f"PASS {operation} shape={actual[0]} "
            f"values_fnv={actual[1]:016x} scales_fnv={actual[3]:016x}"
        )


if __name__ == "__main__":
    main()

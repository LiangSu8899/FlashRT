#!/usr/bin/env python3
import argparse
import subprocess

import ml_dtypes
import numpy as np


def fnv1a(data: bytes) -> int:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def parse_probe(text: str) -> dict[str, str]:
    return dict(field.split("=", 1) for field in text.strip().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    args = parser.parse_args()
    cases = [(2, 200, 10, 1, 37), (3, 256, 50, 2, 256)]
    for views, max_prompt, chunk, pool, prompt in cases:
        vision = views * 256 // (pool * pool)
        encoder_length = vision + max_prompt
        max_positions = encoder_length + chunk
        inverse_frequency = 1.0 / (
            10000 ** (np.arange(0, 256, 2, dtype=np.float64) / 256)
        )
        positions = np.arange(max_positions, dtype=np.float64)
        phase = positions[:, None] * inverse_frequency[None, :]
        cosine = np.cos(phase).astype(ml_dtypes.bfloat16)
        sine = np.sin(phase).astype(ml_dtypes.bfloat16)
        table = np.stack([cosine, sine], axis=-1).reshape(max_positions, 256)
        encoder = np.ascontiguousarray(table[:encoder_length])
        decoder = np.ascontiguousarray(
            table[vision + prompt : vision + prompt + chunk]
        )
        output = subprocess.check_output(
            [
                args.probe,
                str(views),
                str(max_prompt),
                str(chunk),
                str(pool),
                str(prompt),
            ],
            text=True,
        )
        actual = parse_probe(output)
        expected = {
            "encoder_shape": f"{encoder_length},256",
            "encoder_fnv": f"{fnv1a(encoder.tobytes()):016x}",
            "decoder_shape": f"{chunk},256",
            "decoder_fnv": f"{fnv1a(decoder.tobytes()):016x}",
        }
        if actual != expected:
            raise AssertionError(f"C++ {actual} != NumPy {expected}")
        print(
            f"PASS views={views} pool={pool} prompt={prompt} "
            f"encoder_fnv={actual['encoder_fnv']} "
            f"decoder_fnv={actual['decoder_fnv']}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Emit deterministic hashes for all five FA2 adapter or raw C entries."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Shape:
    batch: int
    q: int
    kv: int
    q_heads: int
    kv_heads: int
    dim: int


def _tensor(shape: tuple[int, ...], dtype: torch.dtype, offset: int) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    values = torch.arange(count, dtype=torch.float32)
    values = torch.sin(values * 0.017 + offset * 0.13) * 0.25
    return values.reshape(shape).to(dtype).cuda()


def _digest(*tensors: torch.Tensor) -> str:
    result = hashlib.sha256()
    for tensor in tensors:
        result.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


class Adapter:
    def __init__(self) -> None:
        from flash_rt import flash_rt_fa2

        self.module = flash_rt_fa2

    def standard(self, name: str, tensors: dict, shape: Shape,
                 stream: torch.cuda.Stream) -> None:
        getattr(self.module, name)(
            tensors["q"].data_ptr(), tensors["k"].data_ptr(),
            tensors["v"].data_ptr(), tensors["o"].data_ptr(),
            tensors["lse"].data_ptr(), 0, 0,
            batch=shape.batch, seqlen_q=shape.q, seqlen_k=shape.kv,
            num_heads_q=shape.q_heads, num_heads_kv=shape.kv_heads,
            head_dim=shape.dim, q_strides=tensors["q"].stride()[:3],
            k_strides=tensors["k"].stride()[:3],
            v_strides=tensors["v"].stride()[:3],
            o_strides=tensors["o"].stride()[:3],
            softmax_scale=shape.dim ** -0.5, num_sms=0,
            stream=stream.cuda_stream)

    def seqused(self, tensors: dict, shape: Shape,
                 stream: torch.cuda.Stream, split: bool) -> None:
        kwargs = dict(
            batch=shape.batch, seqlen_q=shape.q, seqlen_k=shape.kv,
            num_heads_q=shape.q_heads, num_heads_kv=shape.kv_heads,
            head_dim=shape.dim, q_strides=tensors["q"].stride()[:3],
            k_strides=tensors["k"].stride()[:3],
            v_strides=tensors["v"].stride()[:3],
            o_strides=tensors["o"].stride()[:3],
            softmax_scale=shape.dim ** -0.5,
            num_sms=torch.cuda.get_device_properties(0).multi_processor_count,
            stream=stream.cuda_stream)
        args = [
            tensors["q"].data_ptr(), tensors["k"].data_ptr(),
            tensors["v"].data_ptr(), tensors["o"].data_ptr(),
            tensors["lse"].data_ptr(), tensors["seqused"].data_ptr(),
        ]
        if split:
            args += [tensors["lse_accum"].data_ptr(), tensors["o_accum"].data_ptr()]
            self.module.fwd_bf16_seqused_splitkv(*args, **kwargs)
        else:
            self.module.fwd_bf16_seqused(*args, **kwargs)


class Raw:
    _void = ctypes.c_void_p
    _int = ctypes.c_int
    _float = ctypes.c_float
    _shape_args = [_int] * 18 + [_float, _int, _void]

    def __init__(self, path: str) -> None:
        self.library = ctypes.CDLL(path)
        for name in ("fvk_attention_fa2_fwd_fp16",
                     "fvk_attention_fa2_fwd_bf16",
                     "fvk_attention_fa2_fwd_bf16_causal"):
            fn = getattr(self.library, name)
            fn.argtypes = [self._void] * 7 + self._shape_args
            fn.restype = None
        self.library.fvk_attention_fa2_fwd_bf16_seqused.argtypes = (
            [self._void] * 6 + self._shape_args)
        self.library.fvk_attention_fa2_fwd_bf16_seqused.restype = None
        self.library.fvk_attention_fa2_fwd_bf16_seqused_splitkv.argtypes = (
            [self._void] * 8 + self._shape_args)
        self.library.fvk_attention_fa2_fwd_bf16_seqused_splitkv.restype = None

    @staticmethod
    def _ptr(value: int) -> ctypes.c_void_p:
        return ctypes.c_void_p(value)

    @staticmethod
    def _shape(shape: Shape, tensors: dict,
               stream: torch.cuda.Stream, num_sms: int) -> list:
        strides = (
            list(tensors["q"].stride()[:3]) + list(tensors["k"].stride()[:3])
            + list(tensors["v"].stride()[:3]) + list(tensors["o"].stride()[:3])
        )
        return [shape.batch, shape.q, shape.kv, shape.q_heads, shape.kv_heads,
                shape.dim, *strides, shape.dim ** -0.5, num_sms,
                ctypes.c_void_p(stream.cuda_stream)]

    def standard(self, name: str, tensors: dict, shape: Shape,
                 stream: torch.cuda.Stream) -> None:
        args = [self._ptr(tensors[key].data_ptr()) for key in
                ("q", "k", "v", "o", "lse")]
        args += [self._ptr(0), self._ptr(0)]
        getattr(self.library, f"fvk_attention_fa2_{name}")(
            *args, *self._shape(shape, tensors, stream, 0))

    def seqused(self, tensors: dict, shape: Shape,
                 stream: torch.cuda.Stream, split: bool) -> None:
        args = [self._ptr(tensors[key].data_ptr()) for key in
                ("q", "k", "v", "o", "lse", "seqused")]
        if split:
            args += [self._ptr(tensors["lse_accum"].data_ptr()),
                     self._ptr(tensors["o_accum"].data_ptr())]
            fn = self.library.fvk_attention_fa2_fwd_bf16_seqused_splitkv
        else:
            fn = self.library.fvk_attention_fa2_fwd_bf16_seqused
        fn(*args, *self._shape(
            shape, tensors, stream,
            torch.cuda.get_device_properties(0).multi_processor_count))


def _buffers(shape: Shape, dtype: torch.dtype, offset: int,
             seqused: int | None = None, split: bool = False) -> dict:
    tensors = {
        "q": _tensor((shape.batch, shape.q, shape.q_heads, shape.dim), dtype, offset),
        "k": _tensor((shape.batch, shape.kv, shape.kv_heads, shape.dim), dtype, offset + 1),
        "v": _tensor((shape.batch, shape.kv, shape.kv_heads, shape.dim), dtype, offset + 2),
        "o": torch.empty((shape.batch, shape.q, shape.q_heads, shape.dim),
                         dtype=dtype, device="cuda"),
        "lse": torch.empty((shape.batch, shape.q_heads, shape.q),
                           dtype=torch.float32, device="cuda"),
    }
    if seqused is not None:
        tensors["seqused"] = torch.full(
            (shape.batch,), seqused, dtype=torch.int32, device="cuda")
    if split:
        max_splits = 128
        rounded_dim = (shape.dim + 31) & ~31
        tensors["lse_accum"] = torch.full(
            (max_splits, shape.batch, shape.q_heads, shape.q),
            float("-inf"), dtype=torch.float32, device="cuda")
        tensors["o_accum"] = torch.zeros(
            (max_splits, shape.batch, shape.q_heads, shape.q, rounded_dim),
            dtype=torch.float32, device="cuda")
    return tensors


def run(backend) -> dict[str, str]:
    results: dict[str, str] = {}
    stream = torch.cuda.Stream()
    offset = 1

    standard_cases = [
        ("fwd_fp16", torch.float16, Shape(1, 3, 7, 4, 2, 48), "fwd_fp16_d48"),
        ("fwd_fp16", torch.float16, Shape(1, 3, 7, 4, 2, 72), "fwd_fp16_d72"),
        ("fwd_fp16", torch.float16, Shape(1, 3, 7, 4, 2, 96), "fwd_fp16_d96"),
        ("fwd_fp16", torch.float16, Shape(1, 3, 7, 4, 2, 128), "fwd_fp16_d128"),
        ("fwd_fp16", torch.float16, Shape(1, 3, 7, 4, 2, 256), "fwd_fp16_d256"),
        ("fwd_bf16", torch.bfloat16, Shape(1, 3, 7, 4, 2, 64), "fwd_bf16_d64"),
        ("fwd_bf16", torch.bfloat16, Shape(1, 3, 7, 4, 2, 96), "fwd_bf16_d96"),
        ("fwd_bf16", torch.bfloat16, Shape(1, 3, 7, 4, 2, 128), "fwd_bf16_d128"),
        ("fwd_bf16", torch.bfloat16, Shape(1, 3, 7, 4, 2, 256), "fwd_bf16_d256"),
        ("fwd_fp16", torch.float16, Shape(3, 3, 7, 2, 2, 96),
         "fwd_fp16_mha_batch3_d96"),
        ("fwd_bf16", torch.bfloat16, Shape(3, 3, 7, 4, 2, 128),
         "fwd_bf16_gqa_batch3_d128"),
    ]
    with torch.cuda.stream(stream):
        for name, dtype, shape, result_name in standard_cases:
            tensors = _buffers(shape, dtype, offset)
            backend.standard(name, tensors, shape, stream)
            stream.synchronize()
            results[result_name] = _digest(tensors["o"], tensors["lse"])
            offset += 3

        for dim in (128, 256):
            shape = Shape(1, 8, 8, 4, 2, dim)
            tensors = _buffers(shape, torch.bfloat16, offset)
            backend.standard("fwd_bf16_causal", tensors, shape, stream)
            stream.synchronize()
            results[f"fwd_bf16_causal_d{dim}"] = _digest(
                tensors["o"], tensors["lse"])
            offset += 3

        for dim in (64, 96, 128, 256):
            shape = Shape(1, 1, 32, 4, 2, dim)
            tensors = _buffers(shape, torch.bfloat16, offset, seqused=13)
            backend.seqused(tensors, shape, stream, split=False)
            stream.synchronize()
            results[f"fwd_bf16_seqused_d{dim}"] = _digest(
                tensors["o"], tensors["lse"])
            offset += 3

        for dim in (96, 256):
            shape = Shape(1, 1, 1024, 4, 2, dim)
            tensors = _buffers(shape, torch.bfloat16, offset,
                               seqused=769, split=True)
            backend.seqused(tensors, shape, stream, split=True)
            stream.synchronize()
            results[f"fwd_bf16_seqused_splitkv_d{dim}"] = _digest(
                tensors["o"], tensors["lse"])
            offset += 3
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-library")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    backend = Raw(args.raw_library) if args.raw_library else Adapter()
    print(json.dumps(run(backend), sort_keys=True))


if __name__ == "__main__":
    main()

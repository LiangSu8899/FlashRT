#!/usr/bin/env python3
"""Compare the native Thor FP8 producer with the shipped Torch producer."""

from __future__ import annotations

import argparse
import os
import subprocess
import types
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import flash_rt.flash_rt_kernels as fvk
from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
from flash_rt.hardware.thor.shared_primitives import encoder_forward_calibrate
from flash_rt.models.pi05.pipeline_thor import decoder_forward_calibrate


def _artifact(path: Path) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
    with safe_open(path, framework="np") as handle:
        metadata = handle.metadata() or {}
        encoder = handle.get_tensor("encoder_scales").astype(
            np.float32, copy=True)
        decoder = handle.get_tensor("decoder_scales").astype(
            np.float32, copy=True)
    return metadata, encoder, decoder


def _sample(index: int, num_views: int
            ) -> tuple[str, np.ndarray, list[np.ndarray], np.ndarray]:
    pixels = np.arange(224 * 224 * 3, dtype=np.uint64)
    image = ((pixels * 3 + index * 17) % 251).astype(np.uint8)
    wrist = ((pixels * 7 + index * 29 + 11) % 253).astype(np.uint8)
    state = np.asarray(
        [((index * 8 + dim) % 17 - 8) / 8.0 for dim in range(8)],
        dtype=np.float32,
    )
    noise = np.asarray(
        [((index * 320 + item) % 31 - 15) / 16.0 for item in range(320)],
        dtype=np.float32,
    ).reshape(10, 32)
    prompt = (
        "move the black bowl to the plate"
        if index & 1
        else "pick up the black bowl"
    )
    images = [image.reshape(224, 224, 3), wrist.reshape(224, 224, 3)]
    return (
        prompt,
        state,
        images[:num_views],
        noise,
    )


def _normalized_state(frontend: Pi05TorchFrontendThor,
                      state: np.ndarray) -> np.ndarray:
    q01 = np.asarray(frontend.norm_stats["state"]["q01"], dtype=np.float32)
    q99 = np.asarray(frontend.norm_stats["state"]["q99"], dtype=np.float32)
    return (
        ((state - q01) / (q99 - q01 + np.float32(1.0e-6)))
        * np.float32(2.0)
        - np.float32(1.0)
    ).astype(np.float32)


def _seed_calibration(
    encoder: np.ndarray,
    decoder: np.ndarray,
):
    def seed(self: Pi05TorchFrontendThor, _sequence: int) -> None:
        self._enc_calib_scales.copy_(torch.from_numpy(encoder))
        self._ae_calib_scales.copy_(torch.from_numpy(decoder))
        weight_scales = self._enc_w_dev.cpu().numpy()
        self._enc_alpha_host = [
            float(np.float32(encoder[i]) * np.float32(weight_scales[i]))
            for i in range(encoder.size)
        ]

    return seed


class CalibrationOracle:
    def __init__(self, frontend: Pi05TorchFrontendThor):
        self.frontend = frontend
        p = frontend
        sequence = p.Se
        width = p.De
        hidden = p.He
        action_width = p.Da
        action_hidden = p.Ha
        self._encoder_norm_scratch = torch.empty(
            sequence * width, dtype=torch.float16, device="cuda")
        self._encoder_x_scratch = torch.empty(
            sequence * width, dtype=torch.float16, device="cuda")
        self._encoder_calib_buffer = torch.zeros(
            p.Le * 4, dtype=torch.float32, device="cuda")
        self._encoder_dynamic_scale = torch.zeros(
            1, dtype=torch.float32, device="cuda")
        self._encoder_fp8_scratch = torch.zeros(
            sequence * max(width, hidden), dtype=torch.uint8, device="cuda")
        self._encoder_ones = torch.ones(
            width, dtype=torch.float16, device="cuda")
        self._decoder_calib_buffer = torch.zeros(
            p.steps * p.La * 4, dtype=torch.float32, device="cuda")
        self._decoder_dynamic_scale = torch.zeros(
            1, dtype=torch.float32, device="cuda")
        self._decoder_hidden_scratch = torch.empty(
            p.Sa * action_hidden, dtype=torch.float16, device="cuda")
        self._decoder_fp8_scratch = torch.zeros(
            p.Sa * max(action_width, action_hidden), dtype=torch.uint8,
            device="cuda")

        self.encoder_buffers = {
            "x": p._enc_x.data_ptr(),
            "x_fp8": p._enc_x_fp8.data_ptr(),
            "qkv": p._enc_qkv_buf.data_ptr(),
            "logits": p._enc_logits.data_ptr(),
            "attn_out": p._enc_attn.data_ptr(),
            "o_fp8": p._enc_o_fp8.data_ptr(),
            "gate": p._enc_gate.data_ptr(),
            "hidden": p._enc_hidden.data_ptr(),
            "hid_fp8": p._enc_hid_fp8.data_ptr(),
            "fg": p._enc_fg.data_ptr(),
            "ctx": p._ctx,
            "norm_scratch": self._encoder_norm_scratch.data_ptr(),
            "x_scratch": self._encoder_x_scratch.data_ptr(),
            "calib_buf": self._encoder_calib_buffer.data_ptr(),
            "d_scale": self._encoder_dynamic_scale.data_ptr(),
            "fp8_scratch": self._encoder_fp8_scratch.data_ptr(),
            "ones": self._encoder_ones.data_ptr(),
        }
        self.encoder_weights = {
            "qkv_w": [weight.data_ptr() for weight in p._enc_qkv_w],
            "o_w": [weight.data_ptr() for weight in p._enc_o_w],
            "gate_w": [weight.data_ptr() for weight in p._enc_gu_w],
            "down_w": [weight.data_ptr() for weight in p._enc_d_w],
            "rope": p._enc_rope.data_ptr(),
            "Kc": p._Kc.reshape(-1).data_ptr(),
            "Vc": p._Vc.reshape(-1).data_ptr(),
            "w_scales": p._enc_w_dev.data_ptr(),
        }
        self.encoder_dims = {
            "Se": sequence,
            "D": width,
            "H": hidden,
            "NH": p.NHe,
            "HD": p.HDe,
            "L": p.Le,
            "total_keys": p.total_keys,
        }

        self.decoder_buffers = {
            "noise": p._g_noise.data_ptr(),
            "x": p._ae_x.data_ptr(),
            "xn": p._ae_xn.data_ptr(),
            "gate": p._ae_gate.data_ptr(),
            "qkv": p._ae_qkv.data_ptr(),
            "logits": p._ae_logits.data_ptr(),
            "attn_out": p._ae_attn.data_ptr(),
            "hid": p._ae_hid.data_ptr(),
            "fg": p._ae_fg.data_ptr(),
            "action_f32": p._ae_action_f32.data_ptr(),
            "xn_fp8": p._ae_xn_fp8.data_ptr(),
            "hid_fp8": p._ae_hid_fp8.data_ptr(),
            "ctx_fp8": p._ae_ctx_fp8.data_ptr(),
            "calib_buf": self._decoder_calib_buffer.data_ptr(),
            "d_scale": self._decoder_dynamic_scale.data_ptr(),
            "hidden_scratch": self._decoder_hidden_scratch.data_ptr(),
            "fp8_scratch": self._decoder_fp8_scratch.data_ptr(),
        }
        self.decoder_weights = {
            "ain_w": p._ain_w.data_ptr(),
            "ain_b": p._ain_b.data_ptr(),
            "sa": p._sa_all.data_ptr(),
            "qw": p._dec_qkv_flat.data_ptr(),
            "Kc": p._Kc.reshape(-1).data_ptr(),
            "Vc": p._Vc.reshape(-1).data_ptr(),
            "dec_devpos": p._attn.dec_devpos.data_ptr(),
            "ow": p._dec_o_flat.data_ptr(),
            "sf": p._sf_all.data_ptr(),
            "gw": p._dec_gu_flat.data_ptr(),
            "dw": p._dec_d_flat.data_ptr(),
            "aow": p._aow.data_ptr(),
            "aob": p._aob.data_ptr(),
            "aob_dt": p._aob_dt.data_ptr(),
            "dt": p._ae_dt,
            "fs": p._fs_all.data_ptr(),
            "rope": p._dec_rope.data_ptr(),
            "w_scales": p._ae_w_dev.data_ptr(),
        }
        self.decoder_dims = {
            "S": p.Sa,
            "D": action_width,
            "H": action_hidden,
            "NH": 8,
            "HD": 256,
            "steps": p.steps,
            "layers": p.La,
            "enc_seq": sequence,
            "total_keys": p.total_keys,
            "fixed_shape": p._fixed_shape_active,
        }

    def observe(self, images: list[np.ndarray], noise: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        p = self.frontend
        normalized = np.stack([
            (image.astype(np.float32) / np.float32(127.5)
             - np.float32(1.0)).astype(np.float16)
            for image in images
        ])
        p._img_buf.upload(normalized)
        p._siglip_graph.replay()
        torch.cuda.synchronize()

        p._enc_calib_scales.zero_()
        p._Kc.zero_()
        p._Vc.zero_()
        encoder_forward_calibrate(
            p._gemm, fvk, self.encoder_buffers, self.encoder_weights,
            self.encoder_dims, p._enc_calib_scales.data_ptr(), stream=0,
            attn=p._attn,
        )
        torch.cuda.synchronize()
        encoder = p._enc_calib_scales.cpu().numpy().copy()

        p._ae_calib_scales.zero_()
        p._g_noise.copy_(torch.from_numpy(noise.astype(np.float16)).cuda())
        decoder_forward_calibrate(
            p._ctx, fvk, self.decoder_buffers, self.decoder_weights,
            self.decoder_dims, p._ae_calib_scales.data_ptr(), stream=0,
            attn=p._attn,
        )
        torch.cuda.synchronize()
        decoder = p._ae_calib_scales.cpu().numpy().copy()
        return encoder, decoder


def _assert_equal(label: str, actual: np.ndarray,
                  expected: np.ndarray) -> None:
    if np.array_equal(actual, expected):
        print(f"PASS {label}: bit-exact ({actual.size} values)")
        return
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    difference = np.abs(
        actual_flat.astype(np.float64) - expected_flat.astype(np.float64)
    )
    index = int(np.argmax(difference))
    first = int(np.flatnonzero(actual_flat != expected_flat)[0])
    mismatches = int(np.count_nonzero(actual_flat != expected_flat))
    raise AssertionError(
        f"{label} mismatch: max_abs={difference[index]:.9g} "
        f"at {index}, actual={actual_flat[index]!r}, "
        f"expected={expected_flat[index]!r}; "
        f"first={first}, actual={actual_flat[first]!r}, "
        f"expected={expected_flat[first]!r}, "
        f"mismatches={mismatches}/{actual.size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--views", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.samples < 1 or args.samples > 256:
        parser.error("--samples must be in [1, 256]")

    probe = args.probe.resolve()
    os.environ["FLASH_RT_PALIGEMMA_TOKENIZER"] = str(
        args.tokenizer.resolve()
    )
    raw_path = args.artifact.with_suffix(args.artifact.suffix + ".raw")
    env = os.environ.copy()
    library_paths = [probe.parent, probe.parent / "exec",
                     probe.parent / "runtime"]
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(Path(env["LD_LIBRARY_PATH"]))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, library_paths))
    subprocess.run(
        [
            str(probe),
            str(args.checkpoint),
            str(args.tokenizer),
            str(args.artifact),
            str(args.samples),
            str(args.views),
            str(raw_path),
        ],
        check=True,
        env=env,
    )
    metadata, expected_encoder, expected_decoder = _artifact(args.artifact)
    _, single_encoder, single_decoder = _artifact(
        Path(str(args.artifact) + ".single")
    )
    if int(metadata.get("sample_count", "0")) != args.samples:
        raise AssertionError("native calibration sample_count is incorrect")
    if int(metadata.get("num_views", "0")) != args.views:
        raise AssertionError("native calibration num_views is incorrect")

    frontend = Pi05TorchFrontendThor(
        str(args.checkpoint), num_views=args.views, use_cuda_graph=True,
        autotune=0,
        use_fp8=True, state_prompt_mode="fixed",
        state_prompt_fixed_max_len=200,
    )
    frontend._calibrate = types.MethodType(
        _seed_calibration(expected_encoder, expected_decoder), frontend
    )
    first_prompt, first_state, _, _ = _sample(0, args.views)
    frontend.set_prompt(
        first_prompt, state=_normalized_state(frontend, first_state)
    )
    oracle = CalibrationOracle(frontend)

    encoder_samples = []
    decoder_samples = []
    samples = []
    for index in range(args.samples):
        prompt, state, images, noise = _sample(index, args.views)
        samples.append((prompt, state, images, noise))
        frontend.set_prompt(
            prompt, state=_normalized_state(frontend, state)
        )
        encoder, decoder = oracle.observe(images, noise)
        # The final encoder layer only emits decoder K/V. Its O/FFN sites are
        # skipped; the native artifact uses 1.0 as their valid neutral value.
        encoder[-3:] = np.float32(1.0)
        encoder_samples.append(encoder)
        decoder_samples.append(decoder)

    _assert_equal("single-sample encoder calibration",
                  encoder_samples[0], single_encoder)
    _assert_equal("single-sample decoder calibration",
                  decoder_samples[0], single_decoder)
    reduced_encoder = np.percentile(
        np.stack(encoder_samples), 99.9, axis=0
    ).astype(np.float32)
    reduced_decoder = np.percentile(
        np.stack(decoder_samples), 99.9, axis=0
    ).astype(np.float32)
    _assert_equal("dataset encoder calibration",
                  reduced_encoder, expected_encoder)
    _assert_equal("dataset decoder calibration",
                  reduced_decoder, expected_decoder)

    _, state, images, noise = samples[-1]
    frontend.set_prompt(
        "pick up the black bowl", state=_normalized_state(frontend, state)
    )
    frontend._enc_calib_scales.copy_(torch.from_numpy(expected_encoder))
    frontend._ae_calib_scales.copy_(torch.from_numpy(expected_decoder))
    weight_scales = frontend._enc_w_dev.cpu().numpy()
    frontend._enc_alpha_host = [
        float(np.float32(expected_encoder[i]) * np.float32(weight_scales[i]))
        for i in range(expected_encoder.size)
    ]
    frontend._capture_enc_ae_graph()
    normalized_images = np.stack([
        (image.astype(np.float32) / np.float32(127.5)
         - np.float32(1.0)).astype(np.float16)
        for image in images
    ])
    frontend._img_buf.upload(normalized_images)
    frontend._siglip_graph.replay()
    frontend._g_noise.copy_(torch.from_numpy(noise.astype(np.float16)).cuda())
    frontend._enc_ae_graph.replay()
    torch.cuda.synchronize()
    python_raw = frontend._g_noise.cpu().numpy()
    native_raw = np.fromfile(raw_path, dtype=np.float16).reshape(10, 32)
    _assert_equal("native raw action", python_raw, native_raw)


if __name__ == "__main__":
    main()

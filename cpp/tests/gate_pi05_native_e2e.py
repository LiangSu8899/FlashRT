"""Compare native SM120 Pi0.5 against the official OpenPI PyTorch policy."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import ml_dtypes
import numpy as np
from PIL import Image
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    return float(x @ y / (nx * ny)) if nx and ny else float("nan")


def _decode_image(cell) -> np.ndarray:
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _task(root: Path, task_index: int) -> str:
    with (root / "meta" / "tasks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if int(item["task_index"]) == task_index:
                return str(item["task"])
    raise KeyError(f"task_index={task_index} is missing")


def _make_fixture(args, fixture: Path) -> None:
    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    chunk = args.episode // int(info.get("chunks_size", 1000))
    relative = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    ).format(episode_chunk=chunk, episode_index=args.episode)
    table = pq.read_table(args.dataset / relative)
    rows = table.filter(pc.equal(table["frame_index"], args.frame)).to_pylist()
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one episode={args.episode} frame={args.frame} row"
        )
    row = rows[0]
    _decode_image(row["image"]).tofile(fixture / "image_0.rgb")
    _decode_image(row["wrist_image"]).tofile(fixture / "image_1.rgb")
    np.asarray(row["state"], dtype=np.float32).tofile(fixture / "state.f32")
    (fixture / "prompt.txt").write_text(
        _task(args.dataset, int(row["task_index"])), encoding="utf-8"
    )
    values = np.random.default_rng(args.seed).standard_normal(10 * 32)
    np.asarray(values, dtype=np.float32).astype(ml_dtypes.bfloat16).tofile(
        fixture / "noise.bf16"
    )


def _official_baseline(checkpoint: Path, fixture: Path, output: Path) -> None:
    import torch

    from openpi.models import model as model_api
    from openpi.models import tokenizer as tokenizer_api
    from openpi.training import config as training_config

    def image(name: str) -> np.ndarray:
        return np.fromfile(fixture / name, dtype=np.uint8).reshape(224, 224, 3)

    state = np.fromfile(fixture / "state.f32", dtype=np.float32)
    prompt = (fixture / "prompt.txt").read_text(encoding="utf-8")
    noise = np.fromfile(fixture / "noise.bf16", dtype=ml_dtypes.bfloat16)
    noise = noise.astype(np.float32).reshape(10, 32)
    stats = json.loads(
        (checkpoint / "assets" / "physical-intelligence" / "libero" /
         "norm_stats.json").read_text()
    )["norm_stats"]
    state_q01 = np.asarray(stats["state"]["q01"], dtype=np.float32)
    state_q99 = np.asarray(stats["state"]["q99"], dtype=np.float32)
    normalized_state = (
        (state - state_q01) / (state_q99 - state_q01 + 1e-6) * 2.0 - 1.0
    )
    tokens, token_mask = tokenizer_api.PaligemmaTokenizer(200).tokenize(
        prompt, normalized_state
    )
    padded_state = np.zeros(32, dtype=np.float32)
    padded_state[:state.size] = normalized_state
    base = image("image_0.rgb")
    wrist = image("image_1.rgb")
    device_inputs = {
        "image": {
            "base_0_rgb": torch.from_numpy(base).to("cuda")[None, ...],
            "left_wrist_0_rgb": torch.from_numpy(wrist).to("cuda")[None, ...],
            "right_wrist_0_rgb": torch.zeros_like(
                torch.from_numpy(base).to("cuda")[None, ...]
            ),
        },
        "image_mask": {
            "base_0_rgb": torch.ones(1, dtype=torch.bool, device="cuda"),
            "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool, device="cuda"),
            "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device="cuda"),
        },
        "state": torch.from_numpy(padded_state).to("cuda")[None, ...],
        "tokenized_prompt": torch.from_numpy(tokens).to("cuda")[None, ...],
        "tokenized_prompt_mask": torch.from_numpy(token_mask).to("cuda")[None, ...],
    }
    model_observation = model_api.Observation.from_dict(device_inputs)
    train_config = training_config.get_config("pi05_libero")
    model = train_config.model.load_pytorch(
        train_config, str(checkpoint / "model.safetensors")
    )
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.to("cuda").eval()
    noise_tensor = torch.from_numpy(noise).to("cuda")[None, ...]
    with torch.inference_mode():
        raw = model.sample_actions(
            "cuda", model_observation, noise=noise_tensor, num_steps=10
        )[0].float().cpu().numpy()
    action_q01 = np.asarray(stats["actions"]["q01"], dtype=np.float32)
    action_q99 = np.asarray(stats["actions"]["q99"], dtype=np.float32)
    clipped = np.clip(raw[:, :action_q01.size], -1.0, 1.0)
    actions = ((clipped + 1.0) * 0.5 *
               (action_q99 - action_q01 + 1e-6) + action_q01)
    np.asarray(raw, dtype=np.float32).tofile(output / "openpi_raw.f32")
    np.asarray(actions, dtype=np.float32).tofile(
        output / "openpi_actions.f32"
    )


def _run(args) -> None:
    with tempfile.TemporaryDirectory(prefix="pi05_native_e2e_") as temp:
        root = Path(temp)
        _make_fixture(args, root)
        env = dict(os.environ)
        env["TORCH_COMPILE_DISABLE"] = "1"
        baseline_prefix = env.get("OPENPI_BASELINE_SITE_PACKAGES")
        if baseline_prefix:
            baseline_packages = Path(baseline_prefix)
            if not baseline_packages.is_dir():
                raise FileNotFoundError(baseline_packages)
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(baseline_packages) + (
                os.pathsep + existing if existing else ""
            )
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--baseline-fixture",
                str(root),
                "--checkpoint",
                str(args.checkpoint),
            ],
            check=True,
            env=env,
        )
        subprocess.run(
            [
                str(args.probe),
                str(args.checkpoint),
                str(args.tokenizer),
                str(root),
                str(root),
            ],
            check=True,
        )
        openpi_raw = np.fromfile(root / "openpi_raw.f32", dtype=np.float32)
        native_raw = np.fromfile(
            root / "native_raw.bf16", dtype=ml_dtypes.bfloat16
        ).astype(np.float32)
        openpi_actions = np.fromfile(
            root / "openpi_actions.f32", dtype=np.float32
        )
        native_actions = np.fromfile(
            root / "native_actions.f32", dtype=np.float32
        )
        sizes = {
            "openpi_raw": openpi_raw.size,
            "native_raw": native_raw.size,
            "openpi_actions": openpi_actions.size,
            "native_actions": native_actions.size,
        }
        expected_sizes = {
            "openpi_raw": 10 * 32,
            "native_raw": 10 * 32,
            "openpi_actions": 10 * 7,
            "native_actions": 10 * 7,
        }
        if sizes != expected_sizes:
            raise RuntimeError(f"unexpected E2E output sizes: {sizes}")
        raw_cos = _cosine(openpi_raw, native_raw)
        action_cos = _cosine(openpi_actions, native_actions)
        raw_max = float(np.max(np.abs(openpi_raw - native_raw)))
        action_max = float(np.max(np.abs(openpi_actions - native_actions)))
        stats = json.loads(
            (args.checkpoint / "assets" / "physical-intelligence" /
             "libero" / "norm_stats.json").read_text()
        )["norm_stats"]["actions"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        native_model = native_raw.reshape(10, 32)[:, :q01.size]
        native_contract_actions = (
            (np.clip(native_model, -1.0, 1.0) + 1.0) * 0.5 *
            (q99 - q01 + 1e-6) + q01
        )
        contract_max = float(np.max(np.abs(
            native_contract_actions.reshape(-1) - native_actions
        )))
        contract_close = bool(
            np.allclose(native_contract_actions.reshape(-1), native_actions,
                        rtol=1e-6, atol=1e-6)
        )
        print("\n===== PI0.5 NATIVE VS OFFICIAL OPENPI =====")
        print(f"episode/frame     : {args.episode}/{args.frame}")
        print(f"raw action cosine : {raw_cos:.8f} max_abs={raw_max:.6g}")
        print(
            f"robot action      : cos={action_cos:.8f} "
            f"max_abs_vs_fp32={action_max:.6g}"
        )
        print(
            f"action postprocess: allclose={contract_close} "
            f"max_abs={contract_max:.6g}"
        )
        if raw_cos < 0.9999:
            raise AssertionError(f"raw action cosine {raw_cos:.8f} < 0.9999")
        if action_cos < 0.9999:
            raise AssertionError(
                f"robot action cosine {action_cos:.8f} < 0.9999"
            )
        if not contract_close:
            raise AssertionError(
                f"native action postprocess differs; max_abs={contract_max:.6g}"
            )
        print("PASS native Pi0.5 real-episode E2E")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--baseline-fixture", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.baseline_fixture is None:
        for name in ("tokenizer", "dataset", "probe"):
            if getattr(args, name) is None:
                parser.error(f"--{name} is required")
    return args


if __name__ == "__main__":
    parsed = _parse_args()
    if parsed.baseline_fixture is not None:
        _official_baseline(
            parsed.checkpoint,
            parsed.baseline_fixture,
            parsed.baseline_fixture,
        )
    else:
        _run(parsed)

"""Cosmos3-Edge Thor official-baseline frontend.

This frontend deliberately runs NVIDIA's Cosmos Framework inference entrypoint
first. It gives FlashRT a reproducible Thor baseline before the optimized
model-local Thor pipeline is ported from the Cosmos3-Nano FlashRT work.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import Any


class Cosmos3EdgeTorchFrontendThor:
    """Thin wrapper around ``cosmos_framework.scripts.inference``.

    ``checkpoint`` is a local Cosmos3-Edge Hugging Face/diffusers directory or
    a Cosmos Framework registered name. Use ``set_prompt(sample=...)`` or
    ``set_prompt(input_json=...)`` to provide official sample JSON, then
    ``infer(output_dir=..., vae_path=...)`` to run the baseline.
    """

    def __init__(self, checkpoint: str, num_views: int = 1,
                 hardware: str | None = None):
        del num_views
        if hardware not in (None, "thor"):
            raise ValueError(
                "Cosmos3-Edge baseline frontend is registered for Thor only; "
                f"got hardware={hardware!r}.")
        self.checkpoint = str(checkpoint)
        self.hardware = "thor"
        self._input_json: str | None = None
        self._sample: dict[str, Any] | None = None

    def set_prompt(self, *, input_json: str | None = None,
                   sample: dict[str, Any] | None = None,
                   **sample_fields: Any) -> None:
        """Set the official Cosmos Framework sample input.

        Exactly one source should be provided: ``input_json`` for an existing
        JSON file, ``sample`` for a dict, or keyword fields for a single sample.
        """
        provided = sum(x is not None for x in (input_json, sample))
        if provided + bool(sample_fields) != 1:
            raise ValueError(
                "Provide exactly one of input_json=..., sample=..., or "
                "sample keyword fields.")
        self._input_json = str(input_json) if input_json is not None else None
        if sample is not None:
            self._sample = dict(sample)
        elif sample_fields:
            self._sample = dict(sample_fields)
        else:
            self._sample = None

    @staticmethod
    def _default_cosmos_root() -> pathlib.Path:
        here = pathlib.Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "external" / "cosmos-framework"
            if candidate.exists():
                return candidate
        return pathlib.Path.cwd()

    @staticmethod
    def _default_config_file(cosmos_root: pathlib.Path) -> pathlib.Path:
        return (cosmos_root / "cosmos_framework" / "inference" / "configs" /
                "model" / "Cosmos3-Edge.yaml")

    @staticmethod
    def _write_sample_json(sample: dict[str, Any], work_dir: pathlib.Path) -> pathlib.Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(
            prefix="cosmos3_edge_sample_", suffix=".json", dir=str(work_dir))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2)
            f.write("\n")
        return pathlib.Path(path)

    @staticmethod
    def _write_action_only_output(action: Any, output_dir: pathlib.Path) -> pathlib.Path:
        """Write a Cosmos-style action-only output without decoded vision files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "success",
            "args": {},
            "outputs": [
                {
                    "content": {"action": action},
                    "files": [],
                }
            ],
        }
        output_path = output_dir / "sample_outputs.json"
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return output_path

    @staticmethod
    def _absolute_output_path(path: str) -> str:
        out = pathlib.Path(path).expanduser()
        if not out.is_absolute():
            out = pathlib.Path.cwd() / out
        return str(out.resolve())

    def infer(self, *, output_dir: str, seed: int = 0,
              backend: str = "official",
              benchmark: bool = True,
              parallelism_preset: str = "latency",
              guardrails: bool = False,
              use_torch_compile: bool = False,
              use_cuda_graphs: bool = False,
              cosmos_root: str | None = None,
              config_file: str | None = None,
              python_executable: str | None = None,
              vae_path: str | None = None,
              reference_dump: str | None = None,
              boundary_dump: str | None = None,
              live_dump_out: str | None = None,
              live_flashrt_handoff: bool = False,
              live_boundary_out: str | None = None,
              live_boundary_in: str | None = None,
              live_boundary_prepare_in: str | None = None,
              live_boundary_prepare_live: bool = False,
              live_handoff_trace_out: str | None = None,
              upstream_trace_out: str | None = None,
              vae_encode_dump_out: str | None = None,
              vae_latent_in: str | None = None,
              vae_encode_dump_input: bool = False,
              vae_encode_profile_out: str | None = None,
              vae_native_rms_silu: bool = False,
              vae_t1_conv2d: bool = False,
              vae_native_avgdown3d: bool = False,
              vae_channels_last3d_conv320: bool = False,
              vae_compile_encode: bool = False,
              vae_compile_trace_out: str | None = None,
              prepare_dump_out: str | None = None,
              prepare_replay_in: str | None = None,
              prepare_inventory_out: str | None = None,
              prepare_slim_no_raw_state_vision: bool = False,
              prepare_slim_derive_condition_reference: bool = False,
              prepare_slim_derive_initial_noise: bool = False,
              live_prelayer_bootstrap: bool = False,
              live_warm_request: bool = False,
              cache_warmup_vae: bool = False,
              cache_warmup_prepare: bool = False,
              warmup: int = 0,
              replay_device: str = "cpu",
              extra_args: list[str] | None = None,
              env: dict[str, str] | None = None,
              timeout: float | None = None) -> dict[str, Any]:
        """Run the official baseline or P1 denoise replay scaffold.

        ``vae_path`` should point at the official ``Wan2.2_VAE.pth`` when the
        container cannot download it. It is passed as a config override with an
        empty bucket name so Cosmos Framework treats it as a local file.
        """
        if backend not in {"official", "official_action_only", "replay", "torch_ref", "flashrt"}:
            raise ValueError(
                "Cosmos3-Edge backend must be one of 'official', "
                "'official_action_only', 'replay', 'torch_ref', or "
                f"'flashrt'; got {backend!r}.")
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        if live_warm_request:
            if backend != "official_action_only":
                raise ValueError("live_warm_request is only supported with backend='official_action_only'")
            live_prelayer_bootstrap = True
            warmup = max(int(warmup), 1)
            cache_warmup_vae = True
        if live_dump_out is not None and backend != "official_action_only":
            raise ValueError("live_dump_out is only supported with backend='official_action_only'")
        if upstream_trace_out is not None and backend != "official_action_only":
            raise ValueError("upstream_trace_out is only supported with backend='official_action_only'")
        if (
            vae_encode_dump_out is not None
            or vae_latent_in is not None
            or vae_encode_dump_input
            or vae_encode_profile_out is not None
            or vae_native_rms_silu
            or vae_t1_conv2d
            or vae_native_avgdown3d
            or vae_channels_last3d_conv320
            or vae_compile_encode
            or vae_compile_trace_out is not None
            or prepare_dump_out is not None
            or prepare_replay_in is not None
            or prepare_inventory_out is not None
            or prepare_slim_no_raw_state_vision
            or prepare_slim_derive_condition_reference
            or prepare_slim_derive_initial_noise
        ) and backend != "official_action_only":
            raise ValueError(
                "VAE/prepare boundary/profile/inventory dump/replay is only supported with backend='official_action_only'"
            )
        if vae_compile_encode and (vae_native_rms_silu or vae_native_avgdown3d):
            raise ValueError("vae_compile_encode cannot be combined with native VAE monkeypatches")
        if cache_warmup_vae and (vae_encode_dump_out is not None or vae_latent_in is not None):
            raise ValueError("cache_warmup_vae cannot be combined with VAE encode boundary dump/replay")
        if cache_warmup_prepare and (
            prepare_dump_out is not None
            or prepare_replay_in is not None
            or prepare_inventory_out is not None
        ):
            raise ValueError("cache_warmup_prepare cannot be combined with prepare boundary dump/replay/inventory")
        if (
            live_flashrt_handoff
            or live_boundary_out is not None
            or live_boundary_in is not None
            or live_boundary_prepare_in is not None
            or live_boundary_prepare_live
            or live_handoff_trace_out is not None
            or live_prelayer_bootstrap
            or live_warm_request
        ) and backend != "official_action_only":
            raise ValueError("live FlashRT handoff is only supported with backend='official_action_only'")
        if live_dump_out is not None and live_flashrt_handoff:
            raise ValueError("live_dump_out cannot be combined with live_flashrt_handoff")
        if live_boundary_in is not None and live_boundary_prepare_in is not None:
            raise ValueError("live_boundary_in cannot be combined with live_boundary_prepare_in")
        if live_boundary_prepare_live and (live_boundary_in is not None or live_boundary_prepare_in is not None):
            raise ValueError("live_boundary_prepare_live cannot be combined with live boundary input artifacts")
        if backend in {"replay", "torch_ref", "flashrt"}:
            if reference_dump is None:
                raise ValueError(f"backend={backend!r} requires reference_dump=...")
            out_dir = pathlib.Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            from flash_rt.models.cosmos3_edge import (
                EdgeBoundaryDump,
                EdgeDenoiseDump,
                EdgeDenoiseFlashRT,
                EdgeDenoiseReplay,
                EdgeDenoiseTorchReference,
                EdgeTransformerWeights,
            )

            t0 = time.perf_counter()
            dump = EdgeDenoiseDump(reference_dump)
            if backend == "replay":
                runner = EdgeDenoiseReplay(dump, device=replay_device)
                replay_result = runner.replay()
                wall_s = time.perf_counter() - t0
                return {
                    "backend": "replay",
                    "reference_dump": str(reference_dump),
                    "output_dir": str(out_dir),
                    "wall_s": wall_s,
                    "num_steps": len(replay_result.timesteps),
                    "timesteps": list(replay_result.timesteps),
                    "flat_dim": int(replay_result.final_flat.numel()),
                    "vision_shape": list(replay_result.final_parts.vision.shape),
                    "action_model_shape": list(replay_result.final_parts.action_model.shape),
                    "max_input_abs_diff": replay_result.max_input_abs_diff,
                }

            if boundary_dump is None:
                raise ValueError(f"backend={backend!r} requires boundary_dump=...")
            runner_cls = EdgeDenoiseFlashRT if backend == "flashrt" else EdgeDenoiseTorchReference
            runner = runner_cls(
                dump,
                EdgeBoundaryDump(boundary_dump),
                EdgeTransformerWeights(self.checkpoint),
                device=replay_device,
                **({"use_cuda_graphs": use_cuda_graphs} if backend == "flashrt" else {}),
            )
            replay_result = runner.run()
            wall_s = time.perf_counter() - t0
            result = {
                "backend": backend,
                "reference_dump": str(reference_dump),
                "boundary_dump": str(boundary_dump),
                "output_dir": str(out_dir),
                "wall_s": wall_s,
                "num_steps": replay_result.steps_run,
                "timesteps": list(replay_result.timesteps),
                "flat_dim": int(replay_result.final_flat.numel()),
                "vision_shape": list(replay_result.final_parts.vision.shape),
                "action_model_shape": list(replay_result.final_parts.action_model.shape),
                "max_input_abs_diff": replay_result.max_input_abs_diff,
                "max_velocity_abs_diff": replay_result.max_velocity_abs_diff,
            }
            if backend == "flashrt":
                result["use_cuda_graphs"] = bool(use_cuda_graphs)
            if runner.static_engine is not None:
                result["native_attention"] = bool(
                    getattr(runner.static_engine.reference, "native_attention_available", False)
                )
                result["graph_attention"] = bool(
                    getattr(runner.static_engine.reference, "graph_attention_available", False)
                )
            official_action = dump.final_action.float()
            if tuple(official_action.shape) == tuple(replay_result.final_action.shape):
                import torch

                diff = replay_result.final_action.float() - official_action
                result["final_action_cos"] = float(
                    torch.nn.functional.cosine_similarity(
                        replay_result.final_action.flatten(),
                        official_action.flatten(),
                        dim=0,
                    ).item()
                )
                result["final_action_rel_l2"] = float(
                    (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(official_action)).item()
                )
                result["final_action_max_abs"] = float(diff.abs().max().item())
            if backend == "flashrt":
                action_json = self._write_action_only_output(
                    replay_result.final_action.detach().cpu().tolist(),
                    out_dir,
                )
                result["sample_outputs"] = str(action_json)
                result["action_only_output"] = True
            return result

        if self._input_json is None and self._sample is None:
            raise ValueError(
                "Call set_prompt(input_json=...) or set_prompt(sample=...) "
                "before infer().")

        out_dir = pathlib.Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self._input_json is not None:
            input_json = pathlib.Path(self._input_json)
        else:
            input_json = self._write_sample_json(self._sample or {}, out_dir)

        cosmos_root_path = pathlib.Path(cosmos_root) if cosmos_root else self._default_cosmos_root()
        config_path = pathlib.Path(config_file) if config_file else self._default_config_file(cosmos_root_path)
        py = python_executable or sys.executable
        live_dump_path = self._absolute_output_path(live_dump_out) if live_dump_out is not None else None
        live_boundary_path = self._absolute_output_path(live_boundary_out) if live_boundary_out is not None else None
        live_boundary_in_path = self._absolute_output_path(live_boundary_in) if live_boundary_in is not None else None
        live_boundary_prepare_in_path = (
            self._absolute_output_path(live_boundary_prepare_in)
            if live_boundary_prepare_in is not None
            else None
        )
        live_trace_path = (
            self._absolute_output_path(live_handoff_trace_out)
            if live_handoff_trace_out is not None
            else None
        )
        upstream_trace_path = (
            self._absolute_output_path(upstream_trace_out)
            if upstream_trace_out is not None
            else None
        )
        vae_encode_dump_path = (
            self._absolute_output_path(vae_encode_dump_out)
            if vae_encode_dump_out is not None
            else None
        )
        vae_latent_path = (
            self._absolute_output_path(vae_latent_in)
            if vae_latent_in is not None
            else None
        )
        vae_encode_profile_path = (
            self._absolute_output_path(vae_encode_profile_out)
            if vae_encode_profile_out is not None
            else None
        )
        vae_compile_trace_path = (
            self._absolute_output_path(vae_compile_trace_out)
            if vae_compile_trace_out is not None
            else None
        )
        prepare_dump_path = (
            self._absolute_output_path(prepare_dump_out)
            if prepare_dump_out is not None
            else None
        )
        prepare_replay_path = (
            self._absolute_output_path(prepare_replay_in)
            if prepare_replay_in is not None
            else None
        )
        prepare_inventory_path = (
            self._absolute_output_path(prepare_inventory_out)
            if prepare_inventory_out is not None
            else None
        )

        module = (
            "flash_rt.models.cosmos3_edge.action_only_official"
            if backend == "official_action_only"
            else "cosmos_framework.scripts.inference"
        )
        cmd = [
            py, "-m", module,
            "--parallelism-preset", parallelism_preset,
            "-i", str(input_json),
            "-o", str(out_dir),
            "--checkpoint-path", self.checkpoint,
            "--config-file", str(config_path),
            "--seed", str(seed),
        ]
        cmd.append("--benchmark" if benchmark else "--no-benchmark")
        cmd.append("--guardrails" if guardrails else "--no-guardrails")
        cmd.append("--use-torch-compile" if use_torch_compile else "--no-use-torch-compile")
        cmd.append("--use-cuda-graphs" if use_cuda_graphs else "--no-use-cuda-graphs")
        if warmup:
            cmd.extend(["--warmup", str(int(warmup))])
        if vae_path:
            cmd.extend([
                "--experiment-overrides",
                "model.config.tokenizer.bucket_name=",
                f"model.config.tokenizer.vae_path={vae_path}",
            ])
        if live_dump_path is not None:
            cmd.extend(["--flashrt-live-dump-out", live_dump_path])
        if live_flashrt_handoff:
            cmd.append("--flashrt-live-flashrt-handoff")
        if live_prelayer_bootstrap:
            cmd.append("--flashrt-live-prelayer-bootstrap")
        if live_boundary_path is not None:
            cmd.extend(["--flashrt-live-boundary-out", live_boundary_path])
        if live_boundary_in_path is not None:
            cmd.extend(["--flashrt-live-boundary-in", live_boundary_in_path])
        if live_boundary_prepare_in_path is not None:
            cmd.extend(["--flashrt-live-boundary-prepare-in", live_boundary_prepare_in_path])
        if live_boundary_prepare_live:
            cmd.append("--flashrt-live-boundary-prepare-live")
        if live_trace_path is not None:
            cmd.extend(["--flashrt-live-handoff-trace-out", live_trace_path])
        if upstream_trace_path is not None:
            cmd.extend(["--flashrt-upstream-trace-out", upstream_trace_path])
        if vae_encode_dump_path is not None:
            cmd.extend(["--flashrt-vae-encode-dump-out", vae_encode_dump_path])
        if vae_latent_path is not None:
            cmd.extend(["--flashrt-vae-latent-in", vae_latent_path])
        if vae_encode_dump_input:
            cmd.append("--flashrt-vae-encode-dump-input")
        if vae_encode_profile_path is not None:
            cmd.extend(["--flashrt-vae-encode-profile-out", vae_encode_profile_path])
        if vae_native_rms_silu:
            cmd.append("--flashrt-vae-native-rms-silu")
        if vae_t1_conv2d:
            cmd.append("--flashrt-vae-t1-conv2d")
        if vae_native_avgdown3d:
            cmd.append("--flashrt-vae-native-avgdown3d")
        if vae_channels_last3d_conv320:
            cmd.append("--flashrt-vae-channels-last3d-conv320")
        if vae_compile_encode:
            cmd.append("--flashrt-vae-compile-encode")
        if vae_compile_trace_path is not None:
            cmd.extend(["--flashrt-vae-compile-trace-out", vae_compile_trace_path])
        if prepare_dump_path is not None:
            cmd.extend(["--flashrt-prepare-dump-out", prepare_dump_path])
        if prepare_replay_path is not None:
            cmd.extend(["--flashrt-prepare-replay-in", prepare_replay_path])
        if prepare_inventory_path is not None:
            cmd.extend(["--flashrt-prepare-inventory-out", prepare_inventory_path])
        if prepare_slim_no_raw_state_vision:
            cmd.append("--flashrt-prepare-slim-no-raw-state-vision")
        if prepare_slim_derive_condition_reference:
            cmd.append("--flashrt-prepare-slim-derive-condition-reference")
        if prepare_slim_derive_initial_noise:
            cmd.append("--flashrt-prepare-slim-derive-initial-noise")
        if cache_warmup_vae:
            cmd.append("--flashrt-cache-warmup-vae")
        if cache_warmup_prepare:
            cmd.append("--flashrt-cache-warmup-prepare")
        if extra_args:
            cmd.extend(extra_args)

        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(cosmos_root_path),
            env=run_env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        wall_s = time.perf_counter() - t0
        result: dict[str, Any] = {
            "backend": backend,
            "returncode": proc.returncode,
            "cmd": cmd,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "output_dir": str(out_dir),
            "wall_s": wall_s,
        }
        bench = out_dir / "benchmark.json"
        if bench.exists():
            try:
                result["benchmark"] = json.loads(bench.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                result["benchmark_path"] = str(bench)
        sample_outputs = list(out_dir.glob("*/sample_outputs.json"))
        if sample_outputs:
            result["sample_outputs"] = [str(p) for p in sample_outputs]
        if backend == "official_action_only":
            result["action_only_output"] = True
            if live_dump_path is not None:
                result["live_dump_out"] = live_dump_path
            if live_flashrt_handoff:
                result["live_flashrt_handoff"] = True
            if live_prelayer_bootstrap:
                result["live_prelayer_bootstrap"] = True
            if live_warm_request:
                result["live_warm_request"] = True
            if cache_warmup_vae:
                result["cache_warmup_vae"] = True
            if cache_warmup_prepare:
                result["cache_warmup_prepare"] = True
            if warmup:
                result["warmup"] = int(warmup)
            if live_boundary_path is not None:
                result["live_boundary_out"] = live_boundary_path
            if live_boundary_in_path is not None:
                result["live_boundary_in"] = live_boundary_in_path
            if live_boundary_prepare_in_path is not None:
                result["live_boundary_prepare_in"] = live_boundary_prepare_in_path
            if live_boundary_prepare_live:
                result["live_boundary_prepare_live"] = True
            if live_trace_path is not None:
                result["live_handoff_trace_out"] = live_trace_path
            if upstream_trace_path is not None:
                result["upstream_trace_out"] = upstream_trace_path
            if vae_encode_dump_path is not None:
                result["vae_encode_dump_out"] = vae_encode_dump_path
            if vae_latent_path is not None:
                result["vae_latent_in"] = vae_latent_path
            if vae_encode_dump_input:
                result["vae_encode_dump_input"] = True
            if vae_encode_profile_path is not None:
                result["vae_encode_profile_out"] = vae_encode_profile_path
            if vae_native_rms_silu:
                result["vae_native_rms_silu"] = True
            if vae_t1_conv2d:
                result["vae_t1_conv2d"] = True
            if vae_native_avgdown3d:
                result["vae_native_avgdown3d"] = True
            if vae_channels_last3d_conv320:
                result["vae_channels_last3d_conv320"] = True
            if vae_compile_encode:
                result["vae_compile_encode"] = True
            if vae_compile_trace_path is not None:
                result["vae_compile_trace_out"] = vae_compile_trace_path
            if prepare_dump_path is not None:
                result["prepare_dump_out"] = prepare_dump_path
            if prepare_replay_path is not None:
                result["prepare_replay_in"] = prepare_replay_path
            if prepare_inventory_path is not None:
                result["prepare_inventory_out"] = prepare_inventory_path
            if prepare_slim_no_raw_state_vision:
                result["prepare_slim_no_raw_state_vision"] = True
            if prepare_slim_derive_condition_reference:
                result["prepare_slim_derive_condition_reference"] = True
            if prepare_slim_derive_initial_noise:
                result["prepare_slim_derive_initial_noise"] = True

        if proc.returncode != 0:
            raise RuntimeError(
                "Cosmos3-Edge official baseline failed with return code "
                f"{proc.returncode}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return result

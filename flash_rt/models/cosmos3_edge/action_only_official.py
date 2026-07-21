"""FlashRT-owned Cosmos3-Edge official action-only runner."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


def _extract_output_dir(argv: Sequence[str]) -> Path:
    for idx, arg in enumerate(argv):
        if arg in {"-o", "--output-dir"} and idx + 1 < len(argv):
            return Path(argv[idx + 1])
        for prefix in ("--output-dir=", "--setup.output-dir="):
            if arg.startswith(prefix):
                return Path(arg.split("=", 1)[1])
    raise SystemExit("official action-only runner requires -o/--output-dir")


def _extract_live_dump_out(argv: Sequence[str]) -> tuple[list[str], Path | None]:
    clean: list[str] = []
    live_dump: Path | None = None
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--flashrt-live-dump-out":
            if idx + 1 >= len(argv):
                raise SystemExit("--flashrt-live-dump-out requires a path")
            live_dump = Path(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--flashrt-live-dump-out="):
            live_dump = Path(arg.split("=", 1)[1])
            idx += 1
            continue
        clean.append(arg)
        idx += 1

    env_live_dump = os.environ.get("FLASHRT_COSMOS3_EDGE_LIVE_DUMP_OUT")
    if live_dump is None and env_live_dump:
        live_dump = Path(env_live_dump)
    return clean, live_dump


def _extract_checkpoint_path(argv: Sequence[str]) -> Path:
    for idx, arg in enumerate(argv):
        if arg == "--checkpoint-path" and idx + 1 < len(argv):
            return Path(argv[idx + 1])
        for prefix in ("--checkpoint-path=", "--setup.checkpoint-path="):
            if arg.startswith(prefix):
                return Path(arg.split("=", 1)[1])
    raise SystemExit("FlashRT live handoff requires --checkpoint-path")


def _extract_live_handoff_args(
    argv: Sequence[str],
) -> tuple[list[str], bool, Path | None, Path | None, Path | None, bool, Path | None, bool]:
    clean: list[str] = []
    enabled = False
    boundary_out: Path | None = None
    boundary_in: Path | None = None
    boundary_prepare_in: Path | None = None
    boundary_prepare_live = False
    trace_out: Path | None = None
    prelayer_bootstrap = False
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--flashrt-live-flashrt-handoff":
            enabled = True
            idx += 1
            continue
        if arg == "--flashrt-live-prelayer-bootstrap":
            enabled = True
            prelayer_bootstrap = True
            idx += 1
            continue
        if arg == "--flashrt-live-boundary-out":
            if idx + 1 >= len(argv):
                raise SystemExit("--flashrt-live-boundary-out requires a path")
            boundary_out = Path(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--flashrt-live-boundary-out="):
            boundary_out = Path(arg.split("=", 1)[1])
            idx += 1
            continue
        if arg == "--flashrt-live-boundary-in":
            if idx + 1 >= len(argv):
                raise SystemExit("--flashrt-live-boundary-in requires a path")
            enabled = True
            boundary_in = Path(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--flashrt-live-boundary-in="):
            enabled = True
            boundary_in = Path(arg.split("=", 1)[1])
            idx += 1
            continue
        if arg == "--flashrt-live-boundary-prepare-in":
            if idx + 1 >= len(argv):
                raise SystemExit("--flashrt-live-boundary-prepare-in requires a path")
            enabled = True
            boundary_prepare_in = Path(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--flashrt-live-boundary-prepare-in="):
            enabled = True
            boundary_prepare_in = Path(arg.split("=", 1)[1])
            idx += 1
            continue
        if arg == "--flashrt-live-boundary-prepare-live":
            enabled = True
            boundary_prepare_live = True
            idx += 1
            continue
        if arg == "--flashrt-live-handoff-trace-out":
            if idx + 1 >= len(argv):
                raise SystemExit("--flashrt-live-handoff-trace-out requires a path")
            trace_out = Path(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--flashrt-live-handoff-trace-out="):
            trace_out = Path(arg.split("=", 1)[1])
            idx += 1
            continue
        clean.append(arg)
        idx += 1

    if os.environ.get("FLASHRT_COSMOS3_EDGE_LIVE_FLASHRT_HANDOFF"):
        enabled = True
    if os.environ.get("FLASHRT_COSMOS3_EDGE_LIVE_PRELAYER_BOOTSTRAP"):
        enabled = True
        prelayer_bootstrap = True
    env_boundary = os.environ.get("FLASHRT_COSMOS3_EDGE_LIVE_BOUNDARY_OUT")
    if boundary_out is None and env_boundary:
        boundary_out = Path(env_boundary)
    env_boundary_in = os.environ.get("FLASHRT_COSMOS3_EDGE_LIVE_BOUNDARY_IN")
    if boundary_in is None and env_boundary_in:
        enabled = True
        boundary_in = Path(env_boundary_in)
    env_boundary_prepare = os.environ.get("FLASHRT_COSMOS3_EDGE_LIVE_BOUNDARY_PREPARE_IN")
    if boundary_prepare_in is None and env_boundary_prepare:
        enabled = True
        boundary_prepare_in = Path(env_boundary_prepare)
    if os.environ.get("FLASHRT_COSMOS3_EDGE_LIVE_BOUNDARY_PREPARE_LIVE"):
        enabled = True
        boundary_prepare_live = True
    env_trace = os.environ.get("FLASHRT_COSMOS3_EDGE_LIVE_HANDOFF_TRACE_OUT")
    if trace_out is None and env_trace:
        trace_out = Path(env_trace)
    return clean, enabled, boundary_out, boundary_in, boundary_prepare_in, boundary_prepare_live, trace_out, prelayer_bootstrap


def _extract_upstream_trace_out(argv: Sequence[str]) -> tuple[list[str], Path | None]:
    clean: list[str] = []
    trace_out: Path | None = None
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--flashrt-upstream-trace-out":
            if idx + 1 >= len(argv):
                raise SystemExit("--flashrt-upstream-trace-out requires a path")
            trace_out = Path(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--flashrt-upstream-trace-out="):
            trace_out = Path(arg.split("=", 1)[1])
            idx += 1
            continue
        clean.append(arg)
        idx += 1

    env_trace = os.environ.get("FLASHRT_COSMOS3_EDGE_UPSTREAM_TRACE_OUT")
    if trace_out is None and env_trace:
        trace_out = Path(env_trace)
    return clean, trace_out


def _extract_warmup_vae_cache(argv: Sequence[str]) -> tuple[list[str], bool]:
    clean: list[str] = []
    enabled = False
    for arg in argv:
        if arg == "--flashrt-cache-warmup-vae":
            enabled = True
            continue
        clean.append(arg)
    if os.environ.get("FLASHRT_COSMOS3_EDGE_CACHE_WARMUP_VAE"):
        enabled = True
    return clean, enabled


def _extract_warmup_prepare_cache(argv: Sequence[str]) -> tuple[list[str], bool]:
    clean: list[str] = []
    enabled = False
    for arg in argv:
        if arg == "--flashrt-cache-warmup-prepare":
            enabled = True
            continue
        clean.append(arg)
    if os.environ.get("FLASHRT_COSMOS3_EDGE_CACHE_WARMUP_PREPARE"):
        enabled = True
    return clean, enabled


def _extract_vae_encode_boundary_args(argv: Sequence[str]) -> tuple[list[str], Path | None, Path | None, bool]:
    clean: list[str] = []
    dump_out: Path | None = None
    latent_in: Path | None = None
    dump_input = False
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--flashrt-vae-encode-dump-out":
            if idx + 1 >= len(argv):
                raise SystemExit("--flashrt-vae-encode-dump-out requires a path")
            dump_out = Path(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--flashrt-vae-encode-dump-out="):
            dump_out = Path(arg.split("=", 1)[1])
            idx += 1
            continue
        if arg == "--flashrt-vae-latent-in":
            if idx + 1 >= len(argv):
                raise SystemExit("--flashrt-vae-latent-in requires a path")
            latent_in = Path(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--flashrt-vae-latent-in="):
            latent_in = Path(arg.split("=", 1)[1])
            idx += 1
            continue
        if arg == "--flashrt-vae-encode-dump-input":
            dump_input = True
            idx += 1
            continue
        clean.append(arg)
        idx += 1

    env_dump = os.environ.get("FLASHRT_COSMOS3_EDGE_VAE_ENCODE_DUMP_OUT")
    if dump_out is None and env_dump:
        dump_out = Path(env_dump)
    env_latent = os.environ.get("FLASHRT_COSMOS3_EDGE_VAE_LATENT_IN")
    if latent_in is None and env_latent:
        latent_in = Path(env_latent)
    if os.environ.get("FLASHRT_COSMOS3_EDGE_VAE_ENCODE_DUMP_INPUT"):
        dump_input = True
    return clean, dump_out, latent_in, dump_input


def _extract_vae_encode_profile_out(argv: Sequence[str]) -> tuple[list[str], Path | None]:
    clean: list[str] = []
    profile_out: Path | None = None
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--flashrt-vae-encode-profile-out":
            if idx + 1 >= len(argv):
                raise SystemExit("--flashrt-vae-encode-profile-out requires a path")
            profile_out = Path(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--flashrt-vae-encode-profile-out="):
            profile_out = Path(arg.split("=", 1)[1])
            idx += 1
            continue
        clean.append(arg)
        idx += 1

    env_profile = os.environ.get("FLASHRT_COSMOS3_EDGE_VAE_ENCODE_PROFILE_OUT")
    if profile_out is None and env_profile:
        profile_out = Path(env_profile)
    return clean, profile_out


def _extract_vae_native_rms_silu(argv: Sequence[str]) -> tuple[list[str], bool]:
    clean: list[str] = []
    enabled = False
    for arg in argv:
        if arg == "--flashrt-vae-native-rms-silu":
            enabled = True
            continue
        clean.append(arg)
    if os.environ.get("FLASHRT_COSMOS3_EDGE_VAE_NATIVE_RMS_SILU"):
        enabled = True
    return clean, enabled


def _extract_vae_t1_conv2d(argv: Sequence[str]) -> tuple[list[str], bool]:
    clean: list[str] = []
    enabled = False
    for arg in argv:
        if arg == "--flashrt-vae-t1-conv2d":
            enabled = True
            continue
        clean.append(arg)
    if os.environ.get("FLASHRT_COSMOS3_EDGE_VAE_T1_CONV2D"):
        enabled = True
    return clean, enabled


def _extract_vae_native_avgdown3d(argv: Sequence[str]) -> tuple[list[str], bool]:
    clean: list[str] = []
    enabled = False
    for arg in argv:
        if arg == "--flashrt-vae-native-avgdown3d":
            enabled = True
            continue
        clean.append(arg)
    if os.environ.get("FLASHRT_COSMOS3_EDGE_VAE_NATIVE_AVGDOWN3D"):
        enabled = True
    return clean, enabled


def _extract_vae_channels_last3d_conv320(argv: Sequence[str]) -> tuple[list[str], bool]:
    clean: list[str] = []
    enabled = False
    for arg in argv:
        if arg == "--flashrt-vae-channels-last3d-conv320":
            enabled = True
            continue
        clean.append(arg)
    if os.environ.get("FLASHRT_COSMOS3_EDGE_VAE_CHANNELS_LAST3D_CONV320"):
        enabled = True
    return clean, enabled


def _extract_vae_compile_encode(argv: Sequence[str]) -> tuple[list[str], bool, Path | None]:
    clean: list[str] = []
    enabled = False
    trace_out: Path | None = None
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--flashrt-vae-compile-encode":
            enabled = True
            idx += 1
            continue
        if arg == "--flashrt-vae-compile-trace-out":
            if idx + 1 >= len(argv):
                raise SystemExit("--flashrt-vae-compile-trace-out requires a path")
            trace_out = Path(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--flashrt-vae-compile-trace-out="):
            trace_out = Path(arg.split("=", 1)[1])
            idx += 1
            continue
        clean.append(arg)
        idx += 1
    if os.environ.get("FLASHRT_COSMOS3_EDGE_VAE_COMPILE_ENCODE"):
        enabled = True
    env_trace = os.environ.get("FLASHRT_COSMOS3_EDGE_VAE_COMPILE_TRACE_OUT")
    if trace_out is None and env_trace:
        trace_out = Path(env_trace)
    return clean, enabled, trace_out


def _extract_prepare_boundary_args(
    argv: Sequence[str],
) -> tuple[list[str], Path | None, Path | None, Path | None, bool, bool, bool]:
    clean: list[str] = []
    dump_out: Path | None = None
    replay_in: Path | None = None
    inventory_out: Path | None = None
    slim_no_raw_state_vision = False
    slim_derive_condition_reference = False
    slim_derive_initial_noise = False
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--flashrt-prepare-dump-out":
            if idx + 1 >= len(argv):
                raise SystemExit("--flashrt-prepare-dump-out requires a path")
            dump_out = Path(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--flashrt-prepare-dump-out="):
            dump_out = Path(arg.split("=", 1)[1])
            idx += 1
            continue
        if arg == "--flashrt-prepare-replay-in":
            if idx + 1 >= len(argv):
                raise SystemExit("--flashrt-prepare-replay-in requires a path")
            replay_in = Path(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--flashrt-prepare-replay-in="):
            replay_in = Path(arg.split("=", 1)[1])
            idx += 1
            continue
        if arg == "--flashrt-prepare-inventory-out":
            if idx + 1 >= len(argv):
                raise SystemExit("--flashrt-prepare-inventory-out requires a path")
            inventory_out = Path(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--flashrt-prepare-inventory-out="):
            inventory_out = Path(arg.split("=", 1)[1])
            idx += 1
            continue
        if arg == "--flashrt-prepare-slim-no-raw-state-vision":
            slim_no_raw_state_vision = True
            idx += 1
            continue
        if arg == "--flashrt-prepare-slim-derive-condition-reference":
            slim_derive_condition_reference = True
            idx += 1
            continue
        if arg == "--flashrt-prepare-slim-derive-initial-noise":
            slim_derive_initial_noise = True
            idx += 1
            continue
        clean.append(arg)
        idx += 1
    env_dump = os.environ.get("FLASHRT_COSMOS3_EDGE_PREPARE_DUMP_OUT")
    if dump_out is None and env_dump:
        dump_out = Path(env_dump)
    env_replay = os.environ.get("FLASHRT_COSMOS3_EDGE_PREPARE_REPLAY_IN")
    if replay_in is None and env_replay:
        replay_in = Path(env_replay)
    env_inventory = os.environ.get("FLASHRT_COSMOS3_EDGE_PREPARE_INVENTORY_OUT")
    if inventory_out is None and env_inventory:
        inventory_out = Path(env_inventory)
    if os.environ.get("FLASHRT_COSMOS3_EDGE_PREPARE_SLIM_NO_RAW_STATE_VISION"):
        slim_no_raw_state_vision = True
    if os.environ.get("FLASHRT_COSMOS3_EDGE_PREPARE_SLIM_DERIVE_CONDITION_REFERENCE"):
        slim_derive_condition_reference = True
    if os.environ.get("FLASHRT_COSMOS3_EDGE_PREPARE_SLIM_DERIVE_INITIAL_NOISE"):
        slim_derive_initial_noise = True
    return (
        clean,
        dump_out,
        replay_in,
        inventory_out,
        slim_no_raw_state_vision,
        slim_derive_condition_reference,
        slim_derive_initial_noise,
    )


class _VAECompileEncodePatch:
    def __init__(self, trace_out: Path | None) -> None:
        self.trace_out = trace_out
        self.events: list[dict[str, Any]] = []

    def wrap_setup(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def setup_with_compile(model: Any, *args: Any, **kwargs: Any) -> Any:
            out = original(model, *args, **kwargs)
            tokenizer = getattr(model, "tokenizer_vision_gen", None)
            if tokenizer is None or not hasattr(tokenizer, "compile_encode"):
                self.events.append({"name": "vae_compile_encode", "ok": False, "reason": "missing_tokenizer"})
                return out
            t0 = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="cosmos3_edge_vae_aot_") as tmp:
                tokenizer.compile_encode(["480"], tmp, aspect_ratio="16,9")
                loaded = getattr(getattr(tokenizer, "model", None), "model", None)
                loaded_fns = getattr(loaded, "_aot_chunk_fns", {})
            self.events.append(
                {
                    "name": "vae_compile_encode",
                    "ok": True,
                    "compile_s": time.perf_counter() - t0,
                    "resolution": "480",
                    "aspect_ratio": "16,9",
                    "loaded_aot_functions": len(loaded_fns) if isinstance(loaded_fns, dict) else 0,
                }
            )
            return out

        return setup_with_compile

    def save(self) -> None:
        if self.trace_out is None:
            return
        self.trace_out.parent.mkdir(parents=True, exist_ok=True)
        self.trace_out.write_text(json.dumps({"version": 1, "events": self.events}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _install_vae_compile_encode_patch(trace_out: Path | None) -> _VAECompileEncodePatch:
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    patch = _VAECompileEncodePatch(trace_out)
    OmniMoTModel.set_up_tokenizers = patch.wrap_setup(OmniMoTModel.set_up_tokenizers)
    return patch


def _fake_decode(self: Any, vision_latent: Any) -> Any:
    tokenizer = self.tokenizer_vision_gen
    frames = tokenizer.get_pixel_num_frames(int(vision_latent.shape[2]))
    height = int(vision_latent.shape[3]) * int(tokenizer.spatial_compression_factor)
    width = int(vision_latent.shape[4]) * int(tokenizer.spatial_compression_factor)
    return vision_latent.new_zeros((vision_latent.shape[0], 3, frames, height, width))


def _fake_save_img_or_video(vision: Any, path_without_suffix: str, *, fps: int = 24, quality: int = 7) -> None:
    del fps, quality
    path = Path(path_without_suffix)
    frames = int(vision.shape[1]) if hasattr(vision, "shape") and len(vision.shape) > 1 else 2
    output = path.with_suffix(".png" if frames == 1 else ".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"")


def _rewrite_sample_outputs_action_only(output_dir: Path) -> int:
    rewritten = 0
    for sample_outputs in output_dir.glob("*/sample_outputs.json"):
        payload = json.loads(sample_outputs.read_text(encoding="utf-8"))
        for output in payload.get("outputs", []):
            content = output.get("content")
            if not isinstance(content, dict) or "action" not in content:
                continue
            for rel in output.get("files", []):
                try:
                    path = sample_outputs.parent / rel
                    if path.exists():
                        path.unlink()
                except OSError:
                    pass
            output["files"] = []
            rewritten += 1
        sample_outputs.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rewritten


def _install_action_only_patches() -> None:
    import cosmos_framework.inference.inference as inference_mod
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    inference_mod.save_img_or_video = _fake_save_img_or_video
    OmniMoTModel.decode = _fake_decode


def _clone_for_warmup(value: Any) -> Any:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return value.clone()
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{field.name: _clone_for_warmup(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, dict):
        return {key: _clone_for_warmup(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_for_warmup(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_for_warmup(item) for item in value)
    return value


def _install_warmup_clone_patch() -> None:
    from cosmos_framework.inference.inference import OmniInference

    original = OmniInference.generate_batch

    def generate_batch_clone_warmup(self: Any, sample_args_list: Any, data_batch: dict[str, Any], *, warmup: bool = False):
        if warmup:
            data_batch = _clone_for_warmup(data_batch)
        return original(self, sample_args_list, data_batch, warmup=warmup)

    OmniInference.generate_batch = generate_batch_clone_warmup


class _WarmupVAECache:
    def __init__(self) -> None:
        self.phase = "setup"
        self.cached: list[tuple[tuple[Any, ...], Any]] = []
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _signature(state: Any) -> tuple[Any, ...] | None:
        if not hasattr(state, "shape") or not hasattr(state, "dtype") or not hasattr(state, "device"):
            return None
        signature: list[Any] = [
            tuple(int(dim) for dim in state.shape),
            str(state.dtype),
            str(state.device),
            int(state.numel()) if hasattr(state, "numel") else None,
        ]
        # Guard against same-shape different-video cache hits.  A full content
        # hash would erase too much of the saved VAE time, so sample stable
        # positions across the flattened tensor.
        try:
            import torch

            if isinstance(state, torch.Tensor) and state.numel() > 0:
                flat = state.detach().reshape(-1)
                sample_count = min(32, int(flat.numel()))
                if sample_count == 1:
                    index_values = [0]
                else:
                    last = int(flat.numel()) - 1
                    index_values = [
                        (idx * last + (sample_count - 1) // 2) // (sample_count - 1)
                        for idx in range(sample_count)
                    ]
                indexes = torch.tensor(index_values, device=flat.device, dtype=torch.long)
                sample = flat.index_select(0, indexes).to(device="cpu", dtype=torch.float32)
                signature.append(tuple(float(value) for value in sample.tolist()))
        except Exception:
            signature.append("content_signature_unavailable")
        return tuple(signature)

    def wrap_generate_batch(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def generate_batch_with_phase(self_obj: Any, sample_args_list: Any, data_batch: dict[str, Any], *, warmup: bool = False):
            previous = self.phase
            self.phase = "warmup" if warmup else "measured"
            try:
                return original(self_obj, sample_args_list, data_batch, warmup=warmup)
            finally:
                self.phase = previous

        return generate_batch_with_phase

    def wrap_encode(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def encode_with_cache(model: Any, state: Any) -> Any:
            signature = self._signature(state)
            if self.phase == "measured" and signature is not None and self.cached:
                cached_signature, cached_value = self.cached.pop(0)
                if cached_signature == signature:
                    self.events.append(
                        {"phase": self.phase, "name": "vae_cache_hit", "s": 0.0, "ok": True, "signature": str(signature)}
                    )
                    return cached_value.clone()
                self.events.append(
                    {
                        "phase": self.phase,
                        "name": "vae_cache_miss_signature",
                        "s": 0.0,
                        "ok": True,
                        "expected": str(cached_signature),
                        "actual": str(signature),
                    }
                )
            encoded = original(model, state)
            if self.phase == "warmup" and signature is not None:
                self.cached.append((signature, encoded.detach().clone()))
                self.events.append(
                    {"phase": self.phase, "name": "vae_cache_store", "s": 0.0, "ok": True, "signature": str(signature)}
                )
            elif self.phase == "measured":
                self.events.append(
                    {
                        "phase": self.phase,
                        "name": "vae_cache_miss_empty",
                        "s": 0.0,
                        "ok": True,
                        "signature": str(signature),
                    }
                )
            return encoded

        return encode_with_cache


def _signature_for_value(value: Any) -> Any:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return _WarmupVAECache._signature(value)
    if isinstance(value, dict):
        return tuple((str(key), _signature_for_value(item)) for key, item in sorted(value.items(), key=lambda kv: str(kv[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_signature_for_value(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return tuple((field.name, _signature_for_value(getattr(value, field.name))) for field in fields(value))
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


class _WarmupPrepareCache:
    def __init__(
        self,
        *,
        slim_no_raw_state_vision: bool = False,
        slim_derive_condition_reference: bool = False,
        slim_derive_initial_noise: bool = False,
    ) -> None:
        self.phase = "setup"
        self.cached: list[tuple[tuple[Any, ...], Any]] = []
        self.slim_no_raw_state_vision = slim_no_raw_state_vision
        self.slim_derive_condition_reference = slim_derive_condition_reference
        self.slim_derive_initial_noise = slim_derive_initial_noise
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _signature(data_batch: Any, seed: Any, has_negative_prompt: Any) -> tuple[Any, ...]:
        return (
            _signature_for_value(data_batch),
            _signature_for_value(seed),
            bool(has_negative_prompt),
        )

    def wrap_generate_batch(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def generate_batch_with_phase(self_obj: Any, sample_args_list: Any, data_batch: dict[str, Any], *, warmup: bool = False):
            previous = self.phase
            self.phase = "warmup" if warmup else "measured"
            try:
                return original(self_obj, sample_args_list, data_batch, warmup=warmup)
            finally:
                self.phase = previous

        return generate_batch_with_phase

    def wrap_prepare(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def prepare_with_cache(model: Any, *args: Any, **kwargs: Any) -> Any:
            data_batch = args[0] if args else kwargs.get("data_batch")
            seed = args[1] if len(args) > 1 else kwargs.get("seed")
            has_negative_prompt = (
                args[2] if len(args) > 2 else kwargs.get("has_negative_prompt", False)
            )
            signature = self._signature(data_batch, seed, has_negative_prompt)
            if self.phase == "measured" and self.cached:
                cached_signature, cached_value = self.cached.pop(0)
                if cached_signature == signature:
                    self.events.append(
                        {
                            "phase": self.phase,
                            "name": "prepare_cache_hit",
                            "s": 0.0,
                            "ok": True,
                            "slim_no_raw_state_vision": self.slim_no_raw_state_vision,
                            "slim_derive_condition_reference": self.slim_derive_condition_reference,
                            "slim_derive_initial_noise": self.slim_derive_initial_noise,
                        }
                    )
                    payload = _prepare_payload_with_derived_initial_noise(_clone_for_warmup(cached_value), seed)
                    return _prepare_payload_with_derived_condition_reference(payload)
                self.events.append(
                    {
                        "phase": self.phase,
                        "name": "prepare_cache_miss_signature",
                        "s": 0.0,
                        "ok": True,
                        "expected": str(cached_signature),
                        "actual": str(signature),
                    }
                )
            result = original(model, *args, **kwargs)
            if self.phase == "warmup":
                cached_value = _slim_prepare_payload(
                    result,
                    no_raw_state_vision=self.slim_no_raw_state_vision,
                    derive_condition_reference=self.slim_derive_condition_reference,
                    derive_initial_noise=self.slim_derive_initial_noise,
                )
                self.cached.append((signature, cached_value))
                self.events.append(
                    {
                        "phase": self.phase,
                        "name": "prepare_cache_store",
                        "s": 0.0,
                        "ok": True,
                        "slim_no_raw_state_vision": self.slim_no_raw_state_vision,
                        "slim_derive_condition_reference": self.slim_derive_condition_reference,
                        "slim_derive_initial_noise": self.slim_derive_initial_noise,
                    }
                )
            elif self.phase == "measured":
                self.events.append(
                    {"phase": self.phase, "name": "prepare_cache_miss_empty", "s": 0.0, "ok": True}
                )
            return result

        return prepare_with_cache


def _first_tensor_device(value: Any) -> str | None:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return str(value.device)
    if isinstance(value, dict):
        for item in value.values():
            device = _first_tensor_device(item)
            if device is not None:
                return device
    if isinstance(value, (list, tuple)):
        for item in value:
            device = _first_tensor_device(item)
            if device is not None:
                return device
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            device = _first_tensor_device(getattr(value, field.name))
            if device is not None:
                return device
    return None


def _prepare_model_device(model: Any, data_batch: Any) -> str | None:
    tensor_kwargs = getattr(model, "tensor_kwargs", None)
    if isinstance(tensor_kwargs, dict) and tensor_kwargs.get("device") is not None:
        return str(tensor_kwargs["device"])
    tensor_kwargs_fp32 = getattr(model, "tensor_kwargs_fp32", None)
    if isinstance(tensor_kwargs_fp32, dict) and tensor_kwargs_fp32.get("device") is not None:
        return str(tensor_kwargs_fp32["device"])
    return _first_tensor_device(data_batch)


_PREPARE_PAYLOAD_FIELD_NAMES = (
    "sequence_plans",
    "gen_data_clean",
    "cond_text_tokens",
    "uncond_text_tokens",
    "initial_noise",
    "condition_reference",
    "condition_mask",
)


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def _tensor_inventory(path: str, tensor: Any) -> dict[str, Any]:
    return {
        "path": path,
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": [int(dim) for dim in tensor.stride()],
        "contiguous": bool(tensor.is_contiguous()),
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "bytes": int(tensor.numel()) * int(tensor.element_size()),
    }


def _prepare_inventory_value(
    value: Any,
    *,
    path: str,
    tensors: list[dict[str, Any]],
    seen: set[int],
    depth: int = 0,
    max_depth: int = 6,
    max_children: int = 16,
) -> Any:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        info = _tensor_inventory(path, value)
        tensors.append(info)
        return {key: info[key] for key in ("shape", "dtype", "device", "stride", "contiguous", "numel", "bytes")}
    if isinstance(value, (str, int, float, bool, type(None))):
        return {"type": type(value).__name__, "value": _json_safe_scalar(value)}
    if id(value) in seen:
        return {"type": value.__class__.__name__, "cycle": True}
    seen.add(id(value))
    if depth >= max_depth:
        return {"type": value.__class__.__name__, "truncated": "max_depth"}
    if is_dataclass(value) and not isinstance(value, type):
        children: dict[str, Any] = {}
        for field in fields(value):
            field_value = getattr(value, field.name)
            children[field.name] = _prepare_inventory_value(
                field_value,
                path=f"{path}.{field.name}",
                tensors=tensors,
                seen=seen,
                depth=depth + 1,
                max_depth=max_depth,
                max_children=max_children,
            )
        return {
            "type": value.__class__.__name__,
            "module": value.__class__.__module__,
            "fields": children,
        }
    if isinstance(value, dict):
        children = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_children:
                break
            key_text = str(key)
            children[key_text] = _prepare_inventory_value(
                item,
                path=f"{path}[{key_text!r}]",
                tensors=tensors,
                seen=seen,
                depth=depth + 1,
                max_depth=max_depth,
                max_children=max_children,
            )
        return {
            "type": "dict",
            "len": len(value),
            "truncated_children": max(0, len(value) - len(children)),
            "children": children,
        }
    if isinstance(value, (list, tuple)):
        children = []
        for idx, item in enumerate(value[:max_children]):
            children.append(
                _prepare_inventory_value(
                    item,
                    path=f"{path}[{idx}]",
                    tensors=tensors,
                    seen=seen,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_children=max_children,
                )
            )
        return {
            "type": type(value).__name__,
            "len": len(value),
            "truncated_children": max(0, len(value) - len(children)),
            "children": children,
        }
    return {"type": value.__class__.__name__, "repr": repr(value)[:256]}


def _prepare_field_name(index: int) -> str:
    if index < len(_PREPARE_PAYLOAD_FIELD_NAMES):
        return _PREPARE_PAYLOAD_FIELD_NAMES[index]
    return f"field_{index}"


def _prepare_payload_inventory(payload: Any, *, signature: Any = None, source: str | None = None) -> dict[str, Any]:
    tensors: list[dict[str, Any]] = []
    seen: set[int] = set()
    if isinstance(payload, tuple):
        root = {
            _prepare_field_name(idx): _prepare_inventory_value(
                item,
                path=_prepare_field_name(idx),
                tensors=tensors,
                seen=seen,
            )
            for idx, item in enumerate(payload)
        }
    else:
        root = _prepare_inventory_value(payload, path="payload", tensors=tensors, seen=seen)

    by_top_level: dict[str, dict[str, Any]] = {}
    by_dtype: dict[str, dict[str, Any]] = {}
    for tensor in tensors:
        top = str(tensor["path"]).split(".", 1)[0].split("[", 1)[0]
        field_item = by_top_level.setdefault(top, {"field": top, "tensor_count": 0, "bytes": 0})
        field_item["tensor_count"] += 1
        field_item["bytes"] += int(tensor["bytes"])
        dtype = str(tensor["dtype"])
        dtype_item = by_dtype.setdefault(dtype, {"dtype": dtype, "tensor_count": 0, "bytes": 0})
        dtype_item["tensor_count"] += 1
        dtype_item["bytes"] += int(tensor["bytes"])
    for items in (by_top_level.values(), by_dtype.values()):
        for item in items:
            item["mib"] = float(item["bytes"]) / (1024.0 * 1024.0)
    total_bytes = sum(int(tensor["bytes"]) for tensor in tensors)
    return {
        "version": 1,
        "schema": "flashrt_cosmos3_edge_prepare_inventory_v1",
        "source": source,
        "signature": signature,
        "field_names": list(_PREPARE_PAYLOAD_FIELD_NAMES),
        "payload_type": payload.__class__.__name__,
        "tensor_count": len(tensors),
        "tensor_bytes": total_bytes,
        "tensor_mib": float(total_bytes) / (1024.0 * 1024.0),
        "tensor_bytes_by_dtype": sorted(by_dtype.values(), key=lambda item: -int(item["bytes"])),
        "tensor_bytes_by_top_level": sorted(by_top_level.values(), key=lambda item: -int(item["bytes"])),
        "largest_tensors": sorted(tensors, key=lambda item: -int(item["bytes"]))[:32],
        "tree": root,
    }


def _write_prepare_inventory(path: Path, payload: Any, *, signature: Any = None, source: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _prepare_payload_inventory(payload, signature=signature, source=source),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _as_python_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _indexed_optional_int(values: Any, index: int) -> int | None:
    if values is None:
        return None
    value = values[index] if isinstance(values, (list, tuple)) else values
    if value is None:
        return None
    return _as_python_int(value)


def _indexed_optional_float(values: Any, index: int) -> float | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        value = values[index]
    else:
        try:
            value = values[index]
        except (IndexError, TypeError):
            value = values
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _num_vision_items_for_sample(values: Any, index: int) -> int:
    if values is None:
        return 1
    value = values[index]
    if isinstance(value, (list, tuple)):
        return len(value)
    return _as_python_int(value)


def _derive_condition_reference_from_prepare_payload(payload: Any) -> list[Any]:
    if not isinstance(payload, tuple) or len(payload) <= 5:
        raise RuntimeError("prepare payload cannot derive condition_reference from a non-v1 tuple")
    sequence_plans = payload[0]
    gen_data_clean = payload[1]
    initial_noise = payload[4]
    if not isinstance(sequence_plans, list) or not isinstance(initial_noise, list):
        raise RuntimeError("prepare payload cannot derive condition_reference without sequence_plans and initial_noise")

    x0_tokens_vision = getattr(gen_data_clean, "x0_tokens_vision", None)
    x0_tokens_action = getattr(gen_data_clean, "x0_tokens_action", None)
    x0_tokens_sound = getattr(gen_data_clean, "x0_tokens_sound", None)
    raw_action_dim = getattr(gen_data_clean, "raw_action_dim", None)
    num_vision_items_per_sample = getattr(gen_data_clean, "num_vision_items_per_sample", None)

    references: list[Any] = []
    idx_vision = 0
    idx_action = 0
    idx_sound = 0
    for sample_idx, plan in enumerate(sequence_plans):
        parts = []
        sample_noise = initial_noise[sample_idx]
        if getattr(plan, "has_vision", False):
            if x0_tokens_vision is None:
                raise RuntimeError("prepare payload cannot derive vision condition_reference without x0_tokens_vision")
            for _ in range(_num_vision_items_for_sample(num_vision_items_per_sample, sample_idx)):
                x0_vision = x0_tokens_vision[idx_vision].to(dtype=sample_noise.dtype, device=sample_noise.device)
                parts.append(x0_vision.reshape(-1))
                idx_vision += 1
        if getattr(plan, "has_action", False):
            if x0_tokens_action is None:
                raise RuntimeError("prepare payload cannot derive action condition_reference without x0_tokens_action")
            x0_action = x0_tokens_action[idx_action].to(dtype=sample_noise.dtype, device=sample_noise.device)
            action_dim = _indexed_optional_int(raw_action_dim, idx_action)
            if action_dim is not None:
                x0_action = x0_action.clone()
                x0_action[:, action_dim:] = 0
            parts.append(x0_action.reshape(-1))
            idx_action += 1
        if getattr(plan, "has_sound", False):
            if x0_tokens_sound is None:
                raise RuntimeError("prepare payload cannot derive sound condition_reference without x0_tokens_sound")
            x0_sound = x0_tokens_sound[idx_sound].to(dtype=sample_noise.dtype, device=sample_noise.device)
            parts.append(x0_sound.reshape(-1))
            idx_sound += 1
        if not parts:
            raise RuntimeError("prepare payload cannot derive an empty condition_reference sample")
        import torch

        references.append(torch.cat(parts, dim=0))
    return references


def _seed_list_for_prepare(seed: Any, n_sample: int) -> list[int]:
    if seed is None:
        raise RuntimeError("prepare payload cannot derive initial_noise without seed")
    if isinstance(seed, (list, tuple)):
        values = list(seed)
    elif hasattr(seed, "tolist"):
        raw = seed.tolist()
        values = raw if isinstance(raw, list) else [raw]
    else:
        values = [seed]
    if len(values) != n_sample:
        raise RuntimeError(f"prepare initial_noise seed count mismatch: got {len(values)} expected {n_sample}")
    return [_as_python_int(value) for value in values]


def _arch_invariant_rand_like(shape: Any, *, dtype: Any, device: Any, seed: int) -> Any:
    import numpy as np
    import torch

    random_array = np.random.RandomState(seed).standard_normal(tuple(int(dim) for dim in shape)).astype(np.float32)
    return torch.from_numpy(random_array).to(dtype=dtype, device=device)


def _append_derived_noise_part(
    parts: list[Any],
    *,
    x0: Any,
    flat_mask: Any,
    offset: int,
    seed: int,
    pure_noise_dtype: Any,
    raw_action_dim: int | None = None,
) -> int:
    numel = int(x0.numel())
    mask = flat_mask[offset : offset + numel].reshape(x0.shape).to(device=x0.device, dtype=flat_mask.dtype)
    pure_noise = _arch_invariant_rand_like(x0.shape, dtype=pure_noise_dtype, device=x0.device, seed=seed)
    x0_t = x0.to(device=x0.device, dtype=pure_noise_dtype)
    noise = mask * x0_t + (1.0 - mask) * pure_noise
    if raw_action_dim is not None:
        noise = noise.clone()
        noise[:, raw_action_dim:] = 0
    parts.append(noise.reshape(-1))
    return offset + numel


def _derive_initial_noise_from_prepare_payload(payload: Any, seed: Any) -> list[Any]:
    if not isinstance(payload, tuple) or len(payload) <= 6:
        raise RuntimeError("prepare payload cannot derive initial_noise from a non-v1 tuple")
    sequence_plans = payload[0]
    gen_data_clean = payload[1]
    condition_mask = payload[6]
    if not isinstance(sequence_plans, list) or not isinstance(condition_mask, list):
        raise RuntimeError("prepare payload cannot derive initial_noise without sequence_plans and condition_mask")

    x0_tokens_vision = getattr(gen_data_clean, "x0_tokens_vision", None)
    x0_tokens_action = getattr(gen_data_clean, "x0_tokens_action", None)
    x0_tokens_sound = getattr(gen_data_clean, "x0_tokens_sound", None)
    raw_action_dim = getattr(gen_data_clean, "raw_action_dim", None)
    num_vision_items_per_sample = getattr(gen_data_clean, "num_vision_items_per_sample", None)
    seeds = _seed_list_for_prepare(seed, len(sequence_plans))

    noises: list[Any] = []
    idx_vision = 0
    idx_action = 0
    idx_sound = 0
    for sample_idx, plan in enumerate(sequence_plans):
        flat_mask = condition_mask[sample_idx]
        offset = 0
        parts: list[Any] = []
        sample_seed = seeds[sample_idx]
        if getattr(plan, "has_vision", False):
            if x0_tokens_vision is None:
                raise RuntimeError("prepare payload cannot derive vision initial_noise without x0_tokens_vision")
            for _ in range(_num_vision_items_for_sample(num_vision_items_per_sample, sample_idx)):
                x0_vision = x0_tokens_vision[idx_vision]
                offset = _append_derived_noise_part(
                    parts,
                    x0=x0_vision,
                    flat_mask=flat_mask,
                    offset=offset,
                    seed=sample_seed,
                    pure_noise_dtype=flat_mask.dtype,
                )
                idx_vision += 1
        if getattr(plan, "has_action", False):
            if x0_tokens_action is None:
                raise RuntimeError("prepare payload cannot derive action initial_noise without x0_tokens_action")
            x0_action = x0_tokens_action[idx_action]
            offset = _append_derived_noise_part(
                parts,
                x0=x0_action,
                flat_mask=flat_mask,
                offset=offset,
                seed=sample_seed,
                pure_noise_dtype=x0_action.dtype,
                raw_action_dim=_indexed_optional_int(raw_action_dim, idx_action),
            )
            idx_action += 1
        if getattr(plan, "has_sound", False):
            if x0_tokens_sound is None:
                raise RuntimeError("prepare payload cannot derive sound initial_noise without x0_tokens_sound")
            x0_sound = x0_tokens_sound[idx_sound]
            offset = _append_derived_noise_part(
                parts,
                x0=x0_sound,
                flat_mask=flat_mask,
                offset=offset,
                seed=sample_seed,
                pure_noise_dtype=x0_sound.dtype,
            )
            idx_sound += 1
        if not parts:
            raise RuntimeError("prepare payload cannot derive an empty initial_noise sample")
        if offset != int(flat_mask.numel()):
            raise RuntimeError(
                f"prepare initial_noise derivation consumed {offset} mask values, expected {int(flat_mask.numel())}"
            )
        import torch

        noises.append(torch.cat(parts, dim=0).to(dtype=flat_mask.dtype, device=flat_mask.device))
    return noises


def _replace_prepare_payload_field(payload: Any, index: int, value: Any) -> Any:
    if not isinstance(payload, tuple) or index >= len(payload):
        raise RuntimeError("prepare payload cannot replace field in a non-v1 tuple")
    items = list(payload)
    items[index] = value
    return tuple(items)


def _prepare_payload_with_derived_condition_reference(payload: Any) -> Any:
    if isinstance(payload, tuple) and len(payload) > 5 and payload[5] is None:
        return _replace_prepare_payload_field(payload, 5, _derive_condition_reference_from_prepare_payload(payload))
    return payload


def _prepare_payload_with_derived_initial_noise(payload: Any, seed: Any) -> Any:
    if isinstance(payload, tuple) and len(payload) > 4 and payload[4] is None:
        return _replace_prepare_payload_field(payload, 4, _derive_initial_noise_from_prepare_payload(payload, seed))
    return payload


def _flat_parts_for_edge_av(flat: Any) -> tuple[Any, Any]:
    from flash_rt.models.cosmos3_edge.dump_replay import EDGE_ACTION_MODEL_SHAPE, EDGE_FLAT_DIM, EDGE_VISION_SHAPE

    if int(flat.numel()) != EDGE_FLAT_DIM:
        raise RuntimeError(f"expected Edge flat latent dim {EDGE_FLAT_DIM}, got {int(flat.numel())}")
    vision_dim = 1
    for dim in EDGE_VISION_SHAPE:
        vision_dim *= int(dim)
    flat = flat.reshape(-1)
    return flat[:vision_dim].reshape(EDGE_VISION_SHAPE), flat[vision_dim:].reshape(EDGE_ACTION_MODEL_SHAPE)


def _edge_vae_mrope_ids(
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    fps: float,
    base_fps: float,
    temporal_compression_factor: int,
    base_temporal_compression_factor: int | None = None,
    start_frame_offset: int = 0,
    device: Any,
) -> tuple[Any, int | float]:
    import math
    import torch

    effective_base_tcf = (
        base_temporal_compression_factor
        if base_temporal_compression_factor is not None
        else temporal_compression_factor
    )
    tps = fps / temporal_compression_factor
    base_tps = base_fps / effective_base_tcf
    frame_indices = torch.arange(grid_t, dtype=torch.float32, device=device)
    scaled_t = (frame_indices + start_frame_offset) / tps * base_tps + temporal_offset
    t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    h_index = torch.arange(grid_h, dtype=torch.long, device=device).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long, device=device).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
    mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    return mrope_ids, math.ceil(mrope_ids.max().item()) + 1


def _derive_step0_position_ids_from_prepare_payload(payload: Any) -> Any:
    """Derive Edge AV step-0 3D MRoPE position ids from slim prepare state."""

    import math
    import torch

    if not isinstance(payload, tuple) or len(payload) <= 3:
        raise RuntimeError("prepare payload cannot derive position_ids from a non-v1 tuple")
    sequence_plans = payload[0]
    gen_data_clean = payload[1]
    cond_text_tokens = payload[2]
    if not isinstance(sequence_plans, list) or len(sequence_plans) != 1:
        raise RuntimeError("position_ids derivation currently supports batch size 1")
    if not isinstance(cond_text_tokens, list) or len(cond_text_tokens) != 1:
        raise RuntimeError("position_ids derivation requires one cond_text_tokens entry")

    plan = sequence_plans[0]
    if not (getattr(plan, "has_text", False) and getattr(plan, "has_vision", False) and getattr(plan, "has_action", False)):
        raise RuntimeError("position_ids derivation currently supports Edge AV text+vision+action packs")
    if getattr(plan, "has_sound", False):
        raise RuntimeError("position_ids derivation does not support sound packs")
    if getattr(plan, "share_vision_temporal_positions", False):
        raise RuntimeError("position_ids derivation does not support shared vision temporal positions")

    x0_tokens_vision = getattr(gen_data_clean, "x0_tokens_vision", None)
    x0_tokens_action = getattr(gen_data_clean, "x0_tokens_action", None)
    if not isinstance(x0_tokens_vision, list) or len(x0_tokens_vision) != 1:
        raise RuntimeError("position_ids derivation requires one vision token tensor")
    if not isinstance(x0_tokens_action, list) or len(x0_tokens_action) != 1:
        raise RuntimeError("position_ids derivation requires one action token tensor")

    vision = x0_tokens_vision[0]
    action = x0_tokens_action[0]
    if len(tuple(vision.shape)) != 5:
        raise RuntimeError(f"unexpected vision token shape for position_ids derivation: {tuple(vision.shape)}")
    if len(tuple(action.shape)) != 2:
        raise RuntimeError(f"unexpected action token shape for position_ids derivation: {tuple(action.shape)}")

    device = vision.device
    latent_patch_size = 2
    temporal_compression_factor = 4
    base_fps = 24.0
    temporal_modality_margin = 15000
    text_len = len(cond_text_tokens[0]) + 2  # official Edge packs EOS and start-of-generation after prompt ids.
    _, _, latent_t, latent_h, latent_w = vision.shape
    patch_h = math.ceil(int(latent_h) / latent_patch_size)
    patch_w = math.ceil(int(latent_w) / latent_patch_size)
    action_len = int(action.shape[0])

    vision_fps = _indexed_optional_float(getattr(gen_data_clean, "fps_vision", None), 0)
    action_fps = _indexed_optional_float(getattr(gen_data_clean, "fps_action", None), 0)
    if vision_fps is None or action_fps is None:
        raise RuntimeError("position_ids derivation requires fps_vision and fps_action")

    text_ids = torch.arange(text_len, dtype=torch.float32, device=device).unsqueeze(0).expand(3, -1).contiguous()
    vision_temporal_offset = text_len + temporal_modality_margin
    vision_ids, _ = _edge_vae_mrope_ids(
        grid_t=int(latent_t),
        grid_h=patch_h,
        grid_w=patch_w,
        temporal_offset=vision_temporal_offset,
        fps=vision_fps,
        base_fps=base_fps,
        temporal_compression_factor=temporal_compression_factor,
        device=device,
    )
    action_ids, _ = _edge_vae_mrope_ids(
        grid_t=action_len,
        grid_h=1,
        grid_w=1,
        temporal_offset=vision_temporal_offset,
        fps=action_fps,
        base_fps=base_fps,
        temporal_compression_factor=1,
        base_temporal_compression_factor=temporal_compression_factor,
        start_frame_offset=_as_python_int(getattr(plan, "action_start_frame_offset", 1)),
        device=device,
    )
    return torch.cat([text_ids, vision_ids, action_ids], dim=1).contiguous()


def _derive_step0_rope_from_position_ids(position_ids: Any, causal_indices: Any, full_indices: Any) -> dict[str, Any]:
    import torch

    head_dim = 128
    rope_theta = 100000000.0
    mrope_section = [24, 20, 20]
    device = position_ids.device
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).to(dtype=torch.float32) / head_dim)
    )
    expanded_position_ids = position_ids.unsqueeze(1) if position_ids.ndim == 2 else position_ids
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, expanded_position_ids.shape[1], -1, 1)
    position_ids_expanded = expanded_position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        freqs_t[..., slice(offset, length, 3)] = freqs[dim, ..., slice(offset, length, 3)]
    emb = torch.cat((freqs_t, freqs_t), dim=-1)
    cos = emb.cos().squeeze(0).to(dtype=torch.bfloat16).contiguous()
    sin = emb.sin().squeeze(0).to(dtype=torch.bfloat16).contiguous()
    causal_long = causal_indices.to(dtype=torch.long)
    full_long = full_indices.to(dtype=torch.long)
    return {
        "s00/layers/00/rope/cos/causal_seq": cos.index_select(0, causal_long).contiguous(),
        "s00/layers/00/rope/cos/full_only_seq": cos.index_select(0, full_long).contiguous(),
        "s00/layers/00/rope/sin/causal_seq": sin.index_select(0, causal_long).contiguous(),
        "s00/layers/00/rope/sin/full_only_seq": sin.index_select(0, full_long).contiguous(),
    }


def _derive_step0_causal_seq_from_prepare_payload(payload: Any, text_embedding_weight: Any) -> Any:
    import torch

    if not isinstance(payload, tuple) or len(payload) <= 2:
        raise RuntimeError("prepare payload cannot derive causal_seq from a non-v1 tuple")
    cond_text_tokens = payload[2]
    if not isinstance(cond_text_tokens, list) or len(cond_text_tokens) != 1:
        raise RuntimeError("causal_seq derivation requires one cond_text_tokens entry")
    if len(tuple(text_embedding_weight.shape)) != 2 or int(text_embedding_weight.shape[1]) != 2048:
        raise RuntimeError(f"unexpected text embedding shape: {tuple(text_embedding_weight.shape)}")

    eos_token_id = 11
    start_of_generation_token_id = 20
    token_ids = list(cond_text_tokens[0]) + [eos_token_id, start_of_generation_token_id]
    if len(token_ids) != 125:
        raise RuntimeError(f"unexpected Edge causal token count: {len(token_ids)}")
    ids = torch.tensor(token_ids, dtype=torch.long, device=text_embedding_weight.device)
    return text_embedding_weight.index_select(0, ids).contiguous()


def _derive_step0_vfm_boundary_from_prepare_payload(
    payload: Any,
    seed: Any,
    text_embedding_weight: Any | None = None,
) -> dict[str, Any]:
    """Derive the step-0 VFM/noise boundary tensors from a prepared Edge payload.

    The full-only LM packed hidden states still require native VLM/prefill work;
    this helper isolates the tensors that already follow from the slim prepare
    contract, model text embeddings, and static Edge AV pack/RoPE rules.
    """

    import torch

    payload = _prepare_payload_with_derived_initial_noise(payload, seed)
    if not isinstance(payload, tuple) or len(payload) <= 6:
        raise RuntimeError("prepare payload cannot derive step0 boundary from a non-v1 tuple")
    sequence_plans = payload[0]
    gen_data_clean = payload[1]
    initial_noise = payload[4]
    condition_mask = payload[6]
    if not isinstance(sequence_plans, list) or len(sequence_plans) != 1:
        raise RuntimeError("step0 boundary derivation currently supports batch size 1")
    if not isinstance(initial_noise, list) or len(initial_noise) != 1:
        raise RuntimeError("step0 boundary derivation requires one initial_noise tensor")
    if not isinstance(condition_mask, list) or len(condition_mask) != 1:
        raise RuntimeError("step0 boundary derivation requires one condition_mask tensor")

    x0_tokens_vision = getattr(gen_data_clean, "x0_tokens_vision", None)
    action_domain_id = getattr(gen_data_clean, "action_domain_id", None)
    raw_action_dim = getattr(gen_data_clean, "raw_action_dim", None)
    if not isinstance(x0_tokens_vision, list) or len(x0_tokens_vision) != 1:
        raise RuntimeError("step0 boundary derivation requires one vision token tensor")

    noise_vision, noise_action = _flat_parts_for_edge_av(initial_noise[0])
    mask_vision, mask_action = _flat_parts_for_edge_av(condition_mask[0])
    device = initial_noise[0].device

    vision_condition = mask_vision.to(dtype=torch.float32)
    if tuple(vision_condition.shape) != tuple(noise_vision.shape):
        raise RuntimeError("unexpected vision condition mask shape")
    vision_condition = vision_condition.amax(dim=(0, 1, 3, 4)).reshape(-1, 1, 1).contiguous()
    action_condition = mask_action.to(dtype=torch.float32).amax(dim=1, keepdim=True).contiguous()
    raw_action_dim_value = _indexed_optional_int(raw_action_dim, 0)
    if raw_action_dim_value is None:
        raise RuntimeError("step0 boundary derivation requires raw_action_dim")
    domain_id_value = _indexed_optional_int(action_domain_id, 0)
    if domain_id_value is None:
        raise RuntimeError("step0 boundary derivation requires action_domain_id")

    causal_indices = torch.arange(125, dtype=torch.int32, device=device)
    full_indices = torch.arange(125, 6425, dtype=torch.int32, device=device)
    position_ids = _derive_step0_position_ids_from_prepare_payload(payload).to(device=device)
    derived = {
        "once/num_velocity_calls": torch.tensor([1], dtype=torch.int64, device=device),
        "s00/lm_in/_causal_indices": causal_indices,
        "s00/lm_in/_causal_seq_offsets": torch.tensor([0, 125], dtype=torch.int32, device=device),
        "s00/lm_in/_full_indices": full_indices,
        "s00/lm_in/_full_only_seq_offsets": torch.tensor([0, 6300], dtype=torch.int32, device=device),
        "s00/lm_in/position_ids": position_ids,
        "s00/lm_in/sample_offsets": torch.tensor([0, 6425], dtype=torch.int32, device=device),
        "steps/00/noise_x": initial_noise[0].contiguous(),
        "steps/00/timestep": torch.tensor([[999]], dtype=torch.int64, device=device),
        "s00/vfm_in/vision/tokens/0": noise_vision.contiguous(),
        "s00/vfm_in/vision/condition_mask/0": vision_condition,
        "s00/vfm_in/vision/sequence_indexes": torch.arange(125, 6365, dtype=torch.int64, device=device),
        "s00/vfm_in/vision/timesteps": torch.empty((0,), dtype=torch.float32, device=device),
        "s00/vfm_in/vision/mse_loss_indexes": torch.empty((0,), dtype=torch.int64, device=device),
        "s00/vfm_in/vision/noisy_frame_indexes/0": torch.empty((0,), dtype=torch.int64, device=device),
        "s00/vfm_in/action/tokens/0": noise_action.contiguous(),
        "s00/vfm_in/action/condition_mask/0": action_condition,
        "s00/vfm_in/action/sequence_indexes": torch.arange(6365, 6425, dtype=torch.int64, device=device),
        "s00/vfm_in/action/timesteps": torch.full((60,), 999.0, dtype=torch.float32, device=device),
        "s00/vfm_in/action/mse_loss_indexes": torch.arange(6365, 6425, dtype=torch.int64, device=device),
        "s00/vfm_in/action/noisy_frame_indexes/0": torch.arange(60, dtype=torch.int64, device=device),
        "s00/vfm_in/action/domain_id/0": torch.tensor(domain_id_value, dtype=torch.int64, device=device),
        "s00/vfm_in/action/raw_action_dim/0": torch.tensor(raw_action_dim_value, dtype=torch.int64, device=device),
    }
    derived.update(_derive_step0_rope_from_position_ids(position_ids, causal_indices, full_indices))
    if text_embedding_weight is not None:
        derived["s00/lm_in/causal_seq"] = _derive_step0_causal_seq_from_prepare_payload(payload, text_embedding_weight).to(
            device=device
        )
    return derived


def _prepare_artifact_seed(artifact: dict[str, Any]) -> Any | None:
    signature = artifact.get("signature")
    if isinstance(signature, (list, tuple)) and len(signature) > 1:
        seed_sig = signature[1]
        if isinstance(seed_sig, tuple):
            return list(seed_sig)
        return seed_sig
    return None


def _load_text_embedding_weight_for_checkpoint(checkpoint: Path) -> Any:
    from safetensors import safe_open

    transformer_dir = checkpoint / "transformer"
    candidates = [
        transformer_dir / "diffusion_pytorch_model-00001-of-00002.safetensors",
        *sorted(transformer_dir.glob("*.safetensors")),
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if "embed_tokens.weight" in handle.keys():
                return handle.get_tensor("embed_tokens.weight")
    raise RuntimeError(f"{checkpoint} is missing transformer embed_tokens.weight")


def _derive_step0_executable_boundary_from_prepare_artifact(
    prepare_path: Path,
    checkpoint: Path,
    *,
    seed: Any | None = None,
    device: Any = "cuda",
) -> dict[str, Any]:
    """Build the executable pre-layer boundary from a slim prepare artifact.

    The returned tensor map is sufficient for ``EdgeStaticBufferEngine``. Most
    tensors come from the bit-exact slim-prepare contract; ``full_only_seq`` is
    synthesized through the same Torch reference used by the live engine bring-up.
    """

    import torch

    loaded = torch.load(str(prepare_path), map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict) or loaded.get("version") != 1 or "payload" not in loaded:
        raise RuntimeError(f"{prepare_path} is not a FlashRT prepare boundary v1 artifact")
    effective_seed = seed if seed is not None else _prepare_artifact_seed(loaded)
    if effective_seed is None:
        raise RuntimeError("prepare-derived live boundary requires a seed or a prepare artifact signature")
    return _derive_step0_executable_boundary_from_prepare_payload(
        loaded["payload"],
        checkpoint,
        seed=effective_seed,
        device=device,
        path=f"{prepare_path}:derived",
    )


def _derive_step0_executable_boundary_from_prepare_payload(
    payload: Any,
    checkpoint: Path,
    *,
    seed: Any,
    device: Any = "cuda",
    path: str = "live_prepare_result:derived",
) -> dict[str, Any]:
    """Build the executable pre-layer boundary directly from a prepare payload."""

    import torch

    from flash_rt.models.cosmos3_edge import EdgeBoundaryDump, EdgeTransformerWeights
    from flash_rt.models.cosmos3_edge.layer_ref import EdgeTransformerTorchReference

    text_embedding = _load_text_embedding_weight_for_checkpoint(checkpoint)
    derived = _derive_step0_vfm_boundary_from_prepare_payload(
        payload,
        seed=seed,
        text_embedding_weight=text_embedding,
    )
    tensors = {key: tensor.detach().cpu().contiguous() for key, tensor in derived.items()}
    causal = tensors["s00/lm_in/causal_seq"]
    full_placeholder = torch.zeros((6300, causal.shape[1]), dtype=causal.dtype)
    tensors["s00/lm_in/full_only_seq"] = full_placeholder
    tensors["s00/layers/00/input/causal_seq"] = causal.clone()
    tensors["s00/layers/00/input/full_only_seq"] = full_placeholder.clone()

    boundary = EdgeBoundaryDump.from_tensors(tensors, path=path)
    weights = EdgeTransformerWeights(checkpoint)
    ref = EdgeTransformerTorchReference(weights, device=device, dtype=torch.bfloat16)
    full = ref.full_sequence_for_step(
        boundary,
        tensors["steps/00/noise_x"].to(device=device),
        tensors["steps/00/timestep"].to(device=device),
    ).detach().cpu().contiguous()
    tensors["s00/lm_in/full_only_seq"] = full
    tensors["s00/layers/00/input/full_only_seq"] = full.clone()
    EdgeBoundaryDump.from_tensors(tensors, path=path).validate_geometry()
    return tensors


def _slim_prepare_payload(
    payload: Any,
    *,
    no_raw_state_vision: bool = False,
    derive_condition_reference: bool = False,
    derive_initial_noise: bool = False,
) -> Any:
    slim = _clone_for_warmup(payload)
    if isinstance(slim, tuple) and len(slim) > 1:
        gen_data_clean = slim[1]
        if no_raw_state_vision and hasattr(gen_data_clean, "raw_state_vision"):
            gen_data_clean.raw_state_vision = None
    if derive_condition_reference:
        slim = _replace_prepare_payload_field(slim, 5, None)
    if derive_initial_noise:
        slim = _replace_prepare_payload_field(slim, 4, None)
    return slim


def _slim_prepare_payload_no_raw_state_vision(payload: Any) -> Any:
    return _slim_prepare_payload(payload, no_raw_state_vision=True)


class _PrepareBoundary:
    def __init__(
        self,
        dump_out: Path | None,
        replay_in: Path | None,
        inventory_out: Path | None,
        *,
        slim_no_raw_state_vision: bool = False,
        slim_derive_condition_reference: bool = False,
        slim_derive_initial_noise: bool = False,
    ) -> None:
        self.dump_out = dump_out
        self.replay_in = replay_in
        self.inventory_out = inventory_out
        self.slim_no_raw_state_vision = slim_no_raw_state_vision
        self.slim_derive_condition_reference = slim_derive_condition_reference
        self.slim_derive_initial_noise = slim_derive_initial_noise
        self.events: list[dict[str, Any]] = []
        self._loaded: dict[str, Any] | None = None
        self._last_effective_slim_no_raw_state_vision = slim_no_raw_state_vision
        self._last_effective_slim_derive_condition_reference = slim_derive_condition_reference
        self._last_effective_slim_derive_initial_noise = slim_derive_initial_noise
        self.latest_payload: Any | None = None
        self.latest_signature: tuple[Any, ...] | None = None
        self.latest_seed: Any | None = None
        self.latest_source: str | None = None

    @staticmethod
    def _signature(data_batch: Any, seed: Any, has_negative_prompt: Any) -> tuple[Any, ...]:
        return _WarmupPrepareCache._signature(data_batch, seed, has_negative_prompt)

    @staticmethod
    def _signature_json(signature: tuple[Any, ...]) -> str:
        return json.dumps(signature, sort_keys=True)

    def _remember(self, payload: Any, *, signature: tuple[Any, ...], seed: Any, source: str) -> None:
        self.latest_payload = _clone_for_warmup(payload)
        self.latest_signature = signature
        self.latest_seed = seed
        self.latest_source = source

    def latest_effective_payload(self) -> Any:
        if self.latest_payload is None:
            raise RuntimeError("live prepare boundary requested before _prepare_inference_data produced a payload")
        return self.latest_payload

    def _load(self, *, model: Any, data_batch: Any, seed: Any, has_negative_prompt: Any) -> Any:
        if self.replay_in is None:
            raise RuntimeError("prepare replay path is not configured")
        import torch

        if self._loaded is None:
            map_location = _prepare_model_device(model, data_batch)
            load_kwargs: dict[str, Any] = {"map_location": map_location} if map_location is not None else {}
            try:
                loaded = torch.load(str(self.replay_in), weights_only=False, **load_kwargs)
            except TypeError:
                loaded = torch.load(str(self.replay_in), **load_kwargs)
            if not isinstance(loaded, dict) or loaded.get("version") != 1 or "payload" not in loaded:
                raise RuntimeError(f"{self.replay_in} is not a FlashRT prepare boundary v1 artifact")
            self._loaded = loaded
        signature = self._signature(data_batch, seed, has_negative_prompt)
        loaded_signature = self._loaded.get("signature")
        if loaded_signature is not None:
            loaded_signature_json = self._signature_json(tuple(loaded_signature))
            current_signature_json = self._signature_json(signature)
            if loaded_signature_json != current_signature_json:
                raise RuntimeError("prepare replay signature mismatch; refusing to reuse a different prepared state")
        effective_slim_no_raw_state_vision = self.slim_no_raw_state_vision or bool(
            self._loaded.get("slim_no_raw_state_vision")
        )
        effective_slim_derive_condition_reference = self.slim_derive_condition_reference or bool(
            self._loaded.get("slim_derive_condition_reference")
        )
        effective_slim_derive_initial_noise = self.slim_derive_initial_noise or bool(
            self._loaded.get("slim_derive_initial_noise")
        )
        self._last_effective_slim_no_raw_state_vision = effective_slim_no_raw_state_vision
        self._last_effective_slim_derive_condition_reference = effective_slim_derive_condition_reference
        self._last_effective_slim_derive_initial_noise = effective_slim_derive_initial_noise
        payload = _slim_prepare_payload(
            self._loaded["payload"],
            no_raw_state_vision=effective_slim_no_raw_state_vision,
            derive_condition_reference=effective_slim_derive_condition_reference,
            derive_initial_noise=effective_slim_derive_initial_noise,
        )
        payload = _prepare_payload_with_derived_initial_noise(payload, seed)
        payload = _prepare_payload_with_derived_condition_reference(payload)
        if self.inventory_out is not None:
            _write_prepare_inventory(
                self.inventory_out,
                payload,
                signature=signature,
                source=str(self.replay_in),
            )
        return payload

    def wrap_prepare(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def prepare_with_boundary(model: Any, *args: Any, **kwargs: Any) -> Any:
            data_batch = args[0] if args else kwargs.get("data_batch")
            seed = args[1] if len(args) > 1 else kwargs.get("seed")
            has_negative_prompt = args[2] if len(args) > 2 else kwargs.get("has_negative_prompt", False)
            signature = self._signature(data_batch, seed, has_negative_prompt)
            if self.replay_in is not None:
                t0 = time.perf_counter()
                payload = self._load(model=model, data_batch=data_batch, seed=seed, has_negative_prompt=has_negative_prompt)
                self._remember(
                    payload,
                    signature=signature,
                    seed=seed,
                    source=str(self.replay_in),
                )
                self.events.append(
                    {
                        "name": "prepare_replay_hit",
                        "phase": "measured",
                        "s": time.perf_counter() - t0,
                        "ok": True,
                        "wrote_inventory": self.inventory_out is not None,
                        "slim_no_raw_state_vision": self._last_effective_slim_no_raw_state_vision,
                        "slim_derive_condition_reference": self._last_effective_slim_derive_condition_reference,
                        "slim_derive_initial_noise": self._last_effective_slim_derive_initial_noise,
                    }
                )
                return payload

            t0 = time.perf_counter()
            result = original(model, *args, **kwargs)
            elapsed = time.perf_counter() - t0
            output_payload = _slim_prepare_payload(
                result,
                no_raw_state_vision=self.slim_no_raw_state_vision,
                derive_condition_reference=self.slim_derive_condition_reference,
                derive_initial_noise=self.slim_derive_initial_noise,
            )
            if self.dump_out is not None:
                import torch

                self.dump_out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "version": 1,
                        "signature": signature,
                        "payload": _clone_for_warmup(output_payload),
                        "slim_no_raw_state_vision": self.slim_no_raw_state_vision,
                        "slim_derive_condition_reference": self.slim_derive_condition_reference,
                        "slim_derive_initial_noise": self.slim_derive_initial_noise,
                    },
                    str(self.dump_out),
                )
            if self.inventory_out is not None:
                _write_prepare_inventory(
                    self.inventory_out,
                    output_payload,
                    signature=signature,
                    source=str(self.dump_out) if self.dump_out is not None else "live_prepare_result",
                )
            self.events.append(
                {
                    "name": "prepare_boundary_dump",
                    "phase": "measured",
                    "s": elapsed,
                    "ok": True,
                    "wrote_dump": self.dump_out is not None,
                    "wrote_inventory": self.inventory_out is not None,
                    "slim_no_raw_state_vision": self.slim_no_raw_state_vision,
                    "slim_derive_condition_reference": self.slim_derive_condition_reference,
                    "slim_derive_initial_noise": self.slim_derive_initial_noise,
                }
            )
            output_payload = _prepare_payload_with_derived_initial_noise(output_payload, seed)
            output_payload = _prepare_payload_with_derived_condition_reference(output_payload)
            self._remember(
                output_payload,
                signature=signature,
                seed=seed,
                source=str(self.dump_out) if self.dump_out is not None else "live_prepare_result",
            )
            return output_payload

        return prepare_with_boundary


def _install_prepare_boundary_patch(
    dump_out: Path | None,
    replay_in: Path | None,
    inventory_out: Path | None,
    *,
    slim_no_raw_state_vision: bool = False,
    slim_derive_condition_reference: bool = False,
    slim_derive_initial_noise: bool = False,
) -> _PrepareBoundary:
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    boundary = _PrepareBoundary(
        dump_out,
        replay_in,
        inventory_out,
        slim_no_raw_state_vision=slim_no_raw_state_vision,
        slim_derive_condition_reference=slim_derive_condition_reference,
        slim_derive_initial_noise=slim_derive_initial_noise,
    )
    OmniMoTModel._prepare_inference_data = boundary.wrap_prepare(OmniMoTModel._prepare_inference_data)
    return boundary


class _VAEEncodeBoundary:
    def __init__(
        self,
        dump_out: Path | None,
        latent_in: Path | None,
        *,
        dump_input: bool = False,
    ) -> None:
        self.dump_out = dump_out
        self.latent_in = latent_in
        self.dump_input = dump_input
        self.events: list[dict[str, Any]] = []
        self._loaded_latent: Any = None
        self._loaded_signature: tuple[Any, ...] | None = None
        self._loaded_metadata: dict[str, str] | None = None

    @staticmethod
    def _signature(state: Any) -> tuple[Any, ...] | None:
        return _WarmupVAECache._signature(state)

    @staticmethod
    def _signature_json(signature: tuple[Any, ...] | None) -> str:
        return json.dumps(signature, sort_keys=True)

    def _load_latent(self, state: Any) -> Any:
        if self.latent_in is None:
            raise RuntimeError("latent input path is not configured")
        if self._loaded_latent is None:
            from safetensors.torch import load_file, safe_open

            device = str(state.device) if hasattr(state, "device") else "cpu"
            tensors = load_file(str(self.latent_in), device=device)
            if "vae_encode/output" not in tensors:
                raise RuntimeError(f"{self.latent_in} is missing vae_encode/output")
            with safe_open(str(self.latent_in), framework="pt", device="cpu") as f:
                self._loaded_metadata = dict(f.metadata() or {})
            raw_signature = (self._loaded_metadata or {}).get("state_signature")
            self._loaded_signature = tuple(json.loads(raw_signature)) if raw_signature else None
            self._loaded_latent = tensors["vae_encode/output"]
        signature = self._signature(state)
        if self._loaded_signature is not None and signature is not None:
            loaded_signature_json = self._signature_json(self._loaded_signature)
            current_signature_json = self._signature_json(signature)
            if loaded_signature_json != current_signature_json:
                raise RuntimeError(
                    "VAE latent input signature mismatch; refusing to reuse a latent from a different input"
                )
        return self._loaded_latent.clone()

    def wrap_encode(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def encode_with_boundary(model: Any, state: Any) -> Any:
            signature = self._signature(state)
            if self.latent_in is not None:
                t0 = time.perf_counter()
                latent = self._load_latent(state)
                self.events.append(
                    {
                        "name": "vae_latent_replay_hit",
                        "phase": "measured",
                        "s": time.perf_counter() - t0,
                        "ok": True,
                    }
                )
                return latent

            t0 = time.perf_counter()
            encoded = original(model, state)
            elapsed = time.perf_counter() - t0
            if self.dump_out is not None:
                from safetensors.torch import save_file

                tensors = {"vae_encode/output": encoded.detach().cpu().contiguous()}
                if self.dump_input:
                    tensors["vae_encode/input"] = state.detach().cpu().contiguous()
                metadata = {
                    "state_signature": self._signature_json(signature),
                    "dump_input": "1" if self.dump_input else "0",
                    "version": "1",
                }
                self.dump_out.parent.mkdir(parents=True, exist_ok=True)
                save_file(tensors, str(self.dump_out), metadata=metadata)
            self.events.append(
                {
                    "name": "vae_encode_boundary_dump",
                    "phase": "measured",
                    "s": elapsed,
                    "ok": True,
                    "wrote_dump": self.dump_out is not None,
                    "dump_input": self.dump_input,
                }
            )
            return encoded

        return encode_with_boundary


def _install_vae_encode_boundary_patch(
    dump_out: Path | None,
    latent_in: Path | None,
    *,
    dump_input: bool = False,
) -> _VAEEncodeBoundary:
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    boundary = _VAEEncodeBoundary(dump_out, latent_in, dump_input=dump_input)
    OmniMoTModel.encode = boundary.wrap_encode(OmniMoTModel.encode)
    return boundary


class _VAEEncodeProfiler:
    def __init__(self, profile_out: Path) -> None:
        self.profile_out = profile_out
        self.events: list[dict[str, Any]] = []
        self.modules: list[dict[str, Any]] = []
        self.encode_calls: list[dict[str, Any]] = []
        self._installed_model_ids: set[int] = set()
        self._handles: list[Any] = []
        self._call_index = 0
        self._active_call: int | None = None
        self._torch: Any = None

    @staticmethod
    def _tensor_sig(value: Any) -> Any:
        try:
            import torch
        except Exception:
            torch = None

        if torch is not None and isinstance(value, torch.Tensor):
            return {
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(value.dtype),
                "device": str(value.device),
                "stride": [int(dim) for dim in value.stride()],
                "contiguous": bool(value.is_contiguous()),
            }
        if isinstance(value, (list, tuple)):
            return [self_item for self_item in (_VAEEncodeProfiler._tensor_sig(item) for item in value)]
        if isinstance(value, dict):
            return {str(key): _VAEEncodeProfiler._tensor_sig(item) for key, item in value.items()}
        return None

    @staticmethod
    def _module_inventory(name: str, module: Any) -> dict[str, Any]:
        params = []
        for param_name, param in module.named_parameters(recurse=False):
            params.append(
                {
                    "name": param_name,
                    "shape": [int(dim) for dim in param.shape],
                    "dtype": str(param.dtype),
                    "numel": int(param.numel()),
                }
            )
        attrs: dict[str, Any] = {}
        for attr in (
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "in_dim",
            "out_dim",
            "dim",
            "mode",
            "factor_t",
            "factor_s",
            "group_size",
        ):
            if hasattr(module, attr):
                value = getattr(module, attr)
                if isinstance(value, tuple):
                    value = [int(item) if isinstance(item, int) else item for item in value]
                elif isinstance(value, (int, float, str, bool)):
                    pass
                else:
                    value = repr(value)
                attrs[attr] = value
        return {
            "name": name,
            "class": module.__class__.__name__,
            "children": len(list(module.children())),
            "parameters": params,
            "parameter_numel": sum(int(item["numel"]) for item in params),
            "attrs": attrs,
        }

    @staticmethod
    def _is_timed_leaf(module: Any) -> bool:
        if len(list(module.children())) != 0:
            return False
        return module.__class__.__name__ in {
            "CausalConv3d",
            "RMS_norm",
            "SiLU",
            "Dropout",
            "Identity",
            "AttentionBlock",
            "AvgDown3D",
        } or module.__class__.__module__.startswith("torch.nn.modules")

    def _sync(self) -> None:
        if self._torch is None:
            try:
                import torch

                self._torch = torch
            except Exception:
                self._torch = False
        torch = self._torch
        if (
            torch
            and torch.cuda.is_available()
            and torch.cuda.is_initialized()
        ):
            torch.cuda.synchronize()

    def _install_for_model(self, model: Any) -> None:
        model_id = id(model)
        if model_id in self._installed_model_ids:
            return
        self._installed_model_ids.add(model_id)
        self.modules = []
        for name, module in model.named_modules():
            if name and not (name == "conv1" or name.startswith("encoder")):
                continue
            self.modules.append(self._module_inventory(name or "<wan_vae>", module))
            if not self._is_timed_leaf(module):
                continue

            label = name or "<wan_vae>"
            cls_name = module.__class__.__name__
            param_numel = sum(int(param.numel()) for param in module.parameters(recurse=False))

            def pre_hook(_module: Any, inputs: Any, *, _label: str = label, _cls: str = cls_name):
                if self._active_call is None:
                    return
                self._sync()
                setattr(
                    _module,
                    "_flashrt_vae_profile_pre",
                    {
                        "t": time.perf_counter(),
                        "input": self._tensor_sig(inputs),
                        "call": self._active_call,
                        "label": _label,
                        "class": _cls,
                    },
                )

            def post_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                *,
                _param_numel: int = param_numel,
            ):
                meta = getattr(_module, "_flashrt_vae_profile_pre", None)
                if not isinstance(meta, dict):
                    return
                self._sync()
                self.events.append(
                    {
                        "encode_call": int(meta["call"]),
                        "name": meta["label"],
                        "class": meta["class"],
                        "input": meta["input"],
                        "output": self._tensor_sig(output),
                        "s": time.perf_counter() - float(meta["t"]),
                        "parameter_numel": int(_param_numel),
                    }
                )
                setattr(_module, "_flashrt_vae_profile_pre", None)

            self._handles.append(module.register_forward_pre_hook(pre_hook))
            self._handles.append(module.register_forward_hook(post_hook))

    def wrap_encode(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def encode_with_profile(model: Any, x: Any, scale: Any) -> Any:
            self._install_for_model(model)
            call_index = self._call_index
            self._call_index += 1
            previous = self._active_call
            self._active_call = call_index
            input_sig = self._tensor_sig(x)
            self._sync()
            t0 = time.perf_counter()
            try:
                out = original(model, x, scale)
            finally:
                self._sync()
                elapsed = time.perf_counter() - t0
                self._active_call = previous
            self.encode_calls.append(
                {
                    "encode_call": call_index,
                    "input": input_sig,
                    "output": self._tensor_sig(out),
                    "s": elapsed,
                }
            )
            return out

        return encode_with_profile

    def save(self) -> None:
        by_name: dict[str, dict[str, Any]] = {}
        for event in self.events:
            key = str(event["name"])
            item = by_name.setdefault(
                key,
                {
                    "name": event["name"],
                    "class": event["class"],
                    "count": 0,
                    "total_s": 0.0,
                    "max_s": 0.0,
                    "parameter_numel": event.get("parameter_numel", 0),
                    "input": event.get("input"),
                    "output": event.get("output"),
                },
            )
            seconds = float(event["s"])
            item["count"] += 1
            item["total_s"] += seconds
            item["max_s"] = max(float(item["max_s"]), seconds)
            item["avg_s"] = float(item["total_s"]) / max(1, int(item["count"]))
        summary = sorted(by_name.values(), key=lambda item: -float(item["total_s"]))

        def shape_of(value: Any) -> list[int] | None:
            if isinstance(value, dict) and isinstance(value.get("shape"), list):
                return [int(dim) for dim in value["shape"]]
            return None

        def input_shape_at(event: dict[str, Any], idx: int) -> list[int] | None:
            inputs = event.get("input")
            if not isinstance(inputs, list) or idx >= len(inputs):
                return None
            return shape_of(inputs[idx])

        by_shape: dict[tuple[Any, ...], dict[str, Any]] = {}
        candidate_by_shape: dict[tuple[Any, ...], dict[str, Any]] = {}
        for event in self.events:
            input0_shape = input_shape_at(event, 0)
            cache_shape = input_shape_at(event, 1)
            output_shape = shape_of(event.get("output"))
            key = (
                event.get("class"),
                event.get("name"),
                tuple(input0_shape or ()),
                tuple(cache_shape or ()),
                tuple(output_shape or ()),
            )
            item = by_shape.setdefault(
                key,
                {
                    "name": event.get("name"),
                    "class": event.get("class"),
                    "count": 0,
                    "total_s": 0.0,
                    "max_s": 0.0,
                    "parameter_numel": int(event.get("parameter_numel", 0)),
                    "input_shape": input0_shape,
                    "cache_shape": cache_shape,
                    "output_shape": output_shape,
                },
            )
            seconds = float(event["s"])
            item["count"] += 1
            item["total_s"] += seconds
            item["max_s"] = max(float(item["max_s"]), seconds)
            item["avg_s"] = float(item["total_s"]) / max(1, int(item["count"]))

            if event.get("class") != "CausalConv3d":
                continue
            t_new = input0_shape[2] if input0_shape and len(input0_shape) == 5 else None
            if cache_shape and t_new and t_new > 1:
                candidate_type = "steady_cached_causal_conv3d"
            elif cache_shape is None and t_new == 1:
                candidate_type = "prime_t1_no_cache"
            else:
                candidate_type = "other_causal_conv3d"
            candidate_key = (
                candidate_type,
                event.get("name"),
                tuple(input0_shape or ()),
                tuple(cache_shape or ()),
                tuple(output_shape or ()),
            )
            candidate = candidate_by_shape.setdefault(
                candidate_key,
                {
                    "candidate_type": candidate_type,
                    "name": event.get("name"),
                    "count": 0,
                    "total_s": 0.0,
                    "max_s": 0.0,
                    "parameter_numel": int(event.get("parameter_numel", 0)),
                    "input_shape": input0_shape,
                    "cache_shape": cache_shape,
                    "output_shape": output_shape,
                },
            )
            candidate["count"] += 1
            candidate["total_s"] += seconds
            candidate["max_s"] = max(float(candidate["max_s"]), seconds)
            candidate["avg_s"] = float(candidate["total_s"]) / max(1, int(candidate["count"]))

        shape_summary = sorted(by_shape.values(), key=lambda item: -float(item["total_s"]))
        native_candidate_summary = sorted(
            candidate_by_shape.values(),
            key=lambda item: (
                str(item.get("candidate_type")) != "steady_cached_causal_conv3d",
                -float(item["total_s"]),
            ),
        )
        module_class_counts: dict[str, int] = {}
        for module in self.modules:
            cls = str(module["class"])
            module_class_counts[cls] = module_class_counts.get(cls, 0) + 1
        self.profile_out.parent.mkdir(parents=True, exist_ok=True)
        self.profile_out.write_text(
            json.dumps(
                {
                    "version": 1,
                    "module_class_counts": module_class_counts,
                    "encode_calls": self.encode_calls,
                    "summary": summary,
                    "shape_summary": shape_summary,
                    "native_candidate_summary": native_candidate_summary,
                    "modules": self.modules,
                    "events": self.events,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _install_vae_encode_profile_patch(profile_out: Path) -> _VAEEncodeProfiler:
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import WanVAE_

    profiler = _VAEEncodeProfiler(profile_out)
    WanVAE_.encode = profiler.wrap_encode(WanVAE_.encode)
    return profiler


def _install_warmup_vae_cache_patch() -> _WarmupVAECache:
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    cache = _WarmupVAECache()
    OmniInference.generate_batch = cache.wrap_generate_batch(OmniInference.generate_batch)
    OmniMoTModel.encode = cache.wrap_encode(OmniMoTModel.encode)
    return cache


def _install_warmup_prepare_cache_patch(
    *,
    slim_no_raw_state_vision: bool = False,
    slim_derive_condition_reference: bool = False,
    slim_derive_initial_noise: bool = False,
) -> _WarmupPrepareCache:
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    cache = _WarmupPrepareCache(
        slim_no_raw_state_vision=slim_no_raw_state_vision,
        slim_derive_condition_reference=slim_derive_condition_reference,
        slim_derive_initial_noise=slim_derive_initial_noise,
    )
    OmniInference.generate_batch = cache.wrap_generate_batch(OmniInference.generate_batch)
    OmniMoTModel._prepare_inference_data = cache.wrap_prepare(OmniMoTModel._prepare_inference_data)
    return cache


class _UpstreamTrace:
    def __init__(
        self,
        trace_out: Path,
        *,
        vae_cache: _WarmupVAECache | None = None,
        prepare_cache: _WarmupPrepareCache | None = None,
        vae_boundary: _VAEEncodeBoundary | None = None,
        prepare_boundary: _PrepareBoundary | None = None,
    ):
        self.trace_out = trace_out
        self.phase = "setup"
        self.vae_cache = vae_cache
        self.prepare_cache = prepare_cache
        self.vae_boundary = vae_boundary
        self.prepare_boundary = prepare_boundary
        self.events: list[dict[str, Any]] = []
        try:
            import torch
        except Exception:
            torch = None
        self.torch = torch

    def _sync(self) -> None:
        if (
            self.torch is not None
            and self.torch.cuda.is_available()
            and self.torch.cuda.is_initialized()
        ):
            self.torch.cuda.synchronize()

    @contextmanager
    def record(
        self,
        name: str,
        *,
        phase: str | None = None,
        metadata: dict[str, Any] | None = None,
        expected_exceptions: tuple[str, ...] = (),
    ):
        self._sync()
        t0 = time.perf_counter()
        ok = False
        event_metadata = dict(metadata) if metadata else None
        try:
            yield
            ok = True
        except Exception as exc:
            if exc.__class__.__name__ in expected_exceptions:
                ok = True
                if event_metadata is None:
                    event_metadata = {}
                event_metadata["expected_exception"] = exc.__class__.__name__
            raise
        finally:
            self._sync()
            event = {
                "name": name,
                "phase": phase if phase is not None else self.phase,
                "s": time.perf_counter() - t0,
                "ok": ok,
            }
            if event_metadata:
                event["metadata"] = event_metadata
            self.events.append(event)

    def save(self) -> None:
        events = list(self.events)
        if self.vae_cache is not None:
            events.extend(self.vae_cache.events)
        if self.prepare_cache is not None:
            events.extend(self.prepare_cache.events)
        if self.vae_boundary is not None:
            events.extend(self.vae_boundary.events)
        if self.prepare_boundary is not None:
            events.extend(self.prepare_boundary.events)
        by_key: dict[str, dict[str, Any]] = {}
        for event in events:
            key = f"{event['phase']}::{event['name']}"
            item = by_key.setdefault(
                key,
                {
                    "name": event["name"],
                    "phase": event["phase"],
                    "count": 0,
                    "total_s": 0.0,
                    "max_s": 0.0,
                    "ok_count": 0,
                },
            )
            seconds = float(event["s"])
            item["count"] += 1
            item["total_s"] += seconds
            item["max_s"] = max(float(item["max_s"]), seconds)
            if event.get("ok"):
                item["ok_count"] += 1
        summary = []
        for item in by_key.values():
            item["avg_s"] = float(item["total_s"]) / max(1, int(item["count"]))
            summary.append(item)
        summary.sort(key=lambda item: (str(item["phase"]), -float(item["total_s"]), str(item["name"])))
        self.trace_out.parent.mkdir(parents=True, exist_ok=True)
        self.trace_out.write_text(
            json.dumps({"summary": summary, "events": events}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _install_upstream_trace_patch(
    trace_out: Path,
    *,
    vae_cache: _WarmupVAECache | None = None,
    prepare_cache: _WarmupPrepareCache | None = None,
    vae_boundary: _VAEEncodeBoundary | None = None,
    prepare_boundary: _PrepareBoundary | None = None,
) -> _UpstreamTrace:
    import cosmos_framework.inference.inference as inference_mod
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
    from cosmos_framework.model.generator.mot.unified_mot import (
        Nemotron3DenseVLTextForCausalLM,
        Nemotron3DenseVLTextModel,
        Qwen3VLTextForCausalLM,
        Qwen3VLTextModel,
        Qwen3VLMoeTextForCausalLM,
        Qwen3VLMoeTextModel,
    )

    trace = _UpstreamTrace(
        trace_out,
        vae_cache=vae_cache,
        prepare_cache=prepare_cache,
        vae_boundary=vae_boundary,
        prepare_boundary=prepare_boundary,
    )
    orig_create_batches = OmniInference.create_batches
    orig_generate_batch = OmniInference.generate_batch
    orig_finalize_data_batch = inference_mod._finalize_data_batch
    orig_prepare_inference_data = OmniMoTModel._prepare_inference_data
    orig_get_data_and_condition = OmniMoTModel.get_data_and_condition
    orig_get_inference_text_tokens = OmniMoTModel._get_inference_text_tokens
    orig_pack_input_sequence = OmniMoTModel._pack_input_sequence
    orig_encode = OmniMoTModel.encode
    orig_get_velocity = OmniMoTModel._get_velocity
    orig_vfm_forward = Cosmos3VFMNetwork.forward
    text_model_classes = (Nemotron3DenseVLTextModel, Qwen3VLTextModel, Qwen3VLMoeTextModel)
    lm_classes = (Nemotron3DenseVLTextForCausalLM, Qwen3VLTextForCausalLM, Qwen3VLMoeTextForCausalLM)
    orig_text_forwards = {cls: cls.forward for cls in (*text_model_classes, *lm_classes)}

    def create_batches_traced(self: Any, sample_args_list: Any):
        idx = 0
        iterator = orig_create_batches(self, sample_args_list)
        while True:
            trace._sync()
            t0 = time.perf_counter()
            try:
                item = next(iterator)
            except StopIteration:
                trace._sync()
                return
            trace._sync()
            trace.events.append(
                {
                    "name": "OmniInference.create_batches.next",
                    "phase": "input_prep",
                    "s": time.perf_counter() - t0,
                    "ok": True,
                    "metadata": {"batch_index": idx},
                }
            )
            idx += 1
            yield item

    def generate_batch_traced(self: Any, sample_args_list: Any, data_batch: dict[str, Any], *, warmup: bool = False):
        previous = trace.phase
        trace.phase = "warmup" if warmup else "measured"
        try:
            with trace.record("OmniInference.generate_batch"):
                return orig_generate_batch(self, sample_args_list, data_batch, warmup=warmup)
        finally:
            trace.phase = previous

    def finalize_data_batch_traced(*args: Any, **kwargs: Any):
        with trace.record("inference._finalize_data_batch"):
            return orig_finalize_data_batch(*args, **kwargs)

    def prepare_inference_data_traced(self: Any, *args: Any, **kwargs: Any):
        with trace.record("OmniMoTModel._prepare_inference_data"):
            return orig_prepare_inference_data(self, *args, **kwargs)

    def get_data_and_condition_traced(self: Any, *args: Any, **kwargs: Any):
        with trace.record("OmniMoTModel.get_data_and_condition"):
            return orig_get_data_and_condition(self, *args, **kwargs)

    def get_inference_text_tokens_traced(self: Any, *args: Any, **kwargs: Any):
        with trace.record("OmniMoTModel._get_inference_text_tokens"):
            return orig_get_inference_text_tokens(self, *args, **kwargs)

    def pack_input_sequence_traced(self: Any, *args: Any, **kwargs: Any):
        metadata = {"skip_text_tokens": bool(kwargs.get("skip_text_tokens", False))}
        with trace.record("OmniMoTModel._pack_input_sequence", metadata=metadata):
            return orig_pack_input_sequence(self, *args, **kwargs)

    def encode_traced(self: Any, state: Any):
        metadata = {"shape": list(state.shape)} if hasattr(state, "shape") else None
        with trace.record("OmniMoTModel.encode", metadata=metadata):
            return orig_encode(self, state)

    def get_velocity_traced(self: Any, *args: Any, **kwargs: Any):
        with trace.record("OmniMoTModel._get_velocity", expected_exceptions=("_FlashRTVelocityReady",)):
            return orig_get_velocity(self, *args, **kwargs)

    def vfm_forward_traced(self: Any, *args: Any, **kwargs: Any):
        with trace.record("Cosmos3VFMNetwork.forward", expected_exceptions=("_FlashRTVelocityReady",)):
            return orig_vfm_forward(self, *args, **kwargs)

    def make_text_forward_traced(cls: type[Any]) -> Callable[..., Any]:
        orig = orig_text_forwards[cls]

        def text_forward_traced(self: Any, *args: Any, **kwargs: Any):
            with trace.record(f"{cls.__name__}.forward", expected_exceptions=("_FlashRTVelocityReady",)):
                return orig(self, *args, **kwargs)

        return text_forward_traced

    OmniInference.create_batches = create_batches_traced
    OmniInference.generate_batch = generate_batch_traced
    inference_mod._finalize_data_batch = finalize_data_batch_traced
    OmniMoTModel._prepare_inference_data = prepare_inference_data_traced
    OmniMoTModel.get_data_and_condition = get_data_and_condition_traced
    OmniMoTModel._get_inference_text_tokens = get_inference_text_tokens_traced
    OmniMoTModel._pack_input_sequence = pack_input_sequence_traced
    OmniMoTModel.encode = encode_traced
    OmniMoTModel._get_velocity = get_velocity_traced
    Cosmos3VFMNetwork.forward = vfm_forward_traced
    for cls in (*text_model_classes, *lm_classes):
        cls.forward = make_text_forward_traced(cls)
    return trace


def _dump_tensor(tensor: Any, *, flatten: bool = False) -> Any:
    out = tensor.detach().cpu().contiguous()
    if flatten:
        out = out.reshape(-1).contiguous()
    return out


def _normalise_final_action(action: Any) -> Any:
    out = _dump_tensor(action)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out.squeeze(0).contiguous()
    return out


def _install_live_dump_patch(live_dump_out: Path) -> None:
    import torch
    from safetensors.torch import save_file

    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    original = OmniMoTModel.generate_samples_from_batch

    def wrapped_generate_samples_from_batch(self: Any, *args: Any, **kwargs: Any) -> dict[str, list[Any]]:
        steps: list[dict[str, Any]] = []
        user_builder: Callable[..., Any] | None = kwargs.get("velocity_postprocess_builder")

        def velocity_postprocess_builder(**builder_kwargs: Any) -> Callable[[list[Any], list[Any], Any], list[Any]]:
            user_postprocess = user_builder(**builder_kwargs) if user_builder is not None else None

            def velocity_postprocess(cond_v_full: list[Any], noise_x: list[Any], timestep: Any) -> list[Any]:
                post_v = (
                    user_postprocess(cond_v_full, noise_x, timestep)
                    if user_postprocess is not None
                    else cond_v_full
                )
                if len(noise_x) == 1 and len(post_v) == 1:
                    steps.append(
                        {
                            "noise_x": _dump_tensor(noise_x[0], flatten=True),
                            "velocity": _dump_tensor(post_v[0], flatten=True),
                            "timestep": _dump_tensor(timestep),
                        }
                    )
                return post_v

            return velocity_postprocess

        kwargs["velocity_postprocess_builder"] = velocity_postprocess_builder
        outputs = original(self, *args, **kwargs)

        tensors: dict[str, torch.Tensor] = {
            "once/num_velocity_calls": torch.tensor([len(steps)], dtype=torch.int64)
        }
        for idx, step in enumerate(steps):
            tensors[f"steps/{idx:02d}/noise_x"] = step["noise_x"]
            tensors[f"steps/{idx:02d}/velocity"] = step["velocity"]
            tensors[f"steps/{idx:02d}/timestep"] = step["timestep"]

        if not outputs.get("vision"):
            raise RuntimeError("live dump capture did not receive a final vision latent")
        if not outputs.get("action"):
            raise RuntimeError("live dump capture did not receive a final action tensor")
        tensors["once/final_vision"] = _dump_tensor(outputs["vision"][0])
        tensors["once/final_action"] = _normalise_final_action(outputs["action"][0])

        live_dump_out.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(live_dump_out))
        return outputs

    OmniMoTModel.generate_samples_from_batch = wrapped_generate_samples_from_batch


class _FlashRTVelocityReady(RuntimeError):
    def __init__(self, velocity: list[Any]):
        super().__init__("FlashRT live pre-layer handoff velocity is ready")
        self.velocity = velocity


class _LiveFlashRTHandoff:
    def __init__(
        self,
        checkpoint: Path,
        boundary_out: Path | None,
        boundary_in: Path | None,
        boundary_prepare_in: Path | None,
        boundary_prepare_live: _PrepareBoundary | None,
        trace_out: Path | None,
        *,
        prelayer_bootstrap: bool = False,
    ):
        import torch

        self.checkpoint = checkpoint
        self.boundary_out = boundary_out
        self.boundary_in = boundary_in
        self.boundary_prepare_in = boundary_prepare_in
        self.boundary_prepare_live = boundary_prepare_live
        self.trace_out = trace_out
        self.prelayer_bootstrap = prelayer_bootstrap
        self.torch = torch
        self.tensors: dict[str, torch.Tensor] = {}
        self.active_step: int | None = None
        self.step = 0
        self.engine: Any = None
        self.current_noise_x: list[Any] | None = None
        self.current_timestep: Any | None = None
        self.current_scheduler_seed: Any | None = None
        self.current_skip_text_tokens = False
        self._engine_prepare_signature: tuple[Any, ...] | None = None
        self._captured_vfm = False
        self._captured_lm = False
        self._captured_layer0 = False
        self.trace: dict[str, Any] = {
            "official_velocity_calls": [],
            "flashrt_velocity_calls": [],
            "engine_init": {},
            "native_scheduler": {
                "enabled": os.environ.get("FLASHRT_COSMOS3_EDGE_LIVE_NATIVE_UNIPC", "1") != "0",
                "runs": [],
                "fallbacks": [],
                "failures": [],
            },
            "prelayer_bootstrap": prelayer_bootstrap,
            "boundary_in": str(boundary_in) if boundary_in is not None else None,
            "boundary_prepare_in": str(boundary_prepare_in) if boundary_prepare_in is not None else None,
            "boundary_prepare_live": boundary_prepare_live is not None,
        }

    def _sync_for_trace(self) -> None:
        if (
            self.trace_out is not None
            and self.torch.cuda.is_available()
            and self.torch.cuda.is_initialized()
        ):
            self.torch.cuda.synchronize()

    def _now_for_trace(self) -> float:
        self._sync_for_trace()
        return time.perf_counter()

    def _cpu(self, value: Any) -> Any:
        return value.detach().cpu().contiguous()

    def _add_tensor(self, name: str, value: Any) -> None:
        if isinstance(value, self.torch.Tensor):
            self.tensors[name] = self._cpu(value)

    def _add_tensor_list(self, prefix: str, values: Any) -> None:
        if isinstance(values, list):
            for idx, value in enumerate(values):
                self._add_tensor(f"{prefix}/{idx}", value)

    def _capture_modality(self, prefix: str, modality: Any) -> None:
        if modality is None:
            return
        for field in ("sequence_indexes", "timesteps", "mse_loss_indexes"):
            self._add_tensor(f"{prefix}/{field}", getattr(modality, field, None))
        for field in ("tokens", "condition_mask", "noisy_frame_indexes", "domain_id", "raw_action_dim"):
            self._add_tensor_list(f"{prefix}/{field}", getattr(modality, field, None))

    def capture_vfm_boundary(self, packed_seq: Any) -> None:
        if self.active_step != 0 or self._captured_vfm:
            return
        self._captured_vfm = True
        self._capture_modality("s00/vfm_in/vision", getattr(packed_seq, "vision", None))
        self._capture_modality("s00/vfm_in/action", getattr(packed_seq, "action", None))

    def capture_lm_boundary(self, pack: Any, _attention_mask: Any, position_ids: Any) -> None:
        if self.active_step != 0 or self._captured_lm:
            return
        self._captured_lm = True
        if isinstance(pack, dict):
            for key, value in pack.items():
                self._add_tensor(f"s00/lm_in/{key}", value)
        self._add_tensor("s00/lm_in/position_ids", position_ids)

    def capture_layer0(self, module: Any, input_pack: Any, args: tuple[Any, ...], output: Any) -> None:
        if self.active_step != 0 or self._captured_layer0:
            return
        layer_idx = getattr(getattr(module, "self_attn", None), "layer_idx", None)
        if layer_idx != 0:
            return
        self._captured_layer0 = True
        if isinstance(input_pack, dict):
            for key in ("causal_seq", "full_only_seq", "_causal_indices", "_full_indices"):
                self._add_tensor(f"s00/layers/00/input/{key}", input_pack.get(key))
        packed_position_embeddings = args[1] if len(args) > 1 else None
        if (
            isinstance(packed_position_embeddings, tuple)
            and len(packed_position_embeddings) == 2
            and all(isinstance(item, dict) for item in packed_position_embeddings)
        ):
            for name, pack in (("cos", packed_position_embeddings[0]), ("sin", packed_position_embeddings[1])):
                for key in ("causal_seq", "full_only_seq"):
                    self._add_tensor(f"s00/layers/00/rope/{name}/{key}", pack.get(key))
        hidden = output[0] if isinstance(output, tuple) and output else None
        if isinstance(hidden, dict):
            for key in ("causal_seq", "full_only_seq"):
                self._add_tensor(f"s00/layers/00/output/{key}", hidden.get(key))

    def _initialise_engine(self, noise_x: list[Any], timestep: Any, velocity: list[Any] | None = None) -> None:
        if len(noise_x) != 1 or (velocity is not None and len(velocity) != 1):
            raise RuntimeError("FlashRT live handoff currently supports batch size 1")
        from safetensors.torch import save_file

        from flash_rt.models.cosmos3_edge import EdgeBoundaryDump, EdgeStaticBufferEngine, EdgeTransformerWeights

        t0 = self._now_for_trace()
        self.tensors["steps/00/noise_x"] = self._cpu(noise_x[0])
        if velocity is not None:
            self.tensors["steps/00/velocity"] = self._cpu(velocity[0])
        self.tensors["steps/00/timestep"] = self._cpu(timestep)
        self.tensors.setdefault("once/num_velocity_calls", self.torch.tensor([1], dtype=self.torch.int64))
        t_assembled = self._now_for_trace()
        boundary_path: Path | str = "<live>"
        if self.boundary_out is not None:
            self.boundary_out.parent.mkdir(parents=True, exist_ok=True)
            save_file(self.tensors, str(self.boundary_out))
            boundary_path = self.boundary_out
        t_saved = self._now_for_trace()
        boundary = EdgeBoundaryDump.from_tensors(self.tensors, path=boundary_path)
        boundary.validate_geometry()
        t_validated = self._now_for_trace()
        self.engine = EdgeStaticBufferEngine(
            EdgeTransformerWeights(str(self.checkpoint)),
            boundary,
            device=noise_x[0].device,
        )
        t_done = self._now_for_trace()
        self.trace["engine_init"] = {
            "mode": "captured_live",
            "boundary_assembly_s": t_assembled - t0,
            "boundary_save_s": t_saved - t_assembled,
            "boundary_validate_s": t_validated - t_saved,
            "engine_construct_s": t_done - t_validated,
            "total_s": t_done - t0,
            "wrote_boundary": self.boundary_out is not None,
        }

    def _initialise_engine_from_boundary_in(self, device: Any) -> None:
        if self.boundary_in is None:
            raise RuntimeError("missing FlashRT live boundary input")

        from flash_rt.models.cosmos3_edge import EdgeBoundaryDump, EdgeStaticBufferEngine, EdgeTransformerWeights

        t0 = self._now_for_trace()
        boundary = EdgeBoundaryDump(self.boundary_in)
        t_loaded = self._now_for_trace()
        boundary.validate_geometry()
        t_validated = self._now_for_trace()
        self.engine = EdgeStaticBufferEngine(
            EdgeTransformerWeights(str(self.checkpoint)),
            boundary,
            device=device,
        )
        self._engine_prepare_signature = self.boundary_prepare_live.latest_signature
        t_done = self._now_for_trace()
        self.trace["engine_init"] = {
            "mode": "boundary_in",
            "boundary_in": str(self.boundary_in),
            "boundary_load_s": t_loaded - t0,
            "boundary_validate_s": t_validated - t_loaded,
            "engine_construct_s": t_done - t_validated,
            "total_s": t_done - t0,
            "wrote_boundary": False,
        }

    def _initialise_engine_from_prepare_in(self, device: Any) -> None:
        if self.boundary_prepare_in is None:
            raise RuntimeError("missing FlashRT live prepare-boundary input")
        from safetensors.torch import save_file

        from flash_rt.models.cosmos3_edge import EdgeBoundaryDump, EdgeStaticBufferEngine, EdgeTransformerWeights

        t0 = self._now_for_trace()
        tensors = _derive_step0_executable_boundary_from_prepare_artifact(
            self.boundary_prepare_in,
            self.checkpoint,
            seed=self.current_scheduler_seed,
            device=device,
        )
        t_derived = self._now_for_trace()
        boundary_path: Path | str = f"{self.boundary_prepare_in}:derived"
        if self.boundary_out is not None:
            self.boundary_out.parent.mkdir(parents=True, exist_ok=True)
            save_file({key: tensor.clone() for key, tensor in tensors.items()}, str(self.boundary_out))
            boundary_path = self.boundary_out
        t_saved = self._now_for_trace()
        boundary = EdgeBoundaryDump.from_tensors(tensors, path=boundary_path)
        boundary.validate_geometry()
        t_validated = self._now_for_trace()
        self.engine = EdgeStaticBufferEngine(
            EdgeTransformerWeights(str(self.checkpoint)),
            boundary,
            device=device,
        )
        t_done = self._now_for_trace()
        self.trace["engine_init"] = {
            "mode": "prepare_in",
            "boundary_prepare_in": str(self.boundary_prepare_in),
            "prepare_artifact_mib": self.boundary_prepare_in.stat().st_size / (1024.0 * 1024.0),
            "boundary_derive_s": t_derived - t0,
            "boundary_save_s": t_saved - t_derived,
            "boundary_validate_s": t_validated - t_saved,
            "engine_construct_s": t_done - t_validated,
            "total_s": t_done - t0,
            "wrote_boundary": self.boundary_out is not None,
        }

    def _initialise_engine_from_prepare_live(self, device: Any) -> None:
        if self.boundary_prepare_live is None:
            raise RuntimeError("missing FlashRT live prepare-boundary source")
        from safetensors.torch import save_file

        from flash_rt.models.cosmos3_edge import EdgeBoundaryDump, EdgeStaticBufferEngine, EdgeTransformerWeights

        t0 = self._now_for_trace()
        payload = self.boundary_prepare_live.latest_effective_payload()
        seed = self.current_scheduler_seed
        if seed is None:
            seed = self.boundary_prepare_live.latest_seed
        if seed is None:
            raise RuntimeError("live prepare-derived boundary requires the sampler seed")
        tensors = _derive_step0_executable_boundary_from_prepare_payload(
            payload,
            self.checkpoint,
            seed=seed,
            device=device,
            path="live_prepare_result:derived",
        )
        t_derived = self._now_for_trace()
        boundary_path: Path | str = "live_prepare_result:derived"
        if self.boundary_out is not None:
            self.boundary_out.parent.mkdir(parents=True, exist_ok=True)
            save_file({key: tensor.clone() for key, tensor in tensors.items()}, str(self.boundary_out))
            boundary_path = self.boundary_out
        t_saved = self._now_for_trace()
        boundary = EdgeBoundaryDump.from_tensors(tensors, path=boundary_path)
        boundary.validate_geometry()
        t_validated = self._now_for_trace()
        self.engine = EdgeStaticBufferEngine(
            EdgeTransformerWeights(str(self.checkpoint)),
            boundary,
            device=device,
        )
        t_done = self._now_for_trace()
        self.trace["engine_init"] = {
            "mode": "prepare_live",
            "prepare_source": self.boundary_prepare_live.latest_source,
            "prepare_slim_no_raw_state_vision": self.boundary_prepare_live._last_effective_slim_no_raw_state_vision,
            "prepare_slim_derive_condition_reference": self.boundary_prepare_live._last_effective_slim_derive_condition_reference,
            "prepare_slim_derive_initial_noise": self.boundary_prepare_live._last_effective_slim_derive_initial_noise,
            "boundary_derive_s": t_derived - t0,
            "boundary_save_s": t_saved - t_derived,
            "boundary_validate_s": t_validated - t_saved,
            "engine_construct_s": t_done - t_validated,
            "total_s": t_done - t0,
            "wrote_boundary": self.boundary_out is not None,
        }

    def should_capture_pre_layers(self) -> bool:
        return (
            self.prelayer_bootstrap
            and self.engine is None
            and self.active_step == 0
            and not self.current_skip_text_tokens
            and self.current_noise_x is not None
            and self.current_timestep is not None
        )

    def capture_pre_layers_and_raise(
        self,
        text_model: Any,
        pack: Any,
        _attention_mask: Any,
        position_ids: Any,
    ) -> None:
        if not self.should_capture_pre_layers():
            return
        from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq, get_device_and_dtype

        if isinstance(pack, dict):
            for key, value in pack.items():
                self._add_tensor(f"s00/lm_in/{key}", value)
                if key in {"causal_seq", "full_only_seq"}:
                    self._add_tensor(f"s00/layers/00/input/{key}", value)
        self._add_tensor("s00/lm_in/position_ids", position_ids)

        device, dtype = get_device_and_dtype(pack)
        meta = self.torch.tensor([], dtype=dtype, device=device)
        cos, sin = text_model.rotary_emb(
            meta,
            position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1),
        )
        cos_pack = from_all_seq(cos.squeeze(0), pack)
        sin_pack = from_all_seq(sin.squeeze(0), pack)
        if isinstance(cos_pack, dict) and isinstance(sin_pack, dict):
            for key in ("causal_seq", "full_only_seq"):
                self._add_tensor(f"s00/layers/00/rope/cos/{key}", cos_pack.get(key))
                self._add_tensor(f"s00/layers/00/rope/sin/{key}", sin_pack.get(key))

        assert self.current_noise_x is not None
        assert self.current_timestep is not None
        self._initialise_engine(self.current_noise_x, self.current_timestep)
        assert self.engine is not None
        t0 = self._now_for_trace()
        velocity = self.engine.flat_velocity_for_step(self.current_noise_x[0], self.current_timestep)
        t1 = self._now_for_trace()
        self.tensors["steps/00/velocity"] = self._cpu(velocity)
        self.trace["flashrt_velocity_calls"].append({"step": self.step, "s": t1 - t0, "bootstrap": "prelayer"})
        raise _FlashRTVelocityReady([velocity])

    def get_velocity(self, original: Callable[..., Any], model: Any, *args: Any, **kwargs: Any) -> list[Any]:
        noise_x = kwargs.get("noise_x")
        timestep = kwargs.get("timestep")
        skip_text_tokens = bool(kwargs.get("skip_text_tokens", False))
        if self.engine is None and self.boundary_in is not None and not skip_text_tokens:
            if not isinstance(noise_x, list) or len(noise_x) != 1:
                raise RuntimeError("FlashRT live boundary input currently supports batch size 1")
            self._initialise_engine_from_boundary_in(noise_x[0].device)
        if self.engine is None and self.boundary_prepare_in is not None and not skip_text_tokens:
            if not isinstance(noise_x, list) or len(noise_x) != 1:
                raise RuntimeError("FlashRT live prepare-boundary input currently supports batch size 1")
            self._initialise_engine_from_prepare_in(noise_x[0].device)
        if self.engine is None and self.boundary_prepare_live is not None and not skip_text_tokens:
            if not isinstance(noise_x, list) or len(noise_x) != 1:
                raise RuntimeError("FlashRT live prepare-boundary source currently supports batch size 1")
            self._initialise_engine_from_prepare_live(noise_x[0].device)
        if self.engine is not None and not skip_text_tokens:
            if not isinstance(noise_x, list) or len(noise_x) != 1:
                raise RuntimeError("FlashRT live handoff currently supports batch size 1")
            t0 = self._now_for_trace()
            velocity = self.engine.flat_velocity_for_step(noise_x[0], timestep)
            t1 = self._now_for_trace()
            self.trace["flashrt_velocity_calls"].append({"step": self.step, "s": t1 - t0})
            self.step += 1
            return [velocity]

        self.active_step = self.step
        self.current_noise_x = noise_x if isinstance(noise_x, list) else None
        self.current_timestep = timestep
        self.current_skip_text_tokens = skip_text_tokens
        t0 = self._now_for_trace()
        try:
            velocity = original(model, *args, **kwargs)
        except _FlashRTVelocityReady as ready:
            t1 = self._now_for_trace()
            self.trace["official_velocity_calls"].append(
                {
                    "step": self.step,
                    "s": t1 - t0,
                    "skip_text_tokens": skip_text_tokens,
                    "prelayer_aborted": True,
                }
            )
            self.step += 1
            return ready.velocity
        finally:
            self.active_step = None
            self.current_noise_x = None
            self.current_timestep = None
            self.current_skip_text_tokens = False
        t1 = self._now_for_trace()
        self.trace["official_velocity_calls"].append(
            {"step": self.step, "s": t1 - t0, "skip_text_tokens": skip_text_tokens}
        )
        if self.step == 0 and not skip_text_tokens:
            if not isinstance(noise_x, list) or timestep is None:
                raise RuntimeError("FlashRT live handoff could not see step-0 noise/timestep")
            self._initialise_engine(noise_x, timestep, velocity)
        self.step += 1
        return velocity

    def _record_native_scheduler_fallback(self, reason: str) -> None:
        native = self.trace.get("native_scheduler")
        if isinstance(native, dict):
            native.setdefault("fallbacks", []).append({"reason": reason})

    def _record_native_scheduler_failure(self, reason: str) -> None:
        native = self.trace.get("native_scheduler")
        if isinstance(native, dict):
            native.setdefault("failures", []).append({"reason": reason})

    @staticmethod
    def _single_latent(noise: Any) -> tuple[Any | None, bool, str | None]:
        if isinstance(noise, list):
            if len(noise) != 1:
                return None, True, "noise_list_len_not_1"
            return noise[0], True, None
        return noise, False, None

    def try_native_unipc_forward(
        self,
        velocity_fn: Callable[..., Any],
        noise: Any,
        *,
        num_steps: int,
        shift: float | None,
        seed: Any,
    ) -> tuple[bool, Any]:
        self.current_scheduler_seed = seed
        native_trace = self.trace.get("native_scheduler")
        if not isinstance(native_trace, dict) or not native_trace.get("enabled", False):
            self._record_native_scheduler_fallback("disabled")
            return False, None
        if seed is not None and isinstance(seed, list) and len(seed) != 1:
            self._record_native_scheduler_fallback("seed_list_len_not_1")
            return False, None
        if shift is None:
            self._record_native_scheduler_fallback("missing_shift")
            return False, None

        from flash_rt.models.cosmos3_edge.dump_replay import EDGE_FLAT_DIM, EDGE_NUM_STEPS

        if int(num_steps) != EDGE_NUM_STEPS:
            self._record_native_scheduler_fallback(f"num_steps_{num_steps}")
            return False, None

        latent_in, return_list, reason = self._single_latent(noise)
        if reason is not None:
            self._record_native_scheduler_fallback(reason)
            return False, None
        if not isinstance(latent_in, self.torch.Tensor):
            self._record_native_scheduler_fallback("noise_not_tensor")
            return False, None
        if latent_in.device.type != "cuda":
            self._record_native_scheduler_fallback("noise_not_cuda")
            return False, None
        if latent_in.dtype != self.torch.float32:
            self._record_native_scheduler_fallback(f"noise_dtype_{latent_in.dtype}")
            return False, None
        if int(latent_in.numel()) != EDGE_FLAT_DIM:
            self._record_native_scheduler_fallback(f"noise_numel_{int(latent_in.numel())}")
            return False, None
        if (
            self.boundary_prepare_live is not None
            and self.engine is not None
            and self.boundary_prepare_live.latest_signature != self._engine_prepare_signature
        ):
            self.engine = None
            self._engine_prepare_signature = None

        from cosmos_framework.utils.progress_bar import progress_bar
        from flash_rt.models.cosmos3_edge.static_unipc import EdgeStaticUniPCScheduler

        scheduler = EdgeStaticUniPCScheduler(int(num_steps), device=latent_in.device, shift=float(shift))
        if not scheduler.native_available:
            self._record_native_scheduler_fallback("native_binding_unavailable")
            return False, None

        latent = latent_in.clone()
        scheduler.reset(latent)
        step_total_s = 0.0
        velocity_dtype: str | None = None
        t_run0 = self._now_for_trace()
        try:
            for step_index, timestep in enumerate(
                progress_bar(scheduler.timesteps, desc="Sampling", total=len(scheduler.timesteps))
            ):
                velocity_pred = velocity_fn([latent] if return_list else latent, timestep.reshape(1, 1))
                velocity = velocity_pred[0] if return_list and isinstance(velocity_pred, list) else velocity_pred
                if not isinstance(velocity, self.torch.Tensor):
                    raise TypeError(f"native UniPC expected tensor velocity, got {type(velocity).__name__}")
                if velocity_dtype is None:
                    velocity_dtype = str(velocity.dtype)
                if velocity.dtype != self.torch.bfloat16:
                    raise TypeError(f"native UniPC expected bf16 velocity, got {velocity.dtype}")
                if tuple(velocity.shape) != tuple(latent.shape):
                    raise ValueError(f"native UniPC velocity shape {tuple(velocity.shape)} != latent {tuple(latent.shape)}")
                t_step0 = self._now_for_trace()
                scheduler.step(latent, velocity, step_index)
                t_step1 = self._now_for_trace()
                step_total_s += t_step1 - t_step0
        except Exception as exc:
            self._record_native_scheduler_failure(type(exc).__name__)
            raise
        t_run1 = self._now_for_trace()
        native_trace.setdefault("runs", []).append(
            {
                "num_steps": int(num_steps),
                "shift": float(shift),
                "total_s": t_run1 - t_run0,
                "scheduler_step_total_s": step_total_s,
                "scheduler_step_avg_s": step_total_s / int(num_steps),
                "latent_dtype": str(latent_in.dtype),
                "velocity_dtype": velocity_dtype,
                "return_list": return_list,
            }
        )
        return True, [latent] if return_list else latent

    def save_trace(self) -> None:
        if self.trace_out is None:
            return
        flashrt_calls = self.trace["flashrt_velocity_calls"]
        official_calls = self.trace["official_velocity_calls"]
        flashrt_total = sum(float(item["s"]) for item in flashrt_calls)
        official_total = sum(float(item["s"]) for item in official_calls)
        native_scheduler = self.trace.get("native_scheduler", {})
        native_runs = native_scheduler.get("runs", []) if isinstance(native_scheduler, dict) else []
        native_step_total = sum(float(item.get("scheduler_step_total_s", 0.0)) for item in native_runs)
        summary = {
            "flashrt_velocity_call_count": len(flashrt_calls),
            "flashrt_velocity_total_s": flashrt_total,
            "flashrt_velocity_avg_s": flashrt_total / len(flashrt_calls) if flashrt_calls else 0.0,
            "official_velocity_call_count": len(official_calls),
            "official_velocity_total_s": official_total,
            "native_scheduler_enabled": bool(native_scheduler.get("enabled")) if isinstance(native_scheduler, dict) else False,
            "native_scheduler_run_count": len(native_runs),
            "native_scheduler_step_count": sum(int(item.get("num_steps", 0)) for item in native_runs),
            "native_scheduler_step_total_s": native_step_total,
        }
        self.trace_out.parent.mkdir(parents=True, exist_ok=True)
        self.trace_out.write_text(
            json.dumps({"summary": summary, **self.trace}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _install_live_flashrt_handoff(
    checkpoint: Path,
    boundary_out: Path | None,
    boundary_in: Path | None,
    boundary_prepare_in: Path | None,
    boundary_prepare_live: _PrepareBoundary | None,
    trace_out: Path | None,
    *,
    prelayer_bootstrap: bool = False,
) -> _LiveFlashRTHandoff:
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
    from cosmos_framework.model.generator.mot.unified_mot import (
        MoTDecoderLayer,
        Nemotron3DenseVLTextForCausalLM,
        Qwen3VLTextForCausalLM,
        Qwen3VLTextModel,
        Qwen3VLMoeTextForCausalLM,
        Qwen3VLMoeTextModel,
        Nemotron3DenseVLTextModel,
    )

    handoff = _LiveFlashRTHandoff(
        checkpoint,
        boundary_out,
        boundary_in,
        boundary_prepare_in,
        boundary_prepare_live,
        trace_out,
        prelayer_bootstrap=prelayer_bootstrap,
    )
    orig_unipc_forward = UniPCSampler.forward
    orig_get_velocity = OmniMoTModel._get_velocity
    orig_vfm_forward = Cosmos3VFMNetwork.forward
    orig_layer_forward = MoTDecoderLayer.forward
    lm_classes = (Nemotron3DenseVLTextForCausalLM, Qwen3VLTextForCausalLM, Qwen3VLMoeTextForCausalLM)
    orig_lm_forwards = {cls: cls.forward for cls in lm_classes}
    text_model_classes = (Nemotron3DenseVLTextModel, Qwen3VLTextModel, Qwen3VLMoeTextModel)
    orig_text_model_forwards = {cls: cls.forward for cls in text_model_classes}

    def _get_velocity_hook(self: Any, *args: Any, **kwargs: Any) -> list[Any]:
        return handoff.get_velocity(orig_get_velocity, self, *args, **kwargs)

    def _unipc_forward_hook(
        self: Any,
        velocity_fn: Callable[..., Any],
        noise: Any,
        num_steps: int = 35,
        shift: float | None = None,
        seed: Any = None,
    ) -> Any:
        handled, output = handoff.try_native_unipc_forward(
            velocity_fn,
            noise,
            num_steps=num_steps,
            shift=self.cfg.shift if shift is None else shift,
            seed=seed,
        )
        if handled:
            return output
        return orig_unipc_forward(
            self,
            velocity_fn,
            noise,
            num_steps=num_steps,
            shift=shift,
            seed=seed,
        )

    def _vfm_forward_hook(self: Any, packed_seq: Any, *args: Any, **kwargs: Any) -> Any:
        handoff.capture_vfm_boundary(packed_seq)
        return orig_vfm_forward(self, packed_seq, *args, **kwargs)

    def _layer_forward_hook(self: Any, input_pack: Any, *args: Any, **kwargs: Any) -> Any:
        output = orig_layer_forward(self, input_pack, *args, **kwargs)
        handoff.capture_layer0(self, input_pack, args, output)
        return output

    def _make_lm_forward_hook(cls: type[Any]) -> Callable[..., Any]:
        orig = orig_lm_forwards[cls]

        def _lm_forward_hook(self: Any, pack: Any, attention_mask: Any, position_ids: Any, *args: Any, **kwargs: Any) -> Any:
            handoff.capture_lm_boundary(pack, attention_mask, position_ids)
            return orig(self, pack, attention_mask, position_ids, *args, **kwargs)

        return _lm_forward_hook

    def _make_text_model_forward_hook(cls: type[Any]) -> Callable[..., Any]:
        orig = orig_text_model_forwards[cls]

        def _text_model_forward_hook(
            self: Any,
            pack: Any,
            attention_mask: Any,
            position_ids: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            handoff.capture_pre_layers_and_raise(self, pack, attention_mask, position_ids)
            return orig(self, pack, attention_mask, position_ids, *args, **kwargs)

        return _text_model_forward_hook

    UniPCSampler.forward = _unipc_forward_hook
    OmniMoTModel._get_velocity = _get_velocity_hook
    Cosmos3VFMNetwork.forward = _vfm_forward_hook
    MoTDecoderLayer.forward = _layer_forward_hook
    for cls in lm_classes:
        cls.forward = _make_lm_forward_hook(cls)
    for cls in text_model_classes:
        cls.forward = _make_text_model_forward_hook(cls)
    return handoff


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv, live_dump_out = _extract_live_dump_out(argv)
    argv, upstream_trace_out = _extract_upstream_trace_out(argv)
    argv, cache_warmup_vae = _extract_warmup_vae_cache(argv)
    argv, cache_warmup_prepare = _extract_warmup_prepare_cache(argv)
    argv, vae_encode_dump_out, vae_latent_in, vae_encode_dump_input = _extract_vae_encode_boundary_args(argv)
    argv, vae_encode_profile_out = _extract_vae_encode_profile_out(argv)
    argv, vae_native_rms_silu = _extract_vae_native_rms_silu(argv)
    argv, vae_t1_conv2d = _extract_vae_t1_conv2d(argv)
    argv, vae_native_avgdown3d = _extract_vae_native_avgdown3d(argv)
    argv, vae_channels_last3d_conv320 = _extract_vae_channels_last3d_conv320(argv)
    argv, vae_compile_encode, vae_compile_trace_out = _extract_vae_compile_encode(argv)
    (
        argv,
        prepare_dump_out,
        prepare_replay_in,
        prepare_inventory_out,
        prepare_slim_no_raw_state_vision,
        prepare_slim_derive_condition_reference,
        prepare_slim_derive_initial_noise,
    ) = _extract_prepare_boundary_args(argv)
    (
        argv,
        live_handoff,
        live_boundary_out,
        live_boundary_in,
        live_boundary_prepare_in,
        live_boundary_prepare_live,
        live_handoff_trace_out,
        live_prelayer_bootstrap,
    ) = _extract_live_handoff_args(argv)
    if live_dump_out is not None and live_handoff:
        raise SystemExit("--flashrt-live-dump-out cannot be combined with --flashrt-live-flashrt-handoff")
    if live_boundary_in is not None and live_boundary_prepare_in is not None:
        raise SystemExit("--flashrt-live-boundary-in cannot be combined with --flashrt-live-boundary-prepare-in")
    if live_boundary_prepare_live and (live_boundary_in is not None or live_boundary_prepare_in is not None):
        raise SystemExit(
            "--flashrt-live-boundary-prepare-live cannot be combined with live boundary input artifacts"
        )
    if cache_warmup_vae and (vae_encode_dump_out is not None or vae_latent_in is not None):
        raise SystemExit("--flashrt-cache-warmup-vae cannot be combined with VAE encode boundary dump/replay")
    if cache_warmup_prepare and (
        prepare_dump_out is not None
        or prepare_replay_in is not None
        or prepare_inventory_out is not None
    ):
        raise SystemExit(
            "--flashrt-cache-warmup-prepare cannot be combined with prepare boundary dump/replay/inventory"
        )
    if vae_compile_encode and (vae_native_rms_silu or vae_native_avgdown3d):
        raise SystemExit(
            "--flashrt-vae-compile-encode cannot be combined with native VAE monkeypatches"
        )
    output_dir = _extract_output_dir(argv)
    _install_action_only_patches()
    if live_dump_out is not None:
        _install_live_dump_patch(live_dump_out)
    upstream_trace: _UpstreamTrace | None = None
    handoff: _LiveFlashRTHandoff | None = None
    vae_profiler: _VAEEncodeProfiler | None = None
    vae_compile_patch: _VAECompileEncodePatch | None = None
    if vae_compile_encode:
        vae_compile_patch = _install_vae_compile_encode_patch(vae_compile_trace_out)
    if vae_t1_conv2d:
        from flash_rt.models.cosmos3_edge.vae_native import install_wan_vae_encode_t1_conv2d

        install_wan_vae_encode_t1_conv2d()
    if vae_native_avgdown3d:
        from flash_rt.models.cosmos3_edge.vae_native import install_wan_vae_encode_native_avgdown3d

        install_wan_vae_encode_native_avgdown3d()
    if vae_channels_last3d_conv320:
        from flash_rt.models.cosmos3_edge.vae_native import install_wan_vae_encode_channels_last3d_conv320

        install_wan_vae_encode_channels_last3d_conv320()
    if vae_native_rms_silu:
        from flash_rt.models.cosmos3_edge.vae_native import install_wan_vae_encode_native_rms_silu

        install_wan_vae_encode_native_rms_silu()
    if vae_encode_profile_out is not None:
        vae_profiler = _install_vae_encode_profile_patch(vae_encode_profile_out)
    if live_handoff:
        _install_warmup_clone_patch()
    vae_cache: _WarmupVAECache | None = None
    if cache_warmup_vae:
        vae_cache = _install_warmup_vae_cache_patch()
    prepare_cache: _WarmupPrepareCache | None = None
    if cache_warmup_prepare:
        prepare_cache = _install_warmup_prepare_cache_patch(
            slim_no_raw_state_vision=prepare_slim_no_raw_state_vision,
            slim_derive_condition_reference=prepare_slim_derive_condition_reference,
            slim_derive_initial_noise=prepare_slim_derive_initial_noise,
        )
    prepare_boundary: _PrepareBoundary | None = None
    if (
        prepare_dump_out is not None
        or prepare_replay_in is not None
        or prepare_inventory_out is not None
        or prepare_slim_no_raw_state_vision
        or prepare_slim_derive_condition_reference
        or prepare_slim_derive_initial_noise
        or live_boundary_prepare_live
    ):
        prepare_boundary = _install_prepare_boundary_patch(
            prepare_dump_out,
            prepare_replay_in,
            prepare_inventory_out,
            slim_no_raw_state_vision=prepare_slim_no_raw_state_vision,
            slim_derive_condition_reference=prepare_slim_derive_condition_reference,
            slim_derive_initial_noise=prepare_slim_derive_initial_noise,
        )
    vae_boundary: _VAEEncodeBoundary | None = None
    if vae_encode_dump_out is not None or vae_latent_in is not None:
        vae_boundary = _install_vae_encode_boundary_patch(
            vae_encode_dump_out,
            vae_latent_in,
            dump_input=vae_encode_dump_input,
        )
    if upstream_trace_out is not None:
        upstream_trace = _install_upstream_trace_patch(
            upstream_trace_out,
            vae_cache=vae_cache,
            prepare_cache=prepare_cache,
            vae_boundary=vae_boundary,
            prepare_boundary=prepare_boundary,
        )
    if live_handoff:
        handoff = _install_live_flashrt_handoff(
            _extract_checkpoint_path(argv),
            live_boundary_out,
            live_boundary_in,
            live_boundary_prepare_in,
            prepare_boundary if live_boundary_prepare_live else None,
            live_handoff_trace_out,
            prelayer_bootstrap=live_prelayer_bootstrap,
        )

    from cosmos_framework.scripts.inference import main as official_main

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        official_main()
    finally:
        sys.argv = old_argv

    rewritten = _rewrite_sample_outputs_action_only(output_dir)
    if rewritten == 0:
        raise SystemExit(f"no action outputs were rewritten under {output_dir}")
    if handoff is not None:
        handoff.save_trace()
    if upstream_trace is not None:
        upstream_trace.save()
    if vae_compile_patch is not None:
        vae_compile_patch.save()
    if vae_profiler is not None:
        vae_profiler.save()


if __name__ == "__main__":
    main()

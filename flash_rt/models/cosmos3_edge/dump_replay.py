"""Replay utilities for Cosmos3-Edge AV denoise dumps.

This module is a P1 scaffold: it validates the denoise boundary captured from
Cosmos Framework before the native FlashRT transformer forward is connected.
It deliberately reuses the local self-contained UniPC scheduler so the next
step can replace recorded velocities with FlashRT-computed velocities without
changing scheduler or latent splitting code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file

from flash_rt.models.cosmos3_video.fm_solvers_unipc import FlowUniPCMultistepScheduler


EDGE_VISION_SHAPE = (1, 48, 16, 30, 52)
EDGE_ACTION_MODEL_SHAPE = (60, 64)
EDGE_FLAT_DIM = 1_201_920
EDGE_NUM_STEPS = 30
EDGE_SHIFT = 10.0
EDGE_NUM_TRAIN_TIMESTEPS = 1000


@dataclass(frozen=True)
class EdgeLatentParts:
    vision: torch.Tensor
    action_model: torch.Tensor


@dataclass(frozen=True)
class ReplayResult:
    final_flat: torch.Tensor
    final_parts: EdgeLatentParts
    timesteps: tuple[int, ...]
    max_input_abs_diff: float
    per_step_input_abs_diff: tuple[float, ...]


class EdgeDenoiseDump:
    """Loaded P0 denoise boundary dump for one AV inverse-dynamics sample."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.tensors = load_file(str(self.path), device="cpu")
        self._validate_required_tensors()

    def _validate_required_tensors(self) -> None:
        required = {"once/final_action", "once/final_vision", "once/num_velocity_calls"}
        for step in range(EDGE_NUM_STEPS):
            required.add(f"steps/{step:02d}/noise_x")
            required.add(f"steps/{step:02d}/velocity")
            required.add(f"steps/{step:02d}/timestep")
        missing = sorted(required.difference(self.tensors))
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(f"{self.path} is missing required tensors: {preview}")

    @property
    def num_steps(self) -> int:
        return int(self.tensors["once/num_velocity_calls"].reshape(-1)[0].item())

    @property
    def final_action(self) -> torch.Tensor:
        return self.tensors["once/final_action"]

    @property
    def final_vision(self) -> torch.Tensor:
        return self.tensors["once/final_vision"]

    @property
    def flat_dim(self) -> int:
        return int(self.step_noise(0).numel())

    @property
    def vision_dim(self) -> int:
        return int(torch.tensor(self.final_vision.shape).prod().item())

    @property
    def action_model_dim(self) -> int:
        return self.flat_dim - self.vision_dim

    def step_noise(self, step: int) -> torch.Tensor:
        return self.tensors[f"steps/{step:02d}/noise_x"]

    def step_velocity(self, step: int) -> torch.Tensor:
        return self.tensors[f"steps/{step:02d}/velocity"]

    def step_timestep(self, step: int) -> torch.Tensor:
        return self.tensors[f"steps/{step:02d}/timestep"]

    def timesteps(self) -> tuple[int, ...]:
        return tuple(int(self.step_timestep(i).reshape(-1)[0].item()) for i in range(self.num_steps))

    def validate_geometry(self) -> None:
        if self.num_steps != EDGE_NUM_STEPS:
            raise ValueError(f"expected {EDGE_NUM_STEPS} steps, got {self.num_steps}")
        if tuple(self.final_vision.shape) != EDGE_VISION_SHAPE:
            raise ValueError(f"unexpected final_vision shape: {tuple(self.final_vision.shape)}")
        if tuple(self.final_action.shape) != (60, 9):
            raise ValueError(f"unexpected final_action shape: {tuple(self.final_action.shape)}")
        if self.flat_dim != EDGE_FLAT_DIM:
            raise ValueError(f"unexpected flat latent dim: {self.flat_dim}")
        if self.action_model_dim != EDGE_ACTION_MODEL_SHAPE[0] * EDGE_ACTION_MODEL_SHAPE[1]:
            raise ValueError(f"unexpected action-model latent dim: {self.action_model_dim}")
        for step in range(self.num_steps):
            if tuple(self.step_noise(step).shape) != (EDGE_FLAT_DIM,):
                raise ValueError(f"unexpected noise_x shape at step {step}")
            if tuple(self.step_velocity(step).shape) != (EDGE_FLAT_DIM,):
                raise ValueError(f"unexpected velocity shape at step {step}")

    def split_flat(self, flat: torch.Tensor) -> EdgeLatentParts:
        if flat.numel() != EDGE_FLAT_DIM:
            raise ValueError(f"expected flat latent dim {EDGE_FLAT_DIM}, got {flat.numel()}")
        flat = flat.reshape(-1)
        vision = flat[: self.vision_dim].reshape(self.final_vision.shape)
        action = flat[self.vision_dim :].reshape(EDGE_ACTION_MODEL_SHAPE)
        return EdgeLatentParts(vision=vision, action_model=action)


class EdgeDenoiseReplay:
    """Replay a captured Edge denoise run with recorded velocity tensors."""

    def __init__(
        self,
        dump: EdgeDenoiseDump,
        *,
        device: str | torch.device = "cpu",
        shift: float = EDGE_SHIFT,
    ):
        self.dump = dump
        self.device = torch.device(device)
        self.shift = float(shift)
        self.dump.validate_geometry()

    def _scheduler(self) -> FlowUniPCMultistepScheduler:
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=EDGE_NUM_TRAIN_TIMESTEPS,
            shift=1.0,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(self.dump.num_steps, device=self.device, shift=self.shift)
        expected = tuple(int(t.item()) for t in scheduler.timesteps.cpu())
        actual = self.dump.timesteps()
        if expected != actual:
            raise ValueError(f"dump timesteps {actual} do not match UniPC schedule {expected}")
        return scheduler

    @staticmethod
    def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.float() - b.float()).abs().max().item())

    def replay(self, *, check_inputs: bool = True) -> ReplayResult:
        scheduler = self._scheduler()
        latent = self.dump.step_noise(0).to(device=self.device)
        input_diffs: list[float] = []

        for step, timestep in enumerate(scheduler.timesteps):
            if check_inputs:
                expected_noise = self.dump.step_noise(step).to(device=self.device)
                input_diffs.append(self._max_abs_diff(latent, expected_noise))
            velocity = self.dump.step_velocity(step).to(device=self.device)
            latent = scheduler.step(
                model_output=velocity,
                timestep=timestep,
                sample=latent.unsqueeze(0),
                return_dict=False,
            )[0].squeeze(0)

        final_flat = latent.detach().cpu().contiguous()
        parts = self.dump.split_flat(final_flat)
        max_diff = max(input_diffs) if input_diffs else 0.0
        return ReplayResult(
            final_flat=final_flat,
            final_parts=parts,
            timesteps=self.dump.timesteps(),
            max_input_abs_diff=max_diff,
            per_step_input_abs_diff=tuple(input_diffs),
        )


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1).float()
    b_f = b.reshape(-1).float()
    return float(torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item())


def max_abs(values: Iterable[float]) -> float:
    values = tuple(values)
    return max(values) if values else 0.0

"""Boundary dump loader for Cosmos3-Edge native forward bring-up."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file


EDGE_BOUNDARY_UND_TOKENS = 125
EDGE_BOUNDARY_GEN_TOKENS = 6300
EDGE_BOUNDARY_VISION_TOKENS = 6240
EDGE_BOUNDARY_ACTION_TOKENS = 60
EDGE_HIDDEN_SIZE = 2048
EDGE_HEAD_DIM = 128
EDGE_PATCH_SPATIAL = 2


@dataclass(frozen=True)
class EdgeBoundaryShapes:
    und_tokens: int
    gen_tokens: int
    vision_tokens: int
    action_tokens: int
    hidden_size: int
    head_dim: int


class EdgeBoundaryDump:
    """Step-0 transformer boundary captured from the official Edge pipeline."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.tensors = load_file(str(self.path), device="cpu")
        self._validate_required_tensors()

    @classmethod
    def from_tensors(cls, tensors: dict[str, torch.Tensor], *, path: str | Path = "<memory>") -> "EdgeBoundaryDump":
        obj = cls.__new__(cls)
        obj.path = Path(path)
        obj.tensors = tensors
        obj._validate_required_tensors()
        return obj

    def _validate_required_tensors(self) -> None:
        required = {
            "s00/lm_in/causal_seq",
            "s00/lm_in/full_only_seq",
            "s00/lm_in/position_ids",
            "s00/layers/00/rope/cos/causal_seq",
            "s00/layers/00/rope/cos/full_only_seq",
            "s00/layers/00/rope/sin/causal_seq",
            "s00/layers/00/rope/sin/full_only_seq",
            "s00/layers/00/input/causal_seq",
            "s00/layers/00/input/full_only_seq",
            "s00/vfm_in/vision/tokens/0",
            "s00/vfm_in/action/tokens/0",
            "s00/vfm_in/action/domain_id/0",
            "s00/vfm_in/action/raw_action_dim/0",
        }
        missing = sorted(required.difference(self.tensors))
        if missing:
            raise ValueError(f"{self.path} is missing boundary tensors: {missing[:8]}")

    @property
    def shapes(self) -> EdgeBoundaryShapes:
        return EdgeBoundaryShapes(
            und_tokens=int(self.causal_seq.shape[0]),
            gen_tokens=int(self.full_only_seq.shape[0]),
            vision_tokens=int(
                self.vision_tokens.shape[2]
                * (self.vision_tokens.shape[3] // EDGE_PATCH_SPATIAL)
                * (self.vision_tokens.shape[4] // EDGE_PATCH_SPATIAL)
            ),
            action_tokens=int(self.action_tokens.shape[0]),
            hidden_size=int(self.causal_seq.shape[1]),
            head_dim=int(self.rope_cos_causal.shape[1]),
        )

    @property
    def causal_seq(self) -> torch.Tensor:
        return self.tensors["s00/lm_in/causal_seq"]

    @property
    def full_only_seq(self) -> torch.Tensor:
        return self.tensors["s00/lm_in/full_only_seq"]

    @property
    def position_ids(self) -> torch.Tensor:
        return self.tensors["s00/lm_in/position_ids"]

    @property
    def rope_cos_causal(self) -> torch.Tensor:
        return self.tensors["s00/layers/00/rope/cos/causal_seq"]

    @property
    def rope_cos_full(self) -> torch.Tensor:
        return self.tensors["s00/layers/00/rope/cos/full_only_seq"]

    @property
    def rope_sin_causal(self) -> torch.Tensor:
        return self.tensors["s00/layers/00/rope/sin/causal_seq"]

    @property
    def rope_sin_full(self) -> torch.Tensor:
        return self.tensors["s00/layers/00/rope/sin/full_only_seq"]

    @property
    def layer0_input_causal(self) -> torch.Tensor:
        return self.tensors["s00/layers/00/input/causal_seq"]

    @property
    def layer0_input_full(self) -> torch.Tensor:
        return self.tensors["s00/layers/00/input/full_only_seq"]

    @property
    def layer0_output_causal(self) -> torch.Tensor:
        return self.tensors["s00/layers/00/output/causal_seq"]

    @property
    def layer0_output_full(self) -> torch.Tensor:
        return self.tensors["s00/layers/00/output/full_only_seq"]

    @property
    def vision_tokens(self) -> torch.Tensor:
        return self.tensors["s00/vfm_in/vision/tokens/0"]

    @property
    def action_tokens(self) -> torch.Tensor:
        return self.tensors["s00/vfm_in/action/tokens/0"]

    def validate_geometry(self) -> None:
        shapes = self.shapes
        expected = EdgeBoundaryShapes(
            und_tokens=EDGE_BOUNDARY_UND_TOKENS,
            gen_tokens=EDGE_BOUNDARY_GEN_TOKENS,
            vision_tokens=EDGE_BOUNDARY_VISION_TOKENS,
            action_tokens=EDGE_BOUNDARY_ACTION_TOKENS,
            hidden_size=EDGE_HIDDEN_SIZE,
            head_dim=EDGE_HEAD_DIM,
        )
        if shapes != expected:
            raise ValueError(f"unexpected Edge boundary geometry: expected {expected}, got {shapes}")
        if self.layer0_input_causal.shape != self.causal_seq.shape:
            raise ValueError("layer0 causal input shape does not match lm_in causal_seq")
        if self.layer0_input_full.shape != self.full_only_seq.shape:
            raise ValueError("layer0 full input shape does not match lm_in full_only_seq")
        if self.position_ids.shape != (3, EDGE_BOUNDARY_UND_TOKENS + EDGE_BOUNDARY_GEN_TOKENS):
            raise ValueError(f"unexpected position_ids shape: {tuple(self.position_ids.shape)}")

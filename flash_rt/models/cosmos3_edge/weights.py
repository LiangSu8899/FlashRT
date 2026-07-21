"""Lazy Cosmos3-Edge transformer weight access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from flash_rt.frontends.torch._cosmos3_edge_thor_spec import (
    SPEC,
    iter_expected_shapes,
    layer_key,
    load_transformer_weight_map,
)


@dataclass(frozen=True)
class WeightRef:
    key: str
    shard: Path
    shape: tuple[int, ...]


class EdgeTransformerWeights:
    """Index-backed loader for the Edge diffusion transformer shards.

    The P1/P2 engine needs explicit, stable tensor names without loading the
    entire checkpoint at construction time. This class validates the index once
    and opens only the shard needed for each requested tensor.
    """

    def __init__(self, checkpoint: str | Path):
        self.checkpoint = Path(checkpoint)
        self.transformer_dir = self.checkpoint / "transformer"
        self.weight_map = load_transformer_weight_map(self.checkpoint)
        self.expected_shapes = dict(iter_expected_shapes())
        missing = sorted(set(self.expected_shapes).difference(self.weight_map))
        if missing:
            raise ValueError(f"missing Edge transformer weights: {missing[:8]}")

    def ref(self, key: str) -> WeightRef:
        if key not in self.expected_shapes:
            raise KeyError(f"unknown Cosmos3-Edge transformer weight: {key}")
        shard = self.transformer_dir / self.weight_map[key]
        return WeightRef(key=key, shard=shard, shape=self.expected_shapes[key])

    def tensor_shape(self, key: str) -> tuple[int, ...]:
        ref = self.ref(key)
        with safe_open(str(ref.shard), framework="pt", device="cpu") as f:
            shape = tuple(f.get_slice(key).get_shape())
        if shape != ref.shape:
            raise ValueError(f"{key} shape mismatch: expected {ref.shape}, got {shape}")
        return shape

    def load_tensor(
        self,
        key: str,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        ref = self.ref(key)
        with safe_open(str(ref.shard), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(key)
        if tuple(tensor.shape) != ref.shape:
            raise ValueError(f"{key} shape mismatch: expected {ref.shape}, got {tuple(tensor.shape)}")
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if torch.device(device).type != "cpu":
            tensor = tensor.to(device=device, non_blocking=True)
        return tensor.contiguous()

    def layer_tensor(
        self,
        layer: int,
        suffix: str,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self.load_tensor(layer_key(layer, suffix), device=device, dtype=dtype)

    def validate_indexed_shapes(self) -> None:
        for key in self.expected_shapes:
            self.tensor_shape(key)

    @property
    def num_layers(self) -> int:
        return SPEC.num_layers

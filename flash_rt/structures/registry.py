"""Structure catalog registry — pure lookup, no execution logic.

Loads structure specifications from the on-disk catalog and resolves
their reference implementations. Dispatch, tuning, calibration, and
activation live in separate layers; the registry only answers "what is
structure X and where is its ground truth".
"""

from __future__ import annotations

import importlib
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import yaml

_CATALOG_DIR = pathlib.Path(__file__).resolve().parent / "catalog"


@dataclass(frozen=True)
class StructureSpec:
    """One catalog entry, as declared in ``catalog/<name>/structure.yaml``."""

    name: str
    version: int
    description: str
    boundary: Mapping[str, Any]
    weights: Sequence[Mapping[str, Any]]
    variants: Mapping[str, Sequence[str]]
    calibration: Mapping[str, Any]
    gates: Mapping[str, Any]
    _reference: Mapping[str, str] = field(repr=False)

    @property
    def symbolic_dims(self) -> Sequence[str]:
        return tuple(self.boundary.get("symbolic_dims", ()))

    @property
    def weight_slots(self) -> Sequence[str]:
        return tuple(entry["slot"] for entry in self.weights)

    def reference(self) -> Callable[..., Any]:
        """Resolve the reference entrypoint (ground truth for gates)."""
        module = importlib.import_module(
            f"{__package__}.catalog.{self._reference['module']}"
        )
        return getattr(module, self._reference["entrypoint"])


def list_structures() -> list[str]:
    """Names of all structures present in the catalog."""
    return sorted(
        path.parent.name for path in _CATALOG_DIR.glob("*/structure.yaml")
    )


def load(name: str) -> StructureSpec:
    """Load one structure specification by catalog name."""
    path = _CATALOG_DIR / name / "structure.yaml"
    if not path.is_file():
        raise KeyError(f"unknown structure: {name!r}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    spec = StructureSpec(
        name=data["structure"],
        version=int(data["version"]),
        description=str(data.get("description", "")).strip(),
        boundary=data["boundary"],
        weights=data["weights"],
        variants=data.get("variants", {}),
        calibration=data.get("calibration", {}),
        gates=data["gates"],
        _reference=data["reference"],
    )
    if spec.name != name:
        raise ValueError(
            f"catalog directory {name!r} declares structure {spec.name!r}"
        )
    return spec

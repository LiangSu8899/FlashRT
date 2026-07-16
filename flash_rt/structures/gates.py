"""Qualification gates — parity judgment against a structure's reference.

The harness is structure-agnostic. Implementations follow the structure
calling convention: required boundary inputs in declared order, then
weight tensors in slot order, then variant selections and any optional
boundary inputs as keyword arguments — the same signature the reference
implementation exposes.

A qualification produces a machine-readable record whose ``plan_digest``
binds the spec content, variant, resolved workload dims, thresholds,
implementation identity, and environment. A record certifies exactly one
execution plan; change any component and the record no longer applies.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from flash_rt.structures.registry import StructureSpec, _CATALOG_DIR


@dataclass(frozen=True)
class QualificationCase:
    """One workload to qualify: boundary inputs, weights, and variant."""

    inputs: Mapping[str, torch.Tensor]
    weights: Mapping[str, torch.Tensor]
    variant: Mapping[str, str] = field(default_factory=dict)


def solve_dims(
    spec: StructureSpec,
    inputs: Mapping[str, torch.Tensor],
    weights: Mapping[str, torch.Tensor],
) -> dict[str, int]:
    """Resolve symbolic dims from actual tensors, rejecting inconsistency."""
    dims: dict[str, int] = {}

    def bind(declared: list[str], tensor: torch.Tensor, what: str) -> None:
        if tensor.ndim != len(declared):
            raise ValueError(
                f"{what}: expected rank {len(declared)} {declared}, "
                f"got shape {tuple(tensor.shape)}"
            )
        for name, size in zip(declared, tensor.shape):
            if dims.setdefault(name, int(size)) != int(size):
                raise ValueError(
                    f"{what}: dim {name}={int(size)} conflicts with "
                    f"{name}={dims[name]} resolved earlier"
                )

    for entry in spec.boundary["inputs"]:
        tensor = inputs.get(entry["name"])
        if tensor is None:
            if entry.get("optional", False):
                continue
            raise ValueError(f"missing required input: {entry['name']!r}")
        bind(entry["dims"], tensor, f"input {entry['name']!r}")
    for entry in spec.weights:
        slot = entry["slot"]
        if slot not in weights:
            raise ValueError(f"missing weight slot: {slot!r}")
        bind(entry["dims"], weights[slot], f"weight {slot!r}")
    return dims


def _call(spec: StructureSpec, fn: Callable[..., Any],
          case: QualificationCase, *, bound: bool = False) -> torch.Tensor:
    """Invoke ``fn`` per the structure calling convention.

    ``bound=False`` targets full-signature callables (the reference):
    required inputs, then weight slots, with variants and optional inputs
    as keywords. ``bound=True`` targets bound implementations whose
    weights and variant were baked in at bind time: inputs only.
    """
    args: list[torch.Tensor] = []
    kwargs: dict[str, Any] = {} if bound else dict(case.variant)
    for entry in spec.boundary["inputs"]:
        tensor = case.inputs.get(entry["name"])
        if entry.get("optional", False):
            if tensor is not None:
                kwargs[entry["name"]] = tensor
        else:
            args.append(tensor)
    if not bound:
        args.extend(case.weights[slot] for slot in spec.weight_slots)
    return fn(*args, **kwargs)


def _parity_metrics(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    if got.shape != want.shape:
        raise ValueError(
            f"output shape mismatch: impl {tuple(got.shape)} vs "
            f"reference {tuple(want.shape)}"
        )
    diff = (got.double() - want.double()).abs().flatten()
    cosine = torch.nn.functional.cosine_similarity(
        got.double().flatten(), want.double().flatten(), dim=0
    )
    return {
        "cosine": float(cosine),
        "max_abs": float(diff.max()),
        "p99_abs": float(torch.quantile(diff, 0.99)),
    }


def _spec_digest(spec: StructureSpec) -> str:
    path = _CATALOG_DIR / spec.name / "structure.yaml"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> dict[str, str]:
    env = {"torch": torch.__version__}
    if torch.cuda.is_available():
        env["device"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        env["arch"] = f"sm_{major}{minor}"
    else:
        env["device"] = "cpu"
    return env


def qualify_parity(
    spec: StructureSpec,
    impl: Callable[..., Any],
    case: QualificationCase,
    *,
    impl_id: str,
    thresholds: Mapping[str, float],
    bound: bool = False,
) -> dict[str, Any]:
    """Judge ``impl`` against the structure reference on one workload.

    ``thresholds`` maps metric name to its passing bound (``cosine`` is a
    floor, absolute-error metrics are ceilings). Every thresholded metric
    must pass for a PASS verdict; metrics without thresholds are recorded
    as evidence only. Set ``bound=True`` when ``impl`` was produced by an
    implementation's ``bind`` and takes boundary inputs only.
    """
    for key in case.variant:
        if key not in spec.variants:
            raise ValueError(f"unknown variant key: {key!r}")
    workload = solve_dims(spec, case.inputs, case.weights)
    reference = spec.reference()

    want = _call(spec, reference, case)
    got = _call(spec, impl, case, bound=bound)
    metrics = _parity_metrics(got, want)

    passed = True
    for name, bound in thresholds.items():
        if name not in metrics:
            raise ValueError(f"threshold on unknown metric: {name!r}")
        ok = metrics[name] >= bound if name == "cosine" else metrics[name] <= bound
        passed = passed and ok

    x_dtype = next(iter(case.inputs.values())).dtype
    record = {
        "structure": f"{spec.name}@{spec.version}",
        "spec_digest": _spec_digest(spec),
        "impl": impl_id,
        "variant": dict(case.variant),
        "workload": {**workload, "dtype": str(x_dtype)},
        "env": _environment(),
        "gate": "parity",
        "metrics": metrics,
        "thresholds": dict(thresholds),
        "verdict": "PASS" if passed else "FAIL",
    }
    record["plan_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(
            {k: record[k] for k in
             ("structure", "spec_digest", "impl", "variant", "workload",
              "env", "thresholds")},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return record


def save_record(record: Mapping[str, Any], directory: str | pathlib.Path) -> pathlib.Path:
    """Write one qualification record as JSON, named by its plan digest."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digest = record["plan_digest"].split(":", 1)[1][:16]
    path = directory / f"{record['gate']}_{digest}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path

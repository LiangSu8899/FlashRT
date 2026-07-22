#!/usr/bin/env python
"""Cosmos3-Edge Thor denoise latency + accuracy benchmark.

This benchmark measures the current FlashRT denoise-only path against the P0
official dump. It intentionally does not run Cosmos Framework; pass the
official ``benchmark.json`` only to print the recorded eager baseline next to
the FlashRT numbers.

Run inside a Thor environment with the FlashRT extensions available:

    python benchmarks/cosmos3_edge_thor_denoise.py \
      --checkpoint /path/to/Cosmos3-Edge \
      --reference-dump /path/to/denoise/tensors.safetensors \
      --boundary-dump /path/to/step0_boundary/tensors.safetensors \
      --official-benchmark /path/to/official/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from pathlib import Path
from typing import Any


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile over empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _load_official_benchmark(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"official benchmark does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _official_denoise_s(bench: dict[str, Any] | None) -> float | None:
    return _official_average(bench, "OmniMoTModel.generate_samples_from_batch")


def _official_average(bench: dict[str, Any] | None, key: str) -> float | None:
    if not bench:
        return None
    try:
        return float(bench["average"][key])
    except (KeyError, TypeError, ValueError):
        return None


def _summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "count": float(len(values)),
        "total_s": float(sum(values)),
        "p50_s": float(statistics.median(ordered)),
        "p90_s": float(_percentile(ordered, 0.90)),
        "min_s": float(min(ordered)),
        "max_s": float(max(ordered)),
    }


def _profile_loop_breakdown(runner: Any, *, device: str) -> dict[str, Any]:
    import torch

    if device != "cuda":
        raise RuntimeError("--profile-loop-breakdown requires CUDA")

    latent = runner.denoise_dump.step_noise(0).to(device=runner.device, dtype=torch.float32)
    engine = getattr(runner, "engine", None)
    if engine is not None:
        engine.latent.copy_(latent)
        latent = engine.latent
    if runner.static_scheduler is not None and runner.static_scheduler.prev_m1 is None:
        # Never re-allocate scheduler state that a captured graph already binds.
        runner.static_scheduler.reset(latent)
    timesteps = torch.tensor(runner.denoise_dump.timesteps(), device=runner.device, dtype=torch.int64)

    events: list[tuple[Any, Any, Any]] = []
    for step, timestep in enumerate(timesteps[: runner.denoise_dump.num_steps]):
        start = torch.cuda.Event(enable_timing=True)
        mid = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        velocity = runner.velocity_for_step(latent, timestep.reshape(1, 1), step_index=step)
        mid.record()
        if runner.static_scheduler is None:
            raise RuntimeError("--profile-loop-breakdown requires the native static scheduler")
        latent = runner.static_scheduler.step(latent, velocity, step)
        end.record()
        events.append((start, mid, end))

    torch.cuda.synchronize()
    velocity_s = [start.elapsed_time(mid) / 1000.0 for start, mid, _end in events]
    scheduler_s = [mid.elapsed_time(end) / 1000.0 for _start, mid, end in events]
    step_s = [start.elapsed_time(end) / 1000.0 for start, _mid, end in events]
    return {
        "note": "CUDA event timings inside one profiled run; per-step sync is not inserted.",
        "steps": len(events),
        "velocity": _summarize(velocity_s),
        "scheduler": _summarize(scheduler_s),
        "step_total": _summarize(step_s),
        "native_scheduler": bool(getattr(runner, "native_scheduler_available", False)),
        "use_cuda_graphs": bool(getattr(runner, "use_cuda_graphs", False)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--reference-dump",
        required=True,
    )
    ap.add_argument(
        "--boundary-dump",
        required=True,
    )
    ap.add_argument(
        "--official-benchmark",
        default=None,
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup-steps", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--use-cuda-graphs", action="store_true")
    ap.add_argument(
        "--engine",
        choices=("bf16-eager", "fp8", "quant-bf16"),
        default="bf16-eager",
        help=(
            "bf16-eager: original per-op FlashRT engine. "
            "fp8: quantized static engine (whole-denoise CUDA graph unless --no-quant-graph). "
            "quant-bf16: static engine with all projections in bf16 (quant-path parity check)."
        ),
    )
    ap.add_argument("--no-quant-graph", action="store_true")
    ap.add_argument("--bf16-projs", default="", help="comma-separated projections kept in bf16 (fp8 engine)")
    ap.add_argument("--ffn-fp4", action="store_true", help="NVFP4 W4A4 FFN (up/down) via flash_rt_fp4")
    ap.add_argument("--no-slim-last", action="store_true", help="disable the M=60 slim last layer")
    ap.add_argument(
        "--teacache-computes",
        type=int,
        default=0,
        help="TeaCache: number of computed denoise steps, evenly spaced over the schedule "
        "(always includes step 0 and the final step); 0 disables",
    )
    ap.add_argument(
        "--teacache-steps",
        default="",
        help="TeaCache: explicit comma-separated computed step indices (overrides --teacache-computes)",
    )
    ap.add_argument("--json-out", default=None)
    ap.add_argument(
        "--profile-loop-breakdown",
        action="store_true",
        help="Add CUDA event timings for velocity vs scheduler inside one 30-step run.",
    )
    ap.add_argument("--enforce-gates", action="store_true")
    ap.add_argument("--min-speedup", type=float, default=2.5)
    ap.add_argument("--min-final-action-cos", type=float, default=0.999)
    ap.add_argument("--max-final-action-rel-l2", type=float, default=0.03)
    ap.add_argument(
        "--skip-speedup-gate",
        action="store_true",
        help="Do not fail --enforce-gates on speedup. Useful for additional-seed accuracy checks.",
    )
    args = ap.parse_args()

    if args.iters < 1:
        raise ValueError("--iters must be >= 1")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    import torch

    from flash_rt.hardware.thor import fa4_backend
    from flash_rt.models.cosmos3_edge.boundary_dump import EdgeBoundaryDump
    from flash_rt.models.cosmos3_edge.denoise_ref import EdgeDenoiseFlashRT, EdgeDenoiseFlashRTQuant
    from flash_rt.models.cosmos3_edge.dump_replay import EdgeDenoiseDump
    from flash_rt.models.cosmos3_edge.weights import EdgeTransformerWeights

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is not available")

    checkpoint = Path(args.checkpoint)
    reference_dump = Path(args.reference_dump)
    boundary_dump = Path(args.boundary_dump)
    official_benchmark_path = Path(args.official_benchmark) if args.official_benchmark else None
    for label, path in (
        ("checkpoint", checkpoint),
        ("reference dump", reference_dump),
        ("boundary dump", boundary_dump),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    official_benchmark = _load_official_benchmark(official_benchmark_path)
    official_denoise = _official_denoise_s(official_benchmark)
    official_e2e = _official_average(official_benchmark, "OmniInference.generate_batch")
    official_decode = _official_average(official_benchmark, "OmniMoTModel.decode")

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    teacache_steps: tuple[int, ...] | None = None
    if args.teacache_steps:
        teacache_steps = tuple(int(v) for v in args.teacache_steps.split(",") if v != "")
    elif args.teacache_computes > 1:
        n_steps = 30
        k = args.teacache_computes
        teacache_steps = tuple(sorted({round(i * (n_steps - 1) / (k - 1)) for i in range(k)}))
    if teacache_steps is not None and args.engine == "bf16-eager":
        raise SystemExit("TeaCache requires the quantized static engine (--engine fp8/quant-bf16)")

    t0 = time.perf_counter()
    if args.engine == "bf16-eager":
        runner = EdgeDenoiseFlashRT(
            EdgeDenoiseDump(reference_dump),
            EdgeBoundaryDump(boundary_dump),
            EdgeTransformerWeights(checkpoint),
            device=args.device,
            use_cuda_graphs=args.use_cuda_graphs,
        )
    else:
        runner = EdgeDenoiseFlashRTQuant(
            EdgeDenoiseDump(reference_dump),
            EdgeBoundaryDump(boundary_dump),
            EdgeTransformerWeights(checkpoint),
            device=args.device,
            quant="bf16" if args.engine == "quant-bf16" else "fp8",
            bf16_projs=tuple(p for p in args.bf16_projs.split(",") if p),
            ffn_fp4=args.ffn_fp4,
            slim_last=not args.no_slim_last,
            teacache_steps=teacache_steps,
            use_cuda_graphs=not args.no_quant_graph,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    init_s = time.perf_counter() - t0

    warmup_result = None
    if args.warmup_steps:
        with torch.no_grad():
            warmup_result = runner.run(max_steps=args.warmup_steps)
        if args.device == "cuda":
            torch.cuda.synchronize()

    times: list[float] = []
    last = None
    for _ in range(args.iters):
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            last = runner.run(max_steps=runner.denoise_dump.num_steps)
        if args.device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    assert last is not None
    sorted_times = sorted(times)
    official_action = runner.denoise_dump.final_action.float()
    diff = last.final_action.float() - official_action
    rel_l2 = float((torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(official_action)).item())
    cos = float(
        torch.nn.functional.cosine_similarity(
            last.final_action.flatten(),
            official_action.flatten(),
            dim=0,
        ).item()
    )
    max_abs = float(diff.abs().max().item())
    p50_s = statistics.median(sorted_times)
    official_non_denoise_non_decode = (
        official_e2e - official_denoise - official_decode
        if official_e2e is not None and official_denoise is not None and official_decode is not None
        else None
    )
    estimated_action_only_e2e = (
        official_non_denoise_non_decode + p50_s
        if official_non_denoise_non_decode is not None
        else None
    )
    native_attention = bool(getattr(runner, "native_attention_available", False))
    graph_attention = bool(getattr(runner, "graph_attention_available", False))
    native_scheduler = bool(getattr(runner, "native_scheduler_available", False))
    speedup = official_denoise / p50_s if official_denoise is not None else None
    gate_checks = {
        "native_attention": native_attention,
        "native_scheduler": native_scheduler,
        "speedup_vs_official_denoise_p50": (
            True if args.skip_speedup_gate else speedup is not None and speedup >= args.min_speedup
        ),
        "final_action_cos": cos >= args.min_final_action_cos,
        "final_action_rel_l2": rel_l2 < args.max_final_action_rel_l2,
    }
    result: dict[str, Any] = {
        "backend": "flashrt",
        "engine": args.engine,
        "bf16_projs": args.bf16_projs,
        "ffn_fp4": args.ffn_fp4,
        "quant_graph": bool(args.engine != "bf16-eager" and not args.no_quant_graph),
        "teacache_steps": list(teacache_steps) if teacache_steps is not None else None,
        "teacache_computes": len(teacache_steps) if teacache_steps is not None else 30,
        "checkpoint": str(checkpoint),
        "reference_dump": str(reference_dump),
        "boundary_dump": str(boundary_dump),
        "device": args.device,
        "use_cuda_graphs": bool(args.use_cuda_graphs),
        "fa4_status": fa4_backend.status(),
        "native_attention": native_attention,
        "graph_attention": graph_attention,
        "native_scheduler": native_scheduler,
        "init_s": init_s,
        "warmup_steps": args.warmup_steps,
        "warmup_max_velocity_abs_diff": (
            warmup_result.max_velocity_abs_diff if warmup_result is not None else None
        ),
        "iters": args.iters,
        "times_s": times,
        "p50_s": p50_s,
        "p90_s": _percentile(sorted_times, 0.90),
        "min_s": min(sorted_times),
        "max_s": max(sorted_times),
        "steps": last.steps_run,
        "max_input_abs_diff": last.max_input_abs_diff,
        "max_velocity_abs_diff": last.max_velocity_abs_diff,
        "final_action_cos": cos,
        "final_action_rel_l2": rel_l2,
        "final_action_max_abs": max_abs,
        "official_denoise_s": official_denoise,
        "official_e2e_s": official_e2e,
        "official_decode_s": official_decode,
        "official_non_denoise_non_decode_s": official_non_denoise_non_decode,
        "estimated_action_only_e2e_s": estimated_action_only_e2e,
        "speedup_vs_official_denoise_p50": speedup,
        "estimated_action_only_e2e_speedup": (
            official_e2e / estimated_action_only_e2e
            if official_e2e is not None and estimated_action_only_e2e is not None
            else None
        ),
        "peak_mem_gib": (
            torch.cuda.max_memory_allocated() / (1024**3) if args.device == "cuda" else None
        ),
        "gate_thresholds": {
            "min_speedup": args.min_speedup,
            "min_final_action_cos": args.min_final_action_cos,
            "max_final_action_rel_l2": args.max_final_action_rel_l2,
            "skip_speedup_gate": args.skip_speedup_gate,
        },
        "gate_checks": gate_checks,
        "gates_passed": all(gate_checks.values()),
    }
    if args.profile_loop_breakdown:
        result["loop_breakdown"] = _profile_loop_breakdown(runner, device=args.device)

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.enforce_gates and not result["gates_passed"]:
        failed = ", ".join(name for name, passed in gate_checks.items() if not passed)
        raise SystemExit(f"Cosmos3-Edge FlashRT benchmark gates failed: {failed}")


if __name__ == "__main__":
    main()

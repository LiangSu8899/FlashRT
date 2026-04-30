"""Real-data validation for the lerobot HF norm_stats adapter.

Pulls the policy preprocessor + postprocessor safetensors off Hugging
Face for a published lerobot checkpoint, runs them through
``flash_vla.core.utils.norm_stats.load_norm_stats``, and reports
whether the resulting dict has finite, sane action / state quantiles.

This is **not** a CI test — it talks to the public network and
downloads a few MB of safetensors. Run manually once per release to
confirm the adapter still matches the upstream lerobot schema.

Example::

    python tests/_helpers/validate_lerobot_norm_stats.py
    python tests/_helpers/validate_lerobot_norm_stats.py \\
        --repo lerobot/pi05_libero_finetuned_v044
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import pathlib

import numpy as np

# Default list = HF lerobot releases known to actually ship the
# policy preprocessor / postprocessor safetensors. Other repos in the
# ``lerobot/`` org (e.g. ``lerobot/pi05_libero``) are base / clean
# configs without stats; they're not in scope for this validator.
DEFAULT_REPOS = (
    "lerobot/pi05_libero_finetuned_v044",
)


def _download_to(ckpt_dir: pathlib.Path, repo: str) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(repo)
    wanted = [f for f in files if any(
        s in f for s in (
            "policy_preprocessor", "policy_postprocessor",
            "stats.json", "norm_stats.json",
        ))]
    if not wanted:
        raise RuntimeError(
            f"{repo} carries no recognisable normalisation file "
            f"(saw {files[:5]}...)")
    print(f"  pulling {len(wanted)} stats file(s) from {repo} ...")
    for f in wanted:
        local = hf_hub_download(repo, f)
        target = ckpt_dir / f
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(local)


def _check_stats(stats: dict) -> tuple[bool, list[str]]:
    """Sanity-check the openpi-shaped stats dict the loader returned."""
    findings: list[str] = []
    ok = True

    if not isinstance(stats, dict):
        return False, [f"stats is not a dict (got {type(stats).__name__})"]

    if "actions" not in stats:
        ok = False
        findings.append("missing 'actions' block")
    else:
        a = stats["actions"]
        for key in ("q01", "q99"):
            if key not in a:
                ok = False
                findings.append(f"actions.{key} missing")
                continue
            arr = np.asarray(a[key], dtype=np.float64)
            if arr.size == 0:
                ok = False
                findings.append(f"actions.{key} empty")
                continue
            if not np.isfinite(arr).all():
                ok = False
                findings.append(
                    f"actions.{key} has non-finite entries: {arr.tolist()}")
        if "q01" in a and "q99" in a:
            q01 = np.asarray(a["q01"], dtype=np.float64)
            q99 = np.asarray(a["q99"], dtype=np.float64)
            if q01.shape != q99.shape:
                ok = False
                findings.append(
                    f"actions q01 / q99 shape mismatch: {q01.shape} vs {q99.shape}")
            elif (q01 > q99).any():
                ok = False
                findings.append(
                    "some actions q01 > q99 entries (data corruption?)")
            else:
                findings.append(
                    f"actions: dim={q01.size}, q01 range "
                    f"[{q01.min():.3f}, {q01.max():.3f}], q99 range "
                    f"[{q99.min():.3f}, {q99.max():.3f}]")

    if "state" in stats:
        s = stats["state"]
        if "q01" in s:
            q01 = np.asarray(s["q01"], dtype=np.float64)
            findings.append(f"state: dim={q01.size} (q01 finite={np.isfinite(q01).all()})")
    else:
        findings.append("(state block absent — non-fatal)")

    return ok, findings


def validate_repo(repo: str) -> bool:
    from flash_vla.core.utils.norm_stats import load_norm_stats, lerobot_candidates

    print(f"\n=== {repo} ===")
    with tempfile.TemporaryDirectory(prefix="lerobot_norm_") as td:
        ckpt_dir = pathlib.Path(td)
        try:
            _download_to(ckpt_dir, repo)
        except Exception as e:
            print(f"  [SKIP] download failed: {type(e).__name__}: {e}")
            return False

        candidates = [
            ckpt_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            ckpt_dir / "norm_stats.json",
            *lerobot_candidates(ckpt_dir),
        ]
        try:
            stats = load_norm_stats(candidates, checkpoint_dir=ckpt_dir)
        except FileNotFoundError as e:
            print(f"  [FAIL] load_norm_stats raised: {e}")
            return False
        if stats is None:
            print("  [FAIL] load_norm_stats returned None")
            return False

        ok, findings = _check_stats(stats)
        for line in findings:
            print(f"    {line}")
        print(f"  [{'PASS' if ok else 'FAIL'}]")
        return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo", action="append", default=None,
        help="HF repo id; can be passed multiple times. "
             f"Default: {DEFAULT_REPOS}")
    args = p.parse_args()
    repos = args.repo or list(DEFAULT_REPOS)

    results = [(r, validate_repo(r)) for r in repos]
    print("\n=== summary ===")
    for r, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {r}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())

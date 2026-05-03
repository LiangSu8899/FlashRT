"""LIBERO (LeRobot-v2 layout) dataset loader for multi-sample calibration.

Layout expected (produced by the ``lerobot`` toolchain and bundled with
the OpenPI LIBERO eval suite)::

    <root>/
      meta/
        info.json       # schema + totals (features, fps, total_frames, ...)
        episodes.jsonl  # {episode_index, tasks, length}
        tasks.jsonl     # {task_index, task}
      data/
        chunk-{NNN}/episode_{NNNNNN}.parquet

Each parquet row is one frame with (at least) these columns::

    observation.images.image          # PNG bytes, dict {"bytes": ..., "path": ...}
    observation.images.wrist_image    # PNG bytes, same dict shape
    observation.state                 # float[state_dim]
    task_index, episode_index, frame_index, index

This module provides:

* :class:`LiberoDataset` — builds the metadata DataFrame (episode × frame)
  that :func:`flash_rt.core.calibration.stratified_sample_indices` expects,
  and can ``load_frame(global_index)`` → obs dict for the frontend.
* :func:`load_calibration_obs` — convenience that combines stratified
  sampling + per-frame loading into one call. Returns a list ready to
  pass to ``model.calibrate(obs_list, percentile=...)``.

Keep this module torch-free / jax-free so the loader works in any
frontend env.
"""

from __future__ import annotations

import io
import json
import logging
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# Default LIBERO image column keys. Override via LiberoDataset(...) kwargs.
_DEFAULT_IMAGE_KEY       = "observation.images.image"
_DEFAULT_WRIST_IMAGE_KEY = "observation.images.wrist_image"
_DEFAULT_STATE_KEY       = "observation.state"

# Pi0 / Pi0.5 / GROOT frontends all take 224×224 uint8 RGB. LIBERO files
# are 256×256 — we resize on load so the caller can pass the obs dict
# straight to frontend.infer() / calibrate().
_DEFAULT_IMAGE_SIZE = 224


def _decode_image(
    cell: Any, *, target_size: int = _DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Decode one parquet image cell to uint8 RGB ``[H, W, 3]`` at ``target_size``."""
    # Cell is either raw bytes or dict(bytes=..., path=...). PIL handles both.
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw is None:
            raise ValueError(
                f"image cell dict missing 'bytes' key; keys={list(cell)!r}")
    elif isinstance(cell, (bytes, bytearray)):
        raw = bytes(cell)
    else:
        raise TypeError(
            f"unsupported image cell type {type(cell).__name__!r} "
            f"(expected dict or bytes)")

    from PIL import Image  # local import to avoid mandatory dep at module load
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


class LiberoDataset:
    """Read-only accessor over a LeRobot-v2 LIBERO dataset.

    The constructor scans ``meta/info.json`` + episode parquets once and
    builds an in-memory DataFrame with columns
    ``(task_index, episode_index, frame_index, index)`` — the exact
    shape required by
    :func:`flash_rt.core.calibration.stratified_sample_indices`.

    Per-frame image bytes are **not** loaded until
    :meth:`load_frame` is called — so building the DataFrame is fast
    (O(episodes) parquet handle opens, no image decode).
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        image_key: str = _DEFAULT_IMAGE_KEY,
        wrist_image_key: Optional[str] = _DEFAULT_WRIST_IMAGE_KEY,
        state_key: Optional[str] = _DEFAULT_STATE_KEY,
        image_size: int = _DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.root = pathlib.Path(root)
        self.image_key = image_key
        self.wrist_image_key = wrist_image_key
        self.state_key = state_key
        self.image_size = int(image_size)

        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"LIBERO meta/info.json not found under {self.root!s}")
        with open(info_path, encoding="utf-8") as f:
            self.info: Dict[str, Any] = json.load(f)

        self.fps: int = int(self.info.get("fps", 10))
        self.total_frames: int = int(self.info.get("total_frames", 0))
        self.total_episodes: int = int(self.info.get("total_episodes", 0))
        self._data_path_template: str = self.info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        self._chunks_size: int = int(self.info.get("chunks_size", 1000))

        # Validate that the image column exists in the schema.
        features = self.info.get("features", {})
        if image_key not in features:
            raise ValueError(
                f"image_key={image_key!r} not in info.json 'features' "
                f"(available: {sorted(features)[:8]}…)")
        if wrist_image_key is not None and wrist_image_key not in features:
            logger.warning(
                "wrist_image_key=%r not in info.json; wrist channel will be "
                "dropped from returned obs dicts.", wrist_image_key)
            self.wrist_image_key = None

        self._metadata = None  # lazy
        self._episode_lengths: Dict[int, int] = {}
        self._episode_task: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Metadata DataFrame (for stratified_sample_indices)
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        """Return the ``(task_index, episode_index, frame_index, index)`` table.

        Lazily built on first access. Reads only the four index columns
        from every episode parquet — no image decode.
        """
        if self._metadata is None:
            self._metadata = self._build_metadata()
        return self._metadata

    def _build_metadata(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        import pandas as pd
        import pyarrow.parquet as pq

        frames = []
        for ep in range(self.total_episodes):
            p = self._episode_path(ep)
            if not p.exists():
                continue
            try:
                t = pq.read_table(
                    p, columns=["task_index", "episode_index",
                                "frame_index", "index"])
            except Exception as e:
                logger.warning("skip %s (%s)", p, e)
                continue
            df = t.to_pandas()
            if len(df) == 0:
                continue
            self._episode_lengths[ep] = len(df)
            self._episode_task[ep] = int(df["task_index"].iloc[0])
            frames.append(df)

        if not frames:
            raise RuntimeError(
                f"no episode parquets found under {self.root}/data/")
        meta = pd.concat(frames, ignore_index=True)
        meta[["task_index", "episode_index", "frame_index", "index"]] = \
            meta[["task_index", "episode_index", "frame_index",
                  "index"]].astype(np.int64)
        return meta

    def _episode_path(self, episode_index: int) -> pathlib.Path:
        chunk = episode_index // self._chunks_size
        rel = self._data_path_template.format(
            episode_chunk=chunk, episode_index=episode_index)
        return self.root / rel

    # ------------------------------------------------------------------
    # Per-frame loader
    # ------------------------------------------------------------------

    def load_frame(self, global_index: int) -> Dict[str, Any]:
        """Decode the row with ``index == global_index`` to an obs dict.

        Returned dict shape (matches Pi0 / Pi0.5 / GROOT ``infer`` contract)::

            {"image":       uint8 [H, W, 3],
             "wrist_image": uint8 [H, W, 3],   # omitted if wrist_image_key is None
             "state":       float32 [state_dim]}  # omitted if state_key is None
        """
        import pyarrow.parquet as pq

        ep = self._episode_index_for_global(int(global_index))
        p = self._episode_path(ep)
        cols = [self.image_key, "index"]
        if self.wrist_image_key is not None:
            cols.append(self.wrist_image_key)
        if self.state_key is not None:
            cols.append(self.state_key)
        df = pq.read_table(p, columns=cols).to_pandas()
        row = df[df["index"] == global_index]
        if row.empty:
            raise KeyError(
                f"global_index={global_index} not found in {p} "
                f"(episode {ep})")
        r = row.iloc[0]

        obs: Dict[str, Any] = {
            "image": _decode_image(r[self.image_key],
                                    target_size=self.image_size),
        }
        if self.wrist_image_key is not None:
            obs["wrist_image"] = _decode_image(
                r[self.wrist_image_key], target_size=self.image_size)
        if self.state_key is not None:
            state = np.asarray(r[self.state_key], dtype=np.float32)
            obs["state"] = state
        return obs

    def _episode_index_for_global(self, global_index: int) -> int:
        # Walk the DataFrame to find which episode contains this global idx.
        meta = self.metadata
        hits = meta.index[meta["index"] == global_index]
        if len(hits) == 0:
            raise KeyError(
                f"global_index={global_index} not in this dataset "
                f"(total_frames={self.total_frames})")
        return int(meta.iloc[int(hits[0])]["episode_index"])

    # ------------------------------------------------------------------
    # Task lookup (for logging / prompts)
    # ------------------------------------------------------------------

    @property
    def tasks(self) -> Dict[int, str]:
        """Return ``{task_index: task_string}`` read from ``meta/tasks.jsonl``."""
        out: Dict[int, str] = {}
        p = self.root / "meta" / "tasks.jsonl"
        if not p.exists():
            return out
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out[int(d["task_index"])] = str(d["task"])
        return out


def load_calibration_obs(
    root: Union[str, pathlib.Path],
    *,
    n: int = 8,
    task_filter: Optional[int] = None,
    exclude: Optional[Iterable[int]] = None,
    image_size: int = _DEFAULT_IMAGE_SIZE,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Stratified-sample N observations from a LIBERO dataset.

    Thin wrapper combining :class:`LiberoDataset` +
    :func:`flash_rt.core.calibration.stratified_sample_indices`. The
    returned list can go straight into
    ``frontend.calibrate(obs_list, percentile=99.9)``.

    Args:
        root: dataset root (contains ``meta/`` and ``data/``).
        n: number of frames to sample (default 8 — the openpi-jax-mlir
           toolchain-validated sweet spot).
        task_filter: if set, only sample frames with this ``task_index``.
        exclude: global indices to exclude (e.g. deployment target frame).
        image_size: resize target for both main + wrist images.
        verbose: log chosen indices and episode layout.
    """
    from flash_rt.core.calibration import stratified_sample_indices

    ds = LiberoDataset(root, image_size=image_size)
    picks = stratified_sample_indices(
        ds.metadata, n=n, task_filter=task_filter, exclude=exclude)
    if verbose:
        tasks = ds.tasks
        logger.info(
            "LIBERO stratified sample: %d/%d frames from %s "
            "(tasks=%d, episodes=%d)",
            len(picks), n, ds.root, len(tasks), ds.total_episodes)
        logger.info("picked global indices: %s", picks)
    return [ds.load_frame(i) for i in picks]

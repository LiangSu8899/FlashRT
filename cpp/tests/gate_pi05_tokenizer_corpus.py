"""Token-exact Pi0.5 gate over real LIBERO prompt/state records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq


CORPUS_MAGIC = 0x50303554
OUTPUT_MAGIC = 0x50303549


def _load_openpi_tokenizer():
    prefix = os.environ.get("OPENPI_BASELINE_SITE_PACKAGES")
    if prefix:
        path = Path(prefix)
        if not path.is_dir():
            raise FileNotFoundError(path)
        sys.path.insert(0, str(path))
    from openpi.models import tokenizer as tokenizer_api

    return tokenizer_api.PaligemmaTokenizer(200)


def _tasks(dataset: Path) -> dict[int, str]:
    result = {}
    with (dataset / "meta" / "tasks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            result[int(item["task_index"])] = str(item["task"])
    return result


def _records(dataset: Path, limit: int):
    info = json.loads((dataset / "meta" / "info.json").read_text())
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    chunk_size = int(info.get("chunks_size", 1000))
    total_episodes = int(info["total_episodes"])
    count = 0
    for episode in range(total_episodes):
        path = dataset / template.format(
            episode_chunk=episode // chunk_size,
            episode_index=episode,
        )
        table = pq.read_table(path, columns=["state", "task_index"])
        for row in table.to_pylist():
            yield int(row["task_index"]), np.asarray(
                row["state"], dtype=np.float32
            )
            count += 1
            if count >= limit:
                return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10000)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    tasks = _tasks(args.dataset)
    stats = json.loads(
        (args.checkpoint / "assets" / "physical-intelligence" / "libero" /
         "norm_stats.json").read_text()
    )["norm_stats"]["state"]
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    official = _load_openpi_tokenizer()
    with tempfile.TemporaryDirectory(prefix="pi05_tokenizer_gate_") as temp:
        corpus = Path(temp) / "corpus.bin"
        output = Path(temp) / "ids.bin"
        expected = []
        lengths = set()
        with corpus.open("wb") as stream:
            stream.write(struct.pack("<II", CORPUS_MAGIC, args.count))
            records = 0
            for task_index, state in _records(args.dataset, args.count):
                normalized = (
                    (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
                ).astype(np.float32)
                task = tasks[task_index]
                task_bytes = task.encode("utf-8")
                stream.write(struct.pack("<II", len(task_bytes), state.size))
                stream.write(task_bytes)
                stream.write(normalized.tobytes())
                ids, mask = official.tokenize(task, normalized)
                valid = np.asarray(ids[np.asarray(mask, dtype=bool)],
                                   dtype=np.int32)
                expected.append(valid)
                lengths.add(valid.size)
                records += 1
            if records != args.count:
                raise RuntimeError(f"requested {args.count}, found {records}")
        subprocess.run(
            [str(args.probe), str(args.tokenizer), str(corpus), str(output)],
            check=True,
        )
        with output.open("rb") as stream:
            magic, records = struct.unpack("<II", stream.read(8))
            if magic != OUTPUT_MAGIC or records != args.count:
                raise RuntimeError("invalid tokenizer probe output header")
            for index, reference in enumerate(expected):
                (count,) = struct.unpack("<I", stream.read(4))
                actual = np.frombuffer(stream.read(count * 4), dtype="<i4")
                if not np.array_equal(actual, reference):
                    raise AssertionError(
                        f"token mismatch at record {index}: "
                        f"actual={actual.tolist()} expected={reference.tolist()}"
                    )
            if stream.read(1):
                raise RuntimeError("tokenizer probe output has trailing bytes")
    print({
        "ok": True,
        "records": args.count,
        "distinct_token_lengths": len(lengths),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
    })
    print("PASS Pi0.5 tokenizer is token-exact on LIBERO corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

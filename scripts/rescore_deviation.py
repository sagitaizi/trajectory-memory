"""Re-score deviation detection in the paper's result files from their saved traces, with the
current `metrics.deviation_roc`; nothing is re-run.

    python scripts/rescore_deviation.py

Rewrites the `auc`, `latency_s` and `fp_per_min` columns of every row that has a trace; rows
without one are left as they are and counted. The deviation-only runs have no traces of their
own and are scored from the `tracks` traces of the same runs (same memories, same centroid).
"""
from __future__ import annotations

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from trajmem.metrics import deviation_roc  # noqa: E402

RUNS = pathlib.Path(__file__).resolve().parents[1] / "runs" / "paper"
SOURCES = (("development/scores.csv", "development/traces"), ("held_out/scores.csv", "held_out/traces"),
           ("reverse_swap/development/scores.csv", "reverse_swap/development/traces"),
           ("tracks/tracks.csv", "tracks/traces"), ("held_out_breaks/breaks.csv", "tracks/traces"))


def rescore(csv_path: pathlib.Path, trace_dir: pathlib.Path) -> tuple[int, int]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    cache, done, skipped = {}, 0, 0
    for r in rows:
        f = trace_dir / f"{r['clip'].replace('/', '__')}__{r['memory']}__{r['input']}.npz"
        if not f.exists():
            skipped += 1
            continue
        if f not in cache:
            z = np.load(f)
            cache[f] = deviation_roc(z["score"], z["t"], list(z["deviation_times"]), threshold=z["bar"])
        r.update({k: cache[f][k] for k in ("auc", "latency_s", "fp_per_min")})
        done += 1
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return done, skipped


def main() -> None:
    for name, traces in SOURCES:
        done, skipped = rescore(RUNS / name, RUNS / traces)
        print(f"{name}: {done} rows re-scored, {skipped} without a trace")


if __name__ == "__main__":
    main()

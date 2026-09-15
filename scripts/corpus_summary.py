"""Verify the simulated corpus and summarise it for the paper (§IV-A).

    python scripts/corpus_summary.py --corpus corpus/sim

Loads every clip (events, ground truth at three instants, break time), then prints the
manifest as counts: shapes, deviation kinds, strings, wobble, lag, and the spread of
periods, event counts and file sizes -- as a Markdown table ready to paste.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from trajmem.data import load_clip  # noqa: E402


def verify(path: pathlib.Path) -> dict:
    clip = load_clip(path)
    t = np.array([0.0, clip.duration_us / 2e6, clip.duration_us / 1e6 - 0.01])
    gt = np.atleast_2d(clip.gt(t))
    ok = (len(clip.events) > 0 and np.all(np.diff(clip.events["timestamp"]) >= 0)
          and np.isfinite(gt).all() and (gt >= 0).all() and (gt <= 1).all()
          and all(0 < d < clip.duration_us / 1e6 for d in clip.deviation_times))
    return {"name": path.stem, "ok": bool(ok), "events": int(len(clip.events)),
            "mb": path.stat().st_size / 1e6}


def summarise(rows: list[dict], checks: list[dict]) -> str:
    n = len(rows)
    periods = np.array([float(r["period_s"]) for r in rows])
    events = np.array([c["events"] for c in checks])
    sizes = np.array([c["mb"] for c in checks])
    lines = [
        f"| Clips | {n}, 15 s each |",
        f"| Shapes | " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['shape'] for r in rows).items())) + " |",
        f"| Period | {periods.min():.2f}–{periods.max():.2f} s (median {np.median(periods):.2f}) |",
        f"| With a scripted break | {sum(1 for r in rows if r['deviation'])} ("
        + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['deviation'] for r in rows if r['deviation']).items())) + ") |",
        f"| Exact path / with imperfections | {sum(1 for r in rows if not r['wobble'])} / {sum(1 for r in rows if r['wobble'])} |",
        f"| Target on a string | {sum(1 for r in rows if r['string'])} |",
        f"| Photoreceptor lag | {sum(1 for r in rows if float(r['cutoff_hz']) > 0)} |",
        f"| Events per clip | {events.min() / 1e6:.1f}–{events.max() / 1e6:.1f} M (median {np.median(events) / 1e6:.1f} M) |",
        f"| On disk | {sizes.sum() / 1e3:.2f} GB ({sizes.min():.0f}–{sizes.max():.0f} MB per clip) |",
    ]
    return "| | |\n|---|---|\n" + "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="corpus/sim")
    args = p.parse_args(argv)

    corpus = pathlib.Path(args.corpus)
    with open(corpus / "manifest.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    checks = [verify(corpus / f"{r['name']}.npz") for r in rows]
    bad = [c["name"] for c in checks if not c["ok"]]
    missing = [r["name"] for r in rows if not (corpus / f"{r['name']}.npz").exists()]
    print(f"{len(rows)} clips in the manifest; {len(bad)} failed checks {bad}; {len(missing)} missing {missing}")
    print(summarise(rows, checks))


if __name__ == "__main__":
    main()

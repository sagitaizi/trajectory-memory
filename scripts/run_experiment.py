"""Run one or more memories over a clip and print the scorecard.

    python scripts/run_experiment.py corpus/real/fan/fan_brush_slow_02
    python scripts/run_experiment.py corpus/sim/sim_003.npz --memory kalman --horizon 0.05
    python scripts/run_experiment.py corpus/real/wall_target/loop_break_01 --slice 0 10.5
    python scripts/run_experiment.py --set development

With --set every clip of a named set in corpus/sets.yaml is scored and a pooled row
added; the table is also written to runs/results/<set>_<memory>.csv.

Prediction error is in pixels at the horizon, on the steady part of the clip before
any break; lock-on is seconds until the error stays under --tol-px; the deviation
numbers follow trajmem.metrics.deviation_roc.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from trajmem.data import load_clip, load_recording, slice_clip  # noqa: E402
from trajmem.experiment import (evaluate_clip, make_memory, open_set,  # noqa: E402
                                run_set, score_trace)

MEMORIES = ("kalman", "harmonic")


def open_clip(path, window=None):
    path = pathlib.Path(path)
    clip = load_clip(path) if path.suffix == ".npz" else load_recording(path)
    if clip.gt is None:
        raise SystemExit(f"{path}: no ground truth to score against")
    return slice_clip(clip, *window) if window else clip


HEADER = (f"{'':26} {'err_med':>8} {'err_iqr':>8} {'lock_on_s':>9} {'auc':>7} "
          f"{'latency_s':>9} {'fp/min':>8} {'unseen':>8}")
CSV_FIELDS = ("set", "memory", "clip", "err_median_px", "err_iqr_px", "lock_on_s", "auc",
              "latency_s", "fp_per_min", "unseen_fraction")


def row(name: str, r: dict) -> str:
    e, d = r["error_px"], r["deviation"]
    return (f"{name:26} {e['median']:8.1f} {e['iqr']:8.1f} {r['lock_on_s']:9.2f} "
            f"{d['auc']:7.2f} {d['latency_s']:9.2f} {d['fp_per_min']:8.2f} {r['unseen_fraction']:8.1%}")


def csv_row(set_name: str, memory: str, r: dict) -> dict:
    e, d = r["error_px"], r["deviation"]
    return {"set": set_name, "memory": memory, "clip": r["name"], "err_median_px": e["median"],
            "err_iqr_px": e["iqr"], "lock_on_s": r["lock_on_s"], "auc": d["auc"],
            "latency_s": d["latency_s"], "fp_per_min": d["fp_per_min"],
            "unseen_fraction": r["unseen_fraction"]}


def run_on_set(set_name: str, names, args) -> None:
    import csv

    clips = open_set(set_name)
    print(f"set {set_name}: {len(clips)} clips with ground truth  window {args.window_us} us  "
          f"horizon {args.horizon} s  warm-up {args.warmup} s")
    out_dir = pathlib.Path("runs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        rows, pooled = run_set(lambda: make_memory(name, dt_s=args.window_us / 1e6, warmup_s=args.warmup),
                               clips, args.window_us, args.horizon, args.tol_px, args.settle)
        print(f"\n[{name}]\n{HEADER}")
        for r in rows:
            print(row(r["name"], r))
        print(row("pooled (medians)", pooled))
        path = out_dir / f"{set_name}_{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(csv_row(set_name, name, r) for r in rows + [pooled])
        print(f"wrote {path}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", nargs="?", help="a clip directory, .aedat4, or a saved .npz")
    p.add_argument("--set", metavar="NAME", help="score every clip of a set in corpus/sets.yaml")
    p.add_argument("--memory", default="all", help=f"one of {MEMORIES} or all")
    p.add_argument("--window-us", type=int, default=5000)
    p.add_argument("--horizon", type=float, default=0.1, help="prediction horizon (s)")
    p.add_argument("--tol-px", type=float, default=15.0, help="lock-on tolerance")
    p.add_argument("--warmup", type=float, default=5.0, help="period estimated from this much (s)")
    p.add_argument("--settle", type=float, default=6.0, help="ignore errors before this (s)")
    p.add_argument("--slice", nargs=2, type=float, metavar=("T0", "T1"), help="time window (s)")
    args = p.parse_args(argv)

    names = MEMORIES if args.memory == "all" else (args.memory,)
    if args.set:
        run_on_set(args.set, names, args)
        return
    if not args.clip:
        p.error("give a clip or --set")
    clip = open_clip(args.clip, args.slice)
    print(f"{args.clip}  {clip.duration_us / 1e6:.1f} s  breaks at {clip.deviation_times or '-'}  "
          f"window {args.window_us} us  horizon {args.horizon} s")
    print(HEADER)
    for name in names:
        memory = make_memory(name, dt_s=args.window_us / 1e6, warmup_s=args.warmup)
        trace = evaluate_clip(memory, clip, args.window_us, args.horizon)
        print(row(name, score_trace(trace, clip, args.tol_px, args.settle)))


if __name__ == "__main__":
    main()

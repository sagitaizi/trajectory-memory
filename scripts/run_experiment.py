"""Run one or more memories over a clip and print the scorecard.

    python scripts/run_experiment.py corpus/real/fan/fan_brush_slow_02
    python scripts/run_experiment.py corpus/sim/sim_003.npz --memory kalman --horizon 0.05
    python scripts/run_experiment.py corpus/real/wall_target/loop_break_01 --slice 0 10.5

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
from trajmem.experiment import evaluate_clip, make_memory, score_trace  # noqa: E402

MEMORIES = ("kalman", "harmonic")


def open_clip(path, window=None):
    path = pathlib.Path(path)
    clip = load_clip(path) if path.suffix == ".npz" else load_recording(path)
    if clip.gt is None:
        raise SystemExit(f"{path}: no ground truth to score against")
    return slice_clip(clip, *window) if window else clip


def row(name: str, r: dict) -> str:
    e, d = r["error_px"], r["deviation"]
    return (f"{name:10} {e['median']:8.1f} {e['iqr']:8.1f} {r['lock_on_s']:9.2f} "
            f"{d['auc']:7.2f} {d['latency_s']:9.2f} {d['fp_per_min']:8.2f} {r['unseen_fraction']:8.1%}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", help="a clip directory, .aedat4, or a saved .npz")
    p.add_argument("--memory", default="all", help=f"one of {MEMORIES} or all")
    p.add_argument("--window-us", type=int, default=5000)
    p.add_argument("--horizon", type=float, default=0.1, help="prediction horizon (s)")
    p.add_argument("--tol-px", type=float, default=15.0, help="lock-on tolerance")
    p.add_argument("--settle", type=float, default=4.0, help="ignore errors before this (s)")
    p.add_argument("--slice", nargs=2, type=float, metavar=("T0", "T1"), help="time window (s)")
    args = p.parse_args(argv)

    clip = open_clip(args.clip, args.slice)
    names = MEMORIES if args.memory == "all" else (args.memory,)
    print(f"{args.clip}  {clip.duration_us / 1e6:.1f} s  breaks at {clip.deviation_times or '-'}  "
          f"window {args.window_us} us  horizon {args.horizon} s")
    print(f"{'memory':10} {'err_med':>8} {'err_iqr':>8} {'lock_on_s':>9} {'auc':>7} "
          f"{'latency_s':>9} {'fp/min':>8} {'unseen':>8}")
    for name in names:
        memory = make_memory(name, dt_s=args.window_us / 1e6)
        trace = evaluate_clip(memory, clip, args.window_us, args.horizon)
        print(row(name, score_trace(trace, clip, args.tol_px, args.settle)))


if __name__ == "__main__":
    main()

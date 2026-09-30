"""Deviation detection on real runs that have a marked onset but no position labels.

    python scripts/score_breaks.py --set development --clips wide_break loop_break_01   # check
    python scripts/score_breaks.py --final                                              # held_out_breaks, once

AUC, latency and false alarms need only the onset (`<slug>.deviation.json`, from
`mark_breaks.py`), so these runs are scored on the classical centroid with the same
memories, alarm constant and rule as `paper_results.py`. Writes runs/paper/<set>/breaks.csv.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402
import numpy as np  # noqa: E402

from trajmem.data import load_recording, slice_clip  # noqa: E402
from trajmem.experiment import evaluate_clip, load_set, make_memory  # noqa: E402
from trajmem.metrics import deviation_roc, k_for_input, ratchet_threshold  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "paper"
MEMORIES = ("snn_phasemap", "phasemap", "kalman", "harmonic", "constant_velocity")
WINDOW_US = 5000
FIELDS = ("set", "clip", "setup", "memory", "input", "onset_s", "auc", "latency_s", "fp_per_min", "n_steps")


def open_unlabelled(entry):
    clip = load_recording(pathlib.Path(entry["clip"]))
    if entry["slice"] is not None:
        t0, t1 = entry["slice"]
        clip = slice_clip(clip, t0 or 0.0, clip.duration_us / 1e6 if t1 is None else t1)
    return clip


def run(args) -> None:
    entries = load_set(args.set_name)
    if args.clips:
        entries = [e for e in entries if e["name"] in args.clips]
    out = OUT / args.set_name / "breaks.csv"
    if args.set_name == "held_out_breaks" and out.exists():
        sys.exit(f"already scored once: {out}")
    k = k_for_input("centroid")
    rows = []
    for entry in entries:
        clip = open_unlabelled(entry)
        if not clip.deviation_times:
            sys.exit(f"{entry['name']}: no marked onset; run scripts/mark_breaks.py first")
        parts = pathlib.Path(entry["clip"]).parts
        setup = parts[parts.index("real") + 1]
        for memory in MEMORIES:
            mem = make_memory(memory, dt_s=WINDOW_US / 1e6, warmup_s=5.0)
            tr = evaluate_clip(mem, clip, WINDOW_US, 0.1)
            d = deviation_roc(tr.score, tr.t, clip.deviation_times, threshold=ratchet_threshold(tr.score, tr.t, k=k))
            rows.append({"set": args.set_name, "clip": entry["name"], "setup": setup, "memory": memory,
                         "input": "centroid", "onset_s": min(clip.deviation_times), "auc": d["auc"],
                         "latency_s": d["latency_s"], "fp_per_min": d["fp_per_min"], "n_steps": len(tr.t)})
            print(f"{entry['name']:28} {memory:18} auc {d['auc']:.2f}  latency {d['latency_s']:.2f} s  "
                  f"false alarms {d['fp_per_min']:.1f}/min", flush=True)
        del clip
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", dest="set_name", default="held_out_breaks")
    p.add_argument("--clips", nargs="+", help="only these entries of the set, by name")
    p.add_argument("--final", action="store_true", help="required for held_out_breaks: scored once, ever")
    args = p.parse_args(argv)
    if args.set_name == "held_out_breaks" and not args.final:
        sys.exit("held_out_breaks is scored once, after every onset is marked; pass --final to do it")
    run(args)


if __name__ == "__main__":
    main()

"""Prediction on real runs without position labels, scored against the tracked position.

    python scripts/score_tracks.py                                  # the `tracks` set
    python scripts/score_tracks.py --set development --clips small_01   # check against labels

Each method is scored against the stream it is fed, h later: the spiking localiser's
positions on the pendulum, the classical centroid elsewhere (as in the paper's tables).
Unseen steps are left out of the reference. Writes runs/paper/<set>/tracks.csv.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402
import numpy as np  # noqa: E402

from scripts.paper_results import (HORIZONS, LOCALISER, MEMORIES, OUT, WINDOW_US, Replay,  # noqa: E402
                                   cached_positions, setup_of)
from scripts.score_breaks import open_unlabelled  # noqa: E402
from trajmem.data import attach_labels  # noqa: E402
from trajmem.experiment import evaluate_clip, load_set, make_memory, score_trace  # noqa: E402
from trajmem.metrics import k_for_input, ratchet_threshold  # noqa: E402

FIELDS = ("set", "clip", "setup", "memory", "input", "horizon_s", "fde_px", "fde_iqr_px", "path_median_px",
          "period_ratio", "auc", "latency_s", "fp_per_min", "n_steps", "unseen_fraction")


def run(args) -> None:
    entries = load_set(args.set_name)
    if args.clips:
        entries = [e for e in entries if e["name"] in args.clips]
    out_dir = OUT / args.set_name
    rows = []
    for entry in entries:
        clip = open_unlabelled(entry)
        name, setup = entry["name"].replace("/", "__"), setup_of(entry)
        input_name, ckpt = ("snn", LOCALISER) if setup == "pendulum" else ("centroid", None)
        txy = cached_positions(clip, input_name, ckpt, out_dir / "positions" / f"{name}__{input_name}.npz")
        seen = np.isfinite(txy[:, 1:]).all(axis=1)
        tracked = attach_labels(clip, [tuple(p) for p in txy[seen]])
        k = k_for_input(input_name, ckpt)
        for memory in MEMORIES:
            replay = Replay(txy[:, 1:]) if input_name != "centroid" else None
            mem = make_memory(memory, dt_s=WINDOW_US / 1e6, warmup_s=5.0)
            traces = evaluate_clip(mem, tracked, WINDOW_US, HORIZONS, localiser=replay)
            bar = ratchet_threshold(traces[0].score, traces[0].t, k=k)
            (out_dir / "traces").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / "traces" / f"{name}__{memory}__{input_name}.npz", t=traces[0].t,
                                obs=traces[0].obs, score=traces[0].score, bar=bar, period=traces[0].period,
                                deviation_times=np.array(clip.deviation_times or [], dtype=float), k=k,
                                resolution=np.array(clip.meta["resolution"]))
            for tr in traces:
                r = score_trace(tr, tracked, 15.0, 6.0, threshold=bar)
                d = r["deviation"]
                rows.append({"set": args.set_name, "clip": entry["name"], "setup": setup, "memory": memory,
                             "input": input_name, "horizon_s": tr.horizon_s, "fde_px": r["fde_px"]["median"],
                             "fde_iqr_px": r["fde_px"]["iqr"], "path_median_px": r["path_px"]["median"],
                             "period_ratio": r["period_ratio"], "auc": d["auc"], "latency_s": d["latency_s"],
                             "fp_per_min": d["fp_per_min"], "n_steps": r["n_steps"],
                             "unseen_fraction": r["unseen_fraction"]})
            one = rows[-len(HORIZONS) + HORIZONS.index(0.1)]
            print(f"{entry['name']:28} {input_name:9} {memory:18} 100 ms {one['fde_px']:6.1f} px  "
                  f"path {one['path_median_px']:6.1f}  false alarms {one['fp_per_min']:.1f}/min", flush=True)
        del clip, tracked
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "tracks.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", dest="set_name", default="tracks")
    p.add_argument("--clips", nargs="+", help="only these entries of the set, by name")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()

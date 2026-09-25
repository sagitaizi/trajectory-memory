"""Development-set runs for the paper's Results section, with the per-step traces the
figures need.

    python scripts/paper_results.py run            # traces + scores -> runs/paper/
    python scripts/paper_results.py run --set sim_long --inputs centroid

Each clip's positions are computed once per input and replayed to every memory. The
spiking localiser is `pend_ft` on pendulum clips and `snn` elsewhere (PLAN.md, input per
setup). Every trace is scored twice: against the labels as-is (the primary pendulum
numbers) and with the clip's constant label offset removed.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402
import numpy as np  # noqa: E402

from trajmem.experiment import (evaluate_clip, load_set, make_localiser, make_memory,  # noqa: E402
                                open_one, positions, score_trace)
from trajmem.metrics import k_for_input, ratchet_threshold  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "paper"
MEMORIES = ("snn_phasemap", "phasemap", "kalman", "harmonic", "constant_velocity")
ABLATIONS = {"two_layer": ROOT / "runs/memory/snn_anchor20.pt", "lmu": ROOT / "runs/memory/lmu_quick.pt"}
HORIZONS = (0.025, 0.05, 0.1, 0.2)
PENDULUM_LOCALISER = ROOT / "runs/localiser/pend_ft.pt"
DEFAULT_LOCALISER = ROOT / "runs/localiser/snn.pt"
WINDOW_US = 5000
FIELDS = ("set", "clip", "setup", "memory", "input", "offset", "horizon_s", "fde_px", "fde_iqr_px",
          "lock_on_s", "path_median_px", "path_last_px", "period_ratio", "auc", "latency_s",
          "fp_per_min", "n_steps", "unseen_fraction")


def setup_of(entry) -> str:
    parts = pathlib.Path(entry["clip"]).parts
    return parts[parts.index("real") + 1] if "real" in parts else "sim"


class Replay:
    """A localiser that hands back positions computed earlier, one per window."""

    def __init__(self, xy):
        self.xy, self.i = xy, 0

    def reset(self):
        self.i = 0

    def locate(self, _frame):
        x, y = self.xy[self.i]
        self.i += 1
        return x, y


def cached_positions(clip, input_name, checkpoint, cache: pathlib.Path):
    if cache.exists():
        return np.load(cache)["txy"]
    localiser = make_localiser(input_name, checkpoint) if input_name != "centroid" else None
    txy = np.array(list(positions(clip, WINDOW_US, localiser)), dtype=float)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, txy=txy)
    return txy


def run(args) -> None:
    entries = load_set(args.set_name)
    if args.clips:
        entries = [e for e in entries if e["name"] in args.clips]
    out_dir = OUT / args.set_name
    if args.set_name == "held_out":
        done = out_dir / "scores.csv"
        if done.exists():
            seen = {r["clip"] for r in csv.DictReader(done.open(encoding="utf-8"))}
            if seen & {e["name"] for e in entries}:
                sys.exit(f"already scored once: {sorted(seen & {e['name'] for e in entries})}")
            rows_before = list(csv.DictReader(done.open(encoding="utf-8")))
        else:
            rows_before = []
    else:
        rows_before = []
    rows, t0 = rows_before, time.time()
    for entry in entries:
        clip = open_one(entry)
        if clip is None:
            continue
        name = entry["name"].replace("/", "__")
        setup = setup_of(entry)
        for input_name in args.inputs:
            ckpt = None
            if input_name == "snn":
                ckpt = PENDULUM_LOCALISER if setup == "pendulum" else DEFAULT_LOCALISER
            txy = cached_positions(clip, input_name, ckpt, out_dir / "positions" / f"{name}__{input_name}.npz")
            replay = Replay(txy[:, 1:]) if input_name != "centroid" else None
            k = k_for_input(input_name, ckpt)
            runs = [(m, None) for m in args.memories]
            if args.ablations:
                runs += [(f"snn_{a}", p) for a, p in ABLATIONS.items()]
            for memory, mem_ckpt in runs:
                mem = make_memory("snn" if mem_ckpt else memory, dt_s=WINDOW_US / 1e6, warmup_s=5.0,
                                  checkpoint=mem_ckpt)
                traces = evaluate_clip(mem, clip, WINDOW_US, HORIZONS, localiser=replay)
                ref = traces[0]
                bar = ratchet_threshold(ref.score, ref.t, k=k)
                (out_dir / "traces").mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_dir / "traces" / f"{name}__{memory}__{input_name}.npz",
                    t=ref.t, obs=ref.obs, score=ref.score, bar=bar, period=ref.period,
                    cycles=ref.cycles.astype(np.float32), horizons=np.array(HORIZONS),
                    pred=np.stack([tr.pred for tr in traces]), gt_ahead=np.stack([tr.gt_ahead for tr in traces]),
                    resolution=np.array(clip.meta["resolution"]),
                    deviation_times=np.array(clip.deviation_times or [], dtype=float),
                    gt_now=np.atleast_2d(clip.gt(ref.t)), k=k)
                for tr in traces:
                    for offset in (False, True):
                        r = score_trace(tr, clip, 15.0, 6.0, threshold=ratchet_threshold(tr.score, tr.t, k=k),
                                        subtract_offset=offset)
                        d, e, p = r["deviation"], r["fde_px"], r["path_px"]
                        rows.append({"set": args.set_name, "clip": entry["name"], "setup": setup, "memory": memory,
                                     "input": input_name, "offset": "removed" if offset else "as_is",
                                     "horizon_s": tr.horizon_s, "fde_px": e["median"], "fde_iqr_px": e["iqr"],
                                     "lock_on_s": r["lock_on_s"], "path_median_px": p["median"],
                                     "path_last_px": p["last"], "period_ratio": r["period_ratio"],
                                     "auc": d["auc"], "latency_s": d["latency_s"], "fp_per_min": d["fp_per_min"],
                                     "n_steps": r["n_steps"], "unseen_fraction": r["unseen_fraction"]})
                print(f"{entry['name']:28} {input_name:9} {memory:18} {time.time() - t0:6.0f} s", flush=True)
            del replay
        del clip
        with open(out_dir / "scores.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("run",))
    p.add_argument("--set", dest="set_name", default="development")
    p.add_argument("--inputs", nargs="+", default=["centroid", "snn"])
    p.add_argument("--memories", nargs="+", default=list(MEMORIES))
    p.add_argument("--ablations", action="store_true", help="also run the two learned memories")
    p.add_argument("--clips", nargs="+", help="only these entries of the set, by name")
    p.add_argument("--final", action="store_true",
                   help="required for the held-out set: each held-out clip is scored once, ever")
    args = p.parse_args(argv)
    if args.set_name == "held_out" and not args.final:
        sys.exit("the held-out set is scored once, after the freeze; pass --final to do it")
    run(args)


if __name__ == "__main__":
    main()

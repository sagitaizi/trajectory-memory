"""Every table the paper needs, from one command.

    python scripts/evaluate.py --set development
    python scripts/evaluate.py --set development --inputs centroid snn      # add the Stage 1 row
    python scripts/evaluate.py --set sim --sample 40
    python scripts/evaluate.py --set held_out                               # once, at the end

Each clip is streamed once per (memory, input) -- the memory's state does not depend on
the horizon it is asked about, so all horizons come from the same pass. Writes, under
runs/eval/<set>/:

    per_clip.csv   one row per clip x memory x input x horizon, every metric
    pooled.csv     the same, pooled over clips (medians; detection over break clips)
    tables.md      the paper's tables, readable
    tables.tex     the same as booktabs, to paste into the paper

The deviation alarm is the label-free ratcheting bar (metrics.ratchet_threshold, k frozen
on the development clips); --threshold oracle switches to the label-using percentile.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from trajmem.experiment import (evaluate_clip, load_set, make_localiser,  # noqa: E402
                                make_memory, open_one, score_trace)
from trajmem.metrics import RATCHET_K  # noqa: E402

MEMORIES = ("snn_phasemap", "phasemap", "kalman", "harmonic")
HORIZONS = (0.025, 0.05, 0.1, 0.2)
FIELDS = ("set", "memory", "input", "horizon_s", "clip", "err_median_px", "err_iqr_px", "lock_on_s",
          "path_median_px", "path_last_px", "path_lock_on_s", "period_ratio", "auc", "latency_s",
          "fp_per_min", "unseen_fraction", "offset_x_px", "offset_y_px", "n_steps")


def rows_for(set_name, entries, memory: str, input_name: str, args, localiser) -> list[dict]:
    """One clip at a time: a simulated clip is ~300 MB of events, so the whole set must
    never be resident at once."""
    out = []
    for entry in entries:
        clip = open_one(entry)
        if clip is None:
            continue
        clip_name = entry["name"]
        mem = make_memory(memory, dt_s=args.window_us / 1e6, warmup_s=args.warmup, checkpoint=args.checkpoint)
        traces = evaluate_clip(mem, clip, args.window_us, args.horizons, localiser=localiser)
        for tr in traces:
            r = score_trace(tr, clip, args.tol_px, args.settle, threshold=args.threshold_rule,
                            subtract_offset=args.subtract_offset)
            d, e, p = r["deviation"], r["error_px"], r["path_px"]
            out.append({"set": set_name, "memory": memory, "input": input_name, "horizon_s": tr.horizon_s,
                        "clip": clip_name, "err_median_px": e["median"], "err_iqr_px": e["iqr"],
                        "lock_on_s": r["lock_on_s"], "path_median_px": p["median"], "path_last_px": p["last"],
                        "path_lock_on_s": r["path_lock_on_s"], "period_ratio": r["period_ratio"],
                        "auc": d["auc"], "latency_s": d["latency_s"], "fp_per_min": d["fp_per_min"],
                        "unseen_fraction": r["unseen_fraction"], "offset_x_px": r["offset_px"][0],
                        "offset_y_px": r["offset_px"][1], "n_steps": r["n_steps"]})
    return out


def pool(rows: list[dict]) -> list[dict]:
    """Medians over clips, per memory x input x horizon; detection over the break clips."""
    out = []
    keys = sorted({(r["memory"], r["input"], r["horizon_s"]) for r in rows})
    for memory, input_name, h in keys:
        group = [r for r in rows if (r["memory"], r["input"], r["horizon_s"]) == (memory, input_name, h)]
        entry = {"set": group[0]["set"], "memory": memory, "input": input_name, "horizon_s": h,
                 "clip": f"pooled ({len(group)} clips)"}
        for f in FIELDS:
            if f in entry:
                continue
            vals = np.array([r[f] for r in group], dtype=float)
            vals = vals[np.isfinite(vals)] if f not in ("latency_s",) else vals
            entry[f] = float(np.nanmedian(vals)) if len(vals) else np.nan
        entry["n_clips"] = len(group)
        entry["n_break_clips"] = int(np.isfinite([r["auc"] for r in group]).sum())
        out.append(entry)
    return out


def _fmt(v, nd=1) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--" if v is None or np.isnan(v) else "inf"
    return f"{v:.{nd}f}"


def tables(pooled: list[dict], set_name: str, horizons) -> tuple[str, str]:
    """The paper's two tables: prediction + path per memory and input, and detection."""
    md, tex = [], []
    inputs = sorted({r["input"] for r in pooled})
    md.append(f"# {set_name}: prediction and path memory\n")
    md.append("Median over clips, offset removed. Prediction error in px at each horizon; "
              "path = shape of the remembered cycle; period = memory's period over the true one.\n")
    head = ["memory", "input"] + [f"pred {int(h * 1000)} ms" for h in horizons] + ["path px", "period"]
    md.append("| " + " | ".join(head) + " |")
    md.append("|" + "---|" * len(head))
    tex.append("\\begin{tabular}{ll" + "r" * (len(horizons) + 2) + "}\n\\toprule")
    tex.append(" & ".join(head).replace("_", " ") + " \\\\\n\\midrule")
    for memory in MEMORIES:
        for input_name in inputs:
            got = [r for r in pooled if r["memory"] == memory and r["input"] == input_name]
            if not got:
                continue
            by_h = {r["horizon_s"]: r for r in got}
            cells = [_fmt(by_h[h]["err_median_px"]) if h in by_h else "--" for h in horizons]
            ref = by_h[max(by_h)]
            line = [memory, input_name] + cells + [_fmt(ref["path_median_px"]), _fmt(ref["period_ratio"], 2)]
            md.append("| " + " | ".join(line) + " |")
            tex.append(" & ".join(line).replace("_", " ") + " \\\\")
    tex.append("\\bottomrule\n\\end{tabular}")

    md.append(f"\n# {set_name}: deviation detection\n")
    md.append(f"Ratcheting alarm, k = {RATCHET_K:g}, chosen on the development clips. "
              "AUC and latency over the break clips; false alarms over every clip's quiet part.\n")
    head2 = ["memory", "input", "break clips", "AUC", "latency s", "false alarms/min"]
    md.append("| " + " | ".join(head2) + " |")
    md.append("|" + "---|" * len(head2))
    tex.append("\n\\begin{tabular}{llrrrr}\n\\toprule")
    tex.append(" & ".join(head2).replace("_", " ") + " \\\\\n\\midrule")
    for memory in MEMORIES:
        for input_name in inputs:
            got = [r for r in pooled if r["memory"] == memory and r["input"] == input_name]
            if not got:
                continue
            r = got[0]                                   # detection does not depend on the horizon
            line = [memory, input_name, str(r["n_break_clips"]), _fmt(r["auc"], 2),
                    _fmt(r["latency_s"], 2), _fmt(r["fp_per_min"], 2)]
            md.append("| " + " | ".join(line) + " |")
            tex.append(" & ".join(line).replace("_", " ") + " \\\\")
    tex.append("\\bottomrule\n\\end{tabular}")
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", dest="set_name", default="development", help="a set in corpus/sets.yaml")
    p.add_argument("--memories", nargs="+", default=list(MEMORIES))
    p.add_argument("--inputs", nargs="+", default=["centroid"], choices=("centroid", "frame_centroid", "snn"),
                   help="where positions come from; 'snn' adds the full Stage 1 row")
    p.add_argument("--horizons", type=float, nargs="+", default=list(HORIZONS))
    p.add_argument("--sample", type=int, help="score this many clips (a seeded sample; for the sim corpus)")
    p.add_argument("--threshold", default="ratchet", help="ratchet (default), oracle, or a number in px")
    p.add_argument("--checkpoint", help="memory weights, for --memories snn")
    p.add_argument("--localiser-checkpoint", help="spiking localiser weights")
    p.add_argument("--window-us", type=int, default=5000)
    p.add_argument("--tol-px", type=float, default=15.0)
    p.add_argument("--warmup", type=float, default=5.0)
    p.add_argument("--settle", type=float, default=6.0)
    p.add_argument("--subtract-offset", action="store_true", default=True)
    p.add_argument("--keep-offset", dest="subtract_offset", action="store_false",
                   help="do not remove each clip's constant label-vs-centroid offset")
    p.add_argument("--out", default="runs/eval")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    args.threshold_rule = None if args.threshold == "oracle" else (
        "ratchet" if args.threshold == "ratchet" else float(args.threshold))

    entries = load_set(args.set_name)
    if args.sample and args.sample < len(entries):
        idx = np.random.default_rng(args.seed).permutation(len(entries))[:args.sample]
        entries = [entries[i] for i in sorted(idx)]
    out_dir = pathlib.Path(args.out) / args.set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"set {args.set_name}: {len(entries)} clips  memories {args.memories}  inputs {args.inputs}  "
          f"horizons {args.horizons}  alarm {args.threshold}", flush=True)

    rows, t0 = [], time.time()
    for input_name in args.inputs:
        localiser = make_localiser(input_name, args.localiser_checkpoint) if input_name != "centroid" else None
        for memory in args.memories:
            rows += rows_for(args.set_name, entries, memory, input_name, args, localiser)
            print(f"  {memory:13} {input_name:14} done  {time.time() - t0:5.0f} s", flush=True)

    pooled = pool(rows)
    with open(out_dir / "per_clip.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "pooled.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS) + ["n_clips", "n_break_clips"])
        w.writeheader()
        w.writerows(pooled)
    md, tex = tables(pooled, args.set_name, args.horizons)
    (out_dir / "tables.md").write_text(md, encoding="utf-8")
    (out_dir / "tables.tex").write_text(tex, encoding="utf-8")
    print(md)
    print(f"wrote {out_dir}/per_clip.csv, pooled.csv, tables.md, tables.tex")


if __name__ == "__main__":
    main()

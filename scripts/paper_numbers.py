"""The Results section's in-text numbers, from saved scores and traces; nothing is re-run.

    python scripts/paper_numbers.py     # -> runs/paper/numbers.csv, runs/paper/<set>/localiser.csv

Sources: `paper_results.py run` (development, held_out), `score_breaks.py` (held_out_breaks),
`score_tracks.py` (tracks), `paper_results.py run --out runs/paper/reverse_swap` (the input
swap on the other setups). The tables themselves come from `paper_figures.py`.
"""
from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figures import (LABELLED, RUNS, TABLE_MEMORIES, all_runs, detected, deviation_runs,  # noqa: E402
                           is_pendulum, pick, read_csv, read_scores)

OTHER = ("fan", "wall_target")


def before_break(tr) -> np.ndarray:
    tb = float(tr["deviation_times"][0]) if len(tr["deviation_times"]) else np.inf
    return tr["t"] < tb


def localiser_rows(set_name: str) -> list[dict]:
    """Distance of each input's positions from the labels, before any deviation, per clip."""
    rows = []
    for path in sorted((RUNS / set_name / "traces").glob("*__kalman__*.npz")):
        clip, input_name = path.stem.split("__kalman__")
        tr = np.load(path)
        d = (tr["obs"] - tr["gt_now"]) * tr["resolution"]
        d = d[before_break(tr) & np.isfinite(d).all(axis=1)]
        e, e0 = np.linalg.norm(d, axis=1), np.linalg.norm(d - np.median(d, axis=0), axis=1)
        rows.append({"set": set_name, "clip": clip, "pendulum": is_pendulum(clip), "input": input_name,
                     "median_px": np.median(e), "p90_px": np.percentile(e, 90),
                     "offset_removed_median_px": np.median(e0), "errors": e})
    return rows


def learning_delay(set_names, memory: str, horizon=0.1, span_s=20.0) -> float:
    """When the across-clip median of the 1 s running prediction error first comes within
    10% of its level over 10-20 s, on the pendulum clips before any break."""
    grid = np.arange(0.0, span_s, 0.05)
    curves = []
    for set_name in set_names:
        for path in sorted((RUNS / set_name / "traces").glob(f"*__{memory}__snn.npz")):
            if not is_pendulum(path.name.split(f"__{memory}__")[0]):
                continue
            tr = np.load(path)
            hi = list(tr["horizons"]).index(horizon)
            t = tr["t"] - tr["t"][0]
            e = np.linalg.norm((tr["pred"][hi] - tr["gt_ahead"][hi]) * tr["resolution"], axis=1)
            e[~before_break(tr)] = np.nan
            curves.append([np.nanmedian(w) if np.isfinite(w := e[(t >= g - 0.5) & (t < g + 0.5)]).any() else np.nan
                           for g in grid])
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.array(curves, dtype=float), axis=0)
    steady = np.nanmedian(med[grid >= 10.0])
    return float(grid[np.flatnonzero(med <= 1.1 * steady)[0]])


def recovery(set_name: str, clip: str, memory: str, horizon=0.1, input_name="snn") -> str:
    """Median prediction error in 2 s bins around the deviation, to show re-learning after it."""
    tr = np.load(RUNS / set_name / "traces" / f"{clip}__{memory}__{input_name}.npz")
    hi = list(np.round(tr["horizons"], 3)).index(horizon)
    e = np.linalg.norm((tr["pred"][hi] - tr["gt_ahead"][hi]) * tr["resolution"], axis=1)
    rel = tr["t"] - float(tr["deviation_times"][0])
    edges = np.arange(-2, 14, 2)
    cells = []
    for lo, hi_s in zip(edges[:-1], edges[1:]):
        w = e[(rel >= lo) & (rel < hi_s) & np.isfinite(e)]
        cells.append(f"{lo:+d}..{hi_s:+d} s {np.median(w):.1f}" if len(w) else f"{lo:+d}..{hi_s:+d} s -")
    return "; ".join(cells)


def medians(rows, memory, **sel) -> str:
    got = pick(rows, memory=memory, **sel)
    with np.errstate(all="ignore"):
        err = [np.nanmedian([r["fde_px"] for r in got if r["horizon_s"] == h]) for h in (0.05, 0.1, 0.2)]
        path = np.nanmedian([r["path_median_px"] for r in got if r["horizon_s"] == 0.1])
    return " / ".join(f"{v:.1f}" for v in err) + f" / path {path:.1f}"


def main() -> None:
    sets = ("development", "held_out")
    rows = [r for s in sets for r in read_scores(s)]
    breaks = read_csv(RUNS / "held_out_breaks" / "breaks.csv")
    tracks = read_csv(RUNS / "tracks" / "tracks.csv")
    swap = read_scores_at(RUNS / "reverse_swap" / "development" / "scores.csv")
    loc = {s: localiser_rows(s) for s in sets + ("reverse_swap/development",)}
    out = []

    def put(section, quantity, value):
        out.append({"section": section, "quantity": quantity, "value": value})

    for s, got in loc.items():
        with open(RUNS / s / "localiser.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[k for k in got[0] if k != "errors"])
            w.writeheader()
            w.writerows({k: v for k, v in r.items() if k != "errors"} for r in got)
    both = loc["development"] + loc["held_out"]
    for r in loc["reverse_swap/development"]:
        if r["input"] == "snn" and r["clip"].count("__") == 0:
            put("V.A", f"{r['clip']}, snn: median px (offset removed)",
                f"{r['median_px']:.1f} ({r['offset_removed_median_px']:.1f})")
    for inp in ("snn", "centroid"):
        pend = [r for r in both if r["pendulum"] and r["input"] == inp]
        e = np.concatenate([r["errors"] for r in pend])
        put("V.A", f"pendulum, {inp}: median / p90 px", f"{np.median(e):.1f} / {np.percentile(e, 90):.1f}")
        rng = [r["offset_removed_median_px"] for r in pend]
        put("V.A", f"pendulum per run, {inp}, offset removed: median px range", f"{min(rng):.1f}-{max(rng):.1f}")
        for r in both:
            if not r["pendulum"] and r["input"] == inp and r["clip"].count("__") == 0:
                put("V.A", f"{r['clip']}, {inp}: median px (offset removed)",
                    f"{r['median_px']:.1f} ({r['offset_removed_median_px']:.1f})")

    pend = LABELLED["pendulum"]
    for m in TABLE_MEMORIES:
        put("V.B", f"pendulum, {m}: 50 / 100 / 200 ms px", medians(rows, m, **pend))
        put("V.B", f"other setups, {m}: 50 / 100 / 200 ms px", medians(rows, m, **LABELLED["other"]))
        if tracks:
            put("V.B", f"unlabelled runs, {m}: 50 / 100 / 200 ms px", medians(tracks, m))
    ratios = [r["period_ratio"] for r in pick(rows, memory="snn_phasemap", horizon_s=0.1, **pend)]
    put("V.B", "clock period / true period, pendulum runs", f"{min(ratios):.3f}-{max(ratios):.3f}")
    for m in ("snn_phasemap", "kalman", "harmonic"):
        put("V.B", f"learning delay to steady 100 ms error, {m} (s)", f"{learning_delay(list(sets), m):.2f}")
    for m in ("snn_phasemap", "kalman", "harmonic"):
        put("V.B", f"wide_break, {m}: 100 ms px around the push (recovery)", recovery("development", "wide_break", m))
    for r in pick(rows, memory="snn_phasemap", horizon_s=0.1, **LABELLED["other"]):
        put("V.B", f"{r['clip']}, spiking memory: path px / period ratio", f"{r['path_median_px']:.1f} / {r['period_ratio']:.2f}")
    if tracks:
        for m in ("snn_phasemap", "kalman"):
            for r in pick(tracks, memory=m, horizon_s=0.1):
                put("V.B", f"unlabelled {r['clip']}, {m}: 100 ms px / period ratio",
                    f"{r['fde_px']:.1f} / {r['period_ratio']:.2f}")

    dev, every = deviation_runs(rows, breaks), all_runs(rows, tracks)
    for m in TABLE_MEMORIES:
        d = [r for r in dev if r["memory"] == m]
        hits = [r for r in d if detected(r)]
        fp = [r for r in every if r["memory"] == m]
        put("V.C", f"{m}: detected / runs with a deviation", f"{len(hits)} / {len(d)}")
        put("V.C", f"{m}: median AUC / median latency s", f"{np.median([r['auc'] for r in d]):.2f} / "
            f"{np.median([r['latency_s'] for r in hits]) if hits else np.nan:.2f}")
        put("V.C", f"{m}: false alarms/min mean / max; runs with any, of all",
            f"{np.mean([r['fp_per_min'] for r in fp]):.1f} / {np.max([r['fp_per_min'] for r in fp]):.1f}; "
            f"{sum(r['fp_per_min'] > 0 for r in fp)} of {len(fp)}")
        for r in d:
            put("V.C", f"{r['clip']}, {m}: AUC / latency s / false alarms/min",
                f"{r['auc']:.2f} / {r['latency_s']:.2f} / {r['fp_per_min']:.1f}")

    for m in ("snn_two_layer", "snn_lmu", "snn_phasemap"):
        got = pick(rows, memory=m, horizon_s=0.1, **pend)
        put("V.D", f"pendulum ({len(got)} runs), {m}: median path px / period ratio / 100 ms px",
            f"{np.nanmedian([r['path_median_px'] for r in got]):.1f} / "
            f"{np.nanmedian([r['period_ratio'] for r in got]):.2f} / {np.nanmedian([r['fde_px'] for r in got]):.1f}")
    for inp in ("snn", "centroid"):
        e = [r["fde_px"] for r in pick(rows, memory="snn_phasemap", setup="pendulum", input=inp, offset="as_is",
                                       horizon_s=0.1)]
        put("V.D", f"pendulum, spiking memory, {inp} input: 100 ms px median (range)",
            f"{np.median(e):.1f} ({min(e):.1f}-{max(e):.1f})")
    if swap:
        other_snn = dict(setup=lambda s: s in OTHER, input="snn", offset="removed")
        for m in TABLE_MEMORIES:
            put("V.D", f"other setups, spiking localiser input, {m}: 50 / 100 / 200 ms px", medians(swap, m, **other_snn))
            for r in pick(swap, memory=m, horizon_s=0.1, **other_snn):
                if np.isfinite(r["auc"]):
                    put("V.D", f"{r['clip']}, spiking localiser input, {m}: AUC / latency s / false alarms/min",
                        f"{r['auc']:.2f} / {r['latency_s']:.2f} / {r['fp_per_min']:.1f}")

    with open(RUNS / "numbers.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["section", "quantity", "value"])
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['section']:4} {r['quantity']:82} {r['value']}")


def read_scores_at(path: pathlib.Path) -> list[dict]:
    rows = read_csv(path)
    return [r for r in rows if r["clip"] not in ("loop_break_01/diagonal", "loop_break_01/horizontal")]


if __name__ == "__main__":
    main()

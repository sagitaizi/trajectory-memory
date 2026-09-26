"""Results figures and tables for the paper, from `paper_results.py run` output.

    python scripts/paper_figures.py                 # -> paper/figures/results_*.pdf, runs/paper/tables.tex

Vector PDFs sized for IEEE columns (3.5 in single, 7.16 in double), Times-like text at
8 pt. Every number is read from runs/paper/<set>/scores.csv or the saved traces.
"""
from __future__ import annotations

import csv
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "paper"
FIGS = ROOT / "paper" / "figures"
COL, DCOL = 3.5, 7.16

INK, MUTED, FAINT, GRID = "#0b0b0b", "#52514e", "#bdbcb6", "#e6e5e1"
STYLE = {  # memory -> (label, colour, line style, marker)
    "snn_phasemap": ("Spiking clock-and-map (ours)", "#2a78d6", "-", "o"),
    "phasemap": ("Clock-and-map, arithmetic", "#2a78d6", (0, (3, 1.5)), None),
    "kalman": ("Periodic Kalman", "#eb6834", "-", "s"),
    "harmonic": ("Harmonic fit", "#1baf7a", "-", "^"),
    "constant_velocity": ("Constant velocity", "#8a8985", (0, (1, 1.2)), "D"),
    "snn_two_layer": ("Learned two-timescale SNN", "#4a3aa7", "-", "v"),
    "snn_lmu": ("LMU window SNN", "#e87ba4", "-", "P"),
}
TRUTH = "#8a8985"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


# --- data ------------------------------------------------------------------------------

SEGMENTS = {"loop_break_01/diagonal", "loop_break_01/horizontal"}   # parts of a recording already scored whole


def read_scores(set_name: str) -> list[dict]:
    path = RUNS / set_name / "scores.csv"
    if not path.exists():
        return []
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["clip"] not in SEGMENTS]
    for r in rows:
        for k in ("horizon_s", "fde_px", "fde_iqr_px", "lock_on_s", "path_median_px", "path_last_px",
                  "period_ratio", "auc", "latency_s", "fp_per_min", "unseen_fraction"):
            r[k] = float(r[k]) if r[k] not in ("", "nan") else np.nan
    return rows


def pick(rows, **kw) -> list[dict]:
    return [r for r in rows if all(r[k] == v if not callable(v) else v(r[k]) for k, v in kw.items())]


def trace(set_name: str, clip: str, memory: str, input_name: str):
    path = RUNS / set_name / "traces" / f"{clip.replace('/', '__')}__{memory}__{input_name}.npz"
    return np.load(path) if path.exists() else None


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(RUNS / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", FIGS / f"{name}.pdf")


# --- figures ---------------------------------------------------------------------------

def fig_prediction_vs_horizon(panels, name="results_horizon") -> None:
    """Median prediction error against horizon, one panel per (rows, title, selection)."""
    memories = ["snn_phasemap", "phasemap", "kalman", "harmonic", "constant_velocity"]
    fig, axes = plt.subplots(1, len(panels), figsize=(DCOL, 1.8), sharey=True)
    for ax, (rows, title, sel) in zip(axes, panels):
        for m in memories:
            got = pick(rows, memory=m, **sel)
            hs = sorted({r["horizon_s"] for r in got})
            if not hs:
                continue
            med = [np.nanmedian([r["fde_px"] for r in got if r["horizon_s"] == h]) for h in hs]
            label, colour, ls, marker = STYLE[m]
            ax.plot(np.array(hs) * 1000, med, color=colour, ls=ls, marker=marker, label=label,
                    lw=1.6 if m == "snn_phasemap" else 1.1, zorder=3 if m == "snn_phasemap" else 2)
            for h, v in zip(hs, med):
                if v > 60:
                    ax.annotate(f"{v:.0f}↑", xy=(h * 1000, 60), xytext=(0, -1), textcoords="offset points",
                                ha="center", va="top", fontsize=6.5, color=colour)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Horizon $h$ (ms)")
        ax.set_xticks([25, 50, 100, 200])
        ax.set_ylim(0, 60)
    axes[0].set_ylabel("Median error (px)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels),
               handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(w_pad=1.0)
    save(fig, name)


def fig_break_trace(set_name, clip, name="results_break", window=(-6.0, 4.0), input_name="snn") -> None:
    """One break clip: the target's horizontal position with the 100 ms prediction, then the
    deviation score against its alarm bar, for the spiking memory and the Kalman."""
    ours = trace(set_name, clip, "snn_phasemap", input_name)
    kal = trace(set_name, clip, "kalman", input_name)
    if ours is None or kal is None or not len(ours["deviation_times"]):
        print("skip", name, "(no trace or no break)")
        return
    tb = float(ours["deviation_times"][0])
    t = ours["t"] - tb
    sel = (t >= window[0]) & (t <= window[1])
    res = ours["resolution"]
    hi = list(ours["horizons"]).index(0.1)
    fig, axes = plt.subplots(3, 1, figsize=(COL, 2.9), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1, 1], "hspace": 0.18})
    ax = axes[0]
    ax.plot(t[sel], ours["gt_now"][sel, 0] * res[0], color=TRUTH, lw=2.4, alpha=0.45, label="Label")
    ahead = t + 0.1
    ax.plot(ahead[sel], ours["pred"][hi][sel, 0] * res[0], color=STYLE["snn_phasemap"][1], lw=1.0,
            label="Prediction, 100 ms ahead")
    ax.set_ylabel("$x$ (px)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.6, borderaxespad=0.2)
    for ax, tr, m in ((axes[1], ours, "snn_phasemap"), (axes[2], kal, "kalman")):
        tt = tr["t"] - tb
        s = (tt >= window[0]) & (tt <= window[1])
        ratio = tr["score"] / np.where(tr["bar"] > 0, tr["bar"], np.nan)
        colour = STYLE[m][1]
        ax.plot(tt[s], ratio[s], color=colour, lw=0.9)
        ax.axhline(1.0, color=INK, lw=0.6, ls=(0, (3, 2)))
        flags = alarm_times(tt, ratio)
        for f in flags[(flags >= window[0]) & (flags <= window[1])]:
            ax.axvline(f, color="#d03b3b", lw=0.6, alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 40)
        ax.set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
        ax.minorticks_off()
        ax.set_ylabel("Score / bar")
        ax.text(0.01, 0.92, STYLE[m][0], transform=ax.transAxes, va="top", fontsize=7, color=INK)
    for ax in axes:
        ax.axvline(0.0, color=INK, lw=0.9)
    axes[0].annotate("deviation", xy=(0, 1), xycoords=("data", "axes fraction"), xytext=(3, -2),
                     textcoords="offset points", va="top", fontsize=7)
    axes[-1].set_xlabel("Time from the deviation (s)")
    fig.align_ylabels(axes)
    save(fig, name)


def alarm_times(t, ratio, hold=3) -> np.ndarray:
    """Onsets of the alarm: `hold` consecutive steps above the bar, as the scoring counts it."""
    above = np.nan_to_num(ratio) > 1.0
    run = np.convolve(above.astype(int), np.ones(hold, int), mode="full")[:len(above)] >= hold
    onset = run & ~np.r_[False, run[:-1]]
    return t[onset]


def fig_paths(set_name, clip, name="results_paths", input_name="snn") -> None:
    """The remembered cycle of each memory, at the last step before any break, over the
    labelled path (image coordinates, y down)."""
    memories = ["snn_phasemap", "kalman", "snn_two_layer", "snn_lmu"]
    fig, axes = plt.subplots(1, len(memories), figsize=(DCOL * 0.78, 1.45), sharex=True, sharey=True)
    for ax, m in zip(axes, memories):
        tr = trace(set_name, clip, m, input_name)
        if tr is None:
            ax.set_visible(False)
            continue
        res = tr["resolution"]
        tb = float(tr["deviation_times"][0]) if len(tr["deviation_times"]) else tr["t"][-1]
        i = int(np.searchsorted(tr["t"], tb) - 1)
        gt = tr["gt_now"] * res
        ok = (tr["t"] < tb) & (tr["t"] > tb - 6)
        ax.plot(gt[ok, 0], gt[ok, 1], color=TRUTH, lw=2.6, alpha=0.4, solid_capstyle="round", label="Labelled path")
        cyc = tr["cycles"][i].astype(float) * res
        closed = np.vstack([cyc, cyc[:1]])
        label, colour, _, _ = STYLE[m]
        ax.plot(closed[:, 0], closed[:, 1], color=colour, lw=1.3)
        ax.set_title(label.replace(" (ours)", "\n(ours)").replace(" SNN", "\nSNN").replace("Periodic ", "Periodic\n"),
                     fontsize=7, loc="left")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("$y$ (px)")
    for ax in axes:
        ax.set_xlabel("$x$ (px)")
    fig.tight_layout(w_pad=0.6)
    save(fig, name)


def fig_localiser_cdf(set_names, name="results_localiser") -> None:
    """Distribution of the localisation error against the labels on the pendulum clips:
    classical centroid vs the spiking localiser (pend_ft)."""
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    for input_name, label, colour in (("centroid", "Classical centroid", "#8a8985"),
                                      ("snn", "Spiking localiser", STYLE["snn_phasemap"][1])):
        errs = []
        for set_name in set_names:
            for path in sorted((RUNS / set_name / "traces").glob(f"*__kalman__{input_name}.npz")):
                tr = np.load(path)
                clip = path.name.split("__kalman__")[0]
                if not is_pendulum(clip):
                    continue
                res = tr["resolution"]
                e = np.linalg.norm((tr["obs"] - tr["gt_now"]) * res, axis=1)
                tb = float(tr["deviation_times"][0]) if len(tr["deviation_times"]) else np.inf
                errs.append(e[np.isfinite(e) & (tr["t"] < tb)])
        if not errs:
            continue
        e = np.sort(np.concatenate(errs))
        ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=colour, label=f"{label} (median {np.median(e):.0f} px)")
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Distance from the label (px)")
    ax.set_ylabel("Fraction of steps")
    ax.legend(loc="lower right")
    save(fig, name)


def fig_learning(set_names, name="results_learning", input_name="snn", horizon=0.1, span_s=20.0) -> None:
    """Prediction error against time from the clip's start on the pendulum clips, before any
    break: a 1 s running median per clip, then the median across clips."""
    memories = ["snn_phasemap", "kalman", "harmonic"]
    grid = np.arange(0.0, span_s, 0.05)
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    for m in memories:
        curves = []
        for set_name in set_names:
            for path in sorted((RUNS / set_name / "traces").glob(f"*__{m}__{input_name}.npz")):
                clip = path.name.split(f"__{m}__")[0]
                if not is_pendulum(clip):
                    continue
                tr = np.load(path)
                hi = list(tr["horizons"]).index(horizon)
                t = tr["t"] - tr["t"][0]
                e = np.linalg.norm((tr["pred"][hi] - tr["gt_ahead"][hi]) * tr["resolution"], axis=1)
                tb = float(tr["deviation_times"][0]) - tr["t"][0] if len(tr["deviation_times"]) else np.inf
                e[t + horizon >= tb] = np.nan
                curves.append(np.array([np.nanmedian(e[(t >= g - 0.5) & (t < g + 0.5)])
                                        if np.isfinite(e[(t >= g - 0.5) & (t < g + 0.5)]).any() else np.nan
                                        for g in grid]))
        if not curves:
            continue
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.vstack(curves), axis=0)
        label, colour, ls, _ = STYLE[m]
        ax.plot(grid, med, color=colour, ls=ls, label=label, lw=1.5 if m == "snn_phasemap" else 1.1)
    ax.axvspan(0, 5, color=GRID, alpha=0.6, lw=0)
    ax.text(2.5, 0.04, "baselines' warm-up", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=6.5, color=MUTED)
    ax.set_xlim(0, span_s)
    ax.set_ylim(0, None)
    ax.set_xlabel("Time from the start of the clip (s)")
    ax.set_ylabel("Error at 100 ms (px)")
    ax.legend(loc="upper right")
    save(fig, name)


def fig_teaser(set_name="held_out", clip="small_break", name="teaser", window=(-5.0, 3.0)) -> None:
    """One column: an event frame just after the break with the remembered path, the current
    position and the 100 ms prediction (left); the horizontal position with its prediction and
    the deviation score against its alarm bar around the break (right)."""
    import _thesis_path  # noqa: F401
    from trajmem.data import load_recording

    tr = trace(set_name, clip, "snn_phasemap", "snn")
    if tr is None or not len(tr["deviation_times"]):
        print("skip", name)
        return
    res, t, tb = tr["resolution"], tr["t"], float(tr["deviation_times"][0])
    hi = list(tr["horizons"]).index(0.1)
    i_b = int(np.searchsorted(t, tb) - 1)
    path = tr["cycles"][i_b].astype(float) * res                  # the path remembered at the break
    obs = tr["obs"] * res
    rec = load_recording(ROOT / "corpus/real/pendulum" / clip)
    ev = rec.events
    ts = ev["timestamp"]

    def visible(k):                                                # events within 40 px of the target in its 5 ms
        a = ts[0] + int(t[k] * 1e6) - 2500
        lo, hi_ = np.searchsorted(ts, [a, a + 5000])
        e = ev[lo:hi_]
        return int(np.sum(np.hypot(e["x"] - obs[k][0], e["y"] - obs[k][1]) < 40))

    cand = [k for k in np.where((t > tb + 0.3) & (t < tb + 2.5) & np.isfinite(obs).all(1))[0] if visible(k) >= 500]
    dist = [np.min(np.linalg.norm(path - obs[k], axis=1)) for k in cand]
    i = int(cand[int(np.argmax(dist))])                           # furthest off the path while clearly in view
    t0 = ev["timestamp"][0] + int(t[i] * 1e6) - 2500
    sel = (ev["timestamp"] >= t0) & (ev["timestamp"] < t0 + 5000)
    e = ev[sel]
    img = np.zeros((res[1], res[0], 3)) + np.array(matplotlib.colors.to_rgb("#0d0d10"))
    for pol, c in ((1, "#ff6b5b"), (0, "#4fb3ff")):
        m = e["polarity"] == pol
        img[e["y"][m], e["x"][m]] = matplotlib.colors.to_rgb(c)
    pts = np.vstack([path, obs[i:i + 1] + [[-90, 60]], obs[i:i + 1] + [[60, -60]]])
    cx, cy = np.nanmean(pts[:, 0]), np.nanmean(pts[:, 1])
    half = max(np.ptp(pts[:, 0]), 1.6 * np.ptp(pts[:, 1])) / 2 + 45
    x0, x1 = max(0, cx - half), min(res[0], cx + half)
    y0, y1 = max(0, cy - half / 1.1), min(res[1], cy + half / 1.1)

    fig = plt.figure(figsize=(COL, 1.85))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1.3, 1], wspace=0.45, hspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img, interpolation="nearest")
    closed = np.vstack([path, path[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color="white", lw=2.4, alpha=0.9)
    ax.plot(closed[:, 0], closed[:, 1], color=STYLE["snn_phasemap"][1], lw=1.3, label="remembered path")
    ax.plot(*obs[i], "o", ms=5, color="white", mec="black", mew=0.7, label="$p(t)$")
    ax.plot(*(tr["pred"][hi][i] * res), "o", ms=5, color=STYLE["snn_phasemap"][1], mec="white", mew=0.9,
            label=r"$\hat p(t{+}100\,\mathrm{ms})$")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(INK)
    ax.set_title(f"{t[i] - tb:+.1f} s from the break", fontsize=7, loc="left", pad=2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=1, fontsize=6.5, handlelength=1.4,
              labelspacing=0.2, borderaxespad=0.2, markerscale=0.7)

    tt = t - tb
    s = (tt >= window[0]) & (tt <= window[1])
    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(tt[s], tr["gt_now"][s, 0] * res[0], color=TRUTH, lw=2.2, alpha=0.5, label="label")
    a1.plot(tt[s] + 0.1, tr["pred"][hi][s, 0] * res[0], color=STYLE["snn_phasemap"][1], lw=0.9, label="prediction")
    a1.set_ylabel("$x$ (px)", labelpad=1)
    a1.tick_params(labelbottom=False)
    a1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=6.5, ncol=2, handlelength=1.2,
              columnspacing=0.8, borderaxespad=0.1)
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    ratio = tr["score"] / np.where(tr["bar"] > 0, tr["bar"], np.nan)
    a2.plot(tt[s], ratio[s], color=STYLE["snn_phasemap"][1], lw=0.9)
    a2.axhline(1.0, color=INK, lw=0.6, ls=(0, (3, 2)))
    flags = alarm_times(tt, ratio)
    flags = flags[(flags >= window[0]) & (flags <= window[1])]
    if len(flags):
        a2.axvline(flags[0], color="#d03b3b", lw=0.8)
        a2.annotate("alarm", xy=(flags[0], 8), xytext=(3, 0), textcoords="offset points", fontsize=6.5,
                    color="#d03b3b", va="center")
    a2.set_yscale("log")
    a2.set_ylim(0.05, 20)
    a2.set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
    a2.minorticks_off()
    a2.set_ylabel("score/bar", labelpad=1)
    a2.set_xlabel("time from the break (s)", labelpad=1)
    for a in (a1, a2):
        a.axvline(0.0, color=INK, lw=0.8)
        a.axvline(t[i] - tb, color=MUTED, lw=0.5, ls=":")
    fig.align_ylabels([a1, a2])
    save(fig, name)


PENDULUM = ("small_01", "wide_02", "wide_break", "small_03", "wide_01", "small_break")


def is_pendulum(clip: str) -> bool:
    return clip.split("__")[0] in PENDULUM


# --- tables ----------------------------------------------------------------------------

TABLE_MEMORIES = ["snn_phasemap", "phasemap", "kalman", "harmonic", "constant_velocity"]


def _cell(v, nd=1) -> str:
    return "--" if not np.isfinite(v) else f"{v:.{nd}f}"


def table_block(rows, title: str, **sel) -> list[str]:
    """Rows of the results table for one clip group: median error at 50/100/200 ms and
    path error (medians over clips), AUC and latency (median over the break clips), false
    alarms per minute (mean over clips, so a single noisy clip still shows)."""
    out = [r"\multicolumn{8}{l}{\emph{" + title + r"}} \\"]
    for m in TABLE_MEMORIES:
        got = pick(rows, memory=m, **sel)
        if not got:
            continue
        with np.errstate(all="ignore"):
            err = {h: np.nanmedian([r["fde_px"] for r in got if r["horizon_s"] == h]) for h in (0.05, 0.1, 0.2)}
            one = [r for r in got if r["horizon_s"] == 0.1]
            paths = [r["path_median_px"] for r in one]
            path = np.nanmedian(paths) if np.isfinite(paths).any() else np.nan
            breaks = [r for r in one if np.isfinite(r["auc"])]
            auc = np.median([r["auc"] for r in breaks]) if breaks else np.nan
            lat = np.median([r["latency_s"] for r in breaks]) if breaks else np.nan
            fp = np.mean([r["fp_per_min"] for r in one])
        name = STYLE[m][0].replace(" (ours)", "").replace("Clock-and-map, arithmetic", "Clock-and-map, arith.")
        if m == "snn_phasemap":
            name = r"\textbf{" + name + "}"
        out.append(" & ".join([name, _cell(err[0.05]), _cell(err[0.1]), _cell(err[0.2]), _cell(path),
                               _cell(auc, 2), _cell(lat, 2), _cell(fp)]) + r" \\")
    return out


TABLE_HEAD = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
              r" & \multicolumn{3}{c}{Error at $h$ (px)} & Path & & Latency & False \\",
              r"\cmidrule(lr){2-4}",
              r"Memory & 50\,ms & 100\,ms & 200\,ms & (px) & AUC & (s) & alarms/min \\", r"\midrule"]
TABLE_FOOT = [r"\bottomrule", r"\end{tabular}"]


def tables(dev, held) -> str:
    pend = dict(input="snn", offset="as_is")
    main_table = (TABLE_HEAD + table_block(held, "Pendulum, held-out (3 clips, 1 break)", **pend) + [r"\midrule"]
                  + table_block(dev, "Pendulum, development (3 clips, 1 break)", setup="pendulum", **pend)
                  + TABLE_FOOT)
    other = (TABLE_HEAD + table_block(dev, "Fan and wall target, development (3 recordings, 1 with a deviation)",
                                      setup=lambda s: s in ("fan", "wall_target"), input="centroid",
                                      offset="removed") + TABLE_FOOT)
    return ("% Table: pendulum, full spiking pipeline, labels as-is\n" + "\n".join(main_table)
            + "\n\n% Table: other setups, classical centroid input, label offset removed\n" + "\n".join(other) + "\n")


def main() -> None:
    setup_style()
    dev = read_scores("development")
    held = read_scores("held_out")
    pend = dict(setup="pendulum", input="snn", offset="as_is")
    other = dict(setup=lambda s: s in ("fan", "wall_target"), input="centroid", offset="removed")
    fig_prediction_vs_horizon([(held, "(a) Pendulum, held-out", pend),
                               (dev, "(b) Pendulum, development", pend),
                               (dev, "(c) Fan and wall target, development", other)], "results_horizon")
    fig_break_trace("held_out", "small_break", "results_break")
    fig_break_trace("development", "wide_break", "results_break_dev")
    fig_paths("development", "wide_break", "results_paths")
    fig_localiser_cdf(["development", "held_out"])
    fig_learning(["development", "held_out"])
    (RUNS / "tables.tex").write_text(tables(dev, held), encoding="utf-8")
    print("wrote", RUNS / "tables.tex")


if __name__ == "__main__":
    sys.exit(main())

"""Raster panels for the illustrative Fig. 1 (`pipeline.tex`), from a real pendulum clip:
the 5 ms event frame the localiser sees, and the same moment with the memory's
remembered path, current position and 100 ms prediction drawn over it.

    python figures/architecture/make_assets.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import _thesis_path  # noqa: F401,E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from trajmem.data import load_recording  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "assets"
CLIP = ROOT / "corpus/real/pendulum/wide_break"
TRACE = ROOT / "runs/paper/development/traces/wide_break__snn_phasemap__snn.npz"
BEFORE_BREAK_S = 2.66         # the moment shown, this long before the marked break
WINDOW_US = 5000
CROP = (60, 600, 40, 400)     # x0, x1, y0, y1 in pixels
ON, OFF, BG = "#ff6b5b", "#4fb3ff", "#0d0d10"
OURS = "#c98500"


def frame_image(events, t0_us, t1_us, res):
    sel = (events["timestamp"] >= t0_us) & (events["timestamp"] < t1_us)
    e = events[sel]
    img = np.zeros((res[1], res[0], 3))
    img[:] = matplotlib.colors.to_rgb(BG)
    for pol, colour in ((1, ON), (0, OFF)):
        m = e["polarity"] == pol
        img[e["y"][m], e["x"][m]] = matplotlib.colors.to_rgb(colour)
    return img, len(e)


def crop(img):
    x0, x1, y0, y1 = CROP
    return img[y0:y1, x0:x1]


def save_image(img, name, overlay=None):
    x0, x1, y0, y1 = CROP
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="nearest", extent=(x0, x1, y1, y0))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    if overlay:
        overlay(ax)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    clip = load_recording(CLIP)
    tr = np.load(TRACE)
    res = tr["resolution"]
    t_show = float(tr["deviation_times"][0]) - BEFORE_BREAK_S
    i = int(np.searchsorted(tr["t"], t_show))
    t0_us = int(clip.events["timestamp"][0] + (tr["t"][i] * 1e6 - WINDOW_US / 2))
    img, n = frame_image(clip.events, t0_us, t0_us + WINDOW_US, res)
    print(f"frame at {tr['t'][i]:.3f} s: {n} events")
    # dilate one pixel so single events survive print scaling
    grown = img.copy()
    lit = (img != matplotlib.colors.to_rgb(BG)).any(axis=2)
    for dy, dx in ((0, 1), (1, 0), (1, 1)):
        shifted = np.roll(np.roll(lit, dy, 0), dx, 1) & ~lit
        grown[shifted] = np.roll(np.roll(img, dy, 0), dx, 1)[shifted]
    save_image(crop(grown), "event_frame")

    faded = crop(grown) * 0.35 + 0.65 * np.array(matplotlib.colors.to_rgb(BG))
    cycle = tr["cycles"][i].astype(float) * res
    now = tr["obs"][i] * res
    hi = list(tr["horizons"]).index(0.1)
    pred = tr["pred"][hi][i] * res

    def overlay(ax):
        closed = np.vstack([cycle, cycle[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="white", lw=3.6, alpha=0.9, solid_capstyle="round")
        ax.plot(closed[:, 0], closed[:, 1], color=OURS, lw=2.0, solid_capstyle="round")
        ax.plot(*now, "o", ms=8, color="white", mec="black", mew=1.0)
        ax.plot(*pred, "o", ms=9, color=OURS, mec="white", mew=1.4)
        ax.annotate("", xy=pred, xytext=now,
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6, shrinkA=6, shrinkB=6,
                                    connectionstyle="arc3,rad=0.15"))

    save_image(faded, "prediction_frame", overlay)
    score_strip(tr)


def score_strip(tr, window=(-3.0, 3.0)) -> None:
    """The deviation score over its alarm bar around the break, as a bare strip."""
    tb = float(tr["deviation_times"][0])
    t = tr["t"] - tb
    s = (t >= window[0]) & (t <= window[1])
    ratio = tr["score"] / np.where(tr["bar"] > 0, tr["bar"], np.nan)
    fig = plt.figure(figsize=(1.35, 0.42), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.fill_between(t[s], 1.0, np.clip(ratio[s], 1.0, None), color="#d03b3b", alpha=0.25, lw=0)
    ax.plot(t[s], ratio[s], color=OURS, lw=1.0)
    ax.axhline(1.0, color="#0b0b0b", lw=0.6, ls=(0, (3, 2)))
    ax.axvline(0.0, color="#0b0b0b", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.06, 6)
    ax.set_xlim(*window)
    ax.axis("off")
    fig.savefig(OUT / "score_strip.png", dpi=300, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()

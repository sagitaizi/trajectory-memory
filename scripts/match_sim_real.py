"""Compare a simulated clip with the real clip it copies, to characterise v2e.

    python scripts/match_sim_real.py corpus/real/fan/fan_brush_slow_02 --out runs/match

Fits an ellipse to the real clip's hand-labels, simulates a blob on that same path,
and prints four numbers for each: event rate near the target, ON fraction, how wide
the event cloud around the target is, and the noise rate far from it. A montage PNG
shows both streams over the same instants. Adjust `sim` in params.yaml and rerun
until the numbers agree. Both clips are saved next to it; watch either with
`python scripts/replay_gt.py runs/match/sim.npz`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from scripts.mark_anchors import accumulate  # noqa: E402
from trajmem.data import Clip, load_recording, load_sim, save_clip, sim_params  # noqa: E402
from trajmem.trajectories import fit_ellipse  # noqa: E402

STATS = ("target_rate", "on_fraction", "footprint_px", "noise_rate_per_px")


def event_stats(clip: Clip, resolution, radius_px: float) -> dict:
    """Rate, polarity and spread of events within `radius_px` of gt; noise rate outside.

    Events at instants where gt is unknown (NaN) are left out of both.
    """
    w, h = resolution
    ev = clip.events
    t = ev["timestamp"] / 1e6
    target = np.atleast_2d(clip.gt(t)) * np.array([w, h], dtype=float)
    d = np.hypot(ev["x"] - target[:, 0], ev["y"] - target[:, 1])
    known = np.isfinite(d)
    near = known & (d <= radius_px)
    far = known & ~near

    seconds = clip.duration_us / 1e6
    return {
        "target_rate": float(near.sum() / seconds),
        "on_fraction": float(ev["polarity"][near].mean()) if near.any() else float("nan"),
        "footprint_px": float(np.median(d[near])) if near.any() else float("nan"),
        "noise_rate_per_px": float(far.sum() / seconds / (w * h - np.pi * radius_px ** 2)),
    }


def _montage(real: Clip, sim: Clip, resolution, instants, window_s, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, h = resolution
    fig, axes = plt.subplots(2, len(instants), figsize=(3.2 * len(instants), 5.2))
    for row, (name, clip) in enumerate((("real", real), ("sim", sim))):
        for ax, t0 in zip(axes[row], instants):
            ax.imshow(accumulate(clip.events, t0, window_s, (w, h)), cmap="gray", vmin=0, vmax=255)
            gx, gy = np.asarray(clip.gt(t0 + window_s / 2)) * (w, h)
            ax.plot(gx, gy, "o", mfc="none", mec="lime", ms=14, mew=1.5)
            ax.set_title(f"{name}  {t0:.2f}s", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _print_table(stats: dict[str, dict]) -> None:
    names = list(stats)
    print(f"{'':18}" + "".join(f"{n:>14}" for n in names))
    for key in STATS:
        print(f"{key:18}" + "".join(f"{stats[n][key]:>14.4g}" for n in names))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clip")
    p.add_argument("--params", default="params.yaml")
    p.add_argument("--out", default="runs/match")
    p.add_argument("--duration", type=float, default=5.0, help="seconds compared, from t=0")
    p.add_argument("--fps", type=int, default=500)
    p.add_argument("--radius", type=float, default=60.0, help="px around gt that count as target")
    p.add_argument("--window", type=float, default=0.02, help="montage accumulation window (s)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    real = load_recording(args.clip)
    if "labels" not in real.meta:
        raise SystemExit("the real clip needs hand-labels to fit a path to")
    resolution = real.meta["resolution"]
    keep = real.events["timestamp"] <= int(args.duration * 1e6)
    real.events, real.duration_us = real.events[keep], int(args.duration * 1e6)

    spec = fit_ellipse(real.meta["labels"])
    sim_cfg = sim_params(args.params)
    sim_cfg["duration_s"], sim_cfg["fps"] = args.duration, args.fps
    sim = load_sim(spec, sim_cfg, seed=args.seed, distort=False)

    stats = {"real": event_stats(real, resolution, args.radius),
             "sim": event_stats(sim, resolution, args.radius)}
    _print_table(stats)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    instants = np.linspace(0, args.duration - args.window, 4)
    _montage(real, sim, resolution, instants, args.window, out / "montage.png")
    save_clip(real, out / "real.npz")
    save_clip(sim, out / "sim.npz")
    (out / "stats.json").write_text(json.dumps(
        {"clip": args.clip, "spec": str(spec), "sim": sim_cfg, "stats": stats}, indent=2))
    print(f"wrote montage.png, stats.json, real.npz and sim.npz to {out}/")


if __name__ == "__main__":
    main()

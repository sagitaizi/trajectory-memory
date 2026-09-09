"""Eyeball a simulated clip: scatter every event over the expected path, plus a
time-sliced montage. Debug tool, not part of the pipeline.

    python scripts/preview_sim.py --out runs/preview
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from trajmem.simulate import _path_pixels, load_intrinsics, simulate  # noqa: E402
from trajmem.trajectories import Deviation, TrajectorySpec  # noqa: E402

ON, OFF = "#c1121f", "#2a6f97"


def _cases():
    circle = TrajectorySpec(shape="circle", size=(0.32, 0.32), period_s=0.5, center=(0.5, 0.5))
    fig8 = TrajectorySpec(
        shape="figure8", size=(0.34, 0.22), period_s=0.5, center=(0.5, 0.5),
        deviations=[Deviation(at_t=0.7, kind="speed_change", params={"factor": 1.8})],
    )
    return {"circle": circle, "figure8_dev": fig8}


def _expected_path(spec, sim_cfg, intr):
    w, h = sim_cfg["resolution"]
    t = np.linspace(0.0, sim_cfg["duration_s"], 600)
    return _path_pixels(spec, t, (w, h), intr), t


def _scatter(ax, ev, path_px, w, h, subsample=200_000):
    if len(ev) > subsample:
        ev = ev[np.random.default_rng(0).choice(len(ev), subsample, replace=False)]
    on = ev["polarity"] == 1
    ax.scatter(ev["x"][on], ev["y"][on], s=1, c=ON, alpha=0.25, linewidths=0, label="ON")
    ax.scatter(ev["x"][~on], ev["y"][~on], s=1, c=OFF, alpha=0.25, linewidths=0, label="OFF")
    ax.plot(path_px[:, 0], path_px[:, 1], "k-", lw=1.2, label="expected path")
    ax.plot(*path_px[0], "ko", ms=5)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")


def _preview(name, spec, sim_cfg, camera_cfg, seed, out_dir):
    clip = simulate(spec, camera_cfg=camera_cfg, sim_cfg=sim_cfg, seed=seed)
    ev = clip.events
    w, h = sim_cfg["resolution"]
    intr = load_intrinsics(camera_cfg) if camera_cfg is not None else None
    path_px, _ = _expected_path(spec, sim_cfg, intr)

    span_s = max((ev["timestamp"].max() - ev["timestamp"].min()) / 1e6, 1e-9)
    print(f"[{name}] events={len(ev):,}  rate={len(ev)/span_s/1e3:,.0f}k/s  "
          f"ON/OFF={(ev['polarity']==1).sum():,}/{(ev['polarity']==0).sum():,}  "
          f"x={ev['x'].min()}..{ev['x'].max()} y={ev['y'].min()}..{ev['y'].max()}")

    fig, ax = plt.subplots(figsize=(6, 6 * h / w))
    _scatter(ax, ev, path_px, w, h)
    ax.legend(loc="upper right", markerscale=6, framealpha=0.9)
    ax.set_title(f"{name}: all events + expected path"
                 + ("  (lens-distorted)" if intr is not None else ""))
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_scatter.png", dpi=120)
    plt.close(fig)

    panels = 8
    edges = np.linspace(0, sim_cfg["duration_s"], panels + 1)
    fig, axes = plt.subplots(2, panels // 2, figsize=(3 * panels // 2, 6 * h / w))
    for i, ax in enumerate(axes.ravel()):
        lo, hi = edges[i] * 1e6, edges[i + 1] * 1e6
        win = ev[(ev["timestamp"] >= lo) & (ev["timestamp"] < hi)]
        _scatter(ax, win, path_px, w, h)
        mid = _path_pixels(spec, [0.5 * (edges[i] + edges[i + 1])], (w, h), intr)[0]
        ax.plot(*mid, "o", mfc="none", mec="lime", ms=12, mew=2)
        ax.set_title(f"{edges[i]:.2f}-{edges[i+1]:.2f}s  ({len(win):,} ev)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{name}: events per time window (green = ground-truth position)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_montage.png", dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params", default="params.yaml")
    p.add_argument("--out", default="runs/preview")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=1.0, help="override clip length (s)")
    p.add_argument("--fps", type=int, default=600, help="override render fps")
    p.add_argument("--no-distortion", action="store_true", help="skip the lens model")
    args = p.parse_args()

    sim_cfg = yaml.safe_load(open(args.params))["sim"]
    sim_cfg["duration_s"] = args.duration
    sim_cfg["fps"] = args.fps
    camera_cfg = None if args.no_distortion else {}

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in _cases().items():
        _preview(name, spec, sim_cfg, camera_cfg, args.seed, out_dir)
    print(f"wrote PNGs to {out_dir}/")


if __name__ == "__main__":
    main()

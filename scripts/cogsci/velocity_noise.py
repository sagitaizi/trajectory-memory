"""How noisy is velocity from the localiser's positions? Against a spline through the labels."""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
RES = np.array([640.0, 480.0])
DT = 0.005
RUNS = [("development", "small_01", "pendulum/small_01"),
        ("held_out", "small_03", "pendulum/small_03"),
        ("held_out", "wide_01", "pendulum/wide_01")]
TAUS = [0.0, 0.01, 0.02, 0.05, 0.1]


def labels(rel):
    p = ROOT / "corpus/real" / rel / f"{Path(rel).name}.labels.csv"
    with open(p, newline="") as fh:
        a = np.array([[float(r["t_s"]), float(r["x"]), float(r["y"])] for r in csv.DictReader(fh)])
    return a[:, 0], a[:, 1:] * RES


def ema(v, tau):
    if tau == 0:
        return v
    a = DT / tau
    out, s = np.empty_like(v), np.array([np.nan, np.nan])
    for i, x in enumerate(v):
        if np.isfinite(x).all():
            s = x if not np.isfinite(s).all() else s + a * (x - s)
        out[i] = s
    return out


print(f"{'run':10s} {'front':8s} {'tau ms':>6s} {'speed px/s':>10s} {'err px/s':>9s} {'err/speed':>9s} {'lag-free err/speed':>18s}")
for split, name, rel in RUNS:
    tl, xy = labels(rel)
    spl = CubicSpline(tl, xy, axis=0)
    for front in ("snn", "centroid"):
        txy = np.load(ROOT / f"runs/paper/{split}/positions/{name}__{front}.npz")["txy"]
        t, p = txy[:, 0], txy[:, 1:] * RES
        inside = (t > tl[0] + 0.2) & (t < tl[-1] - 0.2)
        v_raw = np.vstack([[np.nan, np.nan], np.diff(p, axis=0) / DT])
        v_ref = spl(t, 1)
        speed = np.sqrt(np.nanmean(np.sum(v_ref[inside] ** 2, 1)))
        for tau in TAUS:
            v = ema(v_raw, tau)
            ok = inside & np.isfinite(v).all(1)
            err = np.sqrt(np.mean(np.sum((v[ok] - v_ref[ok]) ** 2, 1)))
            # the same smoothing applied to the reference: what is noise, not the smoothing's lag
            v_ref_s = ema(np.where(np.isfinite(v_raw), v_ref, np.nan), tau)
            lf = np.sqrt(np.mean(np.sum((v[ok] - v_ref_s[ok]) ** 2, 1)))
            print(f"{name:10s} {front:8s} {tau*1e3:6.0f} {speed:10.0f} {err:9.0f} {err/speed:9.2f} {lf/speed:18.2f}")
        print(f"{'':10s} {front:8s} missing frames: {np.mean(~np.isfinite(p[inside]).all(1)):.1%}")

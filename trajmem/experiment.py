"""Harness: walk a clip through a memory, keep the per-step trace, score it.

`evaluate_clip` is the one loop every method goes through -- baselines now, the
SNN once it exists -- so the comparison is on identical inputs. The corpus-wide
pretrain -> freeze -> evaluate run is added with Phase C.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Clip
from .frontend import to_position
from .metrics import deviation_roc, lock_on_time, path_shape_error, prediction_error
from .trajectories import _harmonic_fit, search_period

SETS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "sets.yaml"
SNN_CHECKPOINT = Path(__file__).resolve().parent.parent / "runs" / "memory" / "snn.pt"
LOCALISER_CHECKPOINT = Path(__file__).resolve().parent.parent / "runs" / "localiser" / "snn.pt"
FRAME_DOWNSAMPLE = 8
CYCLE_POINTS = 64                  # samples per cycle when comparing a remembered path to the truth


@dataclass
class Trace:
    """What a memory did on one clip, step by step. Positions are normalised."""
    t: np.ndarray                  # window centres, s
    obs: np.ndarray                # (N, 2) what the frontend saw; NaN = target unseen
    pred: np.ndarray               # (N, 2) prediction for t + horizon
    gt_ahead: np.ndarray           # (N, 2) truth at t + horizon; NaN where unknown
    score: np.ndarray              # (N,) deviation score
    horizon_s: float
    window_us: int
    period: np.ndarray = None      # (N,) the memory's period, s; NaN = none yet
    cycles: np.ndarray = None      # (N, CYCLE_POINTS, 2) the memory's path over one true period; NaN = none yet
    blank: tuple | None = None     # (t0, t1) where the observations were hidden from the memory


def make_memory(name: str, dt_s: float, **params):
    """A fresh memory by name. The baselines take their own params (`warmup_s`, ...); the
    SNN is loaded from `checkpoint` (default runs/memory/snn.pt, from train_memory.py;
    the checkpoint's `arch` picks the class) and ignores the rest."""
    from .baseline import HarmonicFit, PeriodicKalman

    if name == "snn":
        import torch

        from .snn import SpikingMemory
        from .snn_lmu import LmuMemory

        path = params.get("checkpoint") or SNN_CHECKPOINT
        arch = torch.load(Path(path), map_location="cpu", weights_only=False).get("arch", "two_layer")
        return {"two_layer": SpikingMemory, "lmu": LmuMemory}[arch].load(path, params.get("device"), dt_s)
    from .phasemap import PhaseMap
    from .snn_phasemap import SpikingPhaseMap

    kinds = {"kalman": PeriodicKalman, "harmonic": HarmonicFit, "phasemap": PhaseMap, "snn_phasemap": SpikingPhaseMap}
    if name not in kinds:
        raise ValueError(f"unknown memory {name!r}; one of {sorted(kinds) + ['snn']}")
    params = {k: v for k, v in params.items() if k not in ("checkpoint", "device")}
    if name in ("phasemap", "snn_phasemap"):
        params.pop("warmup_s", None)
    return kinds[name](dt_s=dt_s, **params)


def clip_period(clip: Clip) -> float:
    """The path's period: exact from the spec on simulated clips, searched over the truth
    before any break otherwise."""
    if "spec" in clip.meta:
        return float(clip.meta["spec"].period_s)
    t_break = min(clip.deviation_times) if clip.deviation_times else clip.duration_us / 1e6
    t = np.arange(0.0, t_break, 0.05)
    xy = np.atleast_2d(clip.gt(t))
    ok = np.isfinite(xy).all(axis=1)
    return search_period(t[ok], xy[ok])


def reference_cycle(clip: Clip, n: int = CYCLE_POINTS, n_harmonics: int = 3,
                    period_s: float | None = None) -> np.ndarray:
    """One cycle of the true path as `n` points, from a harmonic fit of the truth before
    any break -- the same model the path head and the baselines carry."""
    period_s = clip_period(clip) if period_s is None else period_s
    t_break = min(clip.deviation_times) if clip.deviation_times else clip.duration_us / 1e6
    t = np.arange(0.0, t_break, 0.01)
    xy = np.atleast_2d(clip.gt(t))
    ok = np.isfinite(xy).all(axis=1)
    coef, _ = _harmonic_fit(t[ok], xy[ok], period_s, n_harmonics)
    phase = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cols = [np.ones_like(phase)]
    for k in range(1, n_harmonics + 1):
        cols += [np.cos(k * phase), np.sin(k * phase)]
    return np.column_stack(cols) @ coef


def make_localiser(name: str, checkpoint=None):
    """None for the classical centroid (`frontend.to_position`), or a frame localiser:
    `frame_centroid` (the reference) or `snn` (loaded from `checkpoint`)."""
    if name in (None, "centroid"):
        return None
    if name == "frame_centroid":
        from .localise import FrameCentroid

        return FrameCentroid()
    if name == "snn":
        from .snn_localise import SpikingLocaliser

        return SpikingLocaliser.load(checkpoint or LOCALISER_CHECKPOINT)
    raise ValueError(f"unknown localiser {name!r}; centroid, frame_centroid or snn")


def positions(clip: Clip, window_us: int, localiser=None):
    """(t, x, y) per window: the classical centroid, or a frame localiser run over the
    count frames (NaN where it saw nothing)."""
    if localiser is None:
        yield from to_position(clip, window_us)
        return
    from .frontend import to_frames

    localiser.reset()
    for t0, frame in to_frames(clip, window_us, kind="count", downsample=FRAME_DOWNSAMPLE):
        x, y = localiser.locate(frame)
        yield (t0 + window_us / 2) / 1e6, float(x), float(y)


def evaluate_clip(memory, clip: Clip, window_us: int, horizon_s: float, blank=None, localiser=None) -> Trace:
    """Stream the clip through the memory. With `blank` = (t0, t1) the observations in
    that window are hidden (NaN), the test of a memory that runs on its own. With a
    `localiser` the positions come from it instead of the classical centroid."""
    memory.reset()
    period_s = clip_period(clip) if clip.gt is not None else np.nan
    rows, periods, cycles = [], [], []
    for t, x, y in positions(clip, window_us, localiser):
        if blank is not None and blank[0] <= t < blank[1]:
            x = y = np.nan
        memory.observe(x, y)
        rows.append((t, x, y, *memory.predict(horizon_s), memory.deviation_score()))
        periods.append(memory.period())
        cycles.append(remembered_path(memory, period_s))
    a = np.array(rows, dtype=float)
    ahead = a[:, 0] + horizon_s
    gt = np.full((len(a), 2), np.nan)
    inside = ahead <= clip.duration_us / 1e6
    if clip.gt is not None and inside.any():
        gt[inside] = np.atleast_2d(clip.gt(ahead[inside]))
    return Trace(t=a[:, 0], obs=a[:, 1:3], pred=a[:, 3:5], gt_ahead=gt, score=a[:, 5],
                 horizon_s=horizon_s, window_us=window_us, period=np.array(periods), cycles=np.array(cycles),
                 blank=blank)


def remembered_path(memory, period_s: float, n: int = CYCLE_POINTS) -> np.ndarray:
    """The memory's path over one *true* period as `n` points, so the shape is scored
    apart from the period: a memory whose period is 2T covers half its cycle here."""
    p = memory.period()
    if not (np.isfinite(p) and np.isfinite(period_s)) or p <= 0:
        return np.full((n, 2), np.nan)
    points = memory.path_points(np.arange(n) / n * period_s / p)
    return np.full((n, 2), np.nan) if points is None else np.asarray(points, dtype=float)


def _blank_px(trace: Trace, err_all: np.ndarray) -> float:
    """Median prediction error on the steps whose observation was hidden."""
    if trace.blank is None:
        return np.nan
    inside = (trace.t >= trace.blank[0]) & (trace.t < trace.blank[1])
    e = err_all[inside]
    return float(np.nanmedian(e)) if np.isfinite(e).any() else np.nan


def label_offset(trace: Trace, clip: Clip, settle_s: float = 0.0) -> np.ndarray:
    """Median (label - measured position) over the steady part, normalised units. On hand-
    labelled clips the marks sit on a fixed point of the object that is not the event
    centroid (the pendulum's brush centre vs its bright string end); this is that constant."""
    t_break = min(clip.deviation_times) if clip.deviation_times else np.inf
    steady = (trace.t >= settle_s) & (trace.t < t_break)
    gt_now = np.full_like(trace.obs, np.nan)
    if clip.gt is not None and steady.any():
        gt_now[steady] = np.atleast_2d(clip.gt(trace.t[steady]))
    d = gt_now - trace.obs
    d = d[np.isfinite(d).all(axis=1)]
    return np.median(d, axis=0) if len(d) else np.zeros(2)


def score_trace(trace: Trace, clip: Clip, tol_px: float, settle_s: float = 0.0,
                threshold=None, subtract_offset: bool = False) -> dict:
    """Prediction error (px, on steps before any break and after `settle_s`), path-shape
    error (px, same steps; `last` = median over the final cycle before the break), the
    memory's period over the true one, lock-on times (s, from the clip's start), and the
    deviation-detection numbers. With
    `subtract_offset` the clip's constant label-vs-measurement offset is removed from
    the prediction first (reported as `offset_px`), so the error is about prediction,
    not about which point of the object the labels mark."""
    scale = np.array(clip.meta["resolution"], dtype=float)
    offset = label_offset(trace, clip, settle_s) if subtract_offset else np.zeros(2)
    pred = trace.pred + offset
    err_all = prediction_error(pred * scale, trace.gt_ahead * scale)["errors"]
    t_break = min(clip.deviation_times) if clip.deviation_times else np.inf
    steady = (trace.t >= settle_s) & (trace.t + trace.horizon_s < t_break)
    err = prediction_error(pred[steady] * scale, trace.gt_ahead[steady] * scale)
    err.pop("errors")
    before_break = trace.t + trace.horizon_s < t_break
    dt = trace.window_us / 1e6
    period_s = clip_period(clip)
    reference = reference_cycle(clip, period_s=period_s) * scale
    path = np.array([path_shape_error((c + offset) * scale, reference) for c in trace.cycles])
    on_path = (trace.t < t_break) & (trace.t >= min(t_break, trace.t[-1] + dt) - period_s)
    ratio = trace.period[steady] / period_s
    return {
        "error_px": err,
        "path_px": {"median": float(np.nanmedian(path[steady])) if np.isfinite(path[steady]).any() else np.nan,
                    "last": float(np.nanmedian(path[on_path])) if np.isfinite(path[on_path]).any() else np.nan},
        "period_ratio": float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else np.nan,
        "blank_px": _blank_px(trace, err_all),
        "offset_px": (offset * scale).tolist(),
        "lock_on_s": lock_on_time(err_all[before_break], tol_px, dt),
        "path_lock_on_s": lock_on_time(path[trace.t < t_break], tol_px, dt),
        "deviation": deviation_roc(trace.score, trace.t, clip.deviation_times, threshold=threshold),
        "n_steps": int(len(trace.t)),
        "unseen_fraction": float(np.isnan(trace.obs[:, 0]).mean()),
    }


# --- clip sets --------------------------------------------------------------------

def load_set(name: str, path=None) -> list[dict]:
    """Entries of a named clip set from corpus/sets.yaml: clip path, display name, slice.
    An entry `{glob: pattern}` stands for every clip the pattern matches."""
    import yaml

    sets = yaml.safe_load(Path(path or SETS_PATH).read_text(encoding="utf-8"))
    if name not in sets:
        raise ValueError(f"no clip set {name!r}; one of {sorted(sets)}")
    out = []
    for e in sets[name] or []:
        if "glob" in e:                                   # every matching clip, in name order
            root = Path(path or SETS_PATH).resolve().parent.parent      # corpus/sets.yaml -> repo
            found = sorted(root.glob(e["glob"]))
            if not found:
                raise ValueError(f"set {name!r}: {e['glob']!r} matches no clip under {root}")
            out += [{"clip": str(c), "name": c.stem, "slice": None} for c in found]
            continue
        window = e.get("slice")
        out.append({"clip": e["clip"], "name": e.get("name", Path(e["clip"]).name),
                    "slice": None if window is None else (window[0], window[1])})
    return out


def open_set(name: str, path=None) -> list[tuple[str, Clip]]:
    """The set's clips, loaded and sliced; entries without ground truth yet are skipped."""
    from .data import load_clip, load_recording, slice_clip

    clips = []
    for e in load_set(name, path):
        p = Path(e["clip"])
        clip = load_clip(p) if p.suffix == ".npz" else load_recording(p)
        if clip.gt is None:
            continue
        if e["slice"] is not None:
            t0, t1 = e["slice"]
            clip = slice_clip(clip, t0 or 0.0, clip.duration_us / 1e6 if t1 is None else t1)
        clips.append((e["name"], clip))
    return clips


def run_set(make_memory_fn, clips, window_us: int, horizon_s: float, tol_px: float,
            settle_s: float, threshold=None, subtract_offset: bool = False, localiser=None):
    """Score a fresh memory on every clip; return the per-clip rows and a pooled row
    (medians of the per-clip numbers; detection numbers over the break clips only)."""
    rows = []
    for name, clip in clips:
        trace = evaluate_clip(make_memory_fn(), clip, window_us, horizon_s, localiser=localiser)
        rows.append({"name": name, **score_trace(trace, clip, tol_px, settle_s, threshold, subtract_offset)})
    return rows, pool(rows)


def pool(rows: list[dict]) -> dict:
    """Medians of the per-clip numbers. `inf` ("never flagged", "never locked on") stays
    in, so a method that fails on half the clips pools to inf rather than to the hits'
    median; the misses are counted as well."""
    def med(values):
        v = [x for x in values if not np.isnan(x)]
        return float(np.median(v)) if v else np.nan

    breaks = [r for r in rows if np.isfinite(r["deviation"]["auc"])]
    latencies = [r["deviation"]["latency_s"] for r in breaks]
    locks = [r["lock_on_s"] for r in rows]
    return {
        "name": "pooled", "n_clips": len(rows),
        "error_px": {"median": med(r["error_px"]["median"] for r in rows),
                     "iqr": med(r["error_px"]["iqr"] for r in rows)},
        "path_px": {"median": med(r["path_px"]["median"] for r in rows),
                    "last": med(r["path_px"]["last"] for r in rows)},
        "period_ratio": med(r["period_ratio"] for r in rows),
        "blank_px": med(r["blank_px"] for r in rows),
        "lock_on_s": med(locks),
        "never_locked": int(sum(np.isinf(x) for x in locks)),
        "path_lock_on_s": med(r["path_lock_on_s"] for r in rows),
        "deviation": {"auc": med(r["deviation"]["auc"] for r in breaks),
                      "latency_s": med(latencies),
                      "missed": int(sum(np.isinf(x) for x in latencies)),
                      "fp_per_min": med(r["deviation"]["fp_per_min"] for r in rows)},
        "unseen_fraction": med(r["unseen_fraction"] for r in rows),
    }

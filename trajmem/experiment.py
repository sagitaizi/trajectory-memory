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
from .metrics import deviation_roc, lock_on_time, prediction_error

SETS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "sets.yaml"
SNN_CHECKPOINT = Path(__file__).resolve().parent.parent / "runs" / "memory" / "snn.pt"


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


def make_memory(name: str, dt_s: float, **params):
    """A fresh memory by name. The baselines take their own params (`warmup_s`, ...); the
    SNN is loaded from `checkpoint` (default runs/memory/snn.pt, from train_memory.py)
    and ignores the rest."""
    from .baseline import HarmonicFit, PeriodicKalman

    if name == "snn":
        from .snn import SpikingMemory

        return SpikingMemory.load(params.get("checkpoint") or SNN_CHECKPOINT, params.get("device"), dt_s)
    kinds = {"kalman": PeriodicKalman, "harmonic": HarmonicFit}
    if name not in kinds:
        raise ValueError(f"unknown memory {name!r}; one of {sorted(kinds) + ['snn']}")
    params = {k: v for k, v in params.items() if k not in ("checkpoint", "device")}
    return kinds[name](dt_s=dt_s, **params)


def evaluate_clip(memory, clip: Clip, window_us: int, horizon_s: float) -> Trace:
    memory.reset()
    rows = []
    for t, x, y in to_position(clip, window_us):
        memory.observe(x, y)
        rows.append((t, x, y, *memory.predict(horizon_s), memory.deviation_score()))
    a = np.array(rows, dtype=float)
    ahead = a[:, 0] + horizon_s
    gt = np.full((len(a), 2), np.nan)
    inside = ahead <= clip.duration_us / 1e6
    if clip.gt is not None and inside.any():
        gt[inside] = np.atleast_2d(clip.gt(ahead[inside]))
    return Trace(t=a[:, 0], obs=a[:, 1:3], pred=a[:, 3:5], gt_ahead=gt, score=a[:, 5],
                 horizon_s=horizon_s, window_us=window_us)


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
                threshold: float | None = None, subtract_offset: bool = False) -> dict:
    """Prediction error (px, on steps before any break and after `settle_s`), lock-on
    time (s, from the clip's start), and the deviation-detection numbers. With
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
    return {
        "error_px": err,
        "offset_px": (offset * scale).tolist(),
        "lock_on_s": lock_on_time(err_all[before_break], tol_px, trace.window_us / 1e6),
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
            settle_s: float, threshold: float | None = None, subtract_offset: bool = False):
    """Score a fresh memory on every clip; return the per-clip rows and a pooled row
    (medians of the per-clip numbers; detection numbers over the break clips only)."""
    rows = []
    for name, clip in clips:
        trace = evaluate_clip(make_memory_fn(), clip, window_us, horizon_s)
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
        "lock_on_s": med(locks),
        "never_locked": int(sum(np.isinf(x) for x in locks)),
        "deviation": {"auc": med(r["deviation"]["auc"] for r in breaks),
                      "latency_s": med(latencies),
                      "missed": int(sum(np.isinf(x) for x in latencies)),
                      "fp_per_min": med(r["deviation"]["fp_per_min"] for r in rows)},
        "unseen_fraction": med(r["unseen_fraction"] for r in rows),
    }

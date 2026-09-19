# Implementation Plan

Architecture and module design. Read `PLAN.md` for phases and status.

## Repo layout

```
trajectory-memory/
  _thesis_path.py        # prepend D:\Projects\Thesis to sys.path
  conftest.py            # import _thesis_path so pytest sees the main repo
  requirements.txt
  params.yaml            # sim + model params
  trajmem/
    __init__.py
    data.py
    trajectories.py
    simulate.py
    frontend.py
    model.py
    snn.py                   Stage-1 memory: place cells -> fast LIF <-> adaptive LIF -> readouts
    augment.py               track-level augmentation for pretraining (flips, shift, scale, stretch)
    localise.py              Stage-1 localiser harness: FrameSet, evaluate_localiser, FrameCentroid
    baseline.py
    metrics.py
    experiment.py
  scripts/
    make_sim_dataset.py
    run_experiment.py        CLI: one clip or --set; scorecard table + runs/results/*.csv
    check_localiser.py       a localiser (classical centroid, or a frame localiser) vs hand-labels
    match_sim_real.py        sim-vs-real statistics on a labelled clip (v2e calibration)
    make_tracks.py           corpus/sim/tracks.npz: measured + true positions per window
    make_frames.py           corpus/sim/frames_<d>x_<w>us/: uint8 count images per window
    train_memory.py          pretrain snn.SpikingMemory on tracks.npz -> runs/memory/snn.pt
    corpus_summary.py        verify the corpus, print the §IV-A table
    replay_gt.py             player; --model draws a memory's output live
  tests/
  corpus/                # recorded .aedat4 + generated sim clips + GT (gitignored)
  docs/
```

## Reuse from the main thesis repo

Imported, not copied, via `_thesis_path.py`:

- `recording.player` — deterministic `.aedat4` replay → event arrays.
- `camera.calibration` — `CameraGeometry`, distortion model, `px_per_rad`.
- `pipeline.accumulator`, `pipeline.time_surface` — event → frame representations.

The main repo's `config` resolves its calibration and `params.yaml` relative to its own
location, so imports work with only `sys.path` set — no packaging needed.

## Data model

`Clip` (from `data.py`) — the one object everything else consumes:

| Field | Meaning |
|---|---|
| `events` | structured array `(x, y, t, polarity)` |
| `duration_us` | clip length |
| `gt(t) -> (x, y)` | true target position, normalised image coords; `None` for unlabelled real clips |
| `deviation_times` | list of timestamps where the path changes; empty if none |
| `source` | `"sim"` or a recording path |
| `meta` | trajectory spec (sim) or motion mechanism (real) |

## Modules

### `trajectories.py`
`TrajectorySpec` — shape (`circle`, `ellipse`, `figure8`, `lissajous`), size, period, centre,
plus zero or more `Deviation(at_t, kind, params)` (`kind` ∈ shrink, speed-change, drift,
switch-shape). `sample(spec, t) -> (x, y)`. Pure NumPy.

### `simulate.py`
`simulate(spec, camera_cfg) -> Clip`. Renders a blob on the sampled path to a high-rate frame
stack, runs v2e with the DVXplorer's resolution, lens distortion (from the main repo's
calibration), contrast threshold and noise. Ground truth is the spec, exact.

### `data.py`
`load_recording(path) -> Clip` (via `recording.player`), `load_sim(spec) -> Clip` (via
`simulate`). Hand-labelled real clips: `attach_labels(clip, points)` interpolates sparse
`(t, x, y)` marks into `gt`.

### `frontend.py` ✅
`windows(clip, window_us)` — `(t_start_us, events)` per window, one search for all edges.
`to_frames(clip, window_us, kind, downsample, tau_us) -> iterator[(t_us, frame)]` — `count`:
ON/OFF count image `(2, H, W)` float32, block-summed by `downsample`; `surface`: the main
repo's `pipeline.time_surface` (`(H, W)` in [0, 1]). `pipeline.accumulator` is not reused: it
needs a `dv.EventStore`, our clips are NumPy arrays. Stage 1 input.
`to_position(clip, window_us) -> iterator[(t_s, x, y)]` — dense-cell centroid: events per 16 px
cell, cells with ≥ 30 % of the fullest, mean of their events; NaN when too few. Robust to
sensor-wide noise and to a string. `t_s` is the window centre. Baseline/fallback.
`to_raw(clip, bin_us)` — Stage 2 input, still a stub.

### `model.py`
The framework-agnostic boundary.

```
class Localiser(Protocol):
    def locate(self, obs) -> (x, y): ...          # frame or raw window -> position
    def reset(self): ...

class TrajectoryMemory(Protocol):
    def fit(self, tracks): ...                     # offline pretrain on many position tracks
    def observe(self, x, y): ...                   # stream one step
    def predict(self, horizon_s) -> (x, y): ...    # short-horizon future position
    def deviation_score(self) -> float: ...        # 0 = on-pattern, high = break
    def period(self) -> float: ...                 # remembered period, s; NaN before one exists
    def path_points(self, fractions): ...          # remembered path at fractions of its cycle, (N, 2)
    def reset(self): ...

class Model:                                       # Localiser + TrajectoryMemory
    def step(self, obs) -> (prediction, deviation_score): ...
```

The Stage-1 memory behind this Protocol is `snn.SpikingMemory` (design in `PLAN.md` §C):
`PlaceCells` encoder → `TwoLayerNet` (fast and slow recurrent LIF layers, snnTorch) →
horizon heads and a path head; `fit` pretrains by BPTT on `data.Track`s, `save`/`load`
carry the weights, the target standardisation and the deviation scales.

### `baseline.py` ✅
`PeriodicKalman` (harmonic state per coordinate, normalised-innovation surprise) and
`HarmonicFit` (sliding least-squares harmonics) — implement `TrajectoryMemory`; both take
`dt_s` (one observation per window), estimate the period from a `warmup_s` (5 s) of
observations with `trajectories.search_period` and keep it; a NaN observation is skipped.
`fit()` is a no-op: they learn each clip from its own warm-up.

### `metrics.py` ✅
`prediction_error(pred, gt) -> {errors, median, iqr, mean, n}`.
`lock_on_time(errors, tol, dt)` — first time after which the error stays under `tol`; inf if never.
`deviation_roc(scores, times, deviation_times, threshold, hold_n) -> {threshold, auc,
latency_s, fp_per_min}` — Mann–Whitney AUC; a flag is `hold_n` steps above threshold.

### `experiment.py` ✅ (per-clip and per-set; the corpus-wide pretrain run comes with C)
`Trace` — a memory's per-step output on one clip (`t, obs, pred, gt_ahead, score`).
`evaluate_clip(memory, clip, window_us, horizon_s) -> Trace` — the one loop every method
goes through. `score_trace(trace, clip, tol_px, settle_s) -> dict` — the three metrics.
`load_set / open_set(name)` — `corpus/sets.yaml` entries, loaded and sliced.
`run_set(make_memory, clips, ...) -> (rows, pooled)` — fresh memory per clip, medians pooled.
`make_memory(name, dt_s, **params)` — `kalman` / `harmonic`; the SNN registers here.

## Data flow

```
TrajectorySpec ─► simulate ─► Clip ─┐
recording .aedat4 ─► data ─► Clip ─┴─► frontend ─► obs stream
                                          │
                                   model.step(obs) ─► (prediction, deviation_score)
                                          │
                                   metrics vs Clip.gt / deviation_times ─► Report
```

Pretraining: `make_sim_dataset` writes N `Clip`s → `experiment` calls `model.fit` on their
position tracks → weights frozen → evaluation runs on real clips only.

## Frameworks

| Piece | Candidates | Notes |
|---|---|---|
| Memory core | two-timescale recurrent LIF (snnTorch) | G-F: snnTorch. LMU (Nengo) kept as an optional comparison. |
| Localiser (Stage 1/2) | spiking conv / WTA (snnTorch or SpikingJelly) | Topographic → not NEF. |
| Simulator | v2e | ESIM fallback if a 3-D scene is ever needed. |
| Baseline | NumPy / SciPy | No SNN. |

Everything SNN sits behind `model.py`'s Protocols. Swapping frameworks touches `model.py` and
`requirements.txt` only.

## Testing

pytest, hardware-free, deterministic on simulated clips.

- `trajectories.py`: sampled circle/figure-8 match closed-form points.
- `simulate.py`: event count scales with blob speed; events fall on the path.
- `frontend.py`: frame shape and timing; centroid tracks a known sim path.
- `metrics.py`: known-answer cases for error, lock-on, ROC.
- `snn.py`: place cells, path targets round-trip, a tiny network learns a few ellipses and
  predicts ahead; a scripted deviation raises `deviation_score`; checkpoints round-trip.
- `experiment.py`: end-to-end smoke run on 3 tiny sim clips.

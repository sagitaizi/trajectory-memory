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
    baseline.py
    metrics.py
    experiment.py
  scripts/
    make_sim_dataset.py
    run_experiment.py
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

### `frontend.py`
`to_frames(clip, window_us) -> iterator[Frame]` — reuses `pipeline.accumulator` /
`pipeline.time_surface`. Stage 1 input.
`to_raw(clip, bin_us) -> iterator[SpikeTensor]` — fine spatial+temporal binning. Stage 2 input.
`to_position(clip, window_us) -> iterator[(t, x, y)]` — classical centroid. Baseline/fallback.

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
    def reset(self): ...

class Model:                                       # Localiser + TrajectoryMemory
    def step(self, obs) -> (prediction, deviation_score): ...
```

Ships: `ReservoirMemory` (NumPy echo-state + ridge/RLS readout) and `CentroidLocaliser`
(wraps `frontend.to_position`). Framework-specific implementations (LMU, snnTorch) are added
behind the same Protocols once G-F closes.

### `baseline.py`
`PeriodicKalman` and `HarmonicFit` — implement `TrajectoryMemory`. Non-SNN reference numbers.

### `metrics.py`
`prediction_error(pred, gt)` — mean/median distance at the prediction horizon.
`lock_on_time(errors, tol)` — time until error stays below `tol`.
`deviation_roc(scores, deviation_times)` — ROC + median detection latency.

### `experiment.py`
`run(config) -> Report`. Build corpus (sim train split + real test split) → `model.fit` on sim
tracks → freeze → stream each real clip through `model.step` → `metrics` → table + plots.
Deterministic given a seed.

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
| Memory core | reservoir/LSM (NumPy or Brian2), LMU (Nengo), spiking RNN (snnTorch) | G-F, open. Reservoir is the provisional default. |
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
- `model.py`: `ReservoirMemory` predicts a clean sine within tolerance after training; a
  scripted deviation raises `deviation_score` above baseline.
- `experiment.py`: end-to-end smoke run on 3 tiny sim clips.

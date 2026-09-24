# Metrics

All computed on **held-out real clips** (sim is for pretraining). Report per-clip and pooled;
give spread, not just a mean.

## Prediction error — displacement error, ADE / FDE
At horizon `h` (fixed, e.g. 50 and 100 ms), for every step where a prediction and ground
truth exist:
```
e_i = || pred_i(t + h)  −  gt(t + h) ||        # normalised image coords, or px, or rad
```
Report **median and IQR** (robust to the lock-on transient), and optionally the mean.
Sweep `h` to show how error grows with horizon.

In the trajectory-prediction literature's names (Social LSTM onwards) the error at one
horizon is **FDE(h)** and the mean over the horizons up to it is **ADE**; we pool over the
clip's steps rather than over a set of trajectories, and the tables use those names so the
numbers read against that field. The floor every such table carries is the **constant
velocity** extrapolator (`baseline.Extrapolator`, order 1; order 2 is constant
acceleration) — reported, not assumed to be bad.

## Path-shape error
Whether the memory holds the *path*, not just the next step. Every memory exposes its
remembered cycle (`period()`, `path_points(fractions)`); the truth is a 3-harmonic fit of
the ground truth before any break at the clip's period `T` (exact on sim, searched over the
labels on real clips). Per step:
```
shape_i = min over phase shift of  mean_k || path_i(k/M · T)  −  truth(k/M · T + shift) ||   # M = 64 points, px
```
The memory's path is sampled over one *true* period, so a memory with the right curve and
a doubled period scores well on shape; the period is reported separately as `P / T`. Report
the median over the steady steps, the median over the last cycle before any break, and a
path lock-on time (as below, on `shape_i`).

## Lock-on time
How long until the memory is usable. First time `t*` after which `e(t) < tol` holds for the
rest of the clip (or for `K` consecutive cycles). Report in **seconds** and in **cycles**
(`t* · Ω / 2π`). `tol` stated once, e.g. 1.5× the baseline noise floor.

## Deviation detection
Treat the deviation score `s(t)` as a binary detector over time.
- **ROC / AUC**: sweep the threshold on `s`; positives = frames after `t_break`, negatives =
  frames on no-deviation clips (and pre-break frames).
- **Detection latency**: `t_flag − t_break`, where `t_flag` is the first sustained threshold
  crossing (require `n` consecutive frames to avoid spikes). Report median over deviation
  clips at a fixed operating point (e.g. FPR = 0.05 /min).
- **False-positive rate**: crossings per minute on no-deviation clips.
- **Recovery**: does `s(t)` return toward baseline after a transient deviation, or latch?

## Baseline comparison table (paper)
Rows: ConstantVelocity, PeriodicKalman, HarmonicFit, RhythmicDMP, SNN-Stage1 (, SNN-Stage2).
Columns: FDE @50ms, @100ms; ADE; lock-on (cycles); deviation AUC; detection latency;
false alarms per minute.

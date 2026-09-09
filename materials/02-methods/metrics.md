# Metrics

All computed on **held-out real clips** (sim is for pretraining). Report per-clip and pooled;
give spread, not just a mean.

## Prediction error
At horizon `h` (fixed, e.g. 50 and 100 ms), for every step where a prediction and ground
truth exist:
```
e_i = || pred_i(t + h)  −  gt(t + h) ||        # normalised image coords, or px, or rad
```
Report **median and IQR** (robust to the lock-on transient), and optionally the mean.
Sweep `h` to show how error grows with horizon.

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
Rows: PeriodicKalman, HarmonicFit, RhythmicDMP, SNN-Stage1 (, SNN-Stage2).
Columns: median pred. error @50ms, @100ms; lock-on (cycles); deviation AUC; detection latency.

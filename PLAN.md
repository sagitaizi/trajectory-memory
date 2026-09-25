# Plan

Current state of the work, by phase. `TIMELINE.md` has the dates, `docs/implementation-plan.md`
the architecture, `PAPER_PROGRESS.md` what is writable in the manuscript.

## What this is

A spiking network that, offline, learns a target's repetitive trajectory from event-camera
input, predicts it a short horizon ahead, and flags deviations. Pretrain on simulation,
freeze, test on real recordings. The claim (per Dr. Ezra Tsur): the spiking dynamics *solve*
the prediction — not "the same task on a different network." Closest prior work (Debat et al.
2021, the 2025 event ping-pong paper) uses a static camera, ballistic motion, offline batch
training and no deviation signal; the gap is a repetitive path learned online-style from a
freely moving real target, with a break signal.

**Main proof of concept: the string pendulum** — one object, one motion, the cleanest
repetitive trajectory in the corpus. The fan and wall target are the secondary result: the
same pipeline is still a reasonable predictor on other motions. Pendulum positions are scored against the labels as-is (the labelled brush centre is the
target); offset-removed numbers are a secondary column.

## Status

| Phase | State |
|---|---|
| A — Data | 🟨 Real corpus recorded and split; development set labelled; sim corpus generated. **Open: label the held-out clips.** |
| B — Frontend + baselines | ✅ |
| C — Trajectory memory, Stage 1 | 🟨 **In progress.** Memory (clock and map, `snn_phasemap`) built: at the Kalman bar on prediction, better on path and deviation (development). Pendulum localiser `pend_ft.pt` beats the centroid on the pendulum. **Open: the half-period lock; the held-out set.** |
| D — Deviation detection | ✅ Label-free ratcheting alarm, k calibrated per position input on development (centroid 25, `pend_ft` 30): AUC 0.98, 0.23 s, no false alarms (Kalman 0.86, 39.9/min). Held-out pending labels. |
| E — Raw events, Stage 2 | ⬜ Only if Stage 1 lands. |
| F — Paper | 🟨 Written as sections unlock; draft due 2026-10-01. |

## A — Data

- **Real corpus**: 54 clips (`RECORDING_LOG.md`), four setups — fan (brush on a string, a
  rigid ellipse), string pendulum, hand-moved wall target (4b, marginal SNR), motor-swept
  scene (4a, the whole scene moves). 14 carry a deviation. Ground truth is hand-labels +
  interpolation for every setup (`scripts/mark_labels.py`, `mark_breaks.py`; check with
  `replay_gt.py`). Mark spacing: pendulum 0.1 s, fan 0.15 s, 4b 0.25 s, 4a 0.5 s.
- **Split** (`corpus/sets.yaml`). Real clips are never trained on. Development clips are
  looked at freely while building; held-out clips are labelled last and scored once.

  | Set | Pendulum | Fan | 4b | 4a |
  |---|---|---|---|---|
  | Development ✅ labelled | `small_01` `wide_02` (from 6 s) `wide_break` | `fan_brush_slow_02` | `loop_01` `loop_break_01` (+ its two segments) | — |
  | Held-out | `small_03` `wide_01` `small_break` | `fan_brush_fast_01` `fan_string_01` `fan_brush_break_01` | `loop_03` `loop_break_02` | `scan_pan_slow_01` `scan_both_slow_01` `scan_diag_break_01` |

  Promoting a spare clip is fine; promoting a held-out clip after seeing a result on it is not.
- **Simulated corpus**: 100 clips × 15 s in `corpus/sim/` (`scripts/make_sim_dataset.py`,
  seed-reproducible, resumable). v2e through the calibrated DVXplorer lens model, matched
  to `fan_brush_slow_02` on rate, polarity, footprint, trail and noise floor
  (`scripts/match_sim_real.py`; table in `materials/02-methods/simulation-with-v2e.md`).
  Per clip: random path (five families, T 0.8–4 s, half with a scripted break, 35 % exact
  and the rest with small smooth imperfections), random target (disc to brush, texture,
  half on a string) and camera settings. `corpus/sim_pendulum/` (`--kind pendulum`, 100 × 15 s)
  is the pendulum only: a bar ~200 × 40 px on a V of string, swinging from a pivot above the
  frame, geometry drawn from circle fits to the real labels (`params.yaml` `sim.randomise.pendulum`);
  half get a break that narrows or widens the swing. Ground truth is the target's *apparent*
  (lens-distorted) position — what hand-labels record. `corpus/sim/tracks.npz` holds the
  measured and true position per 5 ms window for every clip; `scripts/make_frames.py`
  writes frame sets at a chosen window and downsample.
- **Known**: on ~10 % of sim clips the string is as bright as the target and the classical
  centroid locks onto it (>100 px). This happens in real life too (the pendulum's string is
  the brightest line in its frames), so the clips stay — a learned localiser must handle it.

## B — Frontend + baselines ✅

- `frontend.py`: ON/OFF count frames (optional downsample), time surfaces (main repo's
  `pipeline.time_surface`), and a dense-cell centroid track robust to noise and to a string.
- `baseline.py`: `Extrapolator` (the naive floor — a least-squares polynomial over the last
  0.1 s; order 1 is constant velocity, 2 constant acceleration; no period, no remembered
  path), `PeriodicKalman` (harmonic state, normalised-innovation surprise) and
  `HarmonicFit` (sliding least-squares harmonics). The two periodic ones estimate the
  period from a 5 s warm-up.
- `metrics.py`, `experiment.py` (`evaluate_clip` is the one loop every method goes through),
  `scripts/run_experiment.py --set development`, `scripts/replay_gt.py --model kalman`.
- **Numbers**, 5 ms windows, 100 ms horizon, px median (`runs/results/development_*.csv`):

  | Clip | Centroid vs labels | Kalman | Harmonic |
  |---|---|---|---|
  | sim, matched fan ellipse (exact truth) | 3.0 | 2.6 | 3.1 |
  | `fan_brush_slow_02` | 5.7 | 8.3 | 7.8 |
  | `small_01` | 27.5 | 27.3 | 29.7 |
  | `loop_01` | 21.7 (p90 93) | 63.1 | 55.1 |
  | `loop_break_01` | 22.1 | 21.8 (AUC 0.75) | 19.0 (AUC 0.81) |
  | `wide_break` | 46.3 | 41.6 (AUC 0.86) | 42.6 (AUC 0.67) |
  | pooled, development (8 entries) | — | 24.8 | 36.2 |
  | pooled, all 100 sim clips | 10.8 | 17.5 | 29.5 |
  | sim clips where the centroid tracks (49) | < 15 | 7.5 | 15.3 |

  On sim the localiser is the bottleneck: on 39 of 88 clips the centroid is ≥ 15 px off
  (the string) and the predictors follow it (~49 px); with clean positions the Kalman is
  7.5 px, 5.6 on exact paths. Break detection on sim: AUC 0.91 / 0.87, latency 0.33 /
  0.15 s. Lock-on pools to "never" on both sets at a 15 px tolerance — the tolerance
  must be set per regime (§IV-D).
  Hand-labels mark a fixed point of the object that is not the event centroid (pendulum:
  brush centre, 25–42 px below the centroid; wall target: 12–21 px), a floor under any error
  scored against them. `--subtract-offset` removes each clip's median label−centroid
  vector before scoring and reports it (§IV-D); it is `evaluate.py`'s default. For the
  pendulum the as-is numbers are primary and offset-removed ones secondary (see top).

## C — Trajectory memory, Stage 1 (the goal)

- `model.py`: `Localiser` + `TrajectoryMemory` protocols; the SNN registers in
  `experiment.make_memory` and is scored and watched with the same tools as the baselines.
- The localiser is pretrained on the sim corpus and frozen; the memory learns each clip
  online. Evaluate on development clips while building, on held-out clips once.
- **Design** (background in `materials/01-literature/multi-timescale-memory.md`):
  - *Input*: the position stream, one (x, y) per 5 ms window; its main-axis signal drives
    the clocks.
  - *Core* (**decided 2026-09-20**, after the LMU; spec of the LMU attempt in
    `docs/superpowers/specs/2026-09-19-lmu-memory-design.md`): a **clock and a map**. Six
    clocks of 1500 neurons — one batched 4-D LIF population built by Nengo and run in torch, whose recurrent
    wiring is the adaptive-frequency-oscillator dynamics (Righetti et al. 2006) — lock their
    phase to the motion's main-axis signal; the rate is a slow modulatory scalar. Each clock
    drives a ring of 400 phase cells whose connections to a position readout are the map,
    learned in the clip by the PES rule. Prediction = the map at the rotated phase plus the
    smoothed current residual; period = the rate; shape = the map; the clock is elected by
    the drive's zero-crossing period and committed at the snapshot. Nothing is pretrained:
    the path is learned online within a few laps. `phasemap.py` is the arithmetic reference.
    Ablations kept with their checkpoints: the learned two-timescale network (`snn.py`) and
    the LMU window recording (`snn_lmu.py`) — neither could hold the cycle (path 55–72 px).
  - *Readout*: as above; horizon heads from a learned fast layer are an open addition for
    the 25–50 ms detail (the two-layer network's fast layer reached 5.6 px at 25 ms).
  - *Deviation score*: the mismatch between the observed position and the **snapshot** of
    the map taken once the mismatch has settled after lock-on (the remembered path); the
    working map keeps adapting for prediction. Without the snapshot the memory re-learns a
    new path within ~3 laps and the break vanishes from the score.
  - *Training*: none for the memory. Pretraining on simulation applies to any gradient-trained
    stage added on top (a fast layer), with the same corpus, blanks and checkpoint rules.
- **Built**: `trajmem/phasemap.py` (reference), `trajmem/snn_phasemap.py` (the memory),
  `run_experiment.py --memory snn_phasemap`; the LMU and two-layer memories and their
  checkpoints stay as ablations (`--memory snn --checkpoint …`).
- **Numbers** (development set, centroid input, offset-subtracted, pooled;
  `docs/results/development.md`, from `scripts/evaluate.py --set development`):

  | | pred 100 ms px | path px | period | AUC | latency s | fp/min |
  |---|---|---|---|---|---|---|
  | Kalman | 13.7 | 14.1 | 1.01 | 0.86 | 0.42 | 39.9 |
  | Harmonic | 23.4 | 10.0 | 1.01 | 0.72 | 0.14 | 0.00 |
  | PhaseMap (arithmetic reference) | 14.5 | 13.1 | 1.00 | 1.00 | 0.15 | 0.00 |
  | **SpikingPhaseMap** | **13.5** | **12.9** | 1.00 | **0.98** | **0.23** | **0.00** |

  Prediction ties the Kalman; path and deviation are better. Known weak cases: the
  diagonal sweep locks at half the period; a 3:2 Lissajous has nothing at the fundamental
  to lock to. **Open: the half-period lock (Sagi checks it in the replay first); a fast
  layer for the 25–50 ms horizons; the held-out set (scored once, at the end).**
- **Ablations** (the learned memories that could not hold the cycle; full record in
  `docs/snn-experiments-log.md`, short version `docs/snn-experiments-summary.md`):
  the two-timescale network (`snn.py`, `runs/memory/snn_anchor20.pt`) predicts 10.2 px at
  100 ms on sim but only by short-term extrapolation — its remembered path is 64 px and its
  period ~2× off on the development set; the LMU window memory (`snn_lmu.py`) reaches a
  54–59 px path, limited by the spiking population's 2–4 % state error over 24 Legendre
  orders (a perfect window gives 5.9 px). Path shape is `metrics.path_shape_error`
  (definition in `materials/02-methods/metrics.md`): how far a memory's own picture of one
  cycle sits from the true cycle, px, with the period reported apart as `P / T`.
- **Localiser (Stage 1's other half)**: `trajmem/snn_localise.py` — a spiking conv net over
  the 5 ms count frames (place cells + a "present" unit), trained on 300 simulated clips
  (200 blob targets, 100 "sheet" targets — the wall target on its sheet) with the real
  cameras' quirks added in training: stuck pixels, a noise floor, mirror flips, a quarter
  turn, blank stretches, a brightness scale and an ON/OFF swap (the real camera's polarity
  is the reverse of v2e's). `--localiser snn` in `run_experiment.py`, `replay_gt.py`,
  `check_localiser.py`, `bench.py`; spec in
  `docs/superpowers/specs/2026-09-21-spiking-localiser-design.md`. Against the labels
  (median px, constant offset removed; centroid in brackets): fan 7.8 (5.2), pendulum
  16–26 (7.6–11.9), wall target 16–22 (14–16); no lag beyond the neurons' own ~35 ms.
  Pooled over the development set it feeds the memory worse than the centroid
  (17.7 / 24.4 px prediction / path vs 13.5 / 12.9).
  **Pendulum localiser**: `runs/localiser/pend_ft.pt`, `snn.pt` fine-tuned on
  `corpus/sim_pendulum/` (lr 1e-3, 10 epochs; from scratch was worse). Pendulum
  development, against the labels as-is, 100 ms prediction: 17.9 / 31.5 / 22.6 px vs the
  centroid's 27.2 / 49.2 / 42.8. Input per setup: pendulum → `pend_ft`; fan and wall
  target → the classical centroid, with `snn.pt` as the ablation.
- **Done when:** on real repetitive clips, Stage 1 prediction error beats the classical
  baseline *or* matches it with a stated event-native/latency argument; lock-on within N
  cycles. Bar: the Kalman on the same input (13.7 px pooled at 100 ms, development,
  offset-subtracted).

## D — Deviation detection ✅

- Score = how far the raw observation sits from where the memory's map expects it (px),
  scored before the input gate. A flag is 3 consecutive steps above the bar.
- The bar is **ratcheting** (`metrics.ratchet_threshold`): the lowest `median + k x spread`
  the score has reached over a trailing 2 s, causal and one-way, so it tightens while the
  memory settles and never loosens. No labels, no calibration window to place.
- `k = 25`, the smallest of 3..30 that raises no false alarm on the development clips;
  both breaks still caught, 0.23 s median. Held-out untouched. `--threshold oracle` keeps
  the old label-using percentile as a reference.
- Development set, same rule on each method's own score (pooled): spiking memory AUC 0.98,
  0.23 s, 0.00 false alarms/min; arithmetic prototype 1.00, 0.15 s, 0.00; Kalman 0.86,
  0.42 s, **39.9**; harmonic 0.72, 0.14 s, 0.00.
- **Open:** the same numbers on the held-out break clips, once they are labelled.

## E — Raw events, Stage 2 (upside)

- `frontend.to_raw` (1 ms bins) and an end-to-end variant behind the same interface.
- **Done when:** it runs on real clips with a prediction-error number, even if worse.

## F — Paper

Per-section status in `PAPER_PROGRESS.md`. Results table: baseline vs Stage 1 (vs Stage 2),
prediction + deviation; one external-dataset generalisation check; 8 pages IEEE.

## Decision gate G-F — framework for the SNN core ✅

One framework per part:

- **Localiser: snnTorch** — a conv net trained by surrogate gradients (`snn_localise.py`).
  snnTorch exports to NIR, which supports the hardware-path claim.
- **Memory: Nengo-built, run in PyTorch** — Nengo solves the NEF weights of the clock
  populations at construction (`lmu.build_dynamics`); the LIF neurons (Nengo's model) are
  stepped in torch (`lmu.SpikingLmu`); the PES map rule is written by hand. No snnTorch.
- Ablation memories (`snn.py`, `snn_lmu.py`) are snnTorch.

SpikingJelly was the localiser alternative: 2.3× faster training with its fused CuPy
kernels (0.15 vs 0.35 s per step, T = 200, batch 8, RTX 3060), but its fast path needs an
`np.int` shim on NumPy 2 and its plain-torch path is 6× slower than snnTorch. Norse, Lava,
Sinabs/Rockpool, Brian2, BindsNET ruled out (inactive, hardware-bound, or not for
supervised regression). Installed: `snntorch 1.0.0`, `nengo 4.1.0`, `torch 2.14.0+cu126`
on the RTX 3060; SpikingJelly and CuPy are installed but unpinned.

## Out of scope (future work)

Re-learning a new pattern after a break; attention across multiple objects; closing the loop
to the pan-tilt rig.

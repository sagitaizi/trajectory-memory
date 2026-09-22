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

## Status

| Phase | State |
|---|---|
| A — Data | 🟨 Real corpus recorded and split; development set labelled; sim corpus generated. **Open: label the held-out clips.** |
| B — Frontend + baselines | ✅ |
| C — Trajectory memory, Stage 1 | 🟨 **In progress.** Network, training and evaluation built; 13.6 px at 100 ms on sim (Kalman 7.5), at the Kalman bar on the development set. Open: the cycle memory. |
| D — Deviation detection | ⬜ Scoring exists; thresholds must be chosen on development clips. |
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
  half on a string) and camera settings. Ground truth is the target's *apparent*
  (lens-distorted) position — what hand-labels record. `corpus/sim/tracks.npz` holds the
  measured and true position per 5 ms window for every clip; `scripts/make_frames.py`
  writes frame sets at a chosen window and downsample.
- **Known**: on ~10 % of sim clips the string is as bright as the target and the classical
  centroid locks onto it (>100 px). This happens in real life too (the pendulum's string is
  the brightest line in its frames), so the clips stay — a learned localiser must handle it.

## B — Frontend + baselines ✅

- `frontend.py`: ON/OFF count frames (optional downsample), time surfaces (main repo's
  `pipeline.time_surface`), and a dense-cell centroid track robust to noise and to a string.
- `baseline.py`: `PeriodicKalman` (harmonic state, normalised-innovation surprise) and
  `HarmonicFit` (sliding least-squares harmonics). Both estimate the period from a 5 s warm-up.
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
  scored against them. **Decided (2026-09-17): `--subtract-offset`** removes each clip's
  median label−centroid vector before scoring and reports it (§IV-D). With it, development
  pooled at 100 ms: Kalman 14.8, Harmonic 24.0, SNN (`runs/memory/snn.pt`) 18.3; fan 7.0 /
  6.0 / 11.5; `loop_01` 61.9 / 54.2 / 29.0.

## C — Trajectory memory, Stage 1 (the goal)

- `model.py`: `Localiser` + `TrajectoryMemory` protocols; the SNN registers in
  `experiment.make_memory` and is scored and watched with the same tools as the baselines.
- Pretrain on the sim corpus, freeze, evaluate on development clips while building, on
  held-out clips once.
- **Design** (decided 2026-09-16; rationale in `materials/01-literature/multi-timescale-memory.md`):
  - *Input*: place cells per axis — 32 overlapping Gaussian tuning curves along x and 32
    along y (64 inputs); the encoder is a swappable class so a 2-D grid can replace it.
  - *Core* (**decided 2026-09-20**, after the LMU; spec of the LMU attempt in
    `docs/superpowers/specs/2026-09-19-lmu-memory-design.md`): a **clock and a map**. Five
    clocks — one batched 4-D LIF population built by Nengo and run in torch, whose recurrent
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
- **Numbers** (development set, 100 ms, offset-subtracted; `docs/snn-experiments-log.md` run 12):

  | | pred px | path px | period | AUC | latency s | fp/min |
  |---|---|---|---|---|---|---|
  | Kalman | 14.7 | 14.0 | 1.00 | 0.80 | 0.95 | 13.2 |
  | **SpikingPhaseMap** | **14.3** | **12.4** | 1.00 | **0.99** | **0.11** | **8.1** |

  Prediction is a tie per clip (within 0.3–2 px behind on six, far ahead on two); path
  better on five of eight; deviation clearly better. Known weak cases: the diagonal sweep
  locks at half the period; a 3:2 Lissajous has nothing at the fundamental to lock to.
  **Open: the half-period lock; a fast layer for the short horizons; the held-out set
  (scored once, at the end).**
- **Runs so far** (validation sim clips, px median; persistence = the input repeated):

  | | 25 ms | 50 ms | 100 ms | 200 ms |
  |---|---|---|---|---|
  | persistence | 11 | 18 | 29 | 51 |
  | (x, y) readout, 192 fast, 30 epochs | 30 | 41 | 47 | 64 |
  | place cells out, 384 fast, rate reg, lr 3e-3, 30 epochs | 21 | 27 | 39 | 59 |
  | + velocity cells in, displacement cells out (`m1_base`) | 5.8 | 8.7 | 15.6 | 33.8 |
  | + fast τ 10–25 ms, readout 15 ms, mirror flips | 5.6 | 7.8 | 13.6 | 27.3 |
  | + string-locked clips with truth standing in, 40 epochs cosine | 5.5 | 7.6 | 11.7 | 21.9 |
  | + corpus doubled to 200 sim clips (`runs/memory/snn.pt`) | 5.6 | 7.3 | 10.7 | 19.2 |
  | + 20 ms anchor smoothing, stopped at epoch 34 (`runs/memory/snn_anchor20.pt`) | 5.7 | 7.2 | 10.2 | 17.7 |

  Kalman on the same clips: 7.5 px at 100 ms. Development set at 100 ms, offset-subtracted,
  pooled: SNN 16.6–17.4, Kalman 14.8, Harmonic 24.0; fan 12–13 (Kalman 7.0); `loop_01`
  26–28 (Kalman 61.9). The adaptive (LSNN) slow layer made no difference in two runs. The
  full record of the 2026-09-17 overnight sweep is `docs/snn-experiments-log.md`, the short
  version `docs/snn-experiments-summary.md`. Decided along the way: horizon heads are place
  cells over the displacement from the current position; the input carries velocity cells;
  a firing-rate regulariser keeps both layers active.
- **Path-shape metric** (`metrics.path_shape_error`, columns `path_med` / `path_last` /
  `path_lock` / `period` in `run_experiment.py`; definition in
  `materials/02-methods/metrics.md`): how far each memory's own picture of one cycle sits
  from the true cycle, px, with the period reported apart as `P / T`. This is the direct
  measure of "memory of the path"; the 200 ms horizon is only a proxy for it. Development
  set, offset-subtracted, path median px / period ratio:

  | | Kalman | Harmonic | SNN (`snn_anchor20.pt`) |
  |---|---|---|---|
  | pooled | 14.0 / 1.00 | 10.3 / 1.00 | 64.4 / 1.83 |
  | `fan_brush_slow_02` | 6.8 / 2.00 | 4.5 / 2.00 | 77.3 / 2.68 |
  | `small_01` | 12.1 / 1.01 | 10.4 / 1.01 | 55.6 / 1.97 |
  | `loop_01` | 48.9 / 0.50 | 43.9 / 0.50 | 39.0 / 1.39 |

  On clean sim clips the Kalman's path is 1.5–9 px, the SNN's 30–65 px. **The SNN's path
  head holds nothing usable: its period readout is ~2× off and its shape is at the scale
  of the path itself, on sim as on real clips. Its prediction numbers come from short-term
  extrapolation alone.** The baselines' period search also lands on 2T (fan) or T/2
  (`loop_01`) from a 5 s warm-up; their shape is right regardless. The geometric path loss
  moved the path to ~72 px and routing the long horizons through the slow layer changed
  nothing (`docs/snn-experiments-log.md`, run 6) — hence the LMU core above. First LMU
  run (12 epochs, `runs/memory/lmu_quick.pt`): path **58 px** on the development set (fan
  30, `loop_01` 29), prediction 32 px; sim validation path 56 px. Four LMU runs (linear /
  MLP heads, with / without blanks) all land at 54–59 px path and 32–33 px prediction.
  Offline: a perfect window read by the classical period search gives **5.9 px**, the
  spiking window 74–86 — the population's 2–4 % state error, amplified over 24 Legendre
  orders, is the binding constraint. **Open: a window representation that tolerates that
  error (shorter θ / fewer orders, a bank of short windows, 2-D ensembles); prediction
  needs the two-layer recipe (flips, 40 epochs) before a fair comparison; self-feeding
  through blanks drifts (52–65 px in a 1 s blank).**
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
  End to end it feeds the memory worse than the centroid on the development set
  (pooled 18.7 / 21.4 px vs 13.5 / 12.9), the pendulum being the gap: on an elongated
  target the network's "centre" drifts along the long axis while the densest-cells
  centroid sits on the head. **Open: an elongated head-and-tail target family in the
  simulator; checkpoint selection (real-clip results swing epoch to epoch).** The
  classical centroid stays the default input; the spiking localiser is the ablation.
- **Done when:** on real repetitive clips, Stage 1 prediction error beats the classical
  baseline *or* matches it with a stated event-native/latency argument; lock-on within N
  cycles. Bar: Kalman 24.2 px pooled (8.0 on the fan), ~3 px on sim.

## D — Deviation detection

- Score = prediction error against the learned path; a flag is `hold_n` steps above a
  threshold chosen on development clips and applied unchanged to held-out ones.
- **Done when:** AUC, latency and false-alarm rate reported on real break clips.

## E — Raw events, Stage 2 (upside)

- `frontend.to_raw` (1 ms bins) and an end-to-end variant behind the same interface.
- **Done when:** it runs on real clips with a prediction-error number, even if worse.

## F — Paper

Per-section status in `PAPER_PROGRESS.md`. Results table: baseline vs Stage 1 (vs Stage 2),
prediction + deviation; one external-dataset generalisation check; 8 pages IEEE.

## Decision gate G-F — framework for the SNN core

**snnTorch** (provisional, 2026-09-17; Sagi to confirm). Both parts — localiser and memory — in
one framework; Nengo's LMU stays as an optional memory-core comparison. SpikingJelly was
the alternative: 2.3× faster conv-localiser training with its fused CuPy kernels (0.15 vs
0.35 s per step, T = 200, batch 8, RTX 3060) but its fast path needs an `np.int` shim on
NumPy 2 and its plain-torch path is 6× slower than snnTorch; the recurrent memory gains
nothing from fused kernels. Norse, Lava, Sinabs/Rockpool, Brian2, BindsNET ruled out
(inactive, hardware-bound, or not for supervised regression). snnTorch exports to NIR,
which supports the hardware-path claim. Installed: `snntorch 1.0.0`, `nengo 4.1.0`,
`torch 2.14.0+cu126` on the RTX 3060; SpikingJelly and CuPy are installed but unpinned.

- **Inputs ready**: `corpus/sim/tracks.npz` (300 k position steps, half the clips with a
  break); frame sets via `make_frames.py --downsample d` (8× → 2×60×80 per 5 ms).
- **One evaluation loop** for every candidate: `run_experiment.py --set development`.

## Out of scope (future work)

Re-learning a new pattern after a break; attention across multiple objects; closing the loop
to the pan-tilt rig.

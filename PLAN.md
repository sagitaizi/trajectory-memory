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
| C — Trajectory memory, Stage 1 | ⬜ **Next.** Framework and architecture to be decided together (G-F). |
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
  | sim (exact truth) | 3.0 | 2.6 | 3.1 |
  | `fan_brush_slow_02` | 5.7 | 8.0 | 7.3 |
  | `small_01` | 27.5 | 26.1 | 28.8 |
  | `loop_01` | 21.7 (p90 93) | 55.8 | 55.3 |
  | `loop_break_01` | 22.1 | 22.3 (AUC 0.75) | 18.9 (AUC 0.81) |
  | `wide_break` | 46.3 | 41.3 (AUC 0.86) | 42.2 (AUC 0.67) |
  | pooled, development (8 entries) | — | 24.2 | 32.6 |
  | pooled, all 100 sim clips | 10.8 | 25.1 | 30.8 |
  | sim clips where the centroid tracks (49) | < 15 | 9.9 | 17.0 |

  On sim the localiser is the bottleneck: on 39 of 88 clips the centroid is ≥ 15 px off
  (the string) and the predictors follow it (55 px); with clean positions the Kalman is
  ~10 px, 7 on exact paths. Both baselines use two harmonics, so 3:2 Lissajous paths score
  worse (Kalman 39.5 px). Break detection on sim: AUC 0.91 / 0.86, latency 0.28 / 0.07 s.
  The pendulum numbers are a *constant* vertical offset of 27–43 px between the event
  centroid and the marked brush centre (1–3 px sideways) — a labelling convention, and a
  floor under any pendulum error scored against labels. How to treat it is an open §IV-D
  choice: subtract the per-clip offset, or score prediction against the frontend's own
  track and report localisation separately.

## C — Trajectory memory, Stage 1 (the goal)

- `model.py`: `Localiser` + `TrajectoryMemory` protocols; the SNN registers in
  `experiment.make_memory` and is scored and watched with the same tools as the baselines.
- Pretrain on the sim corpus, freeze, evaluate on development clips while building, on
  held-out clips once.
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

## Decision gate G-F — framework for the SNN core (OPEN)

Decided together, after a short bake-off on simulated tracks. Unlocks §III-C/D.

- **Candidates.** Memory core: reservoir/LSM + online readout, Legendre Memory Unit (Nengo),
  or surrogate-gradient spiking RNN (snnTorch). Localiser: topographic, so a small spiking
  conv / WTA (snnTorch); NEF is the wrong shape for it. Raw events push toward
  snnTorch/SpikingJelly or a custom reservoir rather than Nengo.
- **Installed**: `snntorch 1.0.0`, `nengo 4.1.0`, `torch 2.14.0+cu126` on the RTX 3060
  (6 GB). Not installed: SpikingJelly, Norse, Lava, nengo-dl.
- **Inputs ready**: `corpus/sim/tracks.npz` (300 k position steps, half the clips with a
  break); frame sets via `make_frames.py --downsample d` (8× → 2×60×80 per 5 ms).
- **One evaluation loop** for every candidate: `run_experiment.py --set development`.

## Out of scope (future work)

Re-learning a new pattern after a break; attention across multiple objects; closing the loop
to the pan-tilt rig.

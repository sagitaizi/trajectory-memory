# Plan

Phased plan for the SNN trajectory-memory paper. Start here; `TIMELINE.md` has the dates,
`docs/implementation-plan.md` has the architecture, `PAPER_PROGRESS.md` has what is writable
in the manuscript right now.

## What this is

A spiking network that, offline, learns a target's repetitive trajectory from event-camera
input, predicts it ahead, and flags deviations. Pretrain on simulation, freeze, test on real
recordings. Contribution framing (per Dr. Ezra Tsur): the spiking dynamics *solve* the
prediction problem — not "the same task on a different network."

Novelty rests on the combination, not any one part. Closest prior work — Debat et al. 2021 and
the 2025 event ping-pong paper — does event-SNN trajectory prediction with a **static camera,
constrained ballistic motion, offline batch, no deviation detection**. The gap: a repetitive
path learned online-style from a freely-moving real target, with a break signal, as predictive
inference.

## Current status

| Phase | Status |
|---|---|
| A — Data | 🟨 Corpus recorded (54 clips) and code in place; **labelling is the open work** |
| B — Frontend + baseline | ⬜ |
| C — Trajectory memory (Stage 1, frames) | ⬜ The goal |
| D — Deviation detection | ⬜ In scope |
| E — Raw events (Stage 2) | ⬜ Upside only |
| F — Paper | ⬜ Draft due 2026-10-01 |

## Phases

### A — Data
- `trajectories.py`: analytic path specs + scripted deviations.
- `simulate.py`: v2e wrapper using the main repo's calibration + measured contrast thresholds.
- `data.py`: uniform `Clip` loader for recorded and simulated clips.
- Real corpus recorded 2026-09-10 (`RECORDING_LOG.md`): 54 clips over fan, pendulum,
  motor-swept wall (4a) and hand-moved wall (4b). 14 carry a deviation.
- **Ground truth is hand-labels + interpolation for every setup**, 4a included. The
  encoder ground truth 4a was recorded for does not hold — see `PAPER_PROGRESS.md`,
  "The 4a encoder ground truth was abandoned".
- Marking tools, one per side-car: `scripts/mark_labels.py` → `.labels.csv` (the path),
  `scripts/mark_breaks.py` → `.deviation.json` (break times),
  `scripts/mark_anchors.py` → `.anchor.json` (4a only, now unused by default).
  `scripts/replay_gt.py` draws a clip's ground truth over it to check the result.
- **Open:** 0 of 54 clips labelled, 0 of 14 break times marked. This is now the
  critical path — nothing in B, C or D can be evaluated until enough clips are labelled.
  Interval matters per group: the pendulum at T ≈ 1.2 s needs ~0.1 s, the 4a triangle
  sweeps are fine at 0.5 s. Labelling all 49 in-scope clips is probably unnecessary;
  decide how many Phase C actually needs before starting.
- **Done when:** a simulated clip and a real clip load through the same path; sim events
  visually resemble the real DVXplorer stream on a matched trajectory; enough real clips
  carry ground truth to evaluate on.
- **Unlocks:** §IV-A's contrast-threshold and noise holes; clears the §IV-B ground-truth caveat.

### B — Frontend + baseline
- `frontend.py`: `frames` (accumulate → images/time surfaces, reusing the main repo's
  `pipeline`), `position` (classical centroid).
- `baseline.py`: Kalman with a periodic-motion model; Fourier/harmonic fit.
- `metrics.py`: prediction error, lock-on time, deviation ROC + detection latency.
- **Done when:** the classical baseline predicts a clean simulated circle within a stated
  tolerance and the metrics reproduce on a fixed clip.
- **Unlocks:** §III-B's window values; §IV-C's tuned baseline settings.

### C — Trajectory memory, Stage 1 (the goal)
- `model.py`: `TrajectoryMemory` interface; provisional NumPy reservoir; frame localiser.
- Pretrain on simulated trajectories, freeze, evaluate on held-out real clips.
- Localiser and memory are separate pieces behind one interface (see G-F).
- **Done when:** on real repetitive clips, Stage 1 prediction error beats the classical
  baseline *or* matches it with a stated event-native/latency argument; lock-on within N cycles.
- **Unlocks:** §III-C, §III-D, and the Stage 1 rows of §V.

### D — Deviation detection
- Deviation score = prediction error against the learned path, with a threshold model.
- Score on clips with a scripted mid-recording change (ROC, time-to-detect).
- **Done when:** detection latency and false-positive rate are reported on real deviation clips.
- **Unlocks:** §III-E and the deviation rows of §V.

### E — Raw events, Stage 2 (upside)
- `frontend.raw`: fine-grained spike-tensor binning.
- End-to-end model variant behind the same interface.
- **Done when:** Stage 2 runs end-to-end on real clips with a prediction-error number, even if
  worse than Stage 1.
- **Unlocks:** the Stage 2 row of §V.

### F — Paper
Per-section status lives in `PAPER_PROGRESS.md`; sections get written as phases unlock them,
not all at the end.

- Results table: baseline vs Stage 1 (vs Stage 2 if reached), prediction + deviation.
- External check: one public dataset (EventVOT or EV-IMO2) sequence, generalisation only.
- 8 pages, IEEE format. Draft to supervisor 2026-10-01.

## Decision gates

### G-F — framework for the SNN core (OPEN)
Deferred deliberately. Resolve by ~2026-09-13 after a short bake-off on simulated trajectories.
Needs a machine, so it cannot be settled away from the desk. **Unlocks:** naming the framework
in §III-C and §III-D.

- **Memory core** (low-D path → prediction): reservoir/LSM + online readout, or Legendre Memory
  Unit (Nengo), or surrogate-gradient spiking RNN (snnTorch).
- **Localiser** (frame → position): topographic, so likely a small spiking conv / WTA in
  snnTorch or SpikingJelly — NEF is the wrong shape for this part.
- Raw-event handling narrows it: no CPU/GPU framework is truly asynchronous; frames and raw
  events both push toward snnTorch/SpikingJelly or a custom reservoir, not Nengo.
- Provisional NumPy reservoir stands in until this closes so the pipeline runs.

## Out of scope (future work)
- Re-learning a new pattern after a break.
- Attention-span / habituation across multiple objects; the human eye-tracking comparison.
- Feeding predictions back to the pan-tilt rig (closed loop).

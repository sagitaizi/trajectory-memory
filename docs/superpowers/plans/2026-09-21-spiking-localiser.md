# Spiking Localiser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A spiking convolutional localiser (frames → place cells + present) trained on simulation, evaluated against the classical centroid on the development clips and end to end with the clock-and-map memory.

**Architecture:** `trajmem/snn_localise.py` holds the snnTorch network, the frame augmentation, the trainer and the `SpikingLocaliser` (model.Localiser); `localise.py` keeps the frame sets and scoring; `experiment.evaluate_clip` streams frames through a localiser when given one; the scripts gain `--localiser`.

**Tech Stack:** Python 3.14 (`D:\Programs\Anaconda\envs\thesis\python.exe`), torch 2.14 CPU, snntorch 1.0, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-spiking-localiser-design.md`

## Global Constraints

- Run with `D:/Programs/Anaconda/envs/thesis/python.exe`; never `conda run`.
- Comments minimal; no narrative in source. History in git.
- Simulation only for training; no real clip is ever trained on.
- Frames from `frontend.to_frames(kind="count", downsample=8)` after `without_hot_pixels`; 5 ms windows; the memory's input rate is unchanged.

---

### Task 1: Frame stream with the hot-pixel filter, and the "present" label

**Files:** Modify `trajmem/frontend.py` (`to_frames` applies `without_hot_pixels` for `count` frames), `trajmem/localise.py` (`FrameSet.present`); Test `tests/test_localise.py`.

**Produces:** `FrameSet.present: np.ndarray (N,) bool` — truth on-sensor and at least `PRESENT_MIN_EVENTS = 8` events within a 5-cell (40 px) radius of the truth in the frame; `frame_set(...)` fills it; `load_frame_set` reads it when saved, else recomputes.

- [ ] Test: an analytic clip (as `tests/test_experiment.py::a_clip_of_events`) → `frame_set` has `present` True on frames with events near the truth and False where the events are removed for a stretch.
- [ ] Implement; `make_frames.py` saves `present`; commit "Frames: hot-pixel filter, present label".

### Task 2: Augmentation on frames

**Files:** Create `trajmem/snn_localise.py` (first part); Test `tests/test_snn_localise.py`.

**Produces:** `augment_frames(frames (N,2,H,W) uint8, gt (N,2), present (N,), rng, stuck=(50, 400), rate=(50, 1500), noise_scale=(0.5, 3.0), flips=True) -> (frames float32, gt, present)`: one random set of stuck pixels per call added as Poisson counts per window (rate × 5 ms) at ON or OFF, the whole frame's background scaled, horizontal/vertical flips applied to frames and truth together.

- [ ] Test: shapes preserved; flips move the truth (x → 1 − x); stuck pixels add counts at fixed positions across all frames; `noise_scale` scales the mean count.
- [ ] Implement; commit "Localiser: frame augmentation".

### Task 3: The network

**Files:** Modify `trajmem/snn_localise.py`; Test `tests/test_snn_localise.py`.

**Produces:** `SpikingLocaliserNet(n_cells=32, ch=(8, 16), hidden=128, dt_s=0.005, tau=(0.01, 0.02), seed)` with `init_state(B, device)` and `step(frame (B,2,60,80) float, state) -> (cells (B, 2, n_cells) logits from a 15 ms low-pass of output spikes, present (B,) logit, state, rate)`; conv 5×5 stride 2 → `snn.Leaky` (learn_beta, per-channel betas from `_betas`) → conv → Leaky → flatten → Linear → Leaky → Linear(64 + 1). Input scaled by 1/8 (counts → currents).

- [ ] Test: shapes; state carries; a frame with a blob at (0.25, 0.75) after 20 steps gives a place-cell centre of mass nearer that than a blob at (0.75, 0.25) once trained for a few hundred steps on two blobs (a smoke-training test, tiny net).
- [ ] Implement; commit "Localiser: spiking conv net".

### Task 4: Trainer and the `SpikingLocaliser`

**Files:** Modify `trajmem/snn_localise.py`, create `scripts/train_localiser.py`; Test `tests/test_snn_localise.py`.

**Produces:** `SpikingLocaliser(dt_s, n_cells=32, ch=(8,16), hidden=128, present_threshold=0.5, resolution=(640,480), seed, device)` with `locate(frame) -> (x, y) | (nan, nan)`, `reset()`, `fit(frame_sets, val_sets, epochs, chunk_s=2.0, batch=8, lr=1e-3, augment=True, log_fn) -> log`, `save/load` (checkpoint with `"arch": "localiser"`). Loss: per-axis cross-entropy against `PlaceCells(32).bumps(gt)` masked to `present & isfinite(gt)`, BCE on present, rate regulariser (target 0.1, weight 10). Validation: px error on present frames, present accuracy. `train_localiser.py --corpus corpus/sim --epochs 20 --out runs/localiser/snn.pt` on `load_frame_sets` (writes them with `make_frames.py --downsample 8` if absent).

- [ ] Test: fit on 6 analytic frame sets (tiny net, 10 epochs) lowers the validation px error below the untrained one by half; `locate` returns NaN on an empty frame after training; checkpoint round-trip.
- [ ] Implement; commit "Localiser: trainer, SpikingLocaliser, train_localiser.py".

### Task 5: Wiring — evaluate with a localiser, scripts

**Files:** Modify `trajmem/experiment.py`, `scripts/run_experiment.py`, `scripts/replay_gt.py`, `scripts/check_localiser.py`; Test `tests/test_experiment.py`.

- [ ] `evaluate_clip(memory, clip, window_us, horizon_s, blank=None, localiser=None)`: with a localiser, positions come from `localiser.locate(frame)` over `to_frames(clip, window_us, downsample=8)` instead of `to_position`; the rest unchanged. `make_localiser(name, checkpoint)` → `FrameCentroid`-equivalent "centroid" (the classical `to_position` path) or `SpikingLocaliser.load`.
- [ ] `run_experiment.py --localiser {centroid,snn} --localiser-checkpoint`; `replay_gt.py` the same (the cyan dot is the localiser's output); `check_localiser.py --localiser snn` prints the per-clip centroid-vs-label table for it.
- [ ] Tests: `evaluate_clip` with `FrameCentroid` as the localiser reproduces the centroid path within 2 px on the analytic clip; `make_localiser` dispatch.
- [ ] Commit "Localiser wired into evaluation and the scripts".

### Task 6: Train, score, bookkeeping

- [ ] `make_frames.py --downsample 8` for the 200 sim clips if not present; `train_localiser.py --epochs 20` (background; log to `runs/localiser/train.log`).
- [ ] `check_localiser.py --set development --localiser snn` vs the centroid table; `run_experiment.py --set development --memory snn_phasemap --localiser snn --subtract-offset` vs the centroid-fed numbers.
- [ ] Log (run 15), `PLAN.md` §C/Stage 1 status, `docs/implementation-plan.md` module table, `docs/snn-experiments-summary.md`. Commit.

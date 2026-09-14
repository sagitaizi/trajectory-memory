# Simulating event data with v2e

`v2e` turns ordinary video frames into a realistic DVS event stream. It's how you get
unlimited labelled trajectories through the DVXplorer's camera model while the real camera is
away.

Ref: Hu, Liu & Delbruck, "v2e: From Video Frames to Realistic DVS Events", CVPRW 2021.
Alternative (3-D rendered scenes): ESIM (Rebecq et al., CoRL 2018).

## What it models
Per pixel: log-intensity over time, threshold crossings emit events. Non-idealities you can
set:
- **threshold** — mean `θ` for ON and OFF, plus **σ** (pixel-to-pixel jitter).
- **refractory period** — dead time after an event.
- **leak rate** — spurious ON events from junction leakage.
- **shot noise rate** — noise events, scales with low light.
- **cutoff frequency** — photoreceptor bandwidth (blurs fast changes).
- **hot pixels** — always-firing pixels.

Input: a frame sequence + its frame rate (upsampled internally). Output: DVS events
(`.h5` / AEDAT / text), optionally with a "DVS video" preview.

## Pipeline for this project
1. **Render** a moving blob (or a rendered black square) along a `TrajectorySpec` with
   OpenCV / matplotlib at a **high frame rate** (e.g. 500–1000 fps) so v2e has smooth motion
   to work from. Blob size ≈ the real target's apparent size.
2. **Apply the lens model**: distort the blob's image-plane position with the DVXplorer's
   radial-tangential coefficients (`camera.calibration`, `distort_normalized`) before
   drawing — or render undistorted then warp the frame. Do it on **one** side only, consistently.
3. **Run v2e** at 640×480 with threshold / σ / noise taken from `camera/tuning.py` (the
   DVXplorer contrast-threshold constants) in the Thesis repo.
4. **Ground truth** = the `TrajectorySpec` sample, exact, at every timestamp.

## Practical constraints found wiring `trajmem/simulate.py` (2026-09-10)
- Integrated via the **`v2ecore.emulator.EventEmulator` Python API**, not the CLI:
  `EventEmulator(seed=, device="cpu", pos_thres=, ...)` then `generate_events(frame_f32,
  t_seconds)` per frame (first call returns `None`); concatenate the `(M, 4)` `[t, x, y, pol]`
  chunks. Output is converted to a structured array (`x, y, timestamp_us, polarity∈{0,1}`).
- **`cutoff_hz` must sit well below the render `fps`** or v2e's photoreceptor IIR filter is
  under-sampled ("large maximum update eps" warning). Rule of thumb: `fps >= ~15 * cutoff_hz`.
- **Event count is very sensitive to blob/background contrast.** A 9x intensity ratio at
  1000 fps gave ~3 M ev/s (v2e emits ~ln(ratio)/pos_thres events per edge pixel per frame).
  Start with a modest ratio (~2x) and raise it only to match the measured real rate.

## Match to real (done 2026-09-15, `scripts/match_sim_real.py`)
Against `fan/fan_brush_slow_02`, the one labelled real clip on a clean analytic path: an
ellipse fitted to its hand-labels (semi-axes 37 x 23 px, T = 1.189 s, residual median
4.5 px) is simulated with the same timing at 650 fps and both streams are scored over
the first 5 s, within 60 px of the ground truth and outside it:

| | real | sim |
|---|---|---|
| target event rate (ev/s) | 82.0 k | 89.1 k |
| ON fraction | 0.509 | 0.507 |
| footprint: median event distance from gt (px) | 18.2 | 18.3 |
| noise far from target (ev/px/s) | 0.291 | 0.292 |

Reached with blob radius 18 px, background 30 / blob 170 (8-bit), thresholds 0.2,
sigma 0.03, shot noise 0.33 Hz/px (`params.yaml`). Starting values (radius 12, 64/128,
noise 0.01) were 4.5x short on target rate and 36x short on noise. What the match does
not capture: the string above the brush, and the brush's bristle texture -- the sim blob
is a disc. `runs/match/` keeps the montage, `stats.json`, and both clips as `.npz` for
`scripts/replay_gt.py`.

## Domain randomisation
For each generated clip, sample threshold, σ, noise rate, blob contrast, and a small random
affine on the path within plausible ranges. A model trained across the spread transfers to
real data far better than one trained on one idealised setting.

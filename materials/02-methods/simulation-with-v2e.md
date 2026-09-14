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
| target event rate (ev/s) | 82.0 k | 82.0 k |
| ON fraction | 0.509 | 0.505 |
| footprint: median event distance from gt (px) | 18.2 | 17.9 |
| trail: 90th pct of distance behind the target (px) | 19.6 | 17.4 |
| noise far from target (ev/px/s) | 0.291 | 0.292 |

Reached with blob radius 18 px, background 30 / blob 115 (8-bit), thresholds 0.2,
sigma 0.03, shot noise 0.33 Hz/px, **photoreceptor filter off** (`params.yaml`).
Starting values (radius 12, 64/128, noise 0.01) were 4.5x short on target rate and
36x short on noise.

**Why the filter is off.** v2e scales the photoreceptor time constant by 275/(I+20),
so with a dark background (I = 30) the trailing edge of the blob decays with tau ~ 29 ms
and OFF events dribble out for ~60-90 ms after the target has passed -- a trail the real
camera does not show. On the slow fan clip this was invisible (trail 22.7 vs 19.6 px);
on `pendulum/small_01` (~720 px/s) it was 36 px against 24 real, and 17 with the filter
off. The real DVXplorer's bandwidth in room light is far above our motion, so no filter.

**What the match does not capture**, seen on the pendulum clip: the real target there
is a ~120 x 30 px brush hanging on a visible string, its scene noise is 2.6 ev/px/s
(9x the fan's), and its rate 594 k/s. The corpus randomisation ranges were widened to
span both clips, and the renderer draws elongated targets and strings (below).
`runs/match*/` keep each comparison's montage, `stats.json`, and both clips as `.npz`
for `scripts/replay_gt.py`.

## Domain randomisation (`scripts/make_sim_dataset.py`)
Per clip, from `sim.randomise`: threshold, sigma, shot noise, blob contrast, size, aspect
(1 = disc to 4 = brush), angle, an optional string to a pivot above the frame (the target
then hangs along it), and the path -- shape, size, period, position, rotation, half with
a scripted deviation. 30 % of paths are exact; the rest carry small smooth imperfections
(`trajectories.Wobble`: amplitude beating, slow centre wander, period jitter, slow
growth/decay), each drawn from a range starting at zero, so the corpus runs from perfect
to clearly imperfect without any of it looking like a deviation. A model trained across
the spread transfers to real data far better than one trained on one idealised setting.

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

## Match to real (`scripts/match_sim_real.py`)
Against `fan/fan_brush_slow_02`, the one labelled real clip on a clean analytic path: an
ellipse fitted to its hand-labels (semi-axes 37 x 23 px, T = 1.189 s, residual median
4.5 px) is simulated with the same timing at 650 fps and both streams are scored over
the first 5 s, within 60 px of the ground truth and outside it:

| | real | sim |
|---|---|---|
| target event rate (ev/s) | 82.0 k | 84.5 k |
| ON fraction | 0.509 | 0.504 |
| footprint: median event distance from gt (px) | 18.2 | 18.4 |
| trail: 90th pct of distance behind the target (px) | 19.6 | 17.1 |
| noise floor: median 32-px tile far from target (ev/px/s) | 0.105 | 0.088 |

Matched values (`params.yaml`): blob radius 18 px, texture depth 0.4, background 30 /
blob 90 (8-bit), thresholds 0.2, sigma 0.03, shot noise 0.1 Hz/px, photoreceptor filter
off. The noise floor is the *median tile* rate because the real far-field is uneven (hot
pixels, flicker, the string; 95th-percentile tiles run 1-8 ev/px/s) and a mean would
put that everywhere.

**Why the filter is off.** v2e scales the photoreceptor time constant by 275/(I+20), so
against a dark background the trailing edge decays with tau ~ 29 ms and OFF events
dribble out for 60-90 ms behind a fast target -- a trail the real camera does not show
(on `pendulum/small_01`, ~720 px/s: 36 px with the filter vs 24 real vs 17 without).
The DVXplorer's bandwidth in room light is far above our motion.

**Beyond the fan clip**: the pendulum target is a ~120 x 30 px brush on a visible string,
with 2.6 ev/px/s of scene clutter and 594 k ev/s; the randomisation ranges span both
clips and the renderer draws elongated, textured targets on strings. `runs/match*/`
keep each comparison's montage, `stats.json`, and both clips as `.npz` for `replay_gt.py`.

## Domain randomisation (`scripts/make_sim_dataset.py`)
Per clip, from `sim.randomise`: threshold, sigma, shot noise, an optional photoreceptor
lag (half the clips, 60-200 Hz), blob contrast, size, aspect (1 = disc to 4 = brush),
angle, a body texture (flat to bristly, so events fire inside the target and it leaves
a wake), an optional string to a pivot above the frame (the target then hangs along it),
and the path -- shape, size, period, position, rotation, half with
a scripted deviation. 30 % of paths are exact; the rest carry small smooth imperfections
(`trajectories.Wobble`: amplitude beating, slow centre wander, period jitter, slow
growth/decay), each drawn from a range starting at zero, so the corpus runs from perfect
to clearly imperfect without any of it looking like a deviation. A model trained across
the spread transfers to real data far better than one trained on one idealised setting.

## The corpus (`corpus/sim/`, `scripts/corpus_summary.py`)
| | |
|---|---|
| Clips | 100, 15 s each, 650 fps render, seed-reproducible (`make_sim_dataset.py --seed 0`) |
| Shapes | circle 16, ellipse 21, figure-8 22, Lissajous 18, straight sweep 23 |
| Period | 0.83–3.88 s (median 2.41) |
| With a scripted break | 48 (drift 9, shrink 14, speed change 13, shape switch 12), in the middle third |
| Exact path / with imperfections | 35 / 65 |
| Target on a string | 52 |
| Photoreceptor lag (60–200 Hz) | 52 |
| Events per clip | 0.5–86.9 M (median 7.7 M) |
| On disk | 14.7 GB (6–1129 MB per clip) |

The manifest (`corpus/sim/manifest.csv`) records each clip's draw. The heaviest clips are
fast circles with a large textured target at a low threshold (up to 5.8 M ev/s) --
realistic (the real pendulum runs 1.5 M ev/s). On ~10 % of clips the string is as bright
as the target and the classical centroid locks onto it; kept, as the real pendulum's
string is the brightest line in its frames.

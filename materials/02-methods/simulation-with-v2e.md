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

## Match to real (do this day one, while the camera is here)
Record one real repetitive clip, simulate the same nominal path, and compare:
- total event rate (events/s) and its time profile,
- spatial event histogram (where events land),
- polarity balance.
Adjust threshold / noise until they're close. This is the "characterised parameters"
procedure (Frontiers in Neuroscience 2021).

## Domain randomisation
For each generated clip, sample threshold, σ, noise rate, blob contrast, and a small random
affine on the path within plausible ranges. A model trained across the spread transfers to
real data far better than one trained on one idealised setting.

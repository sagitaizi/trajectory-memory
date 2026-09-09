# Recording Plan — one night only

The camera leaves **2026-09-10**. Everything the paper is validated on comes from tonight.
Room lighting is fine (no flicker handling needed). Record foreground with `--preview`, one
clip per "go", **verify duration after every clip** (USB link is flaky).

Recorder (from the main thesis repo, `D:\Projects\Thesis`):
```
conda activate thesis
python -m recording.recorder --preview --out <name>.aedat4
```
Copy finished clips into `D:\Projects\trajectory-memory\corpus\real\`.

Objects: **black square** (primary), **ball** (second rung). Hand is its own setup.

Target: ~20–30 clips, **20–40 s each** so every clip holds many cycles.

## Setup 1 — object on the ceiling fan
Circular, roughly constant angular velocity. Ground truth: period measured from the data,
radius fixed (measure it), centre at the fan hub.

- [ ] Black square, slow fan speed — 3 clips
- [ ] Black square, fast fan speed — 3 clips
- [ ] Ball, one fan speed — 2 clips
- [ ] **Deviation clip:** start at one fan speed, change it partway — 2 clips

## Setup 2 — object on a string (pendulum)
Planar swing, near-analytic (measure string length and pivot height). Natural amplitude decay
is a slow deviation on its own.

- [ ] Black square, small amplitude — 2 clips
- [ ] Black square, large amplitude — 2 clips
- [ ] Ball — 2 clips
- [ ] **Deviation clip:** re-push mid-swing, or nudge it out of plane — 2 clips

## Setup 3 — hand movement
No analytic ground truth — will need sparse hand-labels + interpolation, so keep these
**shorter (15–20 s)** and fewer. Move along a taped path on a table to keep it repeatable.

- [ ] Repeating loop along a taped path — 3 clips
- [ ] **Deviation clip:** follow one taped path, then switch to a second — 2 clips

## Setup 4 — motors move, camera on a blank wall
Camera sweeps via the pan-tilt motors; a single black square taped to an otherwise blank wall
traces a path across the sensor. Apparent (ego-motion-induced) trajectory with **exact ground
truth** from the encoders through the calibration. Cleanest GT of the night. Note this is
apparent motion, not object motion — best for pretraining the memory core, weaker as a
"real moving target" test.

Use `setup_with_safety()` for motor bring-up. Script the sweep as a repeating path.

- [ ] Circular sweep, slow — 2 clips
- [ ] Circular sweep, fast — 2 clips
- [ ] Figure-eight sweep — 2 clips
- [ ] **Deviation clip:** circle, then switch to figure-eight or change amplitude — 2 clips

## After every clip
- [ ] Replay / check reported duration matches what you recorded.
- [ ] Confirm the target is visible and events are dense enough (use the `inspect-recording`
      skill or `recording.player --dump`).
- [ ] Re-record if the clip stopped short or the stream dropped mid-way.

## Measure and write down
- Fan hub position and blade radius to the object; fan speed settings used.
- String length, pivot height, swing plane.
- Taped-path geometry for the hand clips (photograph it).
- Nothing extra for setup 4 — the encoders and calibration carry it.

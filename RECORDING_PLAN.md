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

Objects available:
- **Printed black square** on a paper sheet — hand-held or taped to a wall only; cannot hang
  from a string or a fan. Use it for setups 3 and 4.
- **Brush** — the object from the earlier brush corpus; hangs from the fan or ties to a
  string. Main object for setups 1 and 2.
- Other small light objects that can be clipped or tied — optional variety for setups 1–2.

Target: ~20–30 clips, **20–40 s each** so every clip holds many cycles.

## Setup 1 — object on the ceiling fan
Brush (or another hangable object) attached to a blade.
- If it's fixed **rigidly** near the blade tip it orbits — a clean circle, and the period /
  radius / hub centre are the ground truth.
- If it hangs **loose** it swings as well as orbits (this is what the brush did before) —
  ground truth is then sparse hand-labels + interpolation, not analytic.
Try a rigid mount first; fall back to loose if that's all that's possible.

- [ ] Brush, slow fan speed — 3 clips
- [ ] Brush, fast fan speed — 3 clips
- [ ] A second object (if available), one fan speed — 2 clips
- [ ] **Deviation clip:** start at one fan speed, change it partway — 2 clips

## Setup 2 — object on a string (pendulum)
Brush tied to a string; other tie-able objects if handy. Planar swing, near-analytic — measure
string length and pivot height, use a damped-pendulum model. Natural amplitude decay is a slow
deviation on its own.

- [ ] Brush, small amplitude — 2 clips
- [ ] Brush, large amplitude — 2 clips
- [ ] A second object (if available) — 2 clips
- [ ] **Deviation clip:** re-push mid-swing, or nudge it out of plane — 2 clips

## Setup 3 — hand movement
Printed black-square sheet (ideal here) and/or a ball, moved by hand along a taped path on a
table. No analytic ground truth — sparse hand-labels + interpolation — so keep these
**shorter (15–20 s)**.

- [ ] Repeating loop along a taped path — 3 clips
- [ ] **Deviation clip:** follow one taped path, then switch to a second — 2 clips

## Setup 4 — clean background (blank wall)
Two ways to get a repeating path with no background clutter. Do both.

**4a — motors move, square static.** Printed square taped to a blank wall; the camera is swept
by the pan-tilt motors in a repeating path. The apparent (ego-motion-induced) trajectory has
**exact ground truth** from the encoders through the calibration — the cleanest GT of the
night. Best for pretraining; weaker as a "real moving target" test. Use `setup_with_safety()`.

- [ ] Circular sweep, slow — 2 clips
- [ ] Circular sweep, fast — 2 clips
- [ ] Figure-eight sweep — 2 clips
- [ ] **Deviation clip:** circle, then switch to figure-eight or change amplitude — 2 clips

**4b — camera static, square hand-moved.** Camera on a tripod, move the printed square by hand
across the blank wall in a repeating loop. Real object motion, and the clean background makes
labelling easy. Ground truth: sparse hand-labels + interpolation.

- [ ] Repeating loop — 3 clips
- [ ] **Deviation clip:** loop, then change size or switch shape partway — 2 clips

## After every clip
- [ ] Replay / check reported duration matches what you recorded.
- [ ] Confirm the target is visible and events are dense enough (use the `inspect-recording`
      skill or `recording.player --dump`).
- [ ] Re-record if the clip stopped short or the stream dropped mid-way.

## Measure and write down
- Fan: how the brush is attached (rigid vs loose), hub position, blade radius to the object,
  fan speed settings used.
- String: length, pivot height, swing plane.
- Taped / hand-moved path geometry for setups 3 and 4b — photograph it.
- Setup 4a needs nothing extra — the encoders and calibration carry it.

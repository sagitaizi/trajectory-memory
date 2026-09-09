# Recording night — condensed checklist

Full detail: `../../RECORDING_PLAN.md`. Room lighting is fine (no flicker handling).

## Before
- [ ] `conda activate thesis`; camera enumerates (`python -m recording.recorder --help`).
- [ ] Direct USB-3 port; reseat if the link looks marginal.
- [ ] Objects ready: **printed black-square sheet** (hand/wall only), **brush** (hangs/ties),
      any other small clippable objects, a ball.
- [ ] `corpus/real/` exists in the trajectory-memory repo.

## Capture — 4 setups, ~20–30 clips total, 20–40 s each
- [ ] **Fan**: brush on the ceiling fan (rigid = clean circle; loose = swing, hand-label GT).
      Slow + fast speed. Another object ×2 if available. One clip: change fan speed partway.
- [ ] **String**: brush tied to a string, pendulum swing. Small + large amplitude. Another
      object ×2 if available. One clip: re-push / nudge out of plane partway.
- [ ] **Hand**: printed square sheet (and/or ball) along a taped path (15–20 s). One clip:
      switch to a second taped path partway.
- [ ] **Blank wall 4a**: square taped to the wall, camera swept by the motors — circle
      slow/fast, figure-eight. One deviation clip. Use `setup_with_safety()`. Exact GT.
- [ ] **Blank wall 4b**: camera on a tripod, hand-move the square across the wall in a loop.
      One deviation clip. Hand-label GT, but clean background.

## After every clip
- [ ] Replay and confirm the reported duration matches.
- [ ] Target visible, event density adequate (`inspect-recording` skill or
      `recording.player --dump`).
- [ ] Re-record anything that stopped short or dropped mid-stream.
- [ ] Copy to `corpus/real/` with a clear name (`fan_square_slow_01.aedat4`, …).

## Measure and note
- [ ] Fan: brush attachment (rigid vs loose), hub position, blade radius, speed settings.
- [ ] String length, pivot height, swing plane.
- [ ] Taped / hand-moved path geometry for setups 3 and 4b — photograph it.
- [ ] (Setup 4a needs nothing extra — encoders + calibration are the ground truth.)

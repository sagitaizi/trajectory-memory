# Recording night — condensed checklist

Full detail: `../../RECORDING_PLAN.md`. Room lighting is fine (no flicker handling).

## Before
- [ ] `conda activate thesis`; camera enumerates (`python -m recording.recorder --help`).
- [ ] Direct USB-3 port; reseat if the link looks marginal.
- [ ] Target objects ready: black square (primary), ball.
- [ ] `corpus/real/` exists in the trajectory-memory repo.

## Capture — 4 setups, ~20–30 clips total, 20–40 s each
- [ ] **Fan**: object on the ceiling fan. Slow + fast speed. Black square ×several, ball ×2.
      One clip: change fan speed partway (deviation).
- [ ] **String**: pendulum swing. Small + large amplitude. Square + ball. One clip: re-push
      / nudge out of plane partway (deviation).
- [ ] **Hand**: repeat a loop along a taped path (15–20 s). One clip: switch to a second
      taped path partway (deviation).
- [ ] **Motor + blank wall**: black square on a blank wall, camera swept by the motors.
      Circle slow/fast, figure-eight. One clip: switch shape / amplitude partway (deviation).
      Use `setup_with_safety()`.

## After every clip
- [ ] Replay and confirm the reported duration matches.
- [ ] Target visible, event density adequate (`inspect-recording` skill or
      `recording.player --dump`).
- [ ] Re-record anything that stopped short or dropped mid-stream.
- [ ] Copy to `corpus/real/` with a clear name (`fan_square_slow_01.aedat4`, …).

## Measure and note
- [ ] Fan hub position, blade radius to the object, speed settings used.
- [ ] String length, pivot height, swing plane.
- [ ] Taped-path geometry for hand clips — photograph it.
- [ ] (Motor setup needs nothing extra — encoders + calibration are the ground truth.)

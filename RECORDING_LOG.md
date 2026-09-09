# Recording Log — 2026-09-10

Clips captured this session, with verification and the measurements `RECORDING_PLAN.md`
asks for. Clips live in `corpus/real/` (gitignored); this log is tracked.

Layout: `corpus/real/<group>/<slug>/<slug>.aedat4` — one directory per clip, holding
its `.aedat4` + `.params.yaml` (+ later hand-labels, notes). Deviation clips go under
`<group>/break/`.

Verify a clip with:  `bash scripts/check <group>/<slug>.aedat4`

## Status

| Setup | Planned | Captured | Verified |
|---|---|---|---|
| 1 — ceiling fan | 10 | 12 (8 + 4 break) | stats only — motion quality unconfirmed |
| 2 — string pendulum | 8 | 0 | 0 |
| 3 — hand-moved (square/ball) | 5 | 0 | 0 |
| 4a — motor-swept camera (square) | 8 | 0 | 0 |
| 4b — camera static, square hand-moved | 5 | 0 | 0 |
| extra — multi-target (out of scope, future work) | — | 4 (3 + 1 break) | stats only |

## Clips — in scope

All 30 s, 640×480, events+imu, no drops. "stats OK" = full length + healthy rate;
motion quality (clean path vs swing/dangle, framing) still needs a frame check.

The fan target is a brush on a **string** tied to a blade → orbit + pendulum swing,
**not a rigid circle**. GT = hand-labels + interpolation, not analytic.

| Clip | Mean rate | Verdict |
|---|---|---|
| fan/fan_brush_slow_01 | 259 k/s | **SUSPECT** — denoised rate rises 10x over the clip (fan spin-up); non-stationary |
| fan/fan_brush_slow_02 | 165 k/s | stats OK; frame check pending |
| fan/fan_brush_slow_03 | 166 k/s | stats OK; frame check pending |
| fan/fan_brush_fast_01 | 218 k/s | stats OK; frame check pending |
| fan/fan_brush_fast_02 | 200 k/s | stats OK; frame check pending |
| fan/fan_string_01 | 213 k/s | stats OK; frame check pending |
| fan/fan_two_strings_01 | 256 k/s | stats OK; frame check pending |
| fan/fan_blade_01 | 615 k/s | one bare blade — extended object, not a blob; questionable fit, frame check pending |
| fan/break/fan_brush_break_01 | 248 k/s | stats OK; frame check pending |
| fan/break/fan_brush_break_02 | 278 k/s | stats OK; frame check pending |
| fan/break/fan_string_break_01 | 248 k/s | stats OK; frame check pending |
| fan/break/fan_two_strings_break_01 | 288 k/s | stats OK; frame check pending |

## Clips — extra (out of scope for paper 4c)

Multiple simultaneous targets — kept for the future attention-span / habituation work.
Not verified beyond stats; GT would be per-target hand-labels.

| Clip | Mean rate | Notes |
|---|---|---|
| extra/dual_fan_brush_and_hand_01 | 467 k/s | fan-string brush + hand-moved object |
| extra/dual_fan_brush_and_square_01 | 378 k/s | fan-string brush + printed square |
| extra/triple_fan_blade_01 | 1759 k/s | 3 bare fan blades — near sensor throughput limit |
| extra/break/triple_fan_blade_break_01 | 1068 k/s | 3 blades, speed change partway |

## Measurements

### Setup 1 — fan
- Brush attachment (rigid / loose):
- Hub position in frame:
- Blade radius to the object:
- Fan speed settings used:

### Setup 2 — string
- String length:
- Pivot height:
- Swing plane:

### Setup 3 — hand path
- Taped path geometry (photo ref):

### Setup 4a — motor sweep
- (encoders + calibration are the ground truth — nothing extra)
- Scan params used per clip:

### Setup 4b — square on blank wall
- Path geometry (photo ref):

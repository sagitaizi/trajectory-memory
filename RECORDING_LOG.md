# Recording Log — 2026-09-10

Clips captured this session, with verification and the measurements `RECORDING_PLAN.md`
asks for. Clips live in `corpus/real/` (gitignored); this log is tracked.

Layout: `corpus/real/<setup>/`, with deviation clips under `corpus/real/<setup>/break/`.

Verify a clip with:  `bash scripts/check <setup>/<name>.aedat4`

## Status

| Setup | Planned | Captured | Verified |
|---|---|---|---|
| 1 — ceiling fan | 10 | 11 (7 + 4 break) | stats only — motion quality unconfirmed |
| 2 — string pendulum | 8 | 0 | 0 |
| 3 — hand-moved (square/ball) | 5 | 0 | 0 |
| 4a — motor-swept camera (square) | 8 | 0 | 0 |
| 4b — camera static, square hand-moved | 5 | 0 | 0 |

## Clips

All 30 s, 640×480, events+imu, no drops. "stats OK" = full length + healthy rate;
motion quality (clean circle vs swing/dangle, framing) still needs a frame check.

| File | Mean rate | Verdict |
|---|---|---|
| fan/fan_brush_slow_01 | 259 k/s | **SUSPECT** — denoised rate rises 10x over the clip (fan spin-up?); motion a thin vertical smear at the right frame edge, not a centred circle |
| fan/fan_brush_slow_02 | 165 k/s | stats OK; frame check pending |
| fan/fan_brush_slow_03 | 166 k/s | stats OK; frame check pending |
| fan/fan_brush_fast_01 | 218 k/s | stats OK; frame check pending |
| fan/fan_brush_fast_02 | 200 k/s | stats OK; frame check pending |
| fan/fan_string_01 | 213 k/s | stats OK; frame check pending |
| fan/fan_two_strings_01 | 256 k/s | stats OK; frame check pending |
| fan/break/fan_brush_break_01 | 248 k/s | stats OK; frame check pending |
| fan/break/fan_brush_break_02 | 278 k/s | stats OK; frame check pending |
| fan/break/fan_string_break_01 | 248 k/s | stats OK; frame check pending |
| fan/break/fan_two_strings_break_01 | 288 k/s | stats OK; frame check pending |

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

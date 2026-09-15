# Recording Log — 2026-09-10

Every real clip: setup, event rate, what it shows, label status. Clips live in
`corpus/real/<group>/<slug>/<slug>.aedat4` (git-ignored) with a `.params.yaml` side-car and,
once labelled, `.labels.csv` / `.deviation.json` (tracked). Which clips are development or
held-out: `corpus/sets.yaml`.

## Status

| Setup | Planned | Captured | Verified |
|---|---|---|---|
| 1 — ceiling fan | 10 | 12 (8 + 4 break) | stats only — motion quality unconfirmed |
| 2 — string pendulum | 8 | 9 (7 swing + 2 break) | stats OK + frame check — **good SNR, target clean** |
| 3 — hand-moved (square/ball) | 5 | — | **skipped by decision** — see Measurements |
| 4a — motor-swept camera (square) | 8 | 23 (18 + 5 break) | stats + motor sidecar OK; frame check pending |
| 4b — camera static, square hand-moved | 5 | 5 (3 loop + 2 break) | stats OK + frame check — **high noise floor, marginal SNR** |
| extra — multi-target (out of scope, future work) | — | 5 (4 + 1 break) | stats only |

## Clips — in scope

Fan clips 30 s, 4a clips 40 s; all 640×480, events+imu, no drops. "stats OK" = full
length + healthy rate; motion quality (clean path vs swing/dangle, framing) still
needs a frame check.

The fan target is a brush on a **string** tied to a blade → orbit + pendulum swing,
**not a rigid circle**. Sagi confirmed each clip is a *settled, repetitive* dangle
(waited for it to stabilise before recording) — usable, just a harder path to
predict than a clean circle. GT = hand-labels + interpolation, not analytic.

| Clip | Mean rate | Verdict |
|---|---|---|
| fan/fan_brush_slow_01 | 259 k/s | denoised rate rises ~10x over the clip (fan spin-up) — non-stationary, use with care or trim the ramp |
| fan/fan_brush_slow_02 | 165 k/s | **Labelled** — a clean, rigid **ellipse** ~85 × 45 px, T = 1.20 s (50 rpm), 25 laps at constant amplitude. The only true 2-D loop in the development set and the closest real clip to the simulator's analytic paths. Brush behaved as if on a rod, not a swinging string |
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

## Clips — Setup 4a (motor-swept camera, printed square on blank wall)

Ego-motion clips: the target is a static printed square and the pan-tilt rig sweeps a
repeating path, so the whole scene moves. Ground truth is hand-labels like every other
setup (the encoder track does not reproduce the sweep through the calibration; the
`.motors.csv` side-cars remain). Scan parameters are not in `.params.yaml`, so they are
recorded here per clip (`axis / period s / range`): triangle waves, pan over `period`, tilt
over `1.6×period` for `both`, tilt phase-locked at `period` for `diag`/`antidiag`. All
clips 40 s; motors slew to the start for the first ~0.2 s, trimmed on load.

## Plain sweeps (18)

| Clip | axis / period / range | Rate | Span (pan/tilt ticks) | Notes |
|---|---|---|---|---|
| wall/scan_pan_slow_01 | pan / 8.0 / 0.10 | 215 k/s | 200 / — | clean |
| wall/scan_pan_slow_02 | pan / 6.0 / 0.12 | 237 k/s | 241 / — | 2 encoder samples dropped (transient) |
| wall/scan_pan_slow_03 | pan / 5.0 / 0.11 | 239 k/s | 219 / — | clean |
| wall/scan_pan_slow_04 | pan / 4.0 / 0.10 | 234 k/s | 198 / — | "slow" is a misnomer (period 4 s) |
| wall/scan_pan_slow_05 | pan / 3.0 / 0.12 | 308 k/s | 237 / — | "slow" is a misnomer (period 3 s) |
| wall/scan_pan_fast_01 | pan / 1.0 / 0.05 | 296 k/s | 92 / — | small amplitude |
| wall/scan_pan_fast_02 | pan / 0.1 / 0.13 | 302 k/s | 88 / — | **motor can't track 10 Hz command — amplitude collapsed, ragged. Drop or ignore.** |
| wall/scan_pan_fast_03 | pan / 0.7 / 0.11 | 639 k/s | 194 / — | tracks ~86% of commanded amplitude — borderline |
| wall/scan_tilt_slow_01 | tilt / 6.0 / 0.20 | 213 k/s | 31 / 118 | pan held at centre |
| wall/scan_tilt_slow_02 | tilt / 5.0 / 0.20 | 265 k/s | 31 / 117 | |
| wall/scan_tilt_slow_03 | tilt / 4.0 / 0.17 | 325 k/s | 31 / 99 | |
| wall/scan_tilt_fast_01 | tilt / 1.0 / 0.10 | 258 k/s | 30 / 71 | |
| wall/scan_tilt_fast_02 | tilt / 1.0 / 0.19 | 440 k/s | 29 / 108 | |
| wall/scan_tilt_fast_03 | tilt / 2.0 / 0.15 | 279 k/s | 30 / 88 | |
| wall/scan_both_slow_01 | both / 4.0 / 0.12 | 341 k/s | 238 / 75 | 2D Lissajous (tilt period 6.4 s) |
| wall/scan_both_fast_01 | both / 1.0 / 0.11 | 722 k/s | 208 / 73 | recorded as `scan_both_slow_01`, renamed |
| wall/scan_diag_01 | diag / 4.0 / 0.12 | 381 k/s | 238 / 76 | single diagonal, ~16° slope |
| wall/scan_diag_02 | diag / 3.0 / 0.12 | 458 k/s | 236 / 77 | |

### Deviation clips (5)

Deviation introduced by hand mid-clip. **Detection-only GT** — encoders give camera
pose, not the square's hand-moved path, so post-break GT is a hand-labelled break
time, not an analytic path. Break times TBD from the label pass.

| Clip | axis / period / range | Dur | Rate | Notes |
|---|---|---|---|---|
| wall/scan_pan_slow_break_01 | pan / 4.0 / 0.12 | 30.7 s | 579 k/s | stopped early; ~2.5× base rate at the break |
| wall/scan_pan_slow_break_02 | pan / ? / ? | 40 s | 357 k/s | **scan params not captured — confirm period/range** |
| wall/scan_pan_fast_break_01 | pan / 1.0 / 0.05 | 25.9 s | 308 k/s | stopped early — short but usable |
| wall/scan_tilt_fast_break_01 | tilt / 1.0 / 0.20 | 19.5 s | 509 k/s | stopped early — quite short |
| wall/scan_diag_break_01 | diag / 3.0 / 0.12 | 40 s | 502 k/s | full length |

4a is complete — past the planned 8, covering pan / tilt / 2D / diagonal paths.

## Clips — Setup 2 (string pendulum)

`corpus/real/pendulum/`. Static camera, brush on a ~30 cm string swinging against
the wall (measurements above). Real object motion, near-analytic GT (damped
pendulum, T ≈ 1.2 s). No motors (`--no-motors`). All 25 s, no drops.

**Frame check (`runs/check/pendulum/`, small_01 + wide_02):**
- **Good SNR** — unlike 4b, the accumulated frames keep 67–76 % of events after
  denoise (4b kept 10–50 %). The high mean rates below are mostly real: the brush
  bristles carry a lot of edge length and a wide swing sweeps many pixels.
- Brush is clean, isolated, high-contrast; no visible flicker banding despite the
  false-positive flicker detections during probing (999 / 588 / 4975 Hz on
  high-rate probes — detector tripping on the noise floor; retakes read clean).
- **Pivot sits at/near the top frame edge** — the bob is always in frame but the
  pivot point itself may be clipped in some clips (minor: complicates reading exact
  pivot height off the video; the swing arc is fully captured).

| Clip | Dur | Mean rate | Notes |
|---|---|---|---|
| pendulum/small_01 | 25.0 s | 1331 k/s | small amplitude. **Labelled** — T = 1.181 s, settled from t = 0, no net decay, amplitude beating (see GT note below) |
| pendulum/small_02 | 25.0 s | 734 k/s | small amplitude |
| pendulum/small_03 | 25.0 s | 821 k/s | small amplitude |
| pendulum/wide_01 | 25.0 s | 1218 k/s | large amplitude |
| pendulum/wide_02 | 25.1 s | 1529 k/s | large amplitude. **Labelled** — T = 1.274 s, 7.9% longer than small_01 (anharmonic, real). First ~5 s is the push settling; steady from t ≈ 6 s |
| pendulum/wide_03 | 25.0 s | 1429 k/s | large amplitude (recorded as `wide_break`, no real break — renamed) |
| pendulum/updown_02 | 25.0 s | 428 k/s | toward/away or vertical variation; only one such clip (no `updown_01`) |
| pendulum/small_break | 25.0 s | 603 k/s | deviation — re-push / out-of-plane nudge mid-swing |
| pendulum/wide_break | 25.0 s | 1021 k/s | deviation. **Labelled** — steady swing to 12.9 s (T = 1.27 s, as wide_02), push at ~13.0 s overshoots to x = 600, out of frame 13.9–14.1 s, wild to 17 s, then caught and held ~110 px lower and near-static from 18 s. Two things to detect: the path departing (13 s) and the path ceasing (18 s) |

GT: dense hand-labels at 0.1 s, linearly interpolated — model-free by choice, so the
ground truth assumes nothing a deviation detector is later judged on. `small_01`
(251 marks, 2026-09-12) gives T = 1.181 s against ~1.2 s predicted, and **no net
amplitude decay** over 21 swings (135 → 136 px) — so the decay is *not* a slow
deviation here. What it does show is amplitude *beating*: peaks ~432 px, sagging to
~405 px around t = 15–17 s, back to ~435 px by t = 24 s — a brush on a string is a
double pendulum, not a simple one. About one beat cycle fits in 25 s. Expect a
detector to flag this if its idea of "the path" is a single cycle. Break clips:
hand-mark the break time (`scripts/mark_breaks.py`) for detection scoring.

## Clips — Setup 4b (static camera, printed square hand-moved on blank wall)

`corpus/real/wall_target/`. Static camera on tripod, square moved by hand along a
repeating loop against the wall. **Real object motion** (the complement to 4a's
ego-motion). No motors (`--no-motors`, no sidecar). GT = hand-labels of the square
centroid + interpolation; photograph the loop shape (measurement pending). All 25 s.

**Frame check (`runs/check/wall_target/`, loop_02 + loop_break_01):**
- **Noise floor is high** — raw frames are heavy salt-and-pepper across the whole
  sensor; denoise removes ~50–90% of events (e.g. loop_break_01 late frame
  13.6k raw → 1.6k den). Rates below are mostly noise, not signal. Consistent with
  the flicker-probe counts that climbed through the batch (275k → 937k over 1.5 s)
  — the DVXplorer noise floor drifted up late in the session.
- A faint **persistent vertical edge** sits mid-frame in every clip (room
  corner / doorframe / cable) — the "blank" wall isn't fully clean, and a static
  scene feature showing events means minor camera shake too.
- **loop_02** framing is poor — the square rides the top-left corner and clips out
  of frame at times. loop_break_01 is framed better.
- Verdict: **usable but marginal** — the square path is recoverable after denoising;
  SNR is far worse than the fan / 4a clips. Localiser will lean on denoising.

| Clip | Dur | Mean rate (mostly noise) | Notes |
|---|---|---|---|
| wall_target/loop_01 | 25.0 s | 311 k/s | cleanest rate of the batch. **Labelled** — **not a loop**: a horizontal sweep, x 250–410 px at T = 1.80 s, y flat within 37 px. Amplitude grows ~90 → 160 px over the clip. Target sits high in frame (y ≈ 70) |
| wall_target/loop_02 | 25.0 s | 776 k/s | square clips out top-left — reframe if retaken |
| wall_target/loop_03 | 25.0 s | 687 k/s | frame check pending |
| wall_target/loop_break_01 | 25.0 s | 1047 k/s | deviation clip; highest rate, 26M events; framed OK. **Labelled** — diagonal sweep to ~10.5 s, break with an excursion to x = 366, then a *horizontal* sweep from ~13 s: the deviation is a change of direction. Lap period 1.0 s on this take, so 0.2 s marks are only 5 per lap — turnarounds undershoot by ~15–25 px. Fine for detection; use loop_01 for prediction error, or densify to 0.1 s if needed. **Usable three ways:** two independent repetitive segments (diagonal ~10 laps, horizontal ~12 laps) plus a break with a known path on *both* sides — and the natural test of re-learning after a break, which is future work, not this paper |
| wall_target/loop_break_02 | 25.0 s | 480 k/s | deviation clip; frame check pending |

For anything recorded after this (Setup 2, 3, fan re-checks): power-cycle the
DVXplorer or add `--desensitize` (contrast thresholds → 17/17) to cut the noise.

## Clips — extra (out of scope for paper 4c)

Multiple simultaneous targets — kept for the future attention-span / habituation work.
Not verified beyond stats; GT would be per-target hand-labels.

| Clip | Mean rate | Notes |
|---|---|---|
| extra/dual_fan_brush_and_hand_01 | 467 k/s | fan-string brush + hand-moved object |
| extra/dual_fan_brush_and_square_01 | 378 k/s | fan-string brush + printed square |
| extra/dual_scan_pan_slow_01 | 604 k/s | pan sweep (pan / 4.0 s / 0.10) with a second hand-moved target — recorded in `wall/`, moved here |
| extra/triple_fan_blade_01 | 1759 k/s | 3 bare fan blades — near sensor throughput limit |
| extra/break/triple_fan_blade_break_01 | 1068 k/s | 3 blades, speed change partway |

## Measurements

### Setup 1 — fan
- **Not measured, by decision** (Sagi, teardown night): the target is a brush on a
  string, not a rigid orbit, so hub/radius measurements would add more error than
  signal. GT for all fan clips = hand-labels + interpolation.

### Setup 2 — string
- Object: brush hung from a string; the brush handle (~12 cm) is the part that swayed.
- String length: ~30 cm (pivot → top of brush).
- Effective pendulum length ≈ 0.36–0.39 m (string + ~half the brush to its CoM)
  → expected period T ≈ 1.2 s (simple-pendulum estimate, √(L/g)).
- Pivot height: not measured — recover from the video (pivot visible in frame).
- Pixel scale: recover from the ~12 cm brush spanning N px in frame.
- Swing plane: intended side-on (parallel to sensor); confirm per clip from the video.
- Damping / amplitude decay over each clip is itself a slow deviation.

### Setup 3 — hand path
- **Deliberately skipped** (Sagi, teardown night): 4a (over-covered) + 4b + the
  pendulum give enough moving-target coverage, and a taped-path hand-move is
  largely a duplicate of 4b.

### Photos
- Tripod position and the 4b wall photographed (files with Sagi).

### Setup 4a — motor sweep
- Encoders + calibration were meant to be the ground truth, so nothing else was
  measured. That turned out to be the gap: the surviving explanation for the scale
  error is parallax from the camera sitting off the rotation axes, which depends on
  scene distance — and **the wall distance was never recorded**, so it cannot be
  tested after the fact. Record camera-to-scene distance in any future session.
- Scan params used per clip: see the Setup 4a clip tables above (axis / period / range).
- Safe travel: pan [1024, 3072], tilt [700, 1300] ticks; speed cap 44 (10.1 rpm), accel cap 70.
- Rig centre: pan ~2048, tilt ~1000 ticks.
- `diag` / `antidiag` axis mode added to `recording/recorder.py` this session (+ tests).

### Setup 4b — square on blank wall
- Path geometry (photo ref):

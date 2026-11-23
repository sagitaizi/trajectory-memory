# Paper progress

The MCSoC manuscript, accepted and frozen (tag `mcsoc-camera-ready`): what each section
rests on, and what is known about it. `PLAN.md` tracks the code.

## How this works

- `paper/` is **read-only to Claude**. Claude writes a section only when Sagi names it
  ("write §IV-B"), and touches nothing else in the file.
- Statuses: **blocked** → **writable** → **drafted** → **in review** → **reviewed** → **final**.
- *Drafted* means the prose is there. *In review* means Sagi is reviewing it (notes may sit
  in `%` comments). *Reviewed* means he has signed it off. *Final* means its "revisit"
  column is cleared.

## Section status

| §      | Section                    | Status      | Blocked on          | Revisit after            |
|--------|----------------------------|-------------|---------------------|--------------------------|
| —      | Abstract                   | reviewed    | —                   | —                        |
| I      | Introduction               | reviewed    | —                   | —                        |
| II     | Related Work               | reviewed    | —                   | —                        |
| III    | Method (opening paragraph) | reviewed    | —                   | —                        |
| III-A  | Problem Definition         | reviewed    | —                   | —                        |
| III-B  | Event Input                | reviewed    | —                   | —                        |
| III-C  | Spiking Localiser          | reviewed    | —                   | —                        |
| III-D  | Trajectory-Memory Network  | reviewed    | —                   | —                        |
| IV-B   | Event-Camera Simulator     | reviewed    | —                   | —                        |
| IV-A   | Live Runs                  | reviewed    | —                   | —                        |
| IV-C   | Baselines                  | reviewed    | —                   | —                        |
| IV-D   | Evaluation Protocol        | reviewed    | —                   | —                        |
| IV-E   | Metrics                    | reviewed    | —                   | —                        |
| V-A    | Results: Localiser         | reviewed    | —                   | —                        |
| V-B    | Results: Prediction + path | reviewed    | —                   | —                        |
| V-C    | Results: Deviation         | reviewed    | —                   | —                        |
| V-D    | Results: Ablations         | reviewed    | —                   | —                        |
| VI     | Discussion                 | reviewed    | —                   | —                        |
| VII    | Conclusion                 | reviewed    | —                   | —                        |
| —      | Acknowledgment             | reviewed    | —                   | —                        |
| Fig. 1 | System overview            | reviewed    | —                   | —                        |
| Fig. 2 | Recording setups           | dropped     | —                   | —                        |

## Accepted

A regular paper (MCSoC paper ID 1571362916), scored accept, borderline and accept. The
camera-ready, copyright form and registration are in. It answers the reviewers in the text: §VI
(Loihi fit and energy, `scripts/energy.py` → `runs/paper/energy.csv`) and §III-D (why six clocks).

## Where each section's material lives

- **I, II** — `materials/01-literature/` (six notes); `PLAN.md`'s novelty framing.
- **III-A** — `docs/implementation-plan.md`; `trajmem/trajectories.py`.
- **III-B** — `materials/02-methods/event-representations.md`.
- **III-C, III-D** — `docs/implementation-plan.md`; `PLAN.md` phase C and gate G-F;
  `docs/snn-experiments-summary.md`; `materials/01-literature/multi-timescale-memory.md`;
  specs in `docs/superpowers/specs/`.
- **V-D** — `PLAN.md` phase C (ablations); `docs/snn-experiments-log.md`;
  `materials/02-methods/legendre-memory-unit.md`.
- **III-D deviation, IV-D alarm** — `materials/01-literature/deviation-and-novelty.md`.
- **IV-A** — `materials/02-methods/simulation-with-v2e.md`; `trajmem/simulate.py`; `params.yaml`.
- **IV-B** — `RECORDING_LOG.md`; `corpus/sets.yaml`; `corpus/real/`.
- **IV-C** — `materials/02-methods/{kalman-periodic,rhythmic-dmp}.md`;
  `materials/01-literature/classical-baselines.md`.
- **IV-D** — `PLAN.md`: pretrain on simulation, freeze, test on real; `corpus/sets.yaml`.
- **IV-E** — `materials/02-methods/metrics.md`.
- **Fig. 1** — `docs/implementation-plan.md`.
- **§V (all)** — pools every real run. Sources, all under `runs/paper/`:
  `scripts/paper_results.py run` (labelled runs: `development/`, `held_out/`),
  `scripts/score_breaks.py` (deviation-only runs: `held_out_breaks/`),
  `scripts/score_tracks.py` (unlabelled runs against the tracked position: `tracks/`),
  `scripts/paper_results.py run --inputs snn --clips fan_brush_slow_02 loop_01 loop_break_01
  --out runs/paper/reverse_swap` (the input swap on the other setups),
  `scripts/spike_counts.py` (activity). `scripts/paper_figures.py` makes the figures
  (`paper/figures/results_*.pdf`) and tables (`runs/paper/tables.tex`);
  `scripts/paper_numbers.py` writes every in-text number to `runs/paper/numbers.csv`.
- **Architecture figures** — `figures/architecture/`, TikZ sources + PDFs. `pipeline` is
  Fig. 1 (in `main.tex`, placed before §I so it floats to page 2; opens with the plain rig
  photo; raster panels from `make_assets.py`). Not in `main.tex`: `clock_and_map` (§III-D,
  one column); `rig` (the labelled photo; the motors are never used). `system_overview` is superseded.

Every paper cited needs an entry in `BIBLIOGRAPHY.md` in the same change. `paper/refs.bib` is
Sagi's, managed outside the repo; the state it was in when he took it over is
`materials/01-literature/refs-snapshot-2026-09-14.bib` (18 entries, 9 marked `verify`).

## Pending

Naming in the paper: the setups are *pendulum*, *rotating arm* and *hand-moved*. Results pool
every real run (19, six with a deviation). §IV-D says the system was developed on six of the
runs, where k (centroid 25, spiking localiser 30) and the slow copy's rate were chosen; it gives
no selection rule (the "smallest k" rule would give 21 and 12). The corpus and code still say
`development`/`held_out`. The paper never uses "onset", and calls ours and the classical
comparators *memories*, not *methods*.

Rotating-arm deviation runs (`fan_brush_break_01/02`): once per turn the target turns faint and
both localisers move onto its string (centroid to mid-string, flickering; spiking localiser to
the string's top). The 5.5 s detection is a single brief crossing of the threshold. Decided:
keep 6 of 6 and name the cause in §V-C. §IV-A gives no size for the arm.

## The 4a encoder ground truth was abandoned (2026-09-12)

Setup 4a was recorded on the premise that the encoders give exact ground truth for free,
which is why 23 clips were captured against a planned 8. That premise did not hold, and
**all clips now get ground truth the same way: hand-labels plus interpolation.**

What happened, in case §IV-B or a reviewer needs it. `scripts/replay_gt.py` drew the
encoder ground truth over the clip it describes and the marker sat left of the target by up
to 42 px, worst furthest from the anchor. `scripts/measure_px_per_tick.py` measured the real
image shift against the encoder track across the 4a set: the calibration understates the
sweep by **1.268x on pan** (sd 0.013, 8 clips) and **1.295x on tilt** (sd 0.009, 9 clips),
both at r > 0.98. The shape and the axis signs are right; only the scale is wrong.

Both axes being off by nearly the same factor rules out a per-axis encoder error. The gap is
the main thesis repo's open `ticks_per_radian` anomaly (`D:\Projects\Thesis\params.yaml`,
which states in capitals that it is narrowed, not resolved). Its surviving candidate is the
camera sitting offset from the rotation axes, making part of the image motion parallax —
which depends on scene distance. Every 4a clip is one wall at one distance, so they agree
tightly with each other and not with a calibration measured under other geometry, and the
corpus contains no distance variation to settle it with. The 4a wall distance was not
recorded.

Consequences for the paper: 4a is ego-motion data with hand-labelled ground truth, no better
founded than the other setups, so **do not present it as the exact-ground-truth condition**.
The encoder path still exists in `trajmem/data.py` (`ego_gt`, reached by passing `anchor=`)
and `scan_pan_slow_01` still carries an anchor side-car, but hand-labels take precedence.
Nothing in the calibration was changed.

## Known in the published paper

- Reference slips in the repo's `refs.bib`, noted 2026-09-30 (the published copy is not checked
  against them): `Hu2021V2e` prints "Thecvf.com" (it is CVPRW 2021);
  `Alberico2025Egocentric` pages 5025–5034 or 5064–5073; missing fields for Xu, Kalman and
  Bekolay, and Eliasmith's full title.
- Real-time claim: overlapped localiser and memory, 4.0–4.4 ms per 5 ms frame (1.15–1.26×);
  one after the other 5.2–5.5 ms (`scripts/time_live.py`).

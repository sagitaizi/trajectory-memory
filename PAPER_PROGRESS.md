# Paper progress

Which sections of `paper/main.tex` can be written now, which are blocked, and on what.
`PLAN.md` tracks the code; this tracks the manuscript. Deadlines are in `TIMELINE.md`.

## How this works

- `paper/` is **read-only to Claude**. Claude writes a section only when Sagi names it
  ("write §IV-B"), and touches nothing else in the file.
- Statuses: **blocked** → **writable** → **drafted** → **in review** → **reviewed** → **final**.
- *Drafted* means the prose is there. *In review* means Sagi is reviewing it (notes may sit
  in `%` comments). *Reviewed* means he has signed it off. *Final* means its "revisit"
  column is cleared.
- The trigger: when a phase's **Done when** in `PLAN.md` is met, this file is updated and
  the newly writable sections are named. Writing then keeps pace with the code instead of
  landing all at once in the last week.

## Section status

| §      | Section                    | Status      | Blocked on          | Revisit after            |
|--------|----------------------------|-------------|---------------------|--------------------------|
| —      | Abstract                   | placeholder | everything          | write last               |
| I      | Introduction               | reviewed    | —                   | results (claims)         |
| II     | Related Work               | reviewed    | —                   | —                        |
| III    | Method (opening paragraph) | reviewed    | —                   | —                        |
| III-A  | Problem Definition         | reviewed    | —                   | —                        |
| III-B  | Event Input                | reviewed    | —                   | —                        |
| III-C  | Spiking Localiser          | reviewed    | —                   | —                        |
| III-D  | Trajectory-Memory Network  | reviewed    | —                   | half-period fix          |
| IV-A   | Event-Camera Simulator     | writable    | —                   | —                        |
| IV-B   | Real Recordings            | writable    | —                   | —                        |
| IV-C   | Baselines                  | writable    | —                   | —                        |
| IV-D   | Evaluation Protocol        | drafted     | —                   | —                        |
| IV-E   | Metrics                    | writable    | —                   | —                        |
| V-A    | Results: Localiser         | drafted     | —                   | other held-out           |
| V-B    | Results: Prediction + path | drafted     | —                   | other held-out           |
| V-C    | Results: Deviation         | drafted     | —                   | other held-out           |
| V-D    | Results: Ablations         | drafted     | —                   | —                        |
| VI     | Discussion                 | writable    | —                   | —                        |
| VII    | Conclusion                 | writable    | —                   | —                        |
| —      | Acknowledgment             | writable    | —                   | —                        |
| Fig. 1 | System overview            | reviewed    | —                   | —                        |
| Fig. 2 | Recording setups           | writable    | —                   | —                        |

Notes on the rows:

- **III-C** — Sagi's twelve review notes sit as `%` comments in the subsection above the
  paragraphs they refer to; he deletes them himself.
- **IV-A** — comments only. Also takes what left III-C: the v2e citation and the 300 simulated
  recordings of about 15 s.
- **IV-C** — comments only, including the classical-centroid definition that left III-B
  (it must be defined before V-A uses it).
- **IV-B, IV-E, VI, VII** — heading only, or comments only.

### Where each section's material lives

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
- **IV-B, Fig. 2** — `RECORDING_LOG.md`; `corpus/sets.yaml`; `corpus/real/`.
- **IV-C** — `materials/02-methods/{kalman-periodic,rhythmic-dmp}.md`;
  `materials/01-literature/classical-baselines.md`.
- **IV-D** — `PLAN.md`: pretrain on simulation, freeze, test on real; `corpus/sets.yaml`.
- **IV-E** — `materials/02-methods/metrics.md`.
- **Fig. 1** — `docs/implementation-plan.md`.
- **§V (all)** — `scripts/paper_results.py run` (traces + scores, `runs/paper/`) and
  `scripts/paper_figures.py` (figures to `paper/figures/results_*.pdf`, tables to
  `runs/paper/tables.tex`). Scores and tables are kept in `docs/results/paper_*`.
  Held-out: the three pendulum clips, each scored once (2026-09-25); the other held-out
  clips are not labelled yet. Unused figure: `results_learning.pdf` (error vs time from
  the clip's start), `results_break_dev.pdf` (the development break).
- **Architecture figures** — `figures/architecture/`, TikZ sources + PDFs. `pipeline` is
  Fig. 1 (in `main.tex`, placed before §I so it floats to page 2; raster panels from
  `make_assets.py`). Not in `main.tex`: `clock_and_map` (§III-D, one column); `rig` (left
  out: the motors are never used). `system_overview` is superseded.

Every paper cited needs an entry in `BIBLIOGRAPHY.md` in the same change. `paper/refs.bib` is
Sagi's, managed outside the repo; the state it was in when he took it over is
`materials/01-literature/refs-snapshot-2026-09-14.bib` (18 entries, 9 marked `verify`).

## Pending

**Review** (drafted, not yet reviewed), in paper order: §IV-D, §V-A–D.

**Write**, in suggested order:

1. **§IV-C Baselines** — the classical centroid first; §V-A already uses it.
2. **§IV-E Metrics.** Definitions, not values: prediction error at the horizon, lock-on time,
   deviation ROC, median detection latency.
3. **§IV-A Simulator** — including the v2e citation and the simulated corpus moved out of §III-C.
4. **§IV-B Real Recordings** and **Fig. 2.** `RECORDING_LOG.md` has the setups, clip
   counts and durations. **Caveat:** every clip's ground truth is hand-labels plus
   interpolation, including the `wall/` (4a) clips — do not claim exact encoder ground truth
   for them. See "The 4a encoder ground truth was abandoned" below.
5. **§VI Discussion**, **§VII Conclusion**, then the **Abstract** last.

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

## Must revisit before submission

Writing early buys speed and costs staleness. Nothing is *final* until these are cleared:

- [ ] §I contribution claims match what the results actually support — in particular
      "no false positives" (development clips only so far) and "within a fraction of a
      motion cycle" (0.23 s median on development) must hold on the held-out clips.
- [ ] Final sweep validating every reference, once all writing is done. Known so far:
      missing `OKeefe1971Hippocampus`, `MacNeil2011Fine`, `Voelker2019Legendre` (each has a
      `% CITE:` comment in `main.tex`); `Alberico2025Egocentric` has the last author
      reversed and is the arXiv version, not CVPR Workshops 2025; `Harvey2002Forecasting` is
      a 2-page *Wilmott* piece, not Harvey's 1989 book (doi:10.1017/CBO9781107049994);
      `Eshraghian2022Training` and `Neftci2019Surrogate` are the arXiv versions, not
      *Proc. IEEE* 2023 and *IEEE Signal Processing Magazine* 2019 (BIBLIOGRAPHY.md has both).
- [x] Real-time claim: `scripts/time_live.py`, mains power, held-out pendulum runs. With
      localiser and memory overlapped (separate processes, identical outputs) a 5 ms frame
      costs 4.0–4.4 ms (1.15–1.26×); one after the other it is 5.2–5.5 ms (0.91–0.96×).
      §V opening quotes the overlapped number. Re-run it if either network changes.
- [ ] §III-B, §IV-A, §IV-C holes filled with real values.
- [ ] §IV-B describes ground truth as hand-labelled and interpolated, for every setup.
- [ ] Abstract written last, matching the results.
- [ ] `\nocite{*}` removed from `main.tex` (it currently lists every bib entry).

## Online-only tasks

Things that need a connection, so they can't be done on the plane:

- [ ] Register the paper on **EDAS** (`edas.info/N34632`) — title, abstract, authors.
      *Deferred by choice: no point registering a template.* The CFP sets no separate abstract
      deadline, so the only real date is Oct 10; do it once there is a real abstract, and
      leave margin in case EDAS or the connection misbehaves.
- [x] One clean LaTeX build (`latexmk -C && latexmk -pdf`) so MiKTeX fetches nothing lazily.
- [x] Paper PDFs pulled into `materials/pdfs/` (gitignored) — 14 of 18. Missing: Sussillo
      2009, Ijspeert 2002 and Itti 2001 (paywalled), Vacuum Spiker 2025 (no identifier on
      record). All four are already summarised in `materials/`.

The rest of `materials/04-checklists/paper-submission.md` still applies at the end.

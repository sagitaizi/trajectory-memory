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
| —      | Abstract                   | drafted     | —                   | —                        |
| I      | Introduction               | reviewed    | —                   | —                        |
| II     | Related Work               | reviewed    | —                   | —                        |
| III    | Method (opening paragraph) | reviewed    | —                   | —                        |
| III-A  | Problem Definition         | reviewed    | —                   | —                        |
| III-B  | Event Input                | reviewed    | —                   | —                        |
| III-C  | Spiking Localiser          | reviewed    | —                   | —                        |
| III-D  | Trajectory-Memory Network  | reviewed    | —                   | half-period fix          |
| IV-B   | Event-Camera Simulator     | reviewed    | —                   | —                        |
| IV-A   | Live Runs                  | reviewed    | —                   | —                        |
| IV-C   | Baselines                  | reviewed    | —                   | —                        |
| IV-D   | Evaluation Protocol        | reviewed    | —                   | —                        |
| IV-E   | Metrics                    | reviewed    | —                   | —                        |
| V-A    | Results: Localiser         | reviewed    | —                   | —                        |
| V-B    | Results: Prediction + path | reviewed    | —                   | —                        |
| V-C    | Results: Deviation         | reviewed    | —                   | —                        |
| V-D    | Results: Ablations         | reviewed    | —                   | —                        |
| VI     | Discussion                 | drafted     | —                   | —                        |
| VII    | Conclusion                 | drafted     | —                   | —                        |
| —      | Acknowledgment             | drafted     | —                   | —                        |
| Fig. 1 | System overview            | reviewed    | —                   | —                        |
| Fig. 2 | Recording setups           | dropped     | —                   | —                        |

Notes on the rows:

- **III-C** — Sagi's twelve review notes sit as `%` comments in the subsection above the
  paragraphs they refer to; he deletes them himself.

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
  Unused figure: `results_learning.pdf` (error vs time from the clip's start).
- **Architecture figures** — `figures/architecture/`, TikZ sources + PDFs. `pipeline` is
  Fig. 1 (in `main.tex`, placed before §I so it floats to page 2; raster panels from
  `make_assets.py`). Not in `main.tex`: `clock_and_map` (§III-D, one column); `rig` (left
  out: the motors are never used). `system_overview` is superseded.

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

**Review** (drafted, not yet reviewed): the Abstract, §VI, §VII. Then the fixes below, and one
last read of the whole paper.

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

- [x] §I contribution claims match the results: the bullets carry no numbers; "stays quiet
      on steady motion" holds on all 13 runs without a deviation.
- [ ] Abstract reviewed by Sagi. Its numbers match §V: 19 runs, 50 ms as accurate as the
      periodic filters, 6 of 6 deviations (median latency 0.32 s), one alarm outside them,
      Kalman 3 of 6 at 36 false alarms/min.
- [ ] `refs.bib` fixes still open (swept 2026-09-30; months, URLs, capitals, formatting slips,
      published versions and the Harvey book are done):
      - author names swapped: `Sho2026A` (Okazaki, Sho … Ota, Jun), `Righetti2006Dynamic`
        (Ijspeert, Auke Jan);
      - `Hu2021V2e` prints "Thecvf.com": make it `@inproceedings`, booktitle CVPRW 2021;
      - `Alberico2025Egocentric` pages 5025–5034 vs 5064–5073 on the CVF copy: check;
      - missing fields: Xu 16:2121–2129; Kalman pp. 35–45; Bekolay article 48; Eliasmith's
        full title ("…: Computation, Representation, and Dynamics in Neurobiological Systems").
      Re-exporting from the reference manager brings back `month`, `day` and `url` unless the
      export drops them.
- [ ] Back to 8 pages (dealt with at the end). With URLs gone, three references still sit on
      page 9. Tested on a copy: IEEE-abbreviated venue names save one, the "Learned memories"
      rewording and dropping Neftci's subtitle a second; about four lines of text cuts remain,
      one more once v2e's venue is fixed.
- [ ] Text fixes left from the 2026-09-30 review, not taken so far (typos, dashes, units,
      PES, notation and the two content points are done): grammar l.71 "actual deviation",
      l.87 "Neither works involve", l.102 "must alert it", l.148 "real life", l.258 "fired …
      fires"; US spellings l.91, 134, 148, 183; l.128 cites §IV where §IV-B is meant.
- [x] Real-time claim: `scripts/time_live.py`, mains power, held-out pendulum runs. With
      localiser and memory overlapped (separate processes, identical outputs) a 5 ms frame
      costs 4.0–4.4 ms (1.15–1.26×); one after the other it is 5.2–5.5 ms (0.91–0.96×).
      §VI ("What is spiking") quotes the overlapped number. Re-run it if either network changes.
- [x] §III-B, §IV-A, §IV-C holes filled with real values.
- [x] Ground truth described as hand-labelled and interpolated (§IV-A).
- [x] Abstract written last, matching the results.
- [x] `\nocite{*}` removed from `main.tex`; every bib entry is cited.

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

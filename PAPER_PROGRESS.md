# Paper progress

Which sections of `paper/main.tex` can be written now, which are blocked, and on what.
`PLAN.md` tracks the code; this tracks the manuscript. Deadlines are in `TIMELINE.md`.

## How this works

- `paper/` is **read-only to Claude**. Claude writes a section only when Sagi names it
  ("write §IV-B"), and touches nothing else in the file.
- Statuses: **blocked** → **writable** → **drafted** → **final**.
- *Drafted* means the prose is there. *Final* means its "revisit" column is cleared.
- The trigger: when a phase's **Done when** in `PLAN.md` is met, this file is updated and
  the newly writable sections are named. Writing then keeps pace with the code instead of
  landing all at once in the last week.

## Section status

`writable*` means write it now, but some values are still missing; the "revisit after" column
says which phase fills them.

| §      | Section                    | Status      | Blocked on          | Revisit after   |
|--------|----------------------------|-------------|---------------------|-----------------|
| —      | Abstract                   | placeholder | everything          | write last      |
| I      | Introduction               | drafted     | —                   | results         |
| II     | Related Work               | writable    | —                   | bib verify      |
| III-A  | Problem Setup              | writable    | —                   | —               |
| III-B  | Event Input                | writable*   | —                   | Phase B         |
| III-C  | Localiser (incl. training) | writable    | —                   | —               |
| III-D  | Memory network             | writable*   | —                   | half-period fix |
| III-E  | Deviation Score            | writable    | —                   | —               |
| III-F  | Evaluation Protocol        | writable    | —                   | —               |
| IV-A   | Simulator                  | writable*   | —                   | Phase A         |
| IV-B   | Real Recordings            | writable    | —                   | labelling       |
| IV-C   | Baselines                  | writable*   | —                   | Phase B         |
| IV-D   | Metrics                    | writable    | —                   | —               |
| V-A    | Results: Localiser         | drafted     | —                   | other held-out  |
| V-B    | Results: Prediction + path | drafted     | —                   | other held-out  |
| V-C    | Results: Deviation         | drafted     | —                   | other held-out  |
| V-D    | Results: Ablations         | drafted     | —                   | —               |
| VI     | Discussion                 | blocked     | results             | —               |
| VII    | Conclusion                 | blocked     | results             | —               |
| —      | Acknowledgment             | writable    | —                   | —               |
| Fig. 1 | System overview            | drafted     | —                   | —               |
| Fig. 2 | Recording setups           | writable    | —                   | —               |

### Where each section's material lives

- **I, II** — `materials/01-literature/` (six notes); `PLAN.md`'s novelty framing.
- **III-A** — `docs/implementation-plan.md`; `trajmem/trajectories.py`.
- **III-B** — `materials/02-methods/event-representations.md`.
- **III-C, III-D** — `docs/implementation-plan.md`; `PLAN.md` phase C and gate G-F;
  `docs/snn-experiments-summary.md`; `materials/01-literature/multi-timescale-memory.md`;
  specs in `docs/superpowers/specs/`.
- **V-D** — `PLAN.md` phase C (ablations); `docs/snn-experiments-log.md`;
  `materials/02-methods/legendre-memory-unit.md`.
- **III-E** — `materials/01-literature/deviation-and-novelty.md`.
- **III-F** — `PLAN.md`: pretrain on simulation, freeze, test on real.
- **IV-A** — `materials/02-methods/simulation-with-v2e.md`; `trajmem/simulate.py`; `params.yaml`.
- **IV-B, Fig. 2** — `RECORDING_LOG.md`; `corpus/sets.yaml`; `corpus/real/`.
- **IV-C** — `materials/02-methods/{kalman-periodic,rhythmic-dmp}.md`;
  `materials/01-literature/classical-baselines.md`.
- **IV-D** — `materials/02-methods/metrics.md`.
- **Fig. 1** — `docs/implementation-plan.md`.
- **§V (all)** — `scripts/paper_results.py run` (traces + scores, `runs/paper/`) and
  `scripts/paper_figures.py` (figures to `paper/figures/results_*.pdf`, tables to
  `runs/paper/tables.tex`). Scores and tables are kept in `docs/results/paper_*`.
  Held-out: the three pendulum clips, each scored once (2026-09-25); the other held-out
  clips are not labelled yet. Unused figure: `results_learning.pdf` (error vs time from
  the clip's start), `results_break_dev.pdf` (the development break).
- **Architecture figures** — `figures/architecture/`: `system_overview` (Fig. 1 candidate,
  full width) and `clock_and_map` (§III-D candidate, one column), TikZ sources + PDFs,
  not yet in `main.tex`.

Every paper cited needs an entry in `BIBLIOGRAPHY.md` in the same change. `paper/refs.bib` is
Sagi's, managed outside the repo; the state it was in when he took it over is
`materials/01-literature/refs-snapshot-2026-09-14.bib` (18 entries, 9 marked `verify`).

## Writable right now

In suggested order. Introduction goes first: it fixes what the paper claims, and Related Work
exists to position those claims — so §II is easier once §I is down.

1. **§I Introduction.** Motivation, problem, why spiking, contributions. Expand the abstract
   placeholder and `PLAN.md`'s "What this is"; write the contributions as the claims you
   intend to support and revisit once results exist.
2. **§II Related Work.** The largest chunk that results can never change. The six notes in
   `materials/01-literature/` carry the pipelines, the numbers, and the gap statement.
3. **§IV-B Real Recordings.** The session already happened; `RECORDING_LOG.md` has the setups,
   clip counts and durations. **Caveat:** every clip's ground truth is hand-labels plus
   interpolation, including the `wall/` (4a) clips — do not claim exact encoder ground truth
   for them. See "The 4a encoder ground truth was abandoned" below.
4. **§III-A Problem Setup.** Definitions only: position over time, what counts as repetition,
   the prediction horizon, what "deviation" means formally.
5. **§IV-D Metrics.** Definitions, not values: prediction error at the horizon, lock-on time,
   deviation ROC, median detection latency.
6. **§III-F Protocol.** Pretrain on simulation, freeze, evaluate on held-out real clips.
7. **Fig. 1.** Block diagram: events → frontend → localiser → memory → prediction + deviation.
   Won't change unless a block is renamed.

Rough page budget (8 pages including references): the above is ~3.5–4 pages.

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

## Blocked, and on what

- **§V-A–C** wait on the held-out set, scored once at the end. **§V-D** can be drafted
  now from the development-set numbers in `PLAN.md` (spiking vs arithmetic clock-and-map,
  and the two learned memories).
- **§VI–VII and the abstract** wait on results. Nothing to do but hold.

## Must revisit before submission

Writing early buys speed and costs staleness. Nothing is *final* until these are cleared:

- [ ] §I contribution claims match what the results actually support — in particular
      "no false positives" (development clips only so far) and "within a fraction of a
      motion cycle" (0.23 s median on development) must hold on the held-out clips.
- [ ] §II citations verified — 9 of 18 `refs.bib` entries are marked `note = {verify}`,
      including both closest-prior-art papers. Needs internet; deferred to post-flight.
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
- [ ] Verify the 9 `refs.bib` entries marked `verify` — *deferred to post-flight by choice*.
- [x] One clean LaTeX build (`latexmk -C && latexmk -pdf`) so MiKTeX fetches nothing lazily.
- [x] Paper PDFs pulled into `materials/pdfs/` (gitignored) — 14 of 18. Missing: Sussillo
      2009, Ijspeert 2002 and Itti 2001 (paywalled), Vacuum Spiker 2025 (no identifier on
      record). All four are already summarised in `materials/`.

The rest of `materials/04-checklists/paper-submission.md` still applies at the end.

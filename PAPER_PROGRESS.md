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

| § | Section | Status | Blocked on | Material | Revisit after |
|---|---|---|---|---|---|
| — | Abstract | placeholder | everything | — | results; write last |
| I | Introduction | writable | — | `PLAN.md` novelty framing, `materials/01-literature/` | results (the contribution claims) |
| II | Related Work | writable | — | `materials/01-literature/*` (6 notes) | bib verification (post-flight) |
| III-A | Problem Setup | writable | — | `docs/implementation-plan.md`, `trajmem/trajectories.py` | — |
| III-B | Event Input | writable, holes | window/polarity values ← `frontend.py` | `materials/02-methods/event-representations.md` | Phase B |
| III-C | Localiser | blocked | **G-F** framework decision | `docs/implementation-plan.md` | — |
| III-D | Trajectory-Memory Network | blocked | **G-F**; Phase C | `materials/02-methods/{reservoir-computing,legendre-memory-unit}.md` | — |
| III-E | Deviation Score | blocked | Phase D | `materials/01-literature/deviation-and-novelty.md` | — |
| III-F | Training + Evaluation Protocol | writable | — | `PLAN.md` (pretrain on sim, freeze, test on real) | — |
| IV-A | Event-Camera Simulator | writable, holes | contrast threshold + noise ← sim-vs-real check | `materials/02-methods/simulation-with-v2e.md`, `trajmem/simulate.py`, `params.yaml` | Phase A close |
| IV-B | Real Recordings | writable | — | `RECORDING_LOG.md`, `corpus/real/`, `RECORDING_PLAN.md` | anchor marking (see caveat) |
| IV-C | Baselines | writable, holes | tuned settings ← `baseline.py` | `materials/02-methods/{kalman-periodic,rhythmic-dmp}.md`, `classical-baselines.md` | Phase B |
| IV-D | Metrics | writable | — | `materials/02-methods/metrics.md` | — |
| V | Results | blocked | Phases C, D | — | — |
| VI | Discussion | blocked | results | — | — |
| VII | Conclusion | blocked | results | — | — |
| — | Acknowledgment | writable | — | — | — |
| Fig. 1 | System overview | writable | — | `docs/implementation-plan.md` | G-F (if it renames a block) |
| Fig. 2 | Recording setups | writable | — | `RECORDING_LOG.md` | — |

## Writable right now

In suggested order — biggest and most independent first:

1. **§II Related Work.** The largest chunk that results can never change. The six notes in
   `materials/01-literature/` carry the pipelines, the numbers, and the gap statement.
2. **§IV-B Real Recordings.** The session already happened; `RECORDING_LOG.md` has the setups,
   clip counts and durations. **Caveat:** do not yet claim exact encoder ground truth for the
   `wall/` clips — anchors are not marked and the pan/tilt sign convention is unconfirmed
   (`PAN_SIGN`/`TILT_SIGN` in `trajmem/data.py` are pinned by test, not by measurement).
3. **§III-A Problem Setup.** Definitions only: position over time, what counts as repetition,
   the prediction horizon, what "deviation" means formally.
4. **§IV-D Metrics.** Definitions, not values: prediction error at the horizon, lock-on time,
   deviation ROC, median detection latency.
5. **§I Introduction.** Motivation, problem, why spiking, contributions. Write the
   contributions as the claims you intend to support; revisit once results exist.
6. **§III-F Protocol.** Pretrain on simulation, freeze, evaluate on held-out real clips.
7. **Fig. 1.** Block diagram: events → frontend → localiser → memory → prediction + deviation.
   Won't change unless G-F renames a block.

Rough page budget (8 pages including references): the above is ~3.5–4 pages.

## Blocked, and on what

- **§III-C/D/E** wait on decision gate **G-F** (framework for the SNN core, `PLAN.md`).
  G-F needs a bake-off on simulated trajectories, so it needs a machine — it cannot be
  settled on the flight, and §III's middle stays blocked until it is.
- **§V–VII and the abstract** wait on results. Nothing to do but hold.

## Must revisit before submission

Writing early buys speed and costs staleness. Nothing is *final* until these are cleared:

- [ ] §I contribution claims match what the results actually support.
- [ ] §II citations verified — 9 of 18 `refs.bib` entries are marked `note = {verify}`,
      including both closest-prior-art papers. Needs internet; deferred to post-flight.
- [ ] §III-B, §IV-A, §IV-C holes filled with real values.
- [ ] §IV-B encoder ground-truth claim settled after anchor marking.
- [ ] Abstract written last, matching the results.
- [ ] `\nocite{*}` removed from `main.tex` (it currently lists every bib entry).

## Online-only tasks

Things that need a connection, so they can't be done on the plane:

- [ ] Register the paper on **EDAS** (`edas.info/N34632`) — title, abstract, authors. The PDF
      can be uploaded and replaced later, up to Oct 10. Use the new title.
- [ ] Verify the 9 `refs.bib` entries marked `verify` — *deferred to post-flight by choice*.
- [x] One clean LaTeX build (`latexmk -C && latexmk -pdf`) so MiKTeX fetches nothing lazily.
- [x] Paper PDFs pulled into `materials/pdfs/` (gitignored) — 14 of 18. Missing: Sussillo
      2009, Ijspeert 2002 and Itti 2001 (paywalled), Vacuum Spiker 2025 (no identifier on
      record). All four are already summarised in `materials/`.

The rest of `materials/04-checklists/paper-submission.md` still applies at the end.

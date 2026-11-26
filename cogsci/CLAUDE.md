# CLAUDE.md — CogSci 2027

Rules for the CogSci project, on top of the root `CLAUDE.md`. Plan and status:
`cogsci/PLAN.md`. Paper status: `cogsci/PAPER_PROGRESS.md`.

## What it is

A spiking model of how people follow a repeating motion with their eyes, an inner **clock**
with a learned map, driving a simulated eye, compared with published human pursuit data.
The question and the match-or-miss rule: `cogsci/materials/question.md`.

## Where things live

| Path | For |
|---|---|
| `cogsci/PLAN.md` | Phases, design questions D1–D6, situations S1–S8, calendar. |
| `cogsci/PAPER_PROGRESS.md` | Section status of the CogSci paper. |
| `cogsci/materials/` | The question, reading notes, the table of human numbers. |
| `cogsci/paper/` | The manuscript, in the CogSci template. |
| `pursuit/` | The code: eye, velocity, oscillator, clock, loop, measures. One file, one job. |
| `tests/pursuit/` | Its tests. |
| `scripts/cogsci/` | One run script per situation. |
| `runs/cogsci/` | Outputs; `runs/cogsci/numbers.csv` holds every number the paper quotes. |

## Rules

- **MCSoC stays as it was published.** `pursuit/` imports from `trajmem/` and does not
  change it. If `trajmem/` must change, the whole test suite still passes and
  `runs/paper/numbers.csv` still comes out the same. Nothing writes to `runs/paper/`.
- **The localiser is frozen** and the paper makes no claim about it.
- **Own interface.** The pursuit memory takes target velocity in and gives eye velocity out,
  so it gets its own protocol in `pursuit/`; `trajmem/model.py` stays the MCSoC one.
- **Adjust and test apart.** The model is adjusted on one part of the human data and tested on
  the rest; the split (D6) is fixed before any run, and nothing is adjusted on test data.
- **SNN design is decided together**, one question at a time (D1–D6 in `cogsci/PLAN.md`).
  Alone, Claude builds the plumbing: the eye loop, hiding the target, measures, scripts.
- **Tests first** for every `pursuit/` module.
- **`cogsci/paper/` is read-only to Claude** unless Sagi names the section to write.
- **Double-blind.** The paper names no author, lab, university, funding or repo, and cites
  the MCSoC paper in the third person. No internal clip, file or checkpoint names in the text.
- **Human numbers are sourced and free to read.** Every human value in the paper has a
  reference checked against the paper itself, recorded in `BIBLIOGRAPHY.md` with where it is
  used.
- **Words for this audience:** "population code over image position", not "place cells";
  "the clock" and "the map", not "method".

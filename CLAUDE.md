# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

SNN-based **trajectory memory** from event-camera input — Sagi Taizi, M.Sc. CS, Open University
of Israel (supervisor: Dr. Elishai Ezra Tsur).

A spiking network watches a target move in a repetitive path, builds a memory of that path,
predicts it a short time ahead, and flags when the motion stops matching. Offline only — no
motors, no closed loop. This is paper 4c in the main thesis repo's `docs/future-directions.md`,
targeted at **IEEE MCSoC 2026** (8 pages, submission 2026-10-10).

## The funnel

The SNN's input gets harder in two stages. Stage 1 is the goal; Stage 2 is upside.

| Stage | SNN input | SNN does | Role |
|---|---|---|---|
| 1 | accumulated event frames | localise target + memorise path + predict + flag deviation | **goal** |
| 2 | raw events, end-to-end | same, no framing step | upside, only if Stage 1 lands |
| fallback | classical centroid → memory core | memory/predict/deviation only, reported with an asterisk | ablation + safety net |

Re-learning a *new* path after a break, and the attention-span mechanism, are **out of scope**
(future work). Deviation detection is **in scope**.

## Where things live

Read `PLAN.md` first for what's next. `TIMELINE.md` has the dated milestones.

| Doc | For |
|---|---|
| `PLAN.md` | Phased plan + current status. Start here. |
| `TIMELINE.md` | Dated milestones with checkmarks. |
| `PAPER_PROGRESS.md` | Which paper sections are writable, blocked, drafted. The paper is written as we go. |
| `docs/implementation-plan.md` | Architecture, modules, data flow, frameworks. |
| `RECORDING_LOG.md` | Every real clip: setup, rate, what it shows, label status. |
| `corpus/sets.yaml` | Which real clips are development and which are held-out, with slices. |
| `BIBLIOGRAPHY.md` | Every paper referenced, with why/where. Update in the same change that uses it. |

## Environment

Conda env **`thesis`** (Python 3.14) — shared with the main thesis repo. Adds `snntorch`,
`v2e`. `nengo` already present. Pinned in `requirements.txt`. Run scripts with the env's
own interpreter (`D:\Programs\Anaconda\envs\thesis\python.exe`); `conda run` swallows flags.

Reuses code from the main thesis repo at `D:\Projects\Thesis` — `recording` (clip load/replay),
`camera` (calibration), `pipeline` (`accumulator`, `time_surface`). `_thesis_path.py` puts that
repo on `sys.path`; `conftest.py` imports it so tests pick it up. Do not copy those modules in.

## Architecture

Package `trajmem/`. One file, one job.

| Module | Role |
|---|---|
| `data.py` | Load a clip (recorded `.aedat4` or simulated) → uniform `Clip` object: events + true trajectory + deviation times. |
| `trajectories.py` | Analytic path specs (circle, ellipse, figure-8, Lissajous) + scripted deviations → exact position-vs-time. Pure math. |
| `simulate.py` | v2e wrapper: `TrajectorySpec` + camera config → synthetic event stream + exact ground truth. |
| `frontend.py` | Events → model input: `to_frames` (Stage 1), `to_raw` (Stage 2, stub), `to_position` centroid (baseline/fallback). |
| `model.py` | `Localiser` + `TrajectoryMemory` protocols. Two swappable jobs: localiser ("where now") and memory ("where next / is this normal"). The framework-agnostic boundary. |
| `baseline.py` | Non-SNN comparators (`PeriodicKalman`, `HarmonicFit`). Same interface as `model.py`. |
| `metrics.py` | Prediction error, lock-on time, deviation AUC / latency / false alarms. |
| `experiment.py` | `evaluate_clip` (the one loop every method goes through), `score_trace`, clip sets, `run_set`, `make_memory`. |
| `scripts/` | `make_sim_dataset`, `match_sim_real`, `run_experiment`, `check_localiser`, `replay_gt`, `make_tracks`, `make_frames`, `corpus_summary`, the `mark_*` labelling tools. `README.md` lists the commands. |

## Key conventions

- **Comments and code stay minimal.** A comment explains a non-obvious *why* in a line or two.
  No large comment blocks, no narrating past decisions or alternatives tried, no changelog prose
  in source. History belongs in git, not in `.py` files. Prefer clear names and small
  functions over explanatory comments.
- **Framework choice is deferred** (decision gate G-F in `PLAN.md`). Everything SNN goes behind
  `model.py`'s interface so the choice can change without touching the rest.
- **Ground truth**: exact from the spec on simulated clips (the apparent, lens-distorted
  position); hand-labels + interpolation on every real clip, motor-driven ones included.
- **Pretrain on simulation, test on real.** Real clips are never trained on; development
  clips are looked at while building, held-out clips are scored once.
- **Markdown stays minimal and current-state.** No dated narrative, no "was X, now Y", no
  changelog; history lives in git. A doc says what is, what is open, and where things are.
- **SNN design is decided together** — framework, architecture, training, parameters.
  Alone, Claude does plumbing, tooling and bookkeeping.
- **Never hand-author a calibration file.** Load the main repo's calibration through `camera`.

## Answering in sessions

- Keep every answer as simple as the topic allows. Open with what we're doing before any
  result or recommendation.
- Don't lean on references to earlier sessions or other files without saying, in a line, what
  they are — the reader may not have them in mind.
- Avoid big words and jargon unless they're the precise term. When a project term is
  unavoidable, define it the first time it appears in the answer.

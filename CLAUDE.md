# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Projects

SNN-based **trajectory memory** from event-camera input — Sagi Taizi, M.Sc. CS, Open University
of Israel (supervisor: Dr. Elishai Ezra Tsur). Two papers share this repo's data, localiser
and code. `ROADMAP.md` ties them together.

| Project | State | Docs | Code |
|---|---|---|---|
| **MCSoC 2026**: a spiking memory that learns a repeating path, predicts it and flags deviations (8 pages, IEEE) | accepted; frozen | root `PLAN.md`, `TIMELINE.md`, `PAPER_PROGRESS.md`; `paper/` | `trajmem/` |
| **CogSci 2027**: two spiking models of how people follow a repeating motion with their eyes (clock and replay), compared with human data (6 pages) | **active** | `cogsci/` (its own `CLAUDE.md`, plan, paper) | `pursuit/` (imports `trajmem/`) |

Offline only in both: no motors, no real closed loop (the CogSci eye is simulated).

### The MCSoC system

| Stage | SNN input | SNN does | Role |
|---|---|---|---|
| 1 | accumulated event frames | localise target + memorise path + predict + flag deviation | **the paper** |
| 2 | raw events, end-to-end | same, no framing step | future work |
| fallback | classical centroid → memory core | memory/predict/deviation only | ablation + safety net |

## Where things live

For CogSci work start at `cogsci/PLAN.md`; `TIMELINE.md` has the dated milestones of both.

| Doc | For |
|---|---|
| `PLAN.md` | MCSoC: phased plan and status. |
| `TIMELINE.md` | Dated milestones of both papers, with checkmarks. |
| `ROADMAP.md` | After acceptance: camera-ready, CogSci version, learning sessions; how the two papers are kept apart. |
| `cogsci/PLAN.md` | CogSci: the claim, design questions, reading, build, runs, outline, calendar. |
| `PAPER_PROGRESS.md` | MCSoC paper: section status, camera-ready list. CogSci's is `cogsci/PAPER_PROGRESS.md`. |
| `docs/implementation-plan.md` | Architecture, modules, data flow, frameworks. |
| `RECORDING_LOG.md` | Every real clip: setup, rate, what it shows, label status. |
| `corpus/sets.yaml` | Which real clips are development and which are held-out, with slices. |
| `BIBLIOGRAPHY.md` | Every paper referenced by either project, with why/where. Update in the same change that uses it. |

## Environment

Conda env **`thesis`** (Python 3.14) — shared with the main thesis repo. Adds `snntorch`,
`v2e`. `nengo` already present. Pinned in `requirements.txt`. Run scripts with the env's
own interpreter (`D:\Programs\Anaconda\envs\thesis\python.exe`); `conda run` swallows flags.

Reuses code from the main thesis repo at `D:\Projects\Thesis` — `recording` (clip load/replay),
`camera` (calibration), `pipeline` (`accumulator`, `time_surface`). `_thesis_path.py` puts that
repo on `sys.path`; `conftest.py` imports it so tests pick it up. Do not copy those modules in.

## Architecture

Package `trajmem/` (shared, and the MCSoC system). One file, one job. The CogSci package
`pursuit/` is described in `cogsci/CLAUDE.md`.

| Module | Role |
|---|---|
| `data.py` | Load a clip (recorded `.aedat4` or simulated) → uniform `Clip` object: events + true trajectory + deviation times. |
| `trajectories.py` | Analytic path specs (circle, ellipse, figure-8, Lissajous, pendulum) + scripted deviations → exact position-vs-time. Pure math. |
| `simulate.py` | v2e wrapper: `TrajectorySpec` + camera config → synthetic event stream + exact ground truth. |
| `frontend.py` | Events → model input: `to_frames` (Stage 1), `to_raw` (Stage 2, stub), `to_position` centroid (baseline/fallback). |
| `model.py` | `Localiser` + `TrajectoryMemory` protocols. Two swappable jobs: localiser ("where now") and memory ("where next / is this normal"). The framework-agnostic boundary. |
| `snn_localise.py` | Stage 1 localiser: spiking conv net over count frames (snnTorch), trained on sim. `localise.py` is its harness. |
| `snn_phasemap.py` | Stage 1 memory: the clock and map (Nengo-built LIF populations run in PyTorch, PES map). `phasemap.py` is its arithmetic reference. |
| `snn.py`, `snn_lmu.py`, `lmu.py` | Ablation memories (two-timescale LIF, LMU window); `lmu.py` also builds and steps the Nengo populations the memory uses. |
| `baseline.py` | Non-SNN comparators (`PeriodicKalman`, `HarmonicFit`, `Extrapolator`). Same interface as `model.py`. |
| `metrics.py` | Prediction error, path shape, lock-on time, deviation AUC / latency / false alarms, the ratcheting alarm. |
| `experiment.py` | `evaluate_clip` (the one loop every method goes through), `score_trace`, clip sets, `run_set`, `make_memory`, `make_localiser`. |
| `scripts/` | `make_sim_dataset`, `match_sim_real`, `evaluate`, `run_experiment`, `train_localiser`, `check_localiser`, `replay_gt`, `bench`, `run_live`, `time_live`, `make_tracks`, `make_frames`, `corpus_summary`, the `mark_*` labelling tools. `README.md` lists the commands. |

## Key conventions

- **Comments and code stay minimal.** A comment explains a non-obvious *why* in a line or two.
  No large comment blocks, no narrating past decisions or alternatives tried, no changelog prose
  in source. History belongs in git, not in `.py` files. Prefer clear names and small
  functions over explanatory comments.
- **Frameworks** (decision gate G-F in `PLAN.md`): snnTorch for the localiser; the memory's
  weights are built by Nengo and its neurons run in PyTorch. Everything SNN goes behind
  `model.py`'s interface so either can change without touching the rest.
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
